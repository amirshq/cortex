"""End-to-end chat workflows: HTTP in, real business wiring, real SQLite.

These are the user-facing journeys the unit and integration tiers can't prove,
because they mock the layer where the journey actually happens: a chat turn
must land in Redis, the vector store AND SQLite, show up in history and the
session list, and disappear when deleted.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage, ToolMessage

import src.api.router as router

pytestmark = pytest.mark.e2e

USER = 7


def chat(client, message, session_id="sess-1", user_id=USER):
    body = {"message": message, "user_id": user_id}
    if session_id is not None:
        body["session_id"] = session_id
    return client.post("/api/v1/chat", json=body)


def sessions(client, user_id=USER):
    return client.get("/api/v1/sessions", params={"user_id": user_id}).json()["sessions"]


def history(client, session_id="sess-1", user_id=USER):
    return client.get("/api/v1/history", params={"user_id": user_id, "session_id": session_id}).json()


class TestSessionLifecycle:
    def test_chat_then_history_then_sessions_then_delete(self, client, chat_stack):
        chat_stack.queue_replies(chat_stack.reply("Nice to meet you, Amir."))

        response = chat(client, "Hi, I'm Amir")
        assert response.status_code == 200
        assert (response.json()["reply"], response.json()["session_id"]) == ("Nice to meet you, Amir.", "sess-1")

        turn = history(client)
        assert turn["total"] == 2
        assert [(m["role"], m["content"]) for m in turn["messages"]] == \
               [("user", "Hi, I'm Amir"), ("assistant", "Nice to meet you, Amir.")]

        assert [(s["id"], s["title"]) for s in sessions(client)] == [("sess-1", "Hi, I'm Amir")]

        deleted = client.delete("/api/v1/sessions/sess-1", params={"user_id": USER})
        assert deleted.status_code == 200
        assert deleted.json()["success"] is True

        assert sessions(client) == []
        assert history(client) == {"messages": [], "total": 0, "session_id": "sess-1"}

    def test_follow_up_turns_join_the_same_session_and_reach_the_model(self, client, chat_stack):
        chat_stack.queue_replies(chat_stack.reply("First answer."), chat_stack.reply("Second answer."))

        chat(client, "First question")
        chat(client, "Second question")

        assert [m["content"] for m in history(client)["messages"]] == \
               ["First question", "First answer.", "Second question", "Second answer."]
        # The session keeps the title from its first message.
        assert [s["title"] for s in sessions(client)] == ["First question"]
        # Short-term memory: the earlier turn was sent to the model with the new one.
        sent = chat_stack.llm.calls[-1]
        assert "First question" in [m.content for m in sent if isinstance(m, HumanMessage)]


class TestMemoryFanOut:
    def test_one_turn_is_written_to_redis_the_vector_store_and_sqlite(self, client, chat_stack):
        chat_stack.queue_replies(chat_stack.reply("Noted: you like sailing."))

        chat(client, "I like sailing")

        assert chat_stack.redis.store["sess-1"] == [
            {"role": "user", "content": "I like sailing"},
            {"role": "assistant", "content": "Noted: you like sailing."},
        ]
        (row,) = chat_stack.memory_rows()
        assert row["metadata"]["user_id"] == str(USER)
        assert row["text"] == "user: I like sailing\nassistant: Noted: you like sailing."
        assert history(client)["total"] == 2

    def test_the_agent_recalls_an_earlier_session_through_its_tool(self, client, chat_stack):
        chat_stack.queue_replies(
            chat_stack.reply("Noted."),
            chat_stack.tool_call("search_vector_db", {"query": "hobbies"}),
            chat_stack.reply("You told me you like sailing."),
        )

        chat(client, "I like sailing", session_id="sess-1")
        response = chat(client, "What do I like?", session_id="sess-2")

        assert response.json()["reply"] == "You told me you like sailing."
        sent = chat_stack.llm.calls[-1]
        tool_results = [m for m in sent if isinstance(m, ToolMessage)]
        assert len(tool_results) == 1
        assert "I like sailing" in tool_results[0].content


class TestSessionOwnership:
    def test_a_user_cannot_delete_another_users_session(self, client, chat_stack):
        chat_stack.queue_replies(chat_stack.reply("Saved."))
        chat(client, "private note", user_id=7)

        response = client.delete("/api/v1/sessions/sess-1", params={"user_id": 8})

        assert response.status_code == 404
        assert response.json() == {"detail": "Session not found"}
        assert [s["id"] for s in sessions(client, user_id=7)] == ["sess-1"]

    def test_sessions_are_listed_per_user(self, client, chat_stack):
        chat_stack.queue_replies(chat_stack.reply("Hi seven."), chat_stack.reply("Hi eight."))

        chat(client, "hello", session_id="sess-a", user_id=7)
        chat(client, "hello", session_id="sess-b", user_id=8)

        assert [s["id"] for s in sessions(client, user_id=7)] == ["sess-a"]
        assert [s["id"] for s in sessions(client, user_id=8)] == ["sess-b"]


class TestRejectedRequestsLeaveNoTrace:
    def test_a_rate_limited_turn_reaches_neither_the_model_nor_storage(self, client, chat_stack, fake_clock):
        capacity = router._rate_limiter.capacity
        chat_stack.queue_replies(*[chat_stack.reply(f"answer {i}") for i in range(capacity)])
        for i in range(capacity):
            assert chat(client, f"question {i}").status_code == 200

        assert chat(client, "one too many").status_code == 429

        contents = [m["content"] for m in history(client)["messages"]]
        assert len(contents) == capacity * 2
        assert "one too many" not in contents
        assert len(chat_stack.llm.calls) == capacity

    def test_invalid_requests_reach_neither_the_model_nor_storage(self, client, chat_stack):
        assert client.post("/api/v1/chat", json={"user_id": USER}).status_code == 422
        assert chat(client, "   ").status_code == 400

        assert chat_stack.llm.calls == []
        assert chat_stack.redis.store == {}
        assert sessions(client) == []

    def test_a_model_failure_is_a_500_and_persists_nothing(self, client, chat_stack):
        chat_stack.queue_replies(RuntimeError("upstream timeout"))

        response = chat(client, "hello")

        assert response.status_code == 500
        assert response.json() == {"detail": "Internal server error: upstream timeout"}
        assert chat_stack.redis.store == {}
        assert chat_stack.memory_rows() == []
        assert sessions(client) == []


# ---------------------------------------------------------------------------
# Known bugs — each asserts the CORRECT behaviour, so each is expected to fail.
# strict=True: a fix turns them into XPASS failures, prompting marker removal.
# ---------------------------------------------------------------------------
class TestKnownIssues:
    @pytest.mark.xfail(strict=True, reason="KNOWN BUG: POST /chat echoes the request's session_id "
                                           "(null) instead of the session the server actually used")
    def test_response_names_the_session_the_server_used(self, client, chat_stack):
        chat_stack.queue_replies(chat_stack.reply("hi"))

        body = chat(client, "hello", session_id=None).json()

        # The server stored the turn under str(user_id)...
        assert history(client, session_id=str(USER))["total"] == 2
        # ...so that is the session id the client needs back.
        assert body["session_id"] == str(USER)

    @pytest.mark.xfail(strict=True, reason="KNOWN BUG: GET /history filters by session_id only, so any "
                                           "user_id can read any session's messages")
    def test_history_is_scoped_to_the_session_owner(self, client, chat_stack):
        chat_stack.queue_replies(chat_stack.reply("Saved."))
        chat(client, "private note", user_id=7)

        other_user = client.get("/api/v1/history", params={"user_id": 8, "session_id": "sess-1"}).json()

        assert other_user["total"] == 0

    @pytest.mark.xfail(strict=True, reason="KNOWN BUG: ChatHistoryManager.delete_session() deletes the "
                                           "messages BEFORE its ownership check, so another user's "
                                           "rejected delete (404) still wipes the owner's messages")
    def test_a_rejected_delete_by_another_user_leaves_the_owners_messages(self, client, chat_stack):
        chat_stack.queue_replies(chat_stack.reply("Saved."))
        chat(client, "private note", user_id=7)

        assert client.delete("/api/v1/sessions/sess-1", params={"user_id": 8}).status_code == 404

        assert history(client, user_id=7)["total"] == 2

    @pytest.mark.xfail(strict=True, reason="KNOWN BUG: DELETE /sessions removes SQLite rows only; the "
                                           "turns stay in Redis and the conversation vector store")
    def test_deleting_a_session_removes_its_turns_from_every_memory_tier(self, client, chat_stack):
        chat_stack.queue_replies(chat_stack.reply("Saved."))
        chat(client, "private note")

        assert client.delete("/api/v1/sessions/sess-1", params={"user_id": USER}).status_code == 200

        assert "sess-1" not in chat_stack.redis.store
        assert chat_stack.memory_rows() == []
