"""Where a bad request is rejected, and with which status.

Two layers validate input, and which one fires is part of the API contract:

    FastAPI / Pydantic → 422   the request doesn't fit the DTO or signature
                               (missing field, wrong type, malformed JSON)
    controller         → 400   the request fits, but breaks a business rule
                               (blank message, user_id <= 0, not a PDF)

Past validation, the error mapping is covered here too: a 403 or 404 passes
through untouched, a ValueError from the chat path becomes 400, and any other
failure becomes a 500 with an endpoint-specific detail.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

pytestmark = pytest.mark.integration


def assert_422_on(response, *loc_tail):
    assert response.status_code == 422, response.text
    locations = [tuple(error["loc"]) for error in response.json()["detail"]]
    assert any(loc[-len(loc_tail):] == loc_tail for loc in locations), locations


# ---------------------------------------------------------------------------
# 422 — FastAPI / Pydantic request validation
# ---------------------------------------------------------------------------
class TestChatRequestValidation:
    def test_missing_message(self, client, chat_business):
        assert_422_on(client.post("/api/v1/chat", json={"session_id": "s1"}), "body", "message")
        chat_business.assert_not_called()

    def test_null_message(self, client, chat_business):
        assert_422_on(client.post("/api/v1/chat", json={"message": None}), "body", "message")

    def test_non_string_message(self, client, chat_business):
        assert_422_on(client.post("/api/v1/chat", json={"message": 123}), "body", "message")

    def test_non_integer_user_id(self, client, chat_business):
        assert_422_on(client.post("/api/v1/chat", json={"message": "hi", "user_id": "seven"}),
                      "body", "user_id")

    def test_malformed_json_body(self, client, chat_business):
        response = client.post("/api/v1/chat", content=b'{"message": ',
                               headers={"content-type": "application/json"})

        assert response.status_code == 422
        assert response.json()["detail"][0]["type"] == "json_invalid"
        chat_business.assert_not_called()

    def test_json_array_instead_of_object(self, client, chat_business):
        assert client.post("/api/v1/chat", json=["hello"]).status_code == 422
        chat_business.assert_not_called()


class TestQueryAndPathValidation:
    def test_history_requires_user_id(self, client, history_business):
        assert_422_on(client.get("/api/v1/history"), "query", "user_id")
        history_business.assert_not_called()

    def test_history_user_id_must_be_an_integer(self, client, history_business):
        assert_422_on(client.get("/api/v1/history", params={"user_id": "abc"}), "query", "user_id")

    def test_sessions_requires_user_id(self, client, session_store):
        assert_422_on(client.get("/api/v1/sessions"), "query", "user_id")
        session_store.list_sessions.assert_not_called()

    def test_delete_session_requires_user_id(self, client, session_store):
        assert_422_on(client.delete("/api/v1/sessions/s1"), "query", "user_id")
        session_store.delete_session.assert_not_called()


class TestRagRequestValidation:
    def test_query_without_question(self, client, rag_query_business):
        assert_422_on(client.post("/api/v1/rag/query", json={}), "body", "question")
        rag_query_business.assert_not_called()

    def test_upload_without_a_file_part(self, client, ingest_business):
        assert_422_on(client.post("/api/v1/rag/upload"), "body", "file")
        ingest_business.assert_not_called()


# ---------------------------------------------------------------------------
# 400 — controller business rules
# ---------------------------------------------------------------------------
class TestBusinessRuleRejections:
    @pytest.mark.parametrize("message", ["", "   ", "\n\t "])
    def test_blank_chat_message(self, client, chat_business, message):
        response = client.post("/api/v1/chat", json={"message": message})

        assert response.status_code == 400
        assert response.json() == {"detail": "Message cannot be empty"}
        chat_business.assert_not_called()

    def test_blank_rag_question(self, client, rag_query_business):
        response = client.post("/api/v1/rag/query", json={"question": "  "})

        assert response.status_code == 400
        assert response.json() == {"detail": "Question cannot be empty"}
        rag_query_business.assert_not_called()

    @pytest.mark.parametrize("user_id", [0, -3])
    def test_non_positive_user_id_on_history(self, client, history_business, user_id):
        response = client.get("/api/v1/history", params={"user_id": user_id})

        assert response.status_code == 400
        assert response.json() == {"detail": "Invalid user_id"}
        history_business.assert_not_called()

    @pytest.mark.parametrize("user_id", [0, -3])
    def test_non_positive_user_id_on_sessions_never_opens_storage(self, client, monkeypatch, user_id):
        manager_cls = MagicMock()
        monkeypatch.setattr("src.api.controller.ChatHistoryManager", manager_cls)

        response = client.get("/api/v1/sessions", params={"user_id": user_id})

        assert response.status_code == 400
        manager_cls.assert_not_called()

    def test_blank_session_id_on_delete_never_opens_storage(self, client, monkeypatch):
        manager_cls = MagicMock()
        monkeypatch.setattr("src.api.controller.ChatHistoryManager", manager_cls)

        response = client.delete("/api/v1/sessions/%20%20", params={"user_id": 1})

        assert response.status_code == 400
        assert response.json() == {"detail": "Invalid session_id"}
        manager_cls.assert_not_called()

    @pytest.mark.parametrize("filename", ["notes.txt", "report.pdf.exe", "pdf"])
    def test_non_pdf_upload_is_rejected_before_anything_is_written(
            self, client, ingest_business, sandboxed_paths, filename):
        response = client.post("/api/v1/rag/upload",
                               files={"file": (filename, b"not a pdf", "application/octet-stream")})

        assert response.status_code == 400
        assert response.json() == {"detail": "Only PDF files are accepted"}
        ingest_business.assert_not_called()
        assert list(sandboxed_paths.uploads_dir.iterdir()) == []

    def test_upload_with_an_empty_filename_is_a_client_error(self, client, ingest_business, sandboxed_paths):
        """Whether FastAPI (422) or the controller (400) rejects it depends on
        how the multipart parser treats an empty filename; the contract is
        that it's refused as a client error and nothing is indexed."""
        response = client.post("/api/v1/rag/upload", files={"file": ("", b"%PDF-1.4", "application/pdf")})

        assert 400 <= response.status_code < 500
        ingest_business.assert_not_called()
        assert list(sandboxed_paths.uploads_dir.iterdir()) == []


# ---------------------------------------------------------------------------
# Error mapping past validation: 403 / 404 / 400-from-ValueError / 500
# ---------------------------------------------------------------------------
class TestErrorMapping:
    def test_403_from_the_history_business_layer_passes_through(self, client, monkeypatch):
        """Cortex has no auth. 403 exists only as a passthrough: an
        HTTPException raised below the controller must keep its status."""
        monkeypatch.setattr("src.api.controller.get_chat_history",
                            AsyncMock(side_effect=HTTPException(status_code=403, detail="Forbidden")))

        response = client.get("/api/v1/history", params={"user_id": 1})

        assert response.status_code == 403
        assert response.json() == {"detail": "Forbidden"}

    def test_deleting_an_unknown_session_is_404(self, client, session_store):
        session_store.delete_session.return_value = False

        response = client.delete("/api/v1/sessions/missing", params={"user_id": 1})

        assert response.status_code == 404
        assert response.json() == {"detail": "Session not found"}

    def test_value_error_from_chat_business_is_400(self, client, monkeypatch):
        monkeypatch.setattr("src.api.controller.process_chat_message",
                            AsyncMock(side_effect=ValueError("unsupported message format")))

        response = client.post("/api/v1/chat", json={"message": "hi"})

        assert response.status_code == 400
        assert response.json() == {"detail": "unsupported message format"}

    def test_chat_failure_is_500(self, client, monkeypatch):
        monkeypatch.setattr("src.api.controller.process_chat_message",
                            AsyncMock(side_effect=RuntimeError("redis unavailable")))

        response = client.post("/api/v1/chat", json={"message": "hi"})

        assert response.status_code == 500
        assert response.json() == {"detail": "Internal server error: redis unavailable"}

    def test_history_failure_is_500(self, client, monkeypatch):
        monkeypatch.setattr("src.api.controller.get_chat_history",
                            AsyncMock(side_effect=RuntimeError("db locked")))

        response = client.get("/api/v1/history", params={"user_id": 1})

        assert response.status_code == 500
        assert response.json() == {"detail": "Failed to retrieve chat history: db locked"}

    def test_list_sessions_failure_is_500(self, client, session_store):
        session_store.list_sessions.side_effect = RuntimeError("db locked")

        response = client.get("/api/v1/sessions", params={"user_id": 1})

        assert response.status_code == 500
        assert response.json() == {"detail": "Failed to list sessions: db locked"}

    def test_delete_session_failure_is_500(self, client, session_store):
        session_store.delete_session.side_effect = RuntimeError("db locked")

        response = client.delete("/api/v1/sessions/s1", params={"user_id": 1})

        assert response.status_code == 500
        assert response.json() == {"detail": "Failed to delete session: db locked"}

    def test_rag_query_failure_is_500(self, client, monkeypatch):
        monkeypatch.setattr("src.api.controller.query_rag",
                            AsyncMock(side_effect=RuntimeError("vector store unavailable")))

        response = client.post("/api/v1/rag/query", json={"question": "q?"})

        assert response.status_code == 500
        assert response.json() == {"detail": "RAG query failed: vector store unavailable"}

    def test_upload_failure_is_500(self, client, monkeypatch):
        monkeypatch.setattr("src.api.controller.ingest_pdfs",
                            AsyncMock(side_effect=RuntimeError("indexing crashed")))

        response = client.post("/api/v1/rag/upload",
                               files={"file": ("doc.pdf", b"%PDF-1.4", "application/pdf")})

        assert response.status_code == 500
        assert response.json() == {"detail": "PDF upload failed: indexing crashed"}


# ---------------------------------------------------------------------------
# Known bug — asserts the CORRECT behaviour, so it is expected to fail today.
# ---------------------------------------------------------------------------
class TestKnownIssueHistoryPaginationRange:
    """KNOWN BUG. The router takes limit/offset as plain ints and builds
    ChatHistoryRequest (ge=1, le=100 / ge=0) inside the handler. A range
    violation therefore raises inside the endpoint and surfaces as a 500,
    instead of being rejected as 422 at the request boundary.

    strict=True: when the bug is fixed these start passing, the run fails
    with XPASS, and the xfail marker should be removed.
    """

    @pytest.mark.xfail(strict=True, reason="KNOWN BUG: out-of-range limit/offset returns 500, not 422")
    @pytest.mark.parametrize("params", [{"limit": 0}, {"limit": 101}, {"offset": -1}])
    def test_out_of_range_pagination_is_rejected_as_422(self, client, history_business, params):
        response = client.get("/api/v1/history", params={"user_id": 1, **params})

        assert response.status_code == 422
