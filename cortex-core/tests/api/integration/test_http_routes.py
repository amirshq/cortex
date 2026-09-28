"""Every /api/v1 route, exercised over real HTTP.

Unlike tests/api/test_controller*.py, nothing here calls a controller method
directly. Each request travels the production path:

    TestClient → Prometheus middleware → CORS → router
               → rate-limit dependency (POST /chat only) → controller
               → business entry point (mocked)

So these tests verify what the controller unit tests bypass: routing,
parameter binding, the /api/v1 prefix, HTTP method enforcement and response
serialisation.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from src.database.dto import ChatHistoryRequest, ChatMessageRequest

pytestmark = pytest.mark.integration


class TestChatRoute:
    def test_body_is_bound_into_chat_message_request(self, client, chat_business):
        response = client.post("/api/v1/chat", json={
            "message": "Hello", "user_id": 7, "session_id": "s1", "context": {"tz": "UTC"},
        })

        assert response.status_code == 200
        (request,), _ = chat_business.call_args
        assert isinstance(request, ChatMessageRequest)
        assert (request.message, request.user_id, request.session_id, request.context) == \
               ("Hello", 7, "s1", {"tz": "UTC"})

    def test_response_body_matches_chat_message_response(self, client, chat_business):
        body = client.post("/api/v1/chat", json={"message": "Hello", "session_id": "s1"}).json()

        assert set(body) == {"reply", "session_id", "timestamp", "model_used", "tokens_used"}
        assert (body["reply"], body["session_id"], body["model_used"], body["tokens_used"]) == \
               ("Hi there", "s1", "gpt-4o", 12)
        datetime.fromisoformat(body["timestamp"])  # serialised as ISO-8601

    def test_unknown_body_fields_are_ignored_not_rejected(self, client, chat_business):
        """The DTOs set no `extra=` policy, so Pydantic's default applies."""
        response = client.post("/api/v1/chat", json={"message": "Hello", "mood": "curious"})

        assert response.status_code == 200
        (request,), _ = chat_business.call_args
        assert not hasattr(request, "mood")


class TestHistoryRoute:
    def test_query_parameters_are_bound_into_chat_history_request(self, client, history_business):
        response = client.get("/api/v1/history",
                              params={"user_id": 3, "session_id": "s1", "limit": 20, "offset": 5})

        assert response.status_code == 200
        (request,), _ = history_business.call_args
        assert isinstance(request, ChatHistoryRequest)
        assert (request.user_id, request.session_id, request.limit, request.offset) == (3, "s1", 20, 5)

    def test_pagination_defaults_apply_when_omitted(self, client, history_business):
        client.get("/api/v1/history", params={"user_id": 3})

        (request,), _ = history_business.call_args
        assert (request.session_id, request.limit, request.offset) == (None, 50, 0)

    def test_response_body_matches_chat_history_response(self, client, history_business):
        body = client.get("/api/v1/history", params={"user_id": 3}).json()

        assert body == {
            "messages": [{"role": "user", "content": "Hello", "timestamp": "2026-09-16T10:00:00"}],
            "total": 1,
            "session_id": "s1",
        }


class TestListSessionsRoute:
    def test_user_id_reaches_storage_as_a_string(self, client, session_store):
        """SQLite stores user_id as TEXT; an int here would match no rows."""
        response = client.get("/api/v1/sessions", params={"user_id": 42})

        assert response.status_code == 200
        session_store.list_sessions.assert_called_once_with("42")

    def test_response_carries_only_the_session_info_fields(self, client, session_store):
        session_store.list_sessions.return_value = [{
            "id": "s1", "title": "First chat", "created_at": "2026-09-01T10:00:00",
            "user_id": "42", "internal_note": "must not leak",
        }]

        body = client.get("/api/v1/sessions", params={"user_id": 42}).json()

        assert body == {"sessions": [{"id": "s1", "title": "First chat", "created_at": "2026-09-01T10:00:00"}]}


class TestDeleteSessionRoute:
    def test_path_and_query_parameters_are_bound(self, client, session_store):
        response = client.delete("/api/v1/sessions/abc-123", params={"user_id": 7})

        assert response.status_code == 200
        assert response.json() == {"success": True, "message": "Session deleted successfully"}
        session_store.delete_session.assert_called_once_with("abc-123", "7")

    def test_url_encoded_session_id_is_decoded_before_reaching_storage(self, client, session_store):
        client.delete("/api/v1/sessions/chat%20one", params={"user_id": 7})

        session_store.delete_session.assert_called_once_with("chat one", "7")


class TestRagQueryRoute:
    def test_question_reaches_query_rag(self, client, rag_query_business):
        response = client.post("/api/v1/rag/query", json={"question": "How long is the warranty?"})

        assert response.status_code == 200
        rag_query_business.assert_awaited_once_with("How long is the warranty?")

    def test_answer_and_sources_are_returned_intact(self, client, rag_query_business):
        body = client.post("/api/v1/rag/query", json={"question": "How long is the warranty?"}).json()

        assert body == {
            "answer": "The warranty is 24 months.",
            "sources": [{"text": "Warranty: 24 months.", "metadata": {"source_id": "manual.pdf"}, "score": 0.91}],
        }


class TestRagUploadRoute:
    def test_multipart_pdf_is_saved_and_handed_to_ingestion(self, client, ingest_business, sandboxed_paths):
        content = b"%PDF-1.4\n% integration test\n"

        response = client.post("/api/v1/rag/upload",
                               files={"file": ("manual.pdf", content, "application/pdf")})

        assert response.status_code == 200
        assert response.json() == {
            "filename": "manual.pdf", "docs_indexed": 1, "chunks_indexed": 12,
            "table_ocr_enabled": True, "message": "PDF indexed successfully",
        }
        assert (sandboxed_paths.uploads_dir / "manual.pdf").read_bytes() == content
        ingest_business.assert_awaited_once_with(sandboxed_paths.uploads_dir)

    def test_a_new_upload_replaces_previous_pdfs(self, client, ingest_business, sandboxed_paths):
        previous = sandboxed_paths.uploads_dir / "old-report.pdf"
        previous.write_bytes(b"%PDF-1.4 old")

        client.post("/api/v1/rag/upload", files={"file": ("new-report.pdf", b"%PDF-1.4 new", "application/pdf")})

        assert not previous.exists()
        assert (sandboxed_paths.uploads_dir / "new-report.pdf").exists()


class TestRouting:
    VERSIONED_OPERATIONS = {
        ("POST", "/api/v1/chat"),
        ("GET", "/api/v1/history"),
        ("GET", "/api/v1/sessions"),
        ("DELETE", "/api/v1/sessions/{session_id}"),
        ("POST", "/api/v1/rag/query"),
        ("POST", "/api/v1/rag/upload"),
    }

    def test_openapi_lists_exactly_the_six_versioned_operations(self, client):
        paths = client.get("/openapi.json").json()["paths"]
        versioned = {(method.upper(), path)
                     for path, operations in paths.items() if path.startswith("/api/v1")
                     for method in operations}

        assert versioned == self.VERSIONED_OPERATIONS

    @pytest.mark.parametrize("method,path", [
        ("POST", "/chat"), ("GET", "/history"), ("GET", "/sessions"),
        ("DELETE", "/sessions/s1"), ("POST", "/rag/query"), ("POST", "/rag/upload"),
    ])
    def test_routes_do_not_exist_without_the_api_v1_prefix(self, client, method, path):
        assert client.request(method, path).status_code == 404

    @pytest.mark.parametrize("method,path", [
        ("GET", "/api/v1/chat"), ("PUT", "/api/v1/chat"), ("POST", "/api/v1/history"),
        ("POST", "/api/v1/sessions"), ("DELETE", "/api/v1/rag/query"), ("GET", "/api/v1/rag/upload"),
    ])
    def test_wrong_http_method_is_405(self, client, method, path):
        assert client.request(method, path).status_code == 405

    def test_unknown_route_is_404(self, client):
        response = client.get("/api/v1/does-not-exist")

        assert response.status_code == 404
        assert response.json() == {"detail": "Not Found"}
