"""Tests for the session-management controller endpoints.

These shipped in the "session management endpoints" commit with no tests —
the suite stayed at 158 because nothing new was covered. This closes that.

Consistent with test_controller.py, ChatHistoryManager is mocked: what's
under test is validation, status-code mapping and DTO shape, not SQL
(which tests/memory/test_chat_history_manager.py covers against real SQLite).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException, status

from src.api.controller import ChatController
from src.database.dto import DeleteSessionResponse, ListSessionsResponse


class TestListSessions:
    def test_returns_the_users_sessions(self):
        manager = MagicMock()
        manager.list_sessions.return_value = [
            {"id": "s1", "title": "First chat", "created_at": "2026-09-01T10:00:00"},
            {"id": "s2", "title": "Second chat", "created_at": "2026-09-02T10:00:00"},
        ]
        with patch("src.api.controller.ChatHistoryManager", return_value=manager):
            result = ChatController.list_sessions(user_id=1)

        assert isinstance(result, ListSessionsResponse)
        assert [s.id for s in result.sessions] == ["s1", "s2"]
        assert result.sessions[0].title == "First chat"

    def test_queries_with_the_user_id_as_a_string(self):
        """The API takes an int; SQLite stores TEXT. A mismatch here returns
        an empty list for every user instead of erroring."""
        manager = MagicMock()
        manager.list_sessions.return_value = []
        with patch("src.api.controller.ChatHistoryManager", return_value=manager):
            ChatController.list_sessions(user_id=42)

        manager.list_sessions.assert_called_once_with("42")

    def test_no_sessions_returns_an_empty_list(self):
        manager = MagicMock()
        manager.list_sessions.return_value = []
        with patch("src.api.controller.ChatHistoryManager", return_value=manager):
            assert ChatController.list_sessions(user_id=1).sessions == []

    def test_zero_user_id_is_rejected(self):
        with pytest.raises(HTTPException) as exc:
            ChatController.list_sessions(user_id=0)
        assert exc.value.status_code == status.HTTP_400_BAD_REQUEST

    def test_negative_user_id_is_rejected(self):
        with pytest.raises(HTTPException) as exc:
            ChatController.list_sessions(user_id=-5)
        assert exc.value.status_code == status.HTTP_400_BAD_REQUEST

    def test_invalid_user_id_does_not_touch_the_database(self):
        with patch("src.api.controller.ChatHistoryManager") as manager_cls:
            with pytest.raises(HTTPException):
                ChatController.list_sessions(user_id=0)
        manager_cls.assert_not_called()

    def test_storage_errors_become_500(self):
        manager = MagicMock()
        manager.list_sessions.side_effect = RuntimeError("db locked")
        with patch("src.api.controller.ChatHistoryManager", return_value=manager):
            with pytest.raises(HTTPException) as exc:
                ChatController.list_sessions(user_id=1)

        assert exc.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "Failed to list sessions" in exc.value.detail

    def test_validation_400_is_not_swallowed_into_a_500(self):
        """The bug this file's sibling endpoint had in send_message: a bare
        `except Exception` after the raise turns every 400 into a 500."""
        with pytest.raises(HTTPException) as exc:
            ChatController.list_sessions(user_id=0)
        assert exc.value.status_code == 400


class TestDeleteSession:
    def test_successful_delete_returns_success(self):
        manager = MagicMock()
        manager.delete_session.return_value = True
        with patch("src.api.controller.ChatHistoryManager", return_value=manager):
            result = ChatController.delete_session(user_id=1, session_id="s1")

        assert isinstance(result, DeleteSessionResponse)
        assert result.success is True
        assert result.message

    def test_passes_session_id_and_stringified_user_id(self):
        manager = MagicMock()
        manager.delete_session.return_value = True
        with patch("src.api.controller.ChatHistoryManager", return_value=manager):
            ChatController.delete_session(user_id=7, session_id="abc")

        manager.delete_session.assert_called_once_with("abc", "7")

    def test_unknown_session_returns_404(self):
        manager = MagicMock()
        manager.delete_session.return_value = False
        with patch("src.api.controller.ChatHistoryManager", return_value=manager):
            with pytest.raises(HTTPException) as exc:
                ChatController.delete_session(user_id=1, session_id="missing")

        assert exc.value.status_code == status.HTTP_404_NOT_FOUND

    def test_another_users_session_returns_404(self):
        """Ownership is enforced in SQL (WHERE id = ? AND user_id = ?), so a
        mismatched owner looks identical to a missing session — which is the
        right thing to expose, since it doesn't confirm the id exists."""
        manager = MagicMock()
        manager.delete_session.return_value = False
        with patch("src.api.controller.ChatHistoryManager", return_value=manager):
            with pytest.raises(HTTPException) as exc:
                ChatController.delete_session(user_id=999, session_id="someone-elses")

        assert exc.value.status_code == status.HTTP_404_NOT_FOUND

    def test_404_is_not_swallowed_into_a_500(self):
        """`raise HTTPException(404)` inside a try with a generic handler is
        exactly the shape that produced the send_message 400→500 bug."""
        manager = MagicMock()
        manager.delete_session.return_value = False
        with patch("src.api.controller.ChatHistoryManager", return_value=manager):
            with pytest.raises(HTTPException) as exc:
                ChatController.delete_session(user_id=1, session_id="missing")

        assert exc.value.status_code == 404
        assert "Failed to delete" not in str(exc.value.detail)

    def test_zero_user_id_is_rejected(self):
        with pytest.raises(HTTPException) as exc:
            ChatController.delete_session(user_id=0, session_id="s1")
        assert exc.value.status_code == status.HTTP_400_BAD_REQUEST

    def test_negative_user_id_is_rejected(self):
        with pytest.raises(HTTPException) as exc:
            ChatController.delete_session(user_id=-1, session_id="s1")
        assert exc.value.status_code == status.HTTP_400_BAD_REQUEST

    def test_empty_session_id_is_rejected(self):
        with pytest.raises(HTTPException) as exc:
            ChatController.delete_session(user_id=1, session_id="")
        assert exc.value.status_code == status.HTTP_400_BAD_REQUEST

    def test_whitespace_only_session_id_is_rejected(self):
        with pytest.raises(HTTPException) as exc:
            ChatController.delete_session(user_id=1, session_id="   ")
        assert exc.value.status_code == status.HTTP_400_BAD_REQUEST

    def test_invalid_input_does_not_touch_the_database(self):
        """A blank session_id reaching delete_session() would run
        `DELETE FROM messages WHERE session_id = ''` — harmless today, but
        the validation is what keeps it that way."""
        with patch("src.api.controller.ChatHistoryManager") as manager_cls:
            with pytest.raises(HTTPException):
                ChatController.delete_session(user_id=1, session_id="  ")
        manager_cls.assert_not_called()

    def test_storage_errors_become_500(self):
        manager = MagicMock()
        manager.delete_session.side_effect = RuntimeError("db locked")
        with patch("src.api.controller.ChatHistoryManager", return_value=manager):
            with pytest.raises(HTTPException) as exc:
                ChatController.delete_session(user_id=1, session_id="s1")

        assert exc.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "Failed to delete session" in exc.value.detail
