import React, { useState, useCallback } from "react";
import ChatWindow from "./components/ChatWindow.jsx";
import InputBar from "./components/InputBar.jsx";
import Sidebar from "./components/Sidebar.jsx";
import { sendMessage } from "./api/chatApi.js";

const USER_ID = 1;

let _msgId = 0;
function nextMsgId() { return ++_msgId; }

function makeSession() {
  return {
    id: `session-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
    title: "New conversation",
    messages: [],
    createdAt: new Date().toISOString(),
  };
}

const initialSession = makeSession();

export default function App() {
  const [sessions, setSessions]   = useState([initialSession]);
  const [activeId, setActiveId]   = useState(initialSession.id);
  const [isTyping, setIsTyping]   = useState(false);
  const [error, setError]         = useState(null);

  const activeSession = sessions.find((s) => s.id === activeId);

  // ── helpers ────────────────────────────────────────────────────────────
  const patchSession = useCallback((id, updater) => {
    setSessions((prev) => prev.map((s) => (s.id === id ? updater(s) : s)));
  }, []);

  // ── actions ────────────────────────────────────────────────────────────
  const handleNew = useCallback(() => {
    const s = makeSession();
    setSessions((prev) => [s, ...prev]);
    setActiveId(s.id);
    setError(null);
  }, []);

  const handleSelect = useCallback((id) => {
    setActiveId(id);
    setError(null);
  }, []);

  const handleDelete = useCallback((id) => {
    setSessions((prev) => {
      const next = prev.filter((s) => s.id !== id);
      if (next.length === 0) {
        const fresh = makeSession();
        setActiveId(fresh.id);
        return [fresh];
      }
      if (id === activeId) {
        setActiveId(next[0].id);
      }
      return next;
    });
  }, [activeId]);

  const handleSend = useCallback(async (text) => {
    const sessionIdAtSend = activeId;

    const userMsg = {
      id: nextMsgId(),
      role: "user",
      content: text,
      timestamp: new Date().toISOString(),
    };

    patchSession(sessionIdAtSend, (s) => ({
      ...s,
      title: s.messages.length === 0 ? text.slice(0, 42) : s.title,
      messages: [...s.messages, userMsg],
    }));

    setIsTyping(true);
    setError(null);

    try {
      const data = await sendMessage(text, sessionIdAtSend, USER_ID);

      patchSession(sessionIdAtSend, (s) => ({
        ...s,
        messages: [
          ...s.messages,
          {
            id: nextMsgId(),
            role: "assistant",
            content: data.reply,
            timestamp: data.timestamp || new Date().toISOString(),
          },
        ],
      }));
    } catch (err) {
      setError(err.message);
    } finally {
      setIsTyping(false);
    }
  }, [activeId, patchSession]);

  // ── render ─────────────────────────────────────────────────────────────
  return (
    <div className="app">
      <header className="app-header">
        <div className="header-inner">
          <div className="header-avatar">🤖</div>
          <div className="header-info">
            <h1>{activeSession?.title || "Personal Chatbot"}</h1>
            <div className="header-status">
              <span className="status-dot" />
              Online
            </div>
          </div>
        </div>
      </header>

      <div className="app-body">
        <Sidebar
          sessions={sessions}
          activeId={activeId}
          onSelect={handleSelect}
          onNew={handleNew}
          onDelete={handleDelete}
        />

        <main className="app-main">
          <ChatWindow
            messages={activeSession?.messages || []}
            isTyping={isTyping}
          />
          {error && <p className="error-banner">{error}</p>}
          <InputBar onSend={handleSend} disabled={isTyping} />
        </main>
      </div>
    </div>
  );
}
