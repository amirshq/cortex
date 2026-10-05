import React, { useState, useCallback, useEffect } from "react";
import ChatWindow from "./components/ChatWindow.jsx";
import InputBar from "./components/InputBar.jsx";
import Sidebar from "./components/Sidebar.jsx";
import ModeSelector from "./components/ModeSelector.jsx";
import RAGPanel from "./components/RAGPanel.jsx";
import { sendMessage, listSessions, deleteSession, fetchAllHistory } from "./api/chatApi.js";

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
  const [mode, setMode]           = useState("chatbot"); // "chatbot" | "rag"
  const [sessions, setSessions]   = useState([initialSession]);
  const [activeId, setActiveId]   = useState(initialSession.id);
  const [isTyping, setIsTyping]   = useState(false);
  const [error, setError]         = useState(null);

  // Load sessions on app startup
  useEffect(() => {
    (async () => {
      try {
        const data = await listSessions(USER_ID);
        if (data.sessions && data.sessions.length > 0) {
          // Convert backend sessions to frontend format
          // Saved sessions arrive WITHOUT their messages (the list endpoint
          // returns titles only). `historyLoaded: false` marks them so the
          // messages are fetched the first time the session is opened.
          const loadedSessions = data.sessions.map(s => ({
            id: s.id,
            title: s.title,
            messages: [],
            createdAt: s.created_at,
            historyLoaded: false,
          }));
          setSessions(loadedSessions);
          setActiveId(loadedSessions[0].id);
        }
      } catch (err) {
        console.warn("Failed to load sessions:", err);
        // Fall back to initial session
      }
    })();
  }, []);

  const activeSession = sessions.find((s) => s.id === activeId);

  // Load the open session's saved messages once, the first time it's shown.
  // (Before this, every saved chat opened empty: the messages were in the
  // database, but nothing ever requested them.)
  useEffect(() => {
    if (!activeSession || activeSession.historyLoaded !== false) return;
    const id = activeSession.id;
    setSessions((prev) => prev.map((s) => (s.id === id ? { ...s, historyLoaded: "loading" } : s)));
    (async () => {
      try {
        const saved = await fetchAllHistory(USER_ID, id);
        setSessions((prev) => prev.map((s) => (s.id !== id ? s : {
          ...s,
          historyLoaded: true,
          // Saved messages first, then anything sent while they were loading.
          messages: [
            ...saved.map((m) => ({ id: nextMsgId(), role: m.role, content: m.content, timestamp: m.timestamp })),
            ...s.messages,
          ],
        })));
      } catch (err) {
        setSessions((prev) => prev.map((s) => (s.id === id ? { ...s, historyLoaded: false } : s)));
        setError(`Couldn't load this conversation: ${err.message}`);
      }
    })();
  }, [activeSession]);

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
    (async () => {
      try {
        await deleteSession(USER_ID, id);
      } catch (err) {
        console.warn("Failed to delete session on backend:", err);
        // Continue with local deletion anyway
      }
    })();

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
      // Only a brand-new chat takes its first message as the title; saved
      // chats (historyLoaded set) keep theirs even before messages load.
      title: s.messages.length === 0 && s.historyLoaded === undefined ? text.slice(0, 42) : s.title,
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

  const handleModeSelect = useCallback((newMode) => {
    setMode(newMode);
    setError(null);
  }, []);

  // ── render ─────────────────────────────────────────────────────────────
  return (
    <div className="app">
      <header className="app-header">
        <div className="header-inner">
          <div className="header-avatar">🤖</div>
          <div className="header-info">
            <h1>{mode === "rag" ? "RAG · PDF Q&A" : (activeSession?.title || "Cortex")}</h1>
            <div className="header-status">
              <span className="status-dot" />
              Online
            </div>
          </div>
          <ModeSelector mode={mode} onSelect={handleModeSelect} />
        </div>
      </header>

      <div className="app-body">
        {mode === "chatbot" && (
          <Sidebar
            sessions={sessions}
            activeId={activeId}
            onSelect={handleSelect}
            onNew={handleNew}
            onDelete={handleDelete}
          />
        )}

        <main className="app-main">
          {mode === "chatbot" && (
            <>
              <ChatWindow
                messages={activeSession?.messages || []}
                isTyping={isTyping}
              />
              {error && <p className="error-banner">{error}</p>}
              <InputBar onSend={handleSend} disabled={isTyping} />
            </>
          )}
          {/* Always mounted, only hidden: switching to the chatbot used to
              unmount this panel, which threw away the file list, the Q&A, and
              the progress of an upload still running. */}
          <div style={{ display: mode === "rag" ? "contents" : "none" }}>
            <RAGPanel userId={USER_ID} />
          </div>
        </main>
      </div>
    </div>
  );
}
