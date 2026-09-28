import React from "react";

const PlusIcon = () => (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
    <line x1="12" y1="5" x2="12" y2="19" />
    <line x1="5" y1="12" x2="19" y2="12" />
  </svg>
);

const ChatBubbleIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
  </svg>
);

const TrashIcon = () => (
  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="3 6 5 6 21 6" />
    <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6" />
    <path d="M10 11v6M14 11v6" />
    <path d="M9 6V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2" />
  </svg>
);

function formatDate(iso) {
  const d = new Date(iso);
  const now = new Date();
  const diffDays = Math.floor((now - d) / 86400000);
  if (diffDays === 0) return "Today";
  if (diffDays === 1) return "Yesterday";
  return d.toLocaleDateString([], { month: "short", day: "numeric" });
}

export default function Sidebar({ sessions, activeId, onSelect, onNew, onDelete }) {
  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <span className="sidebar-label">Conversations</span>
        <button className="new-chat-btn" onClick={onNew} title="New chat">
          <PlusIcon />
        </button>
      </div>

      <nav className="session-list">
        {sessions.length === 0 && (
          <p className="sidebar-empty">No conversations yet</p>
        )}

        {sessions.map((s) => (
          <div
            key={s.id}
            className={`session-item ${s.id === activeId ? "session-active" : ""}`}
            onClick={() => onSelect(s.id)}
          >
            <span className="session-icon"><ChatBubbleIcon /></span>
            <span className="session-meta">
              <span className="session-title">{s.title}</span>
              <span className="session-date">{formatDate(s.createdAt)}</span>
            </span>
            <button
              className="session-delete"
              onClick={(e) => { e.stopPropagation(); onDelete(s.id); }}
              title="Delete"
            >
              <TrashIcon />
            </button>
          </div>
        ))}
      </nav>
    </aside>
  );
}
