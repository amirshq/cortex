import React from "react";

const ChatIcon = () => (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
  </svg>
);

const DocIcon = () => (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
    <polyline points="14 2 14 8 20 8" />
    <line x1="16" y1="13" x2="8" y2="13" />
    <line x1="16" y1="17" x2="8" y2="17" />
    <polyline points="10 9 9 9 8 9" />
  </svg>
);

export default function ModeSelector({ mode, onSelect }) {
  return (
    <div className="mode-selector">
      <button
        className={`mode-tab ${mode === "chatbot" ? "mode-tab-active" : ""}`}
        onClick={() => onSelect("chatbot")}
      >
        <ChatIcon />
        Chatbot
      </button>
      <button
        className={`mode-tab ${mode === "rag" ? "mode-tab-active" : ""}`}
        onClick={() => onSelect("rag")}
      >
        <DocIcon />
        RAG · PDF Q&amp;A
      </button>
    </div>
  );
}
