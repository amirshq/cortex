import React from "react";

export default function MessageBubble({ role, content, timestamp }) {
  const isUser = role === "user";

  return (
    <div className={`message-row ${isUser ? "user-row" : "assistant-row"}`}>
      {!isUser && (
        <div className="msg-avatar assistant-avatar">🤖</div>
      )}

      <div className={`bubble ${isUser ? "user-bubble" : "assistant-bubble"}`}>
        <p className="bubble-text">{content}</p>
        {timestamp && (
          <span className="bubble-time">
            {new Date(timestamp).toLocaleTimeString([], {
              hour: "2-digit",
              minute: "2-digit",
            })}
          </span>
        )}
      </div>

      {isUser && (
        <div className="msg-avatar user-avatar">You</div>
      )}
    </div>
  );
}
