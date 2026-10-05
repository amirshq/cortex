import React from "react";
import ReactMarkdown from "react-markdown";

// Links in answers (news articles, sources) open in a NEW tab, so following
// one never navigates away from the chat. rel="noopener noreferrer" stops the
// opened page from controlling this tab.
const markdownComponents = {
  a: ({ node, ...props }) => <a {...props} target="_blank" rel="noopener noreferrer" />,
};

export default function MessageBubble({ role, content, timestamp }) {
  const isUser = role === "user";

  return (
    <div className={`message-row ${isUser ? "user-row" : "assistant-row"}`}>
      {!isUser && (
        <div className="msg-avatar assistant-avatar">🤖</div>
      )}

      <div className={`bubble ${isUser ? "user-bubble" : "assistant-bubble"}`}>
        {isUser ? (
          <p className="bubble-text">{content}</p>
        ) : (
          <div className="bubble-text bubble-markdown">
            <ReactMarkdown components={markdownComponents}>{content}</ReactMarkdown>
          </div>
        )}
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
