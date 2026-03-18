const BASE = "/api/v1";

/**
 * Send a message to the chatbot.
 * @param {string} message
 * @param {string} sessionId
 * @param {number|null} userId
 * @returns {Promise<{reply: string, model_used: string, timestamp: string}>}
 */
export async function sendMessage(message, sessionId, userId = null) {
  const res = await fetch(`${BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      message,
      session_id: sessionId,
      user_id: userId,
    }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }

  return res.json();
}

/**
 * Fetch chat history for a session.
 * @param {number} userId
 * @param {string} sessionId
 * @param {number} limit
 * @returns {Promise<{messages: Array, total: number}>}
 */
export async function fetchHistory(userId, sessionId, limit = 50) {
  const params = new URLSearchParams({
    user_id: userId,
    session_id: sessionId,
    limit,
  });

  const res = await fetch(`${BASE}/history?${params}`);

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }

  return res.json();
}
