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
 * @param {number} limit   max 100 (the API's cap)
 * @param {number} offset  how many messages to skip (pagination)
 * @returns {Promise<{messages: Array, total: number}>}
 */
export async function fetchHistory(userId, sessionId, limit = 50, offset = 0) {
  const params = new URLSearchParams({
    user_id: userId,
    session_id: sessionId,
    limit,
    offset,
  });

  const res = await fetch(`${BASE}/history?${params}`);

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }

  return res.json();
}

/**
 * Upload a PDF file and trigger RAG indexing.
 * @param {File} file
 * @returns {Promise<{filename: string, docs_indexed: number, chunks_indexed: number, message: string}>}
 */
export async function uploadPdf(file) {
  const form = new FormData();
  form.append("file", file);

  const res = await fetch(`${BASE}/rag/upload`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }

  return res.json();
}

/**
 * List all sessions for a user.
 * @param {number} userId
 * @returns {Promise<{sessions: Array}>}
 */
export async function listSessions(userId) {
  const res = await fetch(`${BASE}/sessions?user_id=${userId}`);

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }

  return res.json();
}

/**
 * Delete a session.
 * @param {number} userId
 * @param {string} sessionId
 * @returns {Promise<{success: boolean, message: string}>}
 */
export async function deleteSession(userId, sessionId) {
  const res = await fetch(`${BASE}/sessions/${sessionId}?user_id=${userId}`, {
    method: "DELETE",
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }

  return res.json();
}

/**
 * Ask a question using the RAG pipeline.
 * @param {string} question
 * @returns {Promise<{answer: string, sources: Array}>}
 */
export async function ragQuery(question, userId = null) {
  const res = await fetch(`${BASE}/rag/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    // With a user id the backend saves the Q&A, so it survives reloads.
    body: JSON.stringify({ question, user_id: userId }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }

  return res.json();
}

/**
 * Fetch EVERY message of a session, oldest first, paging through the API
 * (which returns at most 100 per request).
 * @param {number} userId
 * @param {string} sessionId
 * @returns {Promise<Array<{role: string, content: string, timestamp: string}>>}
 */
export async function fetchAllHistory(userId, sessionId) {
  const pageSize = 100;
  const messages = [];
  for (;;) {
    const page = await fetchHistory(userId, sessionId, pageSize, messages.length);
    messages.push(...page.messages);
    if (page.messages.length < pageSize || messages.length >= page.total) return messages;
  }
}

async function getJson(url, options) {
  const res = await fetch(url, options);
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

/** PDFs currently in the RAG index: {documents: [{source_id, pages, chunks, indexed_at}]} */
export function listRagDocuments() {
  return getJson(`${BASE}/rag/documents`);
}

/** A user's saved PDF Q&A, oldest first: {items: [{id, question, answer, sources, created_at}]} */
export function fetchRagHistory(userId) {
  return getJson(`${BASE}/rag/history?user_id=${userId}`);
}

/** Delete a user's saved PDF Q&A: {deleted: number} */
export function clearRagHistory(userId) {
  return getJson(`${BASE}/rag/history?user_id=${userId}`, { method: "DELETE" });
}
