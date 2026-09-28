import React, { useRef, useState } from "react";
import { ragQuery, uploadPdf } from "../api/chatApi.js";

// ── Icons ─────────────────────────────────────────────────────────────────

const UploadIcon = () => (
  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="16 16 12 12 8 16" />
    <line x1="12" y1="12" x2="12" y2="21" />
    <path d="M20.39 18.39A5 5 0 0 0 18 9h-1.26A8 8 0 1 0 3 16.3" />
  </svg>
);

const PdfIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
    <polyline points="14 2 14 8 20 8" />
  </svg>
);

const SendIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
    <line x1="22" y1="2" x2="11" y2="13" />
    <polygon points="22 2 15 22 11 13 2 9 22 2" />
  </svg>
);

const ChevronIcon = ({ open }) => (
  <svg
    width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor"
    strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"
    style={{ transform: open ? "rotate(180deg)" : "rotate(0deg)", transition: "transform 0.2s" }}
  >
    <polyline points="6 9 12 15 18 9" />
  </svg>
);

// ── Source card ────────────────────────────────────────────────────────────

function SourceCard({ source, index }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="rag-source-card">
      <button className="rag-source-header" onClick={() => setOpen((v) => !v)}>
        <span className="rag-source-label">
          Source {index + 1}
          {source.metadata?.source_id && (
            <span className="rag-source-file"> · {source.metadata.source_id}</span>
          )}
        </span>
        <span className="rag-source-score">score {source.score?.toFixed(3)}</span>
        <ChevronIcon open={open} />
      </button>
      {open && <p className="rag-source-text">{source.text}</p>}
    </div>
  );
}

// ── Main panel ─────────────────────────────────────────────────────────────

export default function RAGPanel() {
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [uploading, setUploading]         = useState(false);
  const [uploadError, setUploadError]     = useState(null);
  const [dragOver, setDragOver]           = useState(false);

  const [question, setQuestion]     = useState("");
  const [asking, setAsking]         = useState(false);
  const [conversations, setConversations] = useState([]); // [{id, question, answer, sources, error}]

  const fileInputRef = useRef(null);

  // ── upload logic ─────────────────────────────────────────────────────────
  const handleFile = async (file) => {
    if (!file || !file.name.toLowerCase().endsWith(".pdf")) {
      setUploadError("Please upload a PDF file.");
      return;
    }
    setUploading(true);
    setUploadError(null);
    try {
      const result = await uploadPdf(file);
      setUploadedFiles((prev) => [
        { name: file.name, chunks: result.chunks_indexed },
        ...prev.filter((f) => f.name !== file.name),
      ]);
    } catch (err) {
      setUploadError(err.message);
    } finally {
      setUploading(false);
    }
  };

  const onFileInput = (e) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
    e.target.value = "";
  };

  const onDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files?.[0];
    if (file) handleFile(file);
  };

  // ── ask logic ─────────────────────────────────────────────────────────────
  const handleAsk = async () => {
    const q = question.trim();
    if (!q || asking) return;

    const id = Date.now();
    setConversations((prev) => [...prev, { id, question: q, answer: null, sources: [], error: null }]);
    setQuestion("");
    setAsking(true);

    try {
      const result = await ragQuery(q);
      setConversations((prev) =>
        prev.map((c) =>
          c.id === id ? { ...c, answer: result.answer, sources: result.sources || [] } : c
        )
      );
    } catch (err) {
      setConversations((prev) =>
        prev.map((c) => (c.id === id ? { ...c, error: err.message } : c))
      );
    } finally {
      setAsking(false);
    }
  };

  const onKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleAsk();
    }
  };

  // ── render ────────────────────────────────────────────────────────────────
  return (
    <div className="rag-panel">

      {/* ── Upload section ── */}
      <section className="rag-section">
        <h2 className="rag-section-title">Upload PDF</h2>

        <div
          className={`rag-dropzone ${dragOver ? "rag-dropzone-over" : ""} ${uploading ? "rag-dropzone-loading" : ""}`}
          onClick={() => !uploading && fileInputRef.current?.click()}
          onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={onDrop}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf"
            style={{ display: "none" }}
            onChange={onFileInput}
          />
          {uploading ? (
            <div className="rag-dropzone-status">
              <div className="rag-spinner" />
              <span>Indexing PDF…</span>
            </div>
          ) : (
            <div className="rag-dropzone-idle">
              <UploadIcon />
              <span>Drop a PDF here or <strong>click to browse</strong></span>
            </div>
          )}
        </div>

        {uploadError && <p className="rag-error">{uploadError}</p>}

        {uploadedFiles.length > 0 && (
          <ul className="rag-file-list">
            {uploadedFiles.map((f) => (
              <li key={f.name} className="rag-file-item">
                <PdfIcon />
                <span className="rag-file-name">{f.name}</span>
                <span className="rag-file-meta">{f.chunks} chunks</span>
              </li>
            ))}
          </ul>
        )}
      </section>

      {/* ── Conversation history ── */}
      {conversations.map((conv) => (
        <section key={conv.id} className="rag-section rag-answer-section">
          <div className="rag-question-bubble">
            <p className="rag-question-text">{conv.question}</p>
          </div>

          {conv.error && <p className="rag-error">{conv.error}</p>}

          {!conv.answer && !conv.error && (
            <div className="rag-answer-bubble">
              <div className="rag-spinner" />
            </div>
          )}

          {conv.answer && (
            <>
              <div className="rag-answer-bubble">
                <p className="rag-answer-text">{conv.answer}</p>
              </div>
              {conv.sources.length > 0 && (
                <div className="rag-sources">
                  <p className="rag-sources-label">
                    {conv.sources.length} source{conv.sources.length !== 1 ? "s" : ""} used
                  </p>
                  {conv.sources.map((s, i) => (
                    <SourceCard key={i} source={s} index={i} />
                  ))}
                </div>
              )}
            </>
          )}
        </section>
      ))}

      {/* ── Question input ── */}
      <section className="rag-section">
        <h2 className="rag-section-title">Ask a Question</h2>
        <div className="rag-input-row">
          <textarea
            className="rag-textarea"
            rows={3}
            placeholder="What does the document say about…?"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={onKeyDown}
            disabled={asking}
          />
          <button
            className="rag-ask-btn"
            onClick={handleAsk}
            disabled={asking || !question.trim()}
          >
            {asking ? <div className="rag-spinner rag-spinner-sm" /> : <SendIcon />}
          </button>
        </div>
      </section>
    </div>
  );
}
