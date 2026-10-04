"""Step 0 — Load documents and cut them into chunks.

Why chunks? An LLM can't read your whole library at once, and a search works
better on small, focused pieces of text than on whole files.
"""

from pathlib import Path

from pypdf import PdfReader


def load_documents(folder):
    """Read every .txt, .md and .pdf file in a folder.

    Returns a list of {"source": file name, "text": the file's text}.
    """
    documents = []
    for path in sorted(Path(folder).iterdir()):
        if path.suffix in (".txt", ".md"):
            text = path.read_text(encoding="utf-8")
        elif path.suffix == ".pdf":
            text = "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
        else:
            continue  # skip anything else
        documents.append({"source": path.name, "text": text})
    return documents


def split_into_chunks(text, chunk_size, overlap):
    """Cut text into overlapping pieces of `chunk_size` characters.

    Example with chunk_size=10, overlap=3:
        "abcdefghijklmnop" -> ["abcdefghij", "hijklmnop"]
    The overlap ("hij") means a sentence cut at a boundary still appears
    whole in at least one chunk.
    """
    chunks = []
    step = chunk_size - overlap
    for start in range(0, len(text), step):
        chunk = text[start:start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        if start + chunk_size >= len(text):
            break
    return chunks
