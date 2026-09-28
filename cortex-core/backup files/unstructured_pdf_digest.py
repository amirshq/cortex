import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from unstructured.partition.pdf import partition_pdf
from openai import OpenAI

# Active the table processing module if available
TABLE_PROCESSING_AVAILABLE = False
try:
    # Try relative import first (when imported as module)
    from .process_table_images import extract_table_text_from_images
    TABLE_PROCESSING_AVAILABLE = True
except (ImportError, ValueError) as e:
    try:
        # Fallback to absolute import (when run as script)
        import sys
        from pathlib import Path
        # Add parent directory to path for absolute import
        script_dir = Path(__file__).parent
        if str(script_dir) not in sys.path:
            sys.path.insert(0, str(script_dir))
        from process_table_images import extract_table_text_from_images
        TABLE_PROCESSING_AVAILABLE = True
    except (ImportError, ValueError) as e2:
        TABLE_PROCESSING_AVAILABLE = False
        if "UNSTRUCTURED_API_KEY" in str(e2):
            print(f"⚠️  Table processing unavailable: UNSTRUCTURED_API_KEY not set")
            print("   Set UNSTRUCTURED_API_KEY in .env to enable table extraction from images")
        else:
            print(f"⚠️  Table processing unavailable: {e2}")


# ============================================================
# Environment
# ============================================================
load_dotenv()
assert os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY not found in .env"

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / "data"
IMAGE_DIR = SCRIPT_DIR / "artifacts" / "images"

IMAGE_DIR.mkdir(parents=True, exist_ok=True)

# Find all PDF files in the data directory
pdf_files = list(DATA_DIR.glob("*.pdf"))
if not pdf_files:
    raise FileNotFoundError(f"No PDF files found in {DATA_DIR}")

print(f"\n📚 Found {len(pdf_files)} PDF file(s) to process:")
for pdf_file in pdf_files:
    print(f"   - {pdf_file.name}")

# ============================================================
# Main Business Logic for PDF Extraction using Unstructured
# ===========================================================

all_elements = []
all_pdf_texts = []

for pdf_file in pdf_files:
    print(f"\n{'=' * 80}")
    print(f"📄 Processing: {pdf_file.name}")
    print(f"{'=' * 80}")
    
    elements = partition_pdf(
        filename=str(pdf_file),
        strategy="hi_res",
        skip_infer_table_types=False,
        languages=["eng"],                  
        extract_images_in_pdf=True,
        extract_image_block_types=["Image", "Table"],
        extract_image_block_to_payload=False,
        extract_image_block_output_dir=str(IMAGE_DIR),
    )
    
    print(f"✅ Extracted {len(elements)} elements from {pdf_file.name}")
    all_elements.extend(elements)
    
    # Store PDF-specific text for reference
    all_pdf_texts.append({
        "filename": pdf_file.name,
        "elements": elements,
        "count": len(elements)
    })

print(f"\n✅ Total elements extracted from all PDFs: {len(all_elements)}\n")

# Show sample of elements from all PDFs
print("Sample of extracted elements (from all PDFs):")
for i, el in enumerate(all_elements[:10]):
    text_preview = (el.text or "").strip()[:200]
    print(f"[{i}] CATEGORY={el.category} | Text length: {len(text_preview)}")
    print(f"    Preview: {text_preview}")
    print("-" * 80)


# ============================================================
# Process Table Images
# ============================================================
TABLE_TEXT = ""

if TABLE_PROCESSING_AVAILABLE:
    print("\n" + "=" * 80)
    print("📊 Processing table images...")
    print("=" * 80)
    try:
        TABLE_TEXT = extract_table_text_from_images(IMAGE_DIR)
    except Exception as e:
        print(f"⚠️  Error processing table images: {e}")
        TABLE_TEXT = ""
else:
    print("\n⚠️  Skipping table image processing (UNSTRUCTURED_API_KEY not set)")

# ============================================================
# Build document text (NO CHUNKING)
# ============================================================
USEFUL_CATEGORIES = {
    "Title",
    "NarrativeText",
    "ListItem",
    "Table",  # Include tables directly from PDF
}

MAX_CONTEXT_CHARS = 12_000

# Debug: Show all categories found across all PDFs
all_categories = {}
for el in all_elements:
    cat = el.category
    all_categories[cat] = all_categories.get(cat, 0) + 1

print(f"\n📋 Categories found across all PDFs:")
for cat, count in sorted(all_categories.items()):
    marker = "✅" if cat in USEFUL_CATEGORIES else "❌"
    print(f"   {marker} {cat}: {count} element(s)")

# Extract text blocks from all PDFs
text_blocks: List[str] = [
    el.text.strip()
    for el in all_elements
    if el.category in USEFUL_CATEGORIES and el.text
]

print(f"\n📝 Extracted {len(text_blocks)} text block(s) from all PDFs")
print(f"   Total PDF text length: {sum(len(block) for block in text_blocks)} characters")

# Show breakdown by PDF
print(f"\n📊 Breakdown by PDF:")
for pdf_info in all_pdf_texts:
    pdf_elements = pdf_info["elements"]
    pdf_text_blocks = [
        el.text.strip()
        for el in pdf_elements
        if el.category in USEFUL_CATEGORIES and el.text
    ]
    print(f"   {pdf_info['filename']}: {len(pdf_text_blocks)} text blocks, {sum(len(b) for b in pdf_text_blocks)} chars")

# Count tables found across all PDFs
pdf_tables = [el for el in all_elements if el.category == "Table" and el.text]
print(f"📊 Found {len(pdf_tables)} table(s) directly in PDFs")

# Combine PDF text with table text
DOCUMENT_TEXT = "\n\n".join(text_blocks)
print(f"📄 PDF text combined: {len(DOCUMENT_TEXT)} characters")

# Add table content from images if available
if TABLE_TEXT:
    print(f"✅ Adding table content from images ({len(TABLE_TEXT)} characters)")
    DOCUMENT_TEXT += "\n\n" + "=" * 80 + "\n"
    DOCUMENT_TEXT += "TABLES FROM DOCUMENT IMAGES:\n"
    DOCUMENT_TEXT += "=" * 80 + "\n"
    DOCUMENT_TEXT += TABLE_TEXT
else:
    print(f"⚠️  No table content extracted from images")

# Debug: Show document preview
print(f"\n📄 Document text preview:")
print("-" * 80)
print(f"First 500 chars of combined PDF text:")
pdf_text_only = "\n\n".join(text_blocks)
print(pdf_text_only[:500])
print("..." if len(pdf_text_only) > 500 else "")
print("-" * 80)
if TABLE_TEXT:
    print(f"\nFirst 300 chars of table text:")
    print(TABLE_TEXT[:300])
    print("..." if len(TABLE_TEXT) > 300 else "")
    print("-" * 80)

# Truncate if too long (prioritize PDF text, then add tables)
if len(DOCUMENT_TEXT) > MAX_CONTEXT_CHARS:
    print(f"\n⚠️  Combined document text ({len(DOCUMENT_TEXT)} chars) exceeds limit ({MAX_CONTEXT_CHARS} chars)")
    
    pdf_text_only = "\n\n".join(text_blocks)
    pdf_text_len = len(pdf_text_only)
    table_text_len = len(TABLE_TEXT) if TABLE_TEXT else 0
    separator_len = len("\n\n" + "=" * 80 + "\nTABLES FROM DOCUMENT IMAGES:\n" + "=" * 80 + "\n")
    
    # Strategy: Keep as much PDF text as possible, then add tables
    if table_text_len > 0:
        available_for_pdf = MAX_CONTEXT_CHARS - table_text_len - separator_len
        if available_for_pdf > 1000:  # Ensure we keep substantial PDF text
            # Truncate PDF text, keep all table text
            truncated_pdf = pdf_text_only[:available_for_pdf]
            DOCUMENT_TEXT = truncated_pdf + "\n\n" + "=" * 80 + "\n" + "TABLES FROM DOCUMENT IMAGES:\n" + "=" * 80 + "\n" + TABLE_TEXT
            print(f"   📄 PDF text: {len(truncated_pdf)} chars (truncated from {pdf_text_len})")
            print(f"   📊 Table text: {table_text_len} chars (preserved)")
        else:
            # PDF text is too large, prioritize it over tables
            DOCUMENT_TEXT = pdf_text_only[:MAX_CONTEXT_CHARS]
            print(f"   ⚠️  Combined PDF text too large, keeping only PDF text ({MAX_CONTEXT_CHARS} chars)")
            print(f"   ⚠️  Table text ({table_text_len} chars) excluded due to size limit")
    else:
        # No table text, just truncate PDF
        DOCUMENT_TEXT = pdf_text_only[:MAX_CONTEXT_CHARS]
        print(f"   📄 Combined PDF text truncated to {MAX_CONTEXT_CHARS} characters")
    
    print(f"   ✅ Final document: {len(DOCUMENT_TEXT)} characters")

print(f"\n📄 Final document text loaded ({len(DOCUMENT_TEXT)} characters)")
pdf_text_in_final = "\n\n".join(text_blocks)
if TABLE_TEXT:
    pdf_text_in_final_len = len(DOCUMENT_TEXT.split("TABLES FROM DOCUMENT IMAGES")[0]) if "TABLES FROM DOCUMENT IMAGES" in DOCUMENT_TEXT else len(DOCUMENT_TEXT)
    print(f"   📝 Combined PDF text in final document: {pdf_text_in_final_len} characters")
    print(f"   📊 Table content from images: {len(TABLE_TEXT)} characters")
else:
    print(f"   📝 Combined PDF text: {len(DOCUMENT_TEXT)} characters")
if pdf_tables:
    print(f"   📊 Includes {len(pdf_tables)} table(s) directly from PDFs")
print(f"   📚 Processed {len(pdf_files)} PDF file(s)")
print()


# ============================================================
# GPT Question Answering
# ============================================================
def ask_gpt(question: str, debug: bool = False) -> str:
    prompt = f"""
You are answering questions about a document.

RULES:
- Use ONLY the document text below
- Do NOT use external knowledge
- If the answer is not present, say: "The document does not specify this."
- Be concise and factual

DOCUMENT:
{DOCUMENT_TEXT}

QUESTION:
{question}
"""

    if debug:
        print("\n" + "=" * 80)
        print("DEBUG: Document text being sent to GPT:")
        print("=" * 80)
        print(f"Length: {len(DOCUMENT_TEXT)} characters")
        print(f"Contains 'TABLE': {'TABLE' in DOCUMENT_TEXT.upper()}")
        print(f"Contains 'Table': {'Table' in DOCUMENT_TEXT}")
        if "TABLES FROM DOCUMENT" in DOCUMENT_TEXT:
            table_section = DOCUMENT_TEXT.split("TABLES FROM DOCUMENT")[1][:500]
            print(f"\nTable section preview:\n{table_section}...")
        print("=" * 80 + "\n")

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a precise document analysis assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )

    return response.choices[0].message.content.strip()


# ============================================================
# Interactive Terminal Loop
# ============================================================
def main():
    print("💬 Ask questions about the PDF (type 'exit' to quit)")
    print("   Type 'debug' before your question to see what's sent to GPT\n")

    while True:
        question = input("❓ Question: ").strip()
        if question.lower() == "exit":
            break

        # Check for debug mode
        debug_mode = False
        if question.lower().startswith("debug"):
            debug_mode = True
            question = question[5:].strip()  # Remove "debug" prefix
        
        if not question:
            continue

        answer = ask_gpt(question, debug=debug_mode)

        print("\n🧠 Answer:")
        print(answer)
        print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()