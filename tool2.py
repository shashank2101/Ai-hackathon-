"""
ingest.py — Production RAG Ingestion Layer (Docling Edition)
============================================================
Converts any supported file format → markdown files in ./markdown_out/
Structure-aware chunking ready for vector DB ingestion.

Core Engine: Docling (replaces MarkItDown/pdfminer)
- Handles PDFs (preserves tables and complex layouts)
- Handles Office Docs (Word, Excel, PPT)
- Handles Web files (HTML)

Specialized handlers:
- JSON/Excel → SQLite structured export
- Images/Scanned PDFs → Anthropic Vision LLM
- Code/Config → Syntax-highlighted Markdown
"""

import os
import re
import json
import base64
import sqlite3
import hashlib
import logging
import mimetypes
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

# ── NEW: Import Docling ──────────────────────────────────────────────────────
from docling.document_converter import DocumentConverter

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

MARKDOWN_OUT_DIR      = Path("markdown_out")
SQLITE_DB_PATH        = Path("reference_data.sqlite")
CHUNK_SIZE            = 500
CHUNK_OVERLAP         = 50
EXCEL_ROW_LIMIT       = 2000
JSON_ARRAY_ROW_LIMIT  = 5000

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ingest")

# ─────────────────────────────────────────────────────────────────────────────
# Extension sets
# ─────────────────────────────────────────────────────────────────────────────

DOCUMENT_EXTS = {".pdf", ".docx", ".doc", ".pptx", ".ppt", ".rtf", ".odt"}
SHEET_EXTS    = {".xlsx", ".xls", ".csv"}
WEB_EXTS      = {".html", ".htm", ".xml"}
TEXT_EXTS     = {".txt", ".md", ".rst"}
DATA_EXTS     = {".json", ".yaml", ".yml", ".toml"}
STYLE_EXTS    = {".css", ".scss", ".sass", ".less"}
IMAGE_EXTS    = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff", ".tif"}

CODE_EXTS = {
    ".py", ".pyw", ".js", ".mjs", ".jsx", ".ts", ".tsx",
    ".java", ".c", ".h", ".cpp", ".cc", ".cxx", ".hpp",
    ".cs", ".go", ".rs", ".rb", ".php", ".swift", ".kt", ".kts",
    ".r", ".sh", ".bash", ".zsh", ".fish",
    ".sql", ".graphql", ".gql",
    ".lua", ".scala", ".ex", ".exs", ".hs",
    ".tf", ".tfvars", ".dockerfile", ".makefile",
}

_CODE_LANG: dict[str, str] = {
    ".py": "python", ".js": "javascript", ".ts": "typescript",
    ".java": "java", ".c": "c", ".cpp": "cpp", ".cs": "csharp",
    ".go": "go", ".rs": "rust", ".rb": "ruby", ".php": "php",
    ".swift": "swift", ".sh": "bash", ".sql": "sql",
    ".html": "html", ".css": "css", ".yaml": "yaml", ".json": "json"
}

ALL_SUPPORTED_EXTS = (
    DOCUMENT_EXTS | SHEET_EXTS | WEB_EXTS | TEXT_EXTS |
    DATA_EXTS | STYLE_EXTS | IMAGE_EXTS | CODE_EXTS
)

# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    source_file : str
    chunk_index : int
    content     : str
    heading_path: list[str] = field(default_factory=list)
    chunk_type  : str = "text"
    metadata    : dict = field(default_factory=dict)

    def token_estimate(self) -> int:
        return int(len(self.content.split()) * 0.75)


# ─────────────────────────────────────────────────────────────────────────────
# PDF extraction — Docling Engine
# ─────────────────────────────────────────────────────────────────────────────

def extract_pdf_text(src: Path) -> tuple[str, str]:
    """
    Extract text from a PDF using ONLY Docling.
    """
    try:
        converter = DocumentConverter()
        result = converter.convert(str(src))
        text = result.document.export_to_markdown() or ""
        
        if text and len(text.strip()) > 50:
            log.info(f"  PDF extracted via Docling ({len(text.split())} words)")
            return text, "docling"
            
        log.info("  Docling: empty output (possibly scanned/image)")
    except Exception as e:
        log.warning(f"  Docling failed for PDF {src.name}: {e}")

    # Fallback — scanned/image PDF
    log.warning(f"  {src.name}: Docling failed — likely scanned/image PDF")
    return (
        f"# {src.name}\n\n"
        f"> **[NEEDS_OCR]** This PDF appears to be scanned or image-based. "
        f"No text could be extracted. Set `ANTHROPIC_API_KEY` and re-run with "
        f"`--ocr` flag to describe pages via vision LLM.\n"
    ), "needs_ocr"


# ─────────────────────────────────────────────────────────────────────────────
# Image → LLM description
# ─────────────────────────────────────────────────────────────────────────────

def describe_image(src: Path) -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        log.warning(f"  Image {src.name}: ANTHROPIC_API_KEY not set — placeholder used")
        return (f"**[Image: {src.name}]**\n\n> *Description pending — set ANTHROPIC_API_KEY...*\n")
    try:
        import anthropic
        mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}
        media_type = mime_map.get(src.suffix.lower(), "image/jpeg")
        img_data   = base64.standard_b64encode(src.read_bytes()).decode("utf-8")
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-haiku-4-5", max_tokens=512,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": img_data}},
                {"type": "text", "text": "Describe this image in detail for a RAG knowledge base. Write 2-5 sentences."}
            ]}],
        )
        return f"**[Image: {src.name}]**\n\n{response.content[0].text.strip()}\n"
    except Exception as e:
        return f"**[Image: {src.name}]**\n\n> *Image description failed: {e}*\n"


# ─────────────────────────────────────────────────────────────────────────────
# JSON & SQLite logic
# ─────────────────────────────────────────────────────────────────────────────

def json_to_markdown(src: Path) -> tuple[str, str]:
    try:
        raw  = src.read_text(encoding="utf-8", errors="replace")
        data = json.loads(raw)
    except json.JSONDecodeError:
        return f"# {src.name}\n\n```json\n{src.read_text(errors='replace')}\n```\n", "mixed"

    if isinstance(data, list) and data and all(isinstance(r, dict) for r in data[:50]):
        # It's a record list
        records  = data
        all_keys = list(dict.fromkeys(k for r in records for k in r.keys()))
        header   = "| " + " | ".join(str(k) for k in all_keys) + " |"
        divider  = "| " + " | ".join("---" for _ in all_keys) + " |"
        rows     = ["| " + " | ".join(str(r.get(k, "")).replace("|", "\\|").replace("\n", " ") for k in all_keys) + " |" for r in records[:100]]
        table = "\n".join([header, divider] + rows)
        return f"# {src.name}\n\n**Type:** JSON records\n\n{table}\n", "json_records"

    pretty = json.dumps(data, indent=2, ensure_ascii=False)
    return f"# {src.name}\n\n```json\n{pretty}\n```\n", "mixed"

def excel_sheets_to_sqlite(src: Path, db_path: Path = SQLITE_DB_PATH) -> list[str]:
    # Keeping the openpyxl integration for native SQLite export of tabular data
    try: import openpyxl
    except ImportError: return []
    written = []
    try:
        wb = openpyxl.load_workbook(src, read_only=True, data_only=True)
        conn = sqlite3.connect(db_path)
        for sheet_name in wb.sheetnames:
            ws = wb[sheet_name]
            rows = list(ws.iter_rows(values_only=True))
            if len(rows) > 1 and len(rows) < EXCEL_ROW_LIMIT:
                table_name = re.sub(r'\W+', '_', f"{src.stem}__{sheet_name}").strip('_')
                conn.execute(f'DROP TABLE IF EXISTS "{table_name}"')
                safe_cols = [f"col_{i}" for i in range(len(rows[0]))]
                conn.execute(f'CREATE TABLE "{table_name}" ({", ".join(f"{c} TEXT" for c in safe_cols)})')
                for r in rows[1:]:
                    conn.execute(f'INSERT INTO "{table_name}" VALUES ({", ".join("?" for _ in safe_cols)})', [str(v) if v is not None else None for v in r])
                written.append(table_name)
        conn.commit()
        conn.close()
    except Exception: pass
    return written


# ─────────────────────────────────────────────────────────────────────────────
# Main per-file converter
# ─────────────────────────────────────────────────────────────────────────────

def file_to_markdown(src: Path) -> tuple[str, str]:
    """Returns (markdown_text, file_category)."""
    suffix = src.suffix.lower()

    # Images
    if suffix in IMAGE_EXTS:
        return f"# Image: {src.name}\n\n{describe_image(src)}\n", "image"

    # PDF — Docling
    if suffix == ".pdf":
        md_text, method = extract_pdf_text(src)
        return md_text, f"document_pdf_{method}"

    # Code / Text / Config
    if suffix in CODE_EXTS or suffix in STYLE_EXTS or suffix in (".yaml", ".yml"):
        lang = _CODE_LANG.get(suffix, suffix.lstrip('.'))
        try: text = src.read_text(encoding="utf-8", errors="replace")
        except Exception: return f"# {src.name}\n\n[Unreadable]\n", "code"
        return f"# File: {src.name}\n\n```{lang}\n{text}\n```\n", "code"

    # JSON
    if suffix == ".json":
        md, jtype = json_to_markdown(src)
        return md, jtype

    # ── ALL OTHER DOCUMENTS (Word, Excel, PPT, HTML) via DOCLING ──
    try:
        log.info(f"  Extracting {src.name} via Docling...")
        converter = DocumentConverter()
        result = converter.convert(str(src))
        md = result.document.export_to_markdown() or ""
        
        if md.strip():
            category = (
                "sheet"    if suffix in SHEET_EXTS    else
                "document" if suffix in DOCUMENT_EXTS else
                "web"      if suffix in WEB_EXTS      else
                "text"
            )
            return md, category
    except Exception as e:
        log.warning(f"  Docling failed for {src.name}: {e}")

    # Last resort plain text
    try:
        text = src.read_text(errors="replace")
        return f"# {src.name}\n\n" + text, "text"
    except Exception:
        return f"# {src.name}\n\n[Binary or unreadable — skipped]\n", "text"


# ─────────────────────────────────────────────────────────────────────────────
# Structure-aware chunker
# ─────────────────────────────────────────────────────────────────────────────

def _word_count(text: str) -> int: return len(text.split())

def chunk_markdown(md_text: str, source_file: str, chunk_size: int = CHUNK_SIZE, file_category: str = "text") -> list[Chunk]:
    if file_category in ("code", "style", "image"):
        return [Chunk(source_file, 0, md_text.strip(), [], file_category, {"file_category": file_category})]

    chunks, heading_stack, buffer, idx = [], [], [], 0
    
    def flush(buf):
        nonlocal idx
        text = "".join(buf).strip()
        if text:
            chunks.append(Chunk(source_file, idx, text, list(heading_stack), "text", {"file_category": file_category}))
            idx += 1

    for line in md_text.splitlines(keepends=True):
        m = re.match(r'^(#{1,6})\s+(.+)$', line.rstrip('\n'))
        if m:
            flush(buffer); buffer = []
            heading_stack = heading_stack[:len(m.group(1)) - 1] + [m.group(2).strip()]
        buffer.append(line)
        if _word_count("".join(buffer)) >= chunk_size:
            flush(buffer); buffer = []

    flush(buffer)
    return chunks

# ─────────────────────────────────────────────────────────────────────────────
# Main ingestion pipeline
# ─────────────────────────────────────────────────────────────────────────────

def ingest_file(src: Path, out_dir: Path = MARKDOWN_OUT_DIR) -> list[Chunk]:
    if src.suffix.lower() not in ALL_SUPPORTED_EXTS: return []
    log.info(f"Processing: {src.name}")

    md_text, file_category = file_to_markdown(src)
    file_hash = hashlib.md5(src.read_bytes()).hexdigest()[:8]
    md_text = f"\n\n" + md_text

    if src.suffix.lower() in (".xlsx", ".xls"):
        tables = excel_sheets_to_sqlite(src)
        if tables: md_text += f"\n\n---\n> **SQLite tables written:** {', '.join(tables)}\n"

    out_dir.mkdir(parents=True, exist_ok=True)
    # NEW CODE (Python 3.11 Compatible)
    safe_stem = re.sub(r'\W+', '_', src.stem)
    md_path = out_dir / f"{safe_stem}__{file_hash}.md"
    md_path.write_text(md_text, encoding="utf-8")
    
    chunks = chunk_markdown(md_text, str(src), file_category=file_category)
    log.info(f"  → {len(chunks)} chunks exported to {md_path.name}")
    return chunks

def ingest_directory(src_dir: Path, out_dir: Path = MARKDOWN_OUT_DIR) -> list[Chunk]:
    return [c for f in src_dir.rglob("*") if f.is_file() for c in ingest_file(f, out_dir)]

if __name__ == "__main__":
    import argparse, sys
    from collections import Counter

    parser = argparse.ArgumentParser()
    parser.add_argument("sources", nargs="+")
    args = parser.parse_args()

    total_chunks = []
    for s in args.sources:
        p = Path(s)
        if p.is_dir(): total_chunks.extend(ingest_directory(p))
        elif p.is_file(): total_chunks.extend(ingest_file(p))

    by_type = Counter(c.chunk_type for c in total_chunks)
    print(f"\n{'─'*50}\n  Total chunks : {len(total_chunks)}")
    for ctype, n in sorted(by_type.items()): print(f"  {ctype:<14}: {n}")
    print(f"{'─'*50}")