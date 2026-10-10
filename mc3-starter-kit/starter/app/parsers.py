"""Robust, defensive document parsers for AMD Mini-Challenge 3.

Handles:
- .pdf  (via pypdf with encrypted file fallback)
- .docx (via standard library zipfile + XML or python-docx)
- .xlsx (via openpyxl)
- .csv  (via standard csv)
- .txt, .log, .py (via utf-8 with error replacement)
- .png, .jpg (image references)

Every parser catches exceptions per-file to ensure the corpus walk never crashes.
"""

from __future__ import annotations

import csv
import logging
import os
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

logger = logging.getLogger(__name__)


def parse_pdf(path: Path) -> str:
    """Extract text from PDF, safely ignoring encrypted/unreadable files."""
    try:
        import pypdf
        reader = pypdf.PdfReader(path)
        if reader.is_encrypted:
            # Try decrypting with empty password
            try:
                if not reader.decrypt(""):
                    return ""
            except Exception:
                return ""
        text_parts = []
        for page in reader.pages:
            try:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
            except Exception:
                continue
        return "\n".join(text_parts).strip()
    except Exception as e:
        logger.warning("Failed to parse PDF %s: %s", path, e)
        return ""


def parse_docx(path: Path) -> str:
    """Extract text and tables from DOCX using standard library zipfile + XML."""
    try:
        with zipfile.ZipFile(path) as z:
            if "word/document.xml" not in z.namelist():
                return ""
            xml_content = z.read("word/document.xml")
            tree = ET.fromstring(xml_content)
            
            # Extract text preserving paragraphs
            namespaces = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
            paragraphs = []
            for p in tree.iterfind(".//w:p", namespaces):
                p_text = "".join(t.text for t in p.iterfind(".//w:t", namespaces) if t.text)
                if p_text.strip():
                    paragraphs.append(p_text.strip())
            return "\n".join(paragraphs).strip()
    except Exception as e:
        logger.warning("Failed to parse DOCX %s: %s", path, e)
        return ""


def parse_xlsx(path: Path) -> str:
    """Extract all sheets and rows from XLSX using openpyxl."""
    try:
        import openpyxl
        wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
        sheets_text = []
        for sheet_name in wb.sheetnames:
            ws = wb[sheet_name]
            sheet_rows = [f"[Sheet: {sheet_name}]"]
            for row in ws.iter_rows(values_only=True):
                if any(row):
                    sheet_rows.append("\t".join(str(cell) for cell in row if cell is not None))
            if len(sheet_rows) > 1:
                sheets_text.append("\n".join(sheet_rows))
        wb.close()
        return "\n\n".join(sheets_text).strip()
    except Exception as e:
        logger.warning("Failed to parse XLSX %s: %s", path, e)
        return ""


def parse_csv(path: Path) -> str:
    """Extract rows from CSV file."""
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            lines = [", ".join(row) for row in reader if any(row)]
            return "\n".join(lines).strip()
    except Exception as e:
        logger.warning("Failed to parse CSV %s: %s", path, e)
        return ""


def parse_text(path: Path) -> str:
    """Extract text from plain text, log, or python source files."""
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except Exception as e:
        logger.warning("Failed to parse text file %s: %s", path, e)
        return ""


def parse_file(path: Path) -> tuple[str, bool]:
    """Parse a single file defensively.
    
    Returns:
        tuple of (extracted_text, is_image)
    """
    try:
        # Check if file is readable
        if not os.access(path, os.R_OK):
            return "", False
    except Exception:
        return "", False

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return parse_pdf(path), False
    elif suffix == ".docx":
        return parse_docx(path), False
    elif suffix in (".xlsx", ".xlsm"):
        return parse_xlsx(path), False
    elif suffix == ".csv":
        return parse_csv(path), False
    elif suffix in (".txt", ".log", ".py", ".md", ".json", ".ini", ".conf"):
        return parse_text(path), False
    elif suffix in (".png", ".jpg", ".jpeg"):
        return "", True
    else:
        # Unknown binary or unsupported extension
        return "", False
