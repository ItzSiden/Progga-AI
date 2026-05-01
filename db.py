"""
EduLumina Knowledge Base — File Storage
Structure:
    database/
        Bangla/
            Class_6/
                Math.csv
                Science.csv
        English/
            Class_9/
                Physics.csv

Each CSV has columns: page | content
Path itself = citation source (version/class/subject)
"""

import os
import csv
import re
import glob
from pathlib import Path
from typing import Optional

DB_ROOT = os.environ.get("DB_ROOT", "database")

VERSIONS = ["Bangla", "English"]
CLASSES  = [6, 7, 8, 9, 10]


# ── Init ──────────────────────────────────────────────────────

def init_db():
    """Create root database folder."""
    os.makedirs(DB_ROOT, exist_ok=True)


# ── Path helpers ──────────────────────────────────────────────

def subject_path(version: str, class_num: int, subject: str) -> str:
    """Get CSV file path for a subject."""
    return os.path.join(DB_ROOT, version, f"Class_{class_num}", f"{subject}.csv")


def subject_exists(version: str, class_num: int, subject: str) -> bool:
    return os.path.exists(subject_path(version, class_num, subject))


def source_label(version: str, class_num: int, subject: str) -> str:
    """Human-readable source citation."""
    return f"{version}/Class_{class_num}/{subject}"


# ── Subject management ────────────────────────────────────────

def list_subjects(version: str, class_num: int) -> list[str]:
    """List all subjects (CSV files) for a version/class."""
    folder = os.path.join(DB_ROOT, version, f"Class_{class_num}")
    if not os.path.exists(folder):
        return []
    return sorted([
        f[:-4] for f in os.listdir(folder)
        if f.endswith(".csv")
    ])


def create_subject(version: str, class_num: int, subject: str) -> tuple[bool, Optional[str]]:
    """Create a new CSV for a subject with page|content header."""
    fpath = subject_path(version, class_num, subject)
    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    if os.path.exists(fpath):
        return False, "Subject already exists."
    with open(fpath, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["page", "content"])
    return True, None


def delete_subject(version: str, class_num: int, subject: str) -> tuple[bool, Optional[str]]:
    """Delete a subject CSV."""
    fpath = subject_path(version, class_num, subject)
    if not os.path.exists(fpath):
        return False, "Subject not found."
    os.remove(fpath)
    return True, None


# ── Page CRUD ─────────────────────────────────────────────────

def get_pages(version: str, class_num: int, subject: str) -> list[dict]:
    """Read all pages from a subject CSV."""
    fpath = subject_path(version, class_num, subject)
    if not os.path.exists(fpath):
        return []
    try:
        with open(fpath, "r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def save_pages(version: str, class_num: int, subject: str, pages: list[dict]) -> tuple[int, Optional[str]]:
    """
    Write pages back to CSV.
    Each page: {"page": int, "content": str}
    Returns (count_written, error).
    """
    fpath = subject_path(version, class_num, subject)
    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    try:
        with open(fpath, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["page", "content"])
            writer.writeheader()
            count = 0
            for p in pages:
                page_num = str(p.get("page", "")).strip()
                content  = str(p.get("content", "")).strip()
                if not content:
                    continue
                writer.writerow({"page": page_num, "content": content})
                count += 1
        return count, None
    except Exception as e:
        return 0, str(e)


def add_page(version: str, class_num: int, subject: str, page_num: int, content: str) -> tuple[bool, Optional[str]]:
    """Append a single page."""
    fpath = subject_path(version, class_num, subject)
    if not os.path.exists(fpath):
        ok, err = create_subject(version, class_num, subject)
        if not ok:
            return False, err
    # Check duplicate
    pages = get_pages(version, class_num, subject)
    if any(str(p["page"]) == str(page_num) for p in pages):
        return False, f"Page {page_num} already exists."
    try:
        with open(fpath, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["page", "content"])
            writer.writerow({"page": page_num, "content": content})
        return True, None
    except Exception as e:
        return False, str(e)


def bulk_import(version: str, class_num: int, subject: str, raw_text: str) -> tuple[int, Optional[str]]:
    """
    Parse [PAGE_N] formatted text and import all pages.
    Returns (pages_imported, error).
    """
    pattern = re.compile(r'\[PAGE_(\d+)\]\s*(.*?)(?=\[PAGE_\d+\]|$)', re.DOTALL)
    matches = pattern.findall(raw_text)
    if not matches:
        return 0, "No [PAGE_N] markers found."

    pages = get_pages(version, class_num, subject)
    existing = {str(p["page"]) for p in pages}

    new_pages = []
    skipped = 0
    for page_num, content in matches:
        content = content.strip()
        if not content:
            continue
        if page_num in existing:
            skipped += 1
            continue
        new_pages.append({"page": page_num, "content": content})
        existing.add(page_num)

    if not new_pages:
        return 0, f"All pages already exist (skipped {skipped})."

    # Append to CSV
    fpath = subject_path(version, class_num, subject)
    if not os.path.exists(fpath):
        create_subject(version, class_num, subject)

    with open(fpath, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["page", "content"])
        for p in new_pages:
            writer.writerow(p)

    return len(new_pages), None


# ── RAG search ────────────────────────────────────────────────

def rag_search(
    query: str,
    limit: int = 5,
    version: Optional[str] = None,
    class_num: Optional[int] = None,
    subject: Optional[str] = None,
) -> list[dict]:
    """
    Search knowledge base with optional filters.
    Returns ranked list of {source, page, content, score, snippet}.
    """
    query_lower = query.lower().strip()
    words = [w for w in re.split(r'\s+', query_lower) if len(w) > 1]
    if not words:
        return []

    # Build glob pattern based on filters
    v_pat  = version  if version   else "*"
    c_pat  = f"Class_{class_num}" if class_num else "Class_*"
    s_pat  = f"{subject}.csv"     if subject   else "*.csv"
    pattern = os.path.join(DB_ROOT, v_pat, c_pat, s_pat)

    results = []
    for fpath in glob.glob(pattern, recursive=False):
        # Parse source from path
        parts = Path(fpath).parts
        # parts: database / Version / Class_N / subject.csv
        try:
            ver  = parts[-3]
            cls  = int(parts[-2].replace("Class_", ""))
            subj = parts[-1][:-4]
        except Exception:
            continue

        source = source_label(ver, cls, subj)

        try:
            with open(fpath, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    content = row.get("content", "")
                    if not content:
                        continue
                    content_lower = content.lower()

                    # Score
                    score = sum(1 for w in words if w in content_lower)
                    if query_lower in content_lower:
                        score += len(words) * 2

                    if score > 0:
                        results.append({
                            "source":  source,
                            "page":    row.get("page", "?"),
                            "content": content,
                            "score":   score,
                            "snippet": _snippet(content, words),
                        })
        except Exception:
            continue

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:limit]


def rag_context(results: list[dict]) -> str:
    """Format RAG results as LLM context string."""
    parts = []
    for r in results:
        parts.append(
            f"[Source: {r['source']}, Page: {r['page']}]\n{r['content']}"
        )
    return "\n\n---\n\n".join(parts)


def _snippet(text: str, words: list[str], max_len: int = 300) -> str:
    """Extract best matching snippet from text."""
    text_lower = text.lower()
    best_pos, best_score = 0, 0
    for i in range(0, len(text), 50):  # Step 50 for speed
        window = text_lower[i:i + max_len]
        score  = sum(1 for w in words if w in window)
        if score > best_score:
            best_score, best_pos = score, i

    snippet = text[best_pos:best_pos + max_len]
    if best_pos > 0:
        snippet = "..." + snippet
    if best_pos + max_len < len(text):
        snippet += "..."
    return snippet


# ── Stats ─────────────────────────────────────────────────────

def get_stats() -> dict:
    """Return database statistics."""
    csvs = glob.glob(os.path.join(DB_ROOT, "*", "Class_*", "*.csv"))
    total_rows = 0
    subjects_by_version: dict[str, int] = {}

    for fpath in csvs:
        parts = Path(fpath).parts
        try:
            ver = parts[-3]
            subjects_by_version[ver] = subjects_by_version.get(ver, 0) + 1
        except Exception:
            pass
        try:
            with open(fpath, encoding="utf-8") as f:
                total_rows += max(0, sum(1 for _ in f) - 1)  # minus header
        except Exception:
            pass

    return {
        "total_subjects": len(csvs),
        "total_pages":    total_rows,
        "by_version":     subjects_by_version,
        "db_root":        DB_ROOT,
    }


def list_all() -> list[dict]:
    """List all version/class/subject combinations."""
    result = []
    for fpath in sorted(glob.glob(os.path.join(DB_ROOT, "*", "Class_*", "*.csv"))):
        parts = Path(fpath).parts
        try:
            ver  = parts[-3]
            cls  = int(parts[-2].replace("Class_", ""))
            subj = parts[-1][:-4]
            # Count pages
            with open(fpath, encoding="utf-8") as f:
                pages = max(0, sum(1 for _ in f) - 1)
            result.append({
                "version": ver,
                "class":   cls,
                "subject": subj,
                "pages":   pages,
                "source":  source_label(ver, cls, subj),
            })
        except Exception:
            continue
    return result
