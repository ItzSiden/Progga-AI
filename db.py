import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any

DB_FILE = Path("nctb_structured.db")


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with get_conn() as conn:
        conn.executescript(
            """
            PRAGMA foreign_keys = ON;

            CREATE TABLE IF NOT EXISTS hierarchy_nodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                node_type TEXT NOT NULL,
                node_name TEXT NOT NULL,
                parent_id INTEGER,
                path TEXT UNIQUE NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(parent_id) REFERENCES hierarchy_nodes(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS textbook_pages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT UNIQUE NOT NULL,
                class INTEGER NOT NULL,
                subject TEXT NOT NULL,
                book_type TEXT NOT NULL DEFAULT 'Textbook',
                chapter INTEGER NOT NULL,
                page INTEGER NOT NULL,
                content TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS qa_pairs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT UNIQUE NOT NULL,
                class INTEGER NOT NULL,
                subject TEXT NOT NULL,
                book_type TEXT NOT NULL DEFAULT 'Guide',
                chapter INTEGER NOT NULL,
                qa_key TEXT NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                reference_path TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS data_tables (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                table_name TEXT NOT NULL,
                path TEXT NOT NULL,
                columns_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(table_name, path)
            );

            CREATE TABLE IF NOT EXISTS data_rows (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                table_id INTEGER NOT NULL,
                row_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(table_id) REFERENCES data_tables(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS search_index (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_type TEXT NOT NULL,
                source_id INTEGER NOT NULL,
                path TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                UNIQUE(source_type, source_id)
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS search_fts USING fts5(
                source_key,
                path,
                content,
                tokenize='porter unicode61'
            );
            """
        )


def fetchall(query: str, params: tuple[Any, ...] = ()):
    with get_conn() as conn:
        return conn.execute(query, params).fetchall()


def execute(query: str, params: tuple[Any, ...] = ()):
    with get_conn() as conn:
        cur = conn.execute(query, params)
        return cur.lastrowid


def ensure_hierarchy_path(path: str) -> None:
    parts = [part.strip() for part in path.split("/") if part.strip()]
    parent_id: int | None = None
    cumulative_parts: list[str] = []
    for part in parts:
        cumulative_parts.append(part)
        partial_path = " / ".join(cumulative_parts)
        row = fetchall("SELECT id FROM hierarchy_nodes WHERE path=?", (partial_path,))
        node_id = int(row[0]["id"]) if row else None
        if node_id is None:
            node_type = "Folder"
            if part.lower().startswith("class "):
                node_type = "Class"
            elif part.lower().startswith("chapter "):
                node_type = "Chapter"
            elif part.lower().startswith("page "):
                node_type = "Page"
            execute(
                """
                INSERT INTO hierarchy_nodes(node_type, node_name, parent_id, path, metadata_json)
                VALUES(?,?,?,?,?)
                """,
                (node_type, part, parent_id, partial_path, "{}"),
            )
            row = fetchall("SELECT id FROM hierarchy_nodes WHERE path=?", (partial_path,))
            node_id = int(row[0]["id"])
        parent_id = node_id


def upsert_node(node_type: str, node_name: str, parent_id: int | None, path: str, metadata: dict[str, Any] | None = None):
    metadata = metadata or {}
    execute(
        """
        INSERT INTO hierarchy_nodes(node_type, node_name, parent_id, path, metadata_json)
        VALUES(?,?,?,?,?)
        ON CONFLICT(path) DO UPDATE SET
            node_type=excluded.node_type,
            node_name=excluded.node_name,
            parent_id=excluded.parent_id,
            metadata_json=excluded.metadata_json
        """,
        (node_type, node_name, parent_id, path, json.dumps(metadata)),
    )


def _refresh_search_document(source_type: str, source_id: int, path: str, content: str, metadata: dict[str, Any]):
    key = f"{source_type}:{source_id}"
    execute(
        """
        INSERT INTO search_index(source_type, source_id, path, content, metadata_json)
        VALUES(?,?,?,?,?)
        ON CONFLICT(source_type, source_id) DO UPDATE SET
            path=excluded.path,
            content=excluded.content,
            metadata_json=excluded.metadata_json
        """,
        (source_type, source_id, path, content, json.dumps(metadata)),
    )
    execute("DELETE FROM search_fts WHERE source_key=?", (key,))
    execute(
        "INSERT INTO search_fts(source_key, path, content) VALUES(?,?,?)",
        (key, path, content),
    )


def upsert_textbook_page(payload: dict[str, Any]) -> int:
    ensure_hierarchy_path(payload["path"])
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO textbook_pages(path, class, subject, book_type, chapter, page, content, metadata_json)
            VALUES(?,?,?,?,?,?,?,?)
            ON CONFLICT(path) DO UPDATE SET
                content=excluded.content,
                metadata_json=excluded.metadata_json,
                updated_at=CURRENT_TIMESTAMP
            """,
            (
                payload["path"],
                payload["class"],
                payload["subject"],
                payload.get("book_type", "Textbook"),
                payload["chapter"],
                payload["page"],
                payload["content"],
                json.dumps(payload.get("metadata", {})),
            ),
        )
        row = conn.execute("SELECT id FROM textbook_pages WHERE path=?", (payload["path"],)).fetchone()
        source_id = int(row["id"])
    _refresh_search_document("textbook_page", source_id, payload["path"], payload["content"], payload.get("metadata", {}))
    return source_id


def upsert_qa_pair(payload: dict[str, Any]) -> int:
    ensure_hierarchy_path(payload["path"])
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO qa_pairs(path, class, subject, book_type, chapter, qa_key, question, answer, reference_path, metadata_json)
            VALUES(?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(path) DO UPDATE SET
                question=excluded.question,
                answer=excluded.answer,
                reference_path=excluded.reference_path,
                metadata_json=excluded.metadata_json,
                updated_at=CURRENT_TIMESTAMP
            """,
            (
                payload["path"],
                payload["class"],
                payload["subject"],
                payload.get("book_type", "Guide"),
                payload["chapter"],
                payload["qa_key"],
                payload["question"],
                payload["answer"],
                payload["reference_path"],
                json.dumps(payload.get("metadata", {})),
            ),
        )
        row = conn.execute("SELECT id FROM qa_pairs WHERE path=?", (payload["path"],)).fetchone()
        source_id = int(row["id"])
    combined = f"Q: {payload['question']}\nA: {payload['answer']}"
    _refresh_search_document("qa_pair", source_id, payload["path"], combined, payload.get("metadata", {}))
    return source_id


def create_custom_table(table_name: str, path: str, columns: list[dict[str, str]]) -> int:
    ensure_hierarchy_path(path)
    return execute(
        "INSERT INTO data_tables(table_name, path, columns_json) VALUES(?,?,?)",
        (table_name, path, json.dumps(columns)),
    )


def insert_custom_row(table_id: int, row_dict: dict[str, Any]) -> int:
    row_id = execute(
        "INSERT INTO data_rows(table_id, row_json) VALUES(?,?)",
        (table_id, json.dumps(row_dict)),
    )
    table = fetchall("SELECT table_name, path FROM data_tables WHERE id=?", (table_id,))[0]
    ensure_hierarchy_path(f"{table['path']} / {table['table_name']}")
    content = " ".join([str(v) for v in row_dict.values()])
    _refresh_search_document(
        "custom_row",
        row_id,
        f"{table['path']} / {table['table_name']} / row_{row_id}",
        content,
        {"table": table["table_name"], "path": table["path"]},
    )
    return row_id


def rag_search(query: str, k: int = 5):
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT source_key, path, bm25(search_fts) AS score, snippet(search_fts, 2, '[', ']', ' ... ', 20) AS snippet
            FROM search_fts
            WHERE search_fts MATCH ?
            ORDER BY score
            LIMIT ?
            """,
            (query, k),
        ).fetchall()
    return rows
