# NCTB Structured Hierarchical Database + RAG Workbench

This project provides a **working local system** (SQLite + Streamlit) to store NCTB curriculum data in strict logical units and run retrieval tests with a GitHub-like repository browsing experience.

## Why this design (vs naive chunking)

- Stores **full textbook pages** and **full Q&A pairs** as first-class rows.
- Enforces a path-like source lineage: `Class > Subject > Book Type > Chapter > Page|Q&A`.
- Keeps source integrity for downstream RAG and future supervised fine-tuning datasets.
- Auto-maps inserted records into hierarchy paths (`hierarchy_nodes`) so structure stays consistent without manual folder pre-creation.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

The database persists to `nctb_structured.db` in the repository root.
This `.db` file is intentionally excluded from version control (generated locally).

## Core Tables

1. `hierarchy_nodes`: explicit folder tree registry.
2. `textbook_pages`: full page units.
3. `qa_pairs`: full guidebook Q&A units with textbook reference path.
4. `data_tables`: visual user-defined custom table schemas at any path.
5. `data_rows`: user-defined row payloads (JSON) for custom tables.
6. `search_fts` + `search_index`: retrieval index over all source rows.

## Example Rows

### Textbook
- Path: `Class 8 / Science / Textbook / Chapter 2 / Page 21`
- Content: full page text.
- Metadata: JSON object.

### Q&A
- Path: `Class 8 / Science / Guide / Chapter 2 / Q&A_45`
- Question + Answer: full pair.
- Reference: `Class 8 / Science / Textbook / Chapter 2 / Page 21`.

## Clean Context Retrieval Example

User query: `What is photosynthesis?`

1. Run FTS retrieval against stored full units.
2. Return top row(s) with:
   - `source_key` (which row),
   - full `path` (source lineage),
   - snippet for quick validation.
3. Build LLM context using full row body, not arbitrary fragments.

Pseudo SQL used by app:

```sql
SELECT source_key, path, bm25(search_fts) AS score,
       snippet(search_fts, 2, '[', ']', ' ... ', 20) AS snippet
FROM search_fts
WHERE search_fts MATCH 'photosynthesis'
ORDER BY score
LIMIT 5;
```

## Dashboard Features (Supabase-like workflow)

- Repository-style folder browsing.
- Create hierarchy folder nodes.
- Create custom table schemas at any path.
- Insert custom rows visually.
- Insert textbook pages and Q&A entries.
- Test RAG retrieval and inspect top relevant rows.
- Snapshot panels for fast inspection.
