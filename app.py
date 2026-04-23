import json
import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

import db

st.set_page_config(page_title="NCTB Structured DB", layout="wide")

db.init_db()

st.title("📚 NCTB Structured Database + RAG Workbench")
st.caption("Logical unit storage (full pages, full Q&A) with strict folder-like hierarchy.")
st.info("Auto-sync is enabled: every create/insert action writes directly to the SQLite file and auto-maps missing hierarchy paths.")

with st.sidebar:
    st.header("Hierarchy Builder")
    node_type = st.selectbox("Node Type", ["Class", "Subject", "Book Type", "Chapter", "Page"])
    node_path = st.text_input("Full Path", placeholder="Class 8 / Science / Textbook / Chapter 2 / Page 21")
    node_name = st.text_input("Node Name", placeholder="Page 21")
    node_parent = st.text_input("Parent Path (optional)")
    metadata_raw = st.text_area("Metadata JSON", value="{}")
    if st.button("Create / Update Folder Node"):
        try:
            metadata = json.loads(metadata_raw)
            parent_id = None
            if node_parent:
                parent_rows = db.fetchall("SELECT id FROM hierarchy_nodes WHERE path=?", (node_parent,))
                parent_id = int(parent_rows[0]["id"]) if parent_rows else None
            db.upsert_node(node_type, node_name, parent_id, node_path, metadata)
            st.success(f"Saved node: {node_path}")
        except json.JSONDecodeError:
            st.error("Invalid JSON metadata")

left, right = st.columns(2)

with left:
    st.subheader("1) Add Textbook Page")
    with st.form("textbook_form"):
        c = st.number_input("Class", min_value=1, max_value=12, value=8)
        subj = st.text_input("Subject", value="Science")
        ch = st.number_input("Chapter", min_value=1, value=2)
        pg = st.number_input("Page", min_value=1, value=21)
        content = st.text_area("Full Page Content", height=180)
        md = st.text_area("Metadata JSON", value='{"language":"bn"}')
        submitted = st.form_submit_button("Save Page")
        if submitted:
            path = f"Class {c} / {subj} / Textbook / Chapter {ch} / Page {pg}"
            payload = {
                "path": path,
                "class": int(c),
                "subject": subj,
                "chapter": int(ch),
                "page": int(pg),
                "content": content,
                "metadata": json.loads(md),
            }
            db.upsert_textbook_page(payload)
            st.success(f"Saved {path}")

    st.subheader("2) Add Guidebook Q&A")
    with st.form("qa_form"):
        c = st.number_input("Q&A Class", min_value=1, max_value=12, value=8)
        subj = st.text_input("Q&A Subject", value="Science")
        ch = st.number_input("Q&A Chapter", min_value=1, value=2)
        qa_key = st.text_input("Q&A Key", value="Q&A_45")
        q = st.text_area("Question")
        a = st.text_area("Answer")
        ref = st.text_input("Reference Path", value="Class 8 / Science / Textbook / Chapter 2 / Page 21")
        md = st.text_area("Q&A Metadata JSON", value='{"difficulty":"easy"}')
        submitted = st.form_submit_button("Save Q&A")
        if submitted:
            path = f"Class {c} / {subj} / Guide / Chapter {ch} / {qa_key}"
            payload = {
                "path": path,
                "class": int(c),
                "subject": subj,
                "chapter": int(ch),
                "qa_key": qa_key,
                "question": q,
                "answer": a,
                "reference_path": ref,
                "metadata": json.loads(md),
            }
            db.upsert_qa_pair(payload)
            st.success(f"Saved {path}")

with right:
    st.subheader("3) Create Custom Table at Any Path")
    with st.form("custom_table_form"):
        table_name = st.text_input("Table Name", value="learning_objectives")
        table_path = st.text_input("Attach to Path", value="Class 8 / Science / Textbook / Chapter 2")
        columns_spec = st.text_area(
            "Columns (JSON list)",
            value='[{"name":"objective_id","type":"text"},{"name":"objective","type":"text"},{"name":"bloom_level","type":"text"}]',
            height=140,
        )
        submitted = st.form_submit_button("Create Table")
        if submitted:
            try:
                table_id = db.create_custom_table(table_name, table_path, json.loads(columns_spec))
                st.success(f"Custom table created (id={table_id})")
            except sqlite3.IntegrityError:
                st.warning("Table already exists at this path")

    tables = db.fetchall("SELECT * FROM data_tables ORDER BY created_at DESC")
    if tables:
        st.subheader("4) Add Row to Custom Table")
        selected = st.selectbox(
            "Select Table",
            options=tables,
            format_func=lambda r: f"#{r['id']} {r['table_name']} @ {r['path']}",
        )
        cols = json.loads(selected["columns_json"])
        row_payload = {}
        for col in cols:
            row_payload[col["name"]] = st.text_input(f"{col['name']} ({col.get('type', 'text')})", key=f"col_{selected['id']}_{col['name']}")
        if st.button("Insert Row"):
            row_id = db.insert_custom_row(int(selected["id"]), row_payload)
            st.success(f"Inserted row_{row_id}")

st.divider()
st.subheader("🔎 RAG Search Test")
query = st.text_input("Prompt / Query", value="What is photosynthesis?")
top_k = st.slider("Top K", min_value=1, max_value=20, value=5)
if st.button("Run RAG Retrieval"):
    try:
        results = db.rag_search(query, top_k)
        if not results:
            st.info("No matches found. Try adding records or use different keywords.")
        else:
            for i, row in enumerate(results, 1):
                st.markdown(f"### {i}. `{row['source_key']}`")
                st.write(f"**Path:** {row['path']}")
                st.write(f"**BM25 score:** {row['score']:.4f}")
                st.write(f"**Snippet:** {row['snippet']}")
    except sqlite3.OperationalError:
        st.error("Query parser error. Try plain keywords (e.g., photosynthesis plants process)")

st.divider()
st.subheader("Data Snapshot")
col1, col2, col3 = st.columns(3)
with col1:
    pages = db.fetchall("SELECT path, class, subject, chapter, page FROM textbook_pages ORDER BY updated_at DESC LIMIT 10")
    st.write("Textbook Pages")
    st.dataframe(pd.DataFrame([dict(r) for r in pages]))
with col2:
    qa = db.fetchall("SELECT path, question, reference_path FROM qa_pairs ORDER BY updated_at DESC LIMIT 10")
    st.write("Q&A Pairs")
    st.dataframe(pd.DataFrame([dict(r) for r in qa]))
with col3:
    nodes = db.fetchall("SELECT node_type, node_name, path FROM hierarchy_nodes ORDER BY created_at DESC LIMIT 10")
    st.write("Hierarchy Nodes")
    st.dataframe(pd.DataFrame([dict(r) for r in nodes]))

st.caption(f"SQLite file: {Path(db.DB_FILE).resolve()}")
