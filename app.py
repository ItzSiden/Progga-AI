import json
import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

import db

st.set_page_config(page_title="NCTB Repo-Style DB", layout="wide")
db.init_db()

st.title("🗂️ NCTB Structured Repository + RAG")
st.caption("GitHub-style browsing + custom tables/rows + retrieval on top relevant rows.")

# ---------- Sidebar: Repository Browser ----------
with st.sidebar:
    st.header("Repository Browser")
    nodes = db.get_all_nodes()
    all_paths = [n["path"] for n in nodes]

    selected_path = st.selectbox(
        "Browse Path",
        options=["/"] + all_paths,
        index=0,
        help="Select a folder-like path similar to browsing a GitHub repository tree.",
    )

    st.markdown("**Quick Create Folder**")
    folder_name = st.text_input("Folder Name", placeholder="Chapter 3")
    if st.button("Create Folder"):
        if not folder_name.strip():
            st.warning("Folder name required")
        else:
            base = "" if selected_path == "/" else selected_path
            full = folder_name.strip() if base == "" else f"{base} / {folder_name.strip()}"
            parent_id = None
            if base:
                parent = db.fetchall("SELECT id FROM hierarchy_nodes WHERE path=?", (base,))
                parent_id = int(parent[0]["id"]) if parent else None
            db.upsert_node("Folder", folder_name.strip(), parent_id, full, {})
            st.success(f"Created: {full}")
            st.rerun()

    st.markdown("---")
    st.markdown("### Tree Preview")
    preview = [p for p in all_paths if selected_path == "/" or p.startswith(selected_path)]
    for p in preview[:80]:
        depth = p.count("/")
        st.write(f"{'  ' * depth}📁 {p}")

# ---------- Main tabs ----------
repo_tab, textbook_tab, qa_tab, custom_tab, rag_tab = st.tabs(
    ["Repo View", "Textbook", "Guide Q&A", "Custom Tables", "RAG Search"]
)

with repo_tab:
    st.subheader("Repository-like File View")
    st.write(f"Current Path: `{selected_path}`")

    child_paths = []
    if selected_path == "/":
        child_paths = [p for p in all_paths if " / " not in p]
    else:
        prefix = selected_path + " / "
        child_paths = [p for p in all_paths if p.startswith(prefix) and p.count(" / ") == selected_path.count(" / ") + 1]

    st.markdown("#### Folders")
    if child_paths:
        st.dataframe(pd.DataFrame({"path": child_paths}), use_container_width=True)
    else:
        st.info("No child folders here yet.")

    st.markdown("#### Data Objects at this path")
    at_path_tables = db.get_custom_tables(None if selected_path == "/" else selected_path)
    if at_path_tables:
        table_df = pd.DataFrame([
            {
                "id": r["id"],
                "table": r["table_name"],
                "path": r["path"],
                "columns": r["columns_json"],
            }
            for r in at_path_tables
        ])
        st.dataframe(table_df, use_container_width=True)
    else:
        st.caption("No custom tables at this path yet.")

with textbook_tab:
    st.subheader("Add Full Textbook Page")
    with st.form("textbook_form"):
        c = st.number_input("Class", min_value=1, max_value=12, value=8)
        subj = st.text_input("Subject", value="Science")
        ch = st.number_input("Chapter", min_value=1, value=2)
        pg = st.number_input("Page", min_value=1, value=21)
        content = st.text_area("Full Page Content", height=200)
        md = st.text_area("Metadata JSON", value='{"source":"nctb"}')
        submitted = st.form_submit_button("Save Page")
        if submitted:
            path = f"Class {c} / {subj} / Textbook / Chapter {ch} / Page {pg}"
            db.upsert_textbook_page(
                {
                    "path": path,
                    "class": int(c),
                    "subject": subj,
                    "chapter": int(ch),
                    "page": int(pg),
                    "content": content,
                    "metadata": json.loads(md),
                }
            )
            st.success(f"Saved and auto-mapped: {path}")

with qa_tab:
    st.subheader("Add Full Guidebook Q&A")
    with st.form("qa_form"):
        c = st.number_input("Class ", min_value=1, max_value=12, value=8)
        subj = st.text_input("Subject ", value="Science")
        ch = st.number_input("Chapter ", min_value=1, value=2)
        qa_key = st.text_input("Q&A Key", value="Q&A_45")
        q = st.text_area("Question")
        a = st.text_area("Answer")
        ref = st.text_input("Reference Path", value="Class 8 / Science / Textbook / Chapter 2 / Page 21")
        md = st.text_area("Metadata JSON ", value='{"difficulty":"easy"}')
        submitted = st.form_submit_button("Save Q&A")
        if submitted:
            path = f"Class {c} / {subj} / Guide / Chapter {ch} / {qa_key}"
            db.upsert_qa_pair(
                {
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
            )
            st.success(f"Saved and auto-mapped: {path}")

with custom_tab:
    st.subheader("Create Custom Table + Add Rows")
    default_path = selected_path if selected_path != "/" else "Class 8 / Science"

    with st.form("create_custom_table"):
        table_name = st.text_input("Table Name", value="learning_objectives")
        attach_path = st.text_input("Attach Path", value=default_path)
        columns_spec = st.text_area(
            "Columns JSON",
            value='[{"name":"objective_id","type":"text"},{"name":"objective","type":"text"},{"name":"bloom_level","type":"text"}]',
            height=120,
        )
        submitted = st.form_submit_button("Create Table")
        if submitted:
            try:
                new_id = db.create_custom_table(table_name, attach_path, json.loads(columns_spec))
                st.success(f"Created table #{new_id} at {attach_path}")
                st.rerun()
            except sqlite3.IntegrityError:
                st.warning("A table with same name already exists at that path")

    tables = db.get_custom_tables(None if selected_path == "/" else selected_path)
    if tables:
        picked = st.selectbox(
            "Select Table",
            options=tables,
            format_func=lambda r: f"#{r['id']} {r['table_name']} @ {r['path']}",
        )
        columns = json.loads(picked["columns_json"])

        st.markdown("#### Insert Row")
        row_payload = {}
        cols = st.columns(min(len(columns), 3) or 1)
        for idx, col in enumerate(columns):
            with cols[idx % len(cols)]:
                row_payload[col["name"]] = st.text_input(col["name"], key=f"row_{picked['id']}_{col['name']}")
        if st.button("Insert Row", type="primary"):
            row_id = db.insert_custom_row(int(picked["id"]), row_payload)
            st.success(f"Inserted row_{row_id}")
            st.rerun()

        st.markdown("#### Table Rows")
        rows = db.get_custom_rows(int(picked["id"]))
        if rows:
            row_df = pd.DataFrame([
                {"row_id": r["id"], **json.loads(r["row_json"]), "created_at": r["created_at"]}
                for r in rows
            ])
            st.dataframe(row_df, use_container_width=True)
        else:
            st.info("No rows yet.")

with rag_tab:
    st.subheader("RAG Retrieval Test (Top Relevant Rows)")
    query = st.text_input("Prompt", value="What is photosynthesis?")
    top_k = st.slider("Top K", min_value=1, max_value=20, value=5)

    if st.button("Run Retrieval"):
        try:
            results = db.rag_search(query, top_k)
            if not results:
                st.info("No relevant rows found.")
            else:
                result_data = []
                for r in results:
                    result_data.append(
                        {
                            "source_key": r["source_key"],
                            "path": r["path"],
                            "bm25_score": round(float(r["score"]), 5),
                            "snippet": r["snippet"],
                        }
                    )
                st.dataframe(pd.DataFrame(result_data), use_container_width=True)
        except sqlite3.OperationalError:
            st.error("Invalid FTS query syntax. Try plain keywords.")

st.divider()
st.caption(f"SQLite file (auto-updated): {Path(db.DB_FILE).resolve()}")
