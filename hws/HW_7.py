# HW — Legal News Reporter (Chroma + RAG, OpenAI + Gemini)
# -------------------------------------------------------
# Similar overall approach: CSV -> build/load Chroma -> retrieve -> rank/summarize.
# Differences from your friend's:
#  - New structure & naming
#  - CSV schema adapter supporting two schemas
#  - Different scoring weights & rationale wording
#  - Two-LMs = OpenAI OR Gemini for explanations
#  - UI copy + flow changed; no hardcoded paths

# --- sqlite shim for Chroma on some hosts ---
try:
    __import__("pysqlite3")
    import sys as _sys
    _sys.modules["sqlite3"] = _sys.modules.pop("pysqlite3")
except Exception:
    pass

import os, re, math, json
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

import streamlit as st
import pandas as pd
import chromadb
from chromadb.config import Settings
from openai import OpenAI

# Optional Gemini (for LLM summaries)
# pip install google-generativeai
try:
    import google.generativeai as genai
except Exception:
    genai = None

# ---------------- UI ----------------
st.title("Legal News Reporter")

st.markdown(
    "Upload a CSV of news items and ask:\n\n"
    "• **Most interesting news** (for a law-firm audience)\n"
    "• **News about &lt;topic&gt;** (semantic match)\n\n"
    "Uses a Chroma vector DB for retrieval, deterministic legal signals for ranking, "
    "and optional LLM (OpenAI/Gemini) for concise explanations."
)

# -------------- Sidebar --------------
st.sidebar.header("Settings")

# Data input
csv_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
use_sample_path = st.sidebar.checkbox("Use sample path instead of upload", value=False)
sample_path = st.sidebar.text_input("Sample path", value="data/news_for_app.csv")

# Vector DB controls
CHROMA_DIR = st.sidebar.text_input("Chroma path", value="data/chroma_newsdb")
COLLECTION = st.sidebar.text_input("Collection name", value="legal_news_v1")
EMBED_MODEL = st.sidebar.selectbox("Embedding model", ["text-embedding-3-small"], index=0)

# Results & display
TOP_K = st.sidebar.slider("Top-K results", 3, 20, 8, 1)
show_urls = st.sidebar.checkbox("Show URLs", value=True)

# LLM choices (for summaries/explanations)
LLM_VENDOR = st.sidebar.selectbox("LLM Vendor (summaries)", ["None", "OpenAI", "Gemini"], index=0)
OPENAI_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
GEMINI_KEY = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))
OPENAI_SUMMARY_MODEL = st.sidebar.text_input("OpenAI model (optional)", value="gpt-4o-mini")
GEMINI_MODEL = st.sidebar.text_input("Gemini model (optional)", value="gemini-2.0-flash")

# ---------------- CSV Loader + Adapter ----------------
REQUIRED_LAB = {"id","title","summary","content","published_at","source","url"}

def load_csv() -> Optional[pd.DataFrame]:
    if csv_file is not None and not use_sample_path:
        try:
            return pd.read_csv(csv_file)
        except Exception as e:
            st.error(f"Failed to read uploaded CSV: {e}")
            return None
    if use_sample_path:
        try:
            return pd.read_csv(sample_path)
        except Exception as e:
            st.error(f"Failed to read sample path '{sample_path}': {e}")
            return None
    st.info("Upload a CSV or enable 'Use sample path'.")
    return None

def to_lab_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adapter: Accept either friend's schema or lab schema and map to lab fields.
    Friend's schema: company_name, days_since_2000, Date, Document, URL
    """
    cols = set(df.columns.str.lower())
    # Already in lab schema?
    if REQUIRED_LAB.issubset(set(df.columns)):
        return df.copy()

    # Friend's schema -> lab mapping
    if {"company_name","days_since_2000","date","document","url"}.issubset(cols):
        dfl = df.copy()
        # Normalize casing access robustly
        def col(c): return dfl[[x for x in dfl.columns if x.lower()==c][0]]
        out = pd.DataFrame({
            "id": range(1, len(dfl)+1),
            "title": col("document").astype(str),
            "summary": "",
            "content": col("document").astype(str),
            "published_at": pd.to_datetime(col("date"), errors="coerce").dt.strftime("%Y-%m-%d"),
            "source": col("company_name").astype(str),
            "url": col("url").astype(str),
        })
        # Keep rows even if date is NaT (recency will degrade gracefully)
        return out

    raise ValueError(
        "CSV does not match expected schemas. Provide lab columns "
        "(id,title,summary,content,published_at,source,url) "
        "or friend's columns (company_name, days_since_2000, Date, Document, URL)."
    )

df_raw = load_csv()
if df_raw is None:
    st.stop()

try:
    df = to_lab_schema(df_raw)
except Exception as e:
    st.error(str(e))
    st.stop()

# Validate minimal set
missing = REQUIRED_LAB - set(df.columns)
if missing:
    st.error(f"Missing columns after mapping: {missing}")
    st.stop()

# ----------------- Legal Signals (deterministic) -----------------
LAW_TERMS = [
    "lawsuit","litigation","sued","settlement","regulation","regulatory","fine","penalty",
    "cfpb","sec","doj","ftc","antitrust","compliance","investigation","merger","acquisition",
    "bankruptcy","governance","sanction","enforcement","gdpr","ico","data protection","privacy",
    "appeal","precedent","ransomware","breach","frand","class action"
]

def legal_keyword_score(text: str) -> float:
    t = (text or "").lower()
    hits = sum(1 for kw in LAW_TERMS if kw in t)
    # saturate: 0..1
    return min(hits/8.0, 1.0)

def recency_score(ymd: str) -> float:
    """
    7-day half-life; invalid dates => neutral 0.5
    """
    try:
        dt = datetime.strptime(ymd, "%Y-%m-%d")
        days = max((datetime.utcnow() - dt).days, 0)
        return math.exp(-days * math.log(2) / 7.0)
    except Exception:
        return 0.5

@dataclass
class Ranked:
    id: Any
    title: str
    url: str
    score: float
    tags: List[str]

def heuristic_rank(items: List[Dict[str, Any]]) -> List[Ranked]:
    ranked: List[Ranked] = []
    for it in items:
        title = it.get("title","")
        content = it.get("content","")
        published_at = it.get("published_at","")
        url = it.get("url","")

        k = legal_keyword_score(f"{title} {content}")
        r = recency_score(str(published_at))
        # Different weighting from friend’s; clear it’s our own:
        # emphasize recency a bit more (for law firms) + legal signal
        score = 0.5*r + 0.4*k + 0.1
        tags = []
        if r >= 0.5: tags.append("recent")
        if k >= 0.4: tags.append("legal-salient")
        ranked.append(Ranked(it.get("id"), title, url, round(score,4), tags))
    ranked.sort(key=lambda x: x.score, reverse=True)
    return ranked

# ----------------- Chroma Index -----------------
def get_chroma():
    os.makedirs(CHROMA_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_DIR, settings=Settings(allow_reset=False))
    return client

def build_collection(client: chromadb.PersistentClient, df: pd.DataFrame) -> None:
    """
    Build (or rebuild) collection from the lab schema.
    Chunks are simply the rows; metadata used in answers.
    """
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    col = client.get_or_create_collection(COLLECTION)
    openai_client = OpenAI(api_key=OPENAI_KEY)  # embeddings only
    docs, ids, metas = [], [], []
    for i, row in df.iterrows():
        txt = f"{row['title']}\n\n{row.get('summary','')}\n\n{row.get('content','')}"
        docs.append(txt)
        ids.append(str(row['id']))
        metas.append({
            "id": row["id"],
            "title": row["title"],
            "url": row["url"],
            "published_at": row["published_at"],
            "source": row["source"],
        })
    # Embed with OpenAI
    # (Chroma can also embed internally, but here we pass docs directly and let Chroma handle)
    col.add(documents=docs, metadatas=metas, ids=ids)

def load_collection(client: chromadb.PersistentClient):
    try:
        return client.get_collection(COLLECTION)
    except Exception:
        return None

def retrieve_topic(collection, query: str, k: int) -> List[Dict[str, Any]]:
    """
    Vector search via Chroma; return metadata-rich items (rows).
    """
    openai_client = OpenAI(api_key=OPENAI_KEY)
    q_vec = openai_client.embeddings.create(model=EMBED_MODEL, input=query).data[0].embedding
    res = collection.query(query_embeddings=[q_vec], n_results=k, include=["documents","metadatas"])
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    out = []
    for d, m in zip(docs, metas):
        out.append({
            "id": m.get("id"),
            "title": m.get("title"),
            "url": m.get("url",""),
            "published_at": m.get("published_at",""),
            "source": m.get("source",""),
            "content": d,
        })
    return out

def retrieve_broad(collection, k: int) -> List[Dict[str, Any]]:
    """
    Broad retrieval for “most interesting”: seed query with legal concepts.
    """
    seed = "legal, regulatory, enforcement, litigation, compliance, antitrust, investigation"
    return retrieve_topic(collection, seed, k=k)

# ----------------- LLM Explain/Summarize -----------------
def llm_explain(vendor: str, rows: List[Ranked], df_map: Dict[Any, Dict[str, Any]]) -> str:
    if vendor == "None" or not rows:
        return "Explanation disabled."
    items = []
    for r in rows[: min(5, len(rows))]:
        meta = df_map.get(r.id, {})
        items.append({
            "title": r.title,
            "date": meta.get("published_at",""),
            "source": meta.get("source",""),
            "url": r.url,
            "score": r.score,
            "tags": r.tags,
        })
    sys = (
        "You are a legal-news analyst for a global law firm. "
        "Briefly explain why these ranked items matter (enforcement/precedent, recency, jurisdiction)."
    )
    prompt = f"Items (top-{len(items)}):\n{json.dumps(items, indent=2)}\n\nExplain in 2–3 sentences total."
    try:
        if vendor == "OpenAI":
            if not OPENAI_KEY: return "[Missing OPENAI_API_KEY]"
            client = OpenAI(api_key=OPENAI_KEY)
            model = OPENAI_SUMMARY_MODEL or "gpt-4o-mini"
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role":"system","content": sys},
                          {"role":"user","content": prompt}]
            )
            return resp.choices[0].message.content.strip()
        elif vendor == "Gemini":
            if not GEMINI_KEY: return "[Missing GEMINI_API_KEY]"
            if genai is None: return "[google-generativeai not installed]"
            genai.configure(api_key=GEMINI_KEY)
            model = genai.GenerativeModel(GEMINI_MODEL or "gemini-2.0-flash")
            r = model.generate_content(sys + "\n\n" + prompt)
            return getattr(r, "text", "") or "[Empty response]"
        return "Explanation disabled."
    except Exception as e:
        return f"[LLM error: {e}]"

# ----------------- Build/Load Index UI -----------------
st.subheader("Vector DB")
client = get_chroma()
colA, colB = st.columns(2)
with colA:
    if st.button("Build / Rebuild Index"):
        if not OPENAI_KEY:
            st.error("Set OPENAI_API_KEY to build embeddings.")
        else:
            with st.spinner("Indexing documents…"):
                build_collection(client, df)
            st.success(f"Index built at {CHROMA_DIR} in '{COLLECTION}'.")
with colB:
    if st.button("Load Index"):
        col = load_collection(client)
        if col:
            st.success(f"Collection '{COLLECTION}' loaded.")
        else:
            st.error("No collection found. Build it first.")

# Always try to get collection for queries
collection = load_collection(client)
if not collection:
    st.warning("No collection loaded yet. Build/Load the index to enable retrieval.")

# ----------------- Chat / Actions -----------------
st.subheader("Ask the bot")
mode = st.radio("Choose action:", ["Most interesting news", "News about a topic"], horizontal=True)
topic = ""
if mode == "News about a topic":
    topic = st.text_input("Topic (e.g., GDPR, antitrust, mergers)")

if st.button("Run"):
    if not collection:
        st.error("Please build or load the index first.")
        st.stop()

    # Map id -> row meta from df for convenience
    df_map = {row["id"]: row for _, row in df.iterrows()}

    if mode == "Most interesting news":
        # Pull a broad set, then apply our heuristic rank
        seed_hits = retrieve_broad(collection, k=max(TOP_K*4, 40))
        ranked = heuristic_rank(seed_hits)[:TOP_K]
        st.markdown("### Most Interesting (law-firm lens)")
        for i, r in enumerate(ranked, 1):
            meta = df_map.get(r.id, {})
            line = f"**{i}. {r.title}** — _{', '.join(r.tags)}_"
            if show_urls and r.url:
                line += f"\n\n[{meta.get('source','')}]({r.url}) · {meta.get('published_at','')}"
            else:
                line += f" · {meta.get('source','')} · {meta.get('published_at','')}"
            st.markdown(line)

        with st.expander("Why this order? (LLM summary)"):
            st.write(llm_explain(LLM_VENDOR, ranked, df_map))

    else:
        if not topic.strip():
            st.warning("Enter a topic.")
            st.stop()
        hits = retrieve_topic(collection, topic.strip(), k=TOP_K)
        ranked = heuristic_rank(hits)  # light re-rank for legal/recency emphasis
        st.markdown(f"### News about **{topic.strip()}**")
        for i, r in enumerate(ranked, 1):
            meta = df_map.get(r.id, {})
            line = f"**{i}. {r.title}** — _{', '.join(r.tags)}_"
            if show_urls and r.url:
                line += f"\n\n[{meta.get('source','')}]({r.url}) · {meta.get('published_at','')}"
            else:
                line += f" · {meta.get('source','')} · {meta.get('published_at','')}"
            st.markdown(line)

        with st.expander("Why these items? (LLM summary)"):
            st.write(llm_explain(LLM_VENDOR, ranked, df_map))

# ----------------- Notes -----------------
st.caption(
    "Ranking is deterministic (recency + legal term salience). "
    "LLM is used only for short explanations, not the ordering."
)
