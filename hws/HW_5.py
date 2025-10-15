# Simplified Legal News Bot — Upload-only + Chroma + OpenAI/Gemini (Normal/Quick)
# - Upload CSV, build Chroma index, ask 2 things:
#   (1) "Most interesting news" — deterministic legal-scoring
#   (2) "News about <topic>" — vector search + light re-rank
# - LLMs (OpenAI/Gemini) used only to explain results:
#     OpenAI: gpt-5 (Normal), gpt-5-nano (Quick)
#     Gemini: gemini-2.5-pro (Normal), gemini-2.5-flash-lite (Quick)
#
# NOTE: You must have keys in .streamlit/secrets.toml or env vars:
#   OPENAI_API_KEY, GEMINI_API_KEY
#
# To avoid embedding dimension mismatches, we use Chroma with
# SentenceTransformerEmbeddingFunction ("all-MiniLM-L6-v2", 384-d) and query_texts.

try:
    __import__("pysqlite3")
    import sys as _sys
    _sys.modules["sqlite3"] = _sys.modules.pop("pysqlite3")
except Exception:
    pass

import os, re, math, json
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import datetime

import streamlit as st
import pandas as pd
import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# LLM clients (explanations only)
from openai import OpenAI
try:
    import google.generativeai as genai
except Exception:
    genai = None

# --------------------------- UI --------------------------------
st.set_page_config(page_title="Legal News — Simple RAG", layout="wide")
st.title("⚖️ Legal News — Simple RAG")

st.markdown(
    "Upload a CSV of news items and then:\n\n"
    "• **Most interesting news** (law-firm lens)\n"
    "• **News about &lt;topic&gt;** (semantic)\n\n"
    "Ranking is deterministic (recency + legal keywords). An LLM can optionally "
    "explain the ordering using your chosen vendor & speed."
)

# Sidebar — minimal controls
st.sidebar.header("Upload CSV")
csv_file = st.sidebar.file_uploader("CSV (required columns below)", type=["csv"])
st.sidebar.caption("Required: id, title, summary, content, published_at (YYYY-MM-DD), source, url")

st.sidebar.header("LLM (explanations only)")
vendor = st.sidebar.selectbox("Vendor", ["OpenAI", "Gemini"])
speed = st.sidebar.radio("Speed", ["Normal", "Quick"], horizontal=True)

OPENAI_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
GEMINI_KEY = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))

# Map (vendor, speed) -> model names (as you requested)
MODEL_MAP = {
    ("OpenAI", "Normal"): "gpt-5",
    ("OpenAI", "Quick"):  "gpt-5-nano",
    ("Gemini", "Normal"): "gemini-2.5-pro",
    ("Gemini", "Quick"):  "gemini-2.5-flash-lite",
}

# Chroma defaults (hidden from UI for simplicity)
CHROMA_DIR = "data/chroma_simple_news"
COLLECTION = "legal_news_simple"
EMBED_FN = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")  # 384-d

# ---------------------- CSV loader & validator ------------------
REQUIRED = {"id","title","summary","content","published_at","source","url"}

def load_csv() -> Optional[pd.DataFrame]:
    if csv_file is None:
        st.info("Upload a CSV to continue.")
        return None
    try:
        return pd.read_csv(csv_file)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        return None

df = load_csv()
if df is None:
    st.stop()

missing = REQUIRED - set(df.columns)
if missing:
    st.error(f"CSV missing columns: {missing}")
    st.stop()

# ---------------------- Deterministic scoring -------------------
LAW_TERMS = [
    "lawsuit","litigation","sued","settlement","regulation","regulatory","fine","penalty",
    "cfpb","sec","doj","ftc","antitrust","compliance","investigation","merger","acquisition",
    "bankruptcy","governance","sanction","enforcement","gdpr","ico","data protection","privacy",
    "appeal","precedent","ransomware","breach","frand","class action"
]

def legal_keyword_score(text: str) -> float:
    t = (text or "").lower()
    hits = sum(1 for kw in LAW_TERMS if kw in t)
    return min(hits/8.0, 1.0)

def recency_score(ymd: str) -> float:
    try:
        dt = datetime.strptime(ymd, "%Y-%m-%d")
        days = max((datetime.utcnow() - dt).days, 0)
        return math.exp(-days * math.log(2) / 7.0)  # ~7-day half-life
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
        score = 0.5*r + 0.4*k + 0.1  # simple & auditable
        tags = []
        if r >= 0.5: tags.append("recent")
        if k >= 0.4: tags.append("legal-salient")
        ranked.append(Ranked(it.get("id"), title, url, round(score,4), tags))
    ranked.sort(key=lambda x: x.score, reverse=True)
    return ranked

# ------------------------- Chroma helpers ----------------------
def get_chroma():
    os.makedirs(CHROMA_DIR, exist_ok=True)
    return chromadb.PersistentClient(path=CHROMA_DIR, settings=Settings(allow_reset=False))

def build_collection(client: chromadb.PersistentClient, df: pd.DataFrame) -> None:
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    col = client.get_or_create_collection(COLLECTION, embedding_function=EMBED_FN)
    docs, ids, metas = [], [], []
    for _, row in df.iterrows():
        txt = f"{row['title']}\n\n{row.get('summary','')}\n\n{row.get('content','')}"
        docs.append(txt)
        ids.append(str(row["id"]))
        metas.append({
            "id": row["id"],
            "title": row["title"],
            "url": row["url"],
            "published_at": row["published_at"],
            "source": row["source"],
        })
    col.add(documents=docs, metadatas=metas, ids=ids)

def load_collection(client: chromadb.PersistentClient):
    try:
        return client.get_collection(COLLECTION, embedding_function=EMBED_FN)
    except Exception:
        return None

def retrieve_topic(collection, query: str, k: int) -> List[Dict[str, Any]]:
    res = collection.query(query_texts=[query], n_results=k, include=["documents","metadatas"])
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
    seed = "legal, regulatory, enforcement, litigation, compliance, antitrust, investigation"
    return retrieve_topic(collection, seed, k=k)

# -------------------- LLM explain (OpenAI/Gemini) --------------
def llm_explain(vendor: str, speed: str, rows: List[Ranked], df_map: Dict[Any, Dict[str, Any]]) -> str:
    if not rows:
        return "No items to explain."
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
    prompt = f"Items:\n{json.dumps(items, indent=2)}\n\nExplain in 2–3 sentences total."
    model = MODEL_MAP.get((vendor, speed))

    try:
        if vendor == "OpenAI":
            if not OPENAI_KEY:
                return "[Missing OPENAI_API_KEY]"
            client = OpenAI(api_key=OPENAI_KEY)
            resp = client.chat.completions.create(
                model=model,  # 'gpt-5' or 'gpt-5-nano'
                messages=[{"role":"system","content": sys},
                          {"role":"user","content": prompt}],
            )
            return resp.choices[0].message.content.strip()

        elif vendor == "Gemini":
            if not GEMINI_KEY:
                return "[Missing GEMINI_API_KEY]"
            if genai is None:
                return "[google-generativeai not installed]"
            genai.configure(api_key=GEMINI_KEY)
            gmodel = genai.GenerativeModel(model)  # 'gemini-2.5-pro' or 'gemini-2.5-flash-lite'
            r = gmodel.generate_content(sys + "\n\n" + prompt)
            return getattr(r, "text", "") or "[Empty response]"

        return "Explanation disabled."
    except Exception as e:
        return f"[LLM error: {e}]"

# ----------------------- Build Index ---------------------------
st.subheader("1) Build index")
client = get_chroma()
if st.button("Build / Rebuild"):
    with st.spinner("Indexing…"):
        build_collection(client, df)
    st.success("Vector index ready.")

collection = load_collection(client)
if not collection:
    st.warning("No collection loaded. Click 'Build / Rebuild' first.")

# ----------------------- Ask the bot ---------------------------
st.subheader("2) Ask")
mode = st.radio("Choose:", ["Most interesting news", "News about a topic"], horizontal=True)
topic = ""
if mode == "News about a topic":
    topic = st.text_input("Topic (e.g., GDPR, antitrust, mergers)")
top_k = st.slider("Top-K results", 3, 20, 8, 1)

if st.button("Run"):
    if not collection:
        st.error("Please build the index first.")
        st.stop()

    df_map = {row["id"]: row for _, row in df.iterrows()}

    if mode == "Most interesting news":
        base = retrieve_broad(collection, k=max(top_k*4, 40))
        ranked = heuristic_rank(base)[:top_k]
        st.markdown("### Most Interesting (law-firm lens)")
        for i, r in enumerate(ranked, 1):
            meta = df_map.get(r.id, {})
            st.markdown(
                f"**{i}. {r.title}** — _{', '.join(r.tags)}_\n\n"
                f"{meta.get('source','')} · {meta.get('published_at','')}  "
                + (f"[link]({r.url})" if r.url else "")
            )
        with st.expander("Why this order? (LLM)"):
            st.write(llm_explain(vendor, speed, ranked, df_map))

    else:
        if not topic.strip():
            st.warning("Enter a topic.")
            st.stop()
        hits = retrieve_topic(collection, topic.strip(), k=top_k)
        ranked = heuristic_rank(hits)
        st.markdown(f"### News about **{topic.strip()}**")
        for i, r in enumerate(ranked, 1):
            meta = df_map.get(r.id, {})
            st.markdown(
                f"**{i}. {r.title}** — _{', '.join(r.tags)}_\n\n"
                f"{meta.get('source','')} · {meta.get('published_at','')}  "
                + (f"[link]({r.url})" if r.url else "")
            )
        with st.expander("Why these items? (LLM)"):
            st.write(llm_explain(vendor, speed, ranked, df_map))

st.caption("Ranking = recency + legal keyword salience. LLM explains; it does not change ordering.")
