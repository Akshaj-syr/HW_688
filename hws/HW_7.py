
try:
    __import__("pysqlite3")
    import sys as _sys
    _sys.modules["sqlite3"] = _sys.modules.pop("pysqlite3")
except Exception:
    pass

import os, math, json
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any, Optional

import streamlit as st
import pandas as pd
import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from openai import OpenAI

try:
    import google.generativeai as genai
except Exception:
    genai = None


st.title("News Info Bot")

st.markdown(
    "Upload your CSV (company_name, days_since_2000, Date, Document, URL) and ask:\n"
    "• **Most interesting news**\n"
    "• **News about a topic**\n\n"
    "Ranking = recency + legal keywords. LLMs (OpenAI/Gemini) explain rankings only."
)

st.sidebar.header("Upload CSV")
csv_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

st.sidebar.header("LLM Options")
vendor = st.sidebar.selectbox("Vendor", ["OpenAI", "Gemini"])
speed = st.sidebar.radio("Speed", ["Normal", "Quick"], horizontal=True)

OPENAI_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
GEMINI_KEY = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))

MODEL_MAP = {
    ("OpenAI", "Normal"): "gpt-5",
    ("OpenAI", "Quick"): "gpt-5-nano",
    ("Gemini", "Normal"): "gemini-2.5-pro",
    ("Gemini", "Quick"): "gemini-2.5-flash-lite",
}

CHROMA_DIR = "data/chroma_simple_news"
COLLECTION = "legal_news_simple"
EMBED_FN = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")


def load_csv() -> Optional[pd.DataFrame]:
    if csv_file is None:
        st.info("Upload your CSV file to continue.")
        return None
    try:
        df = pd.read_csv(csv_file)
        
        cols = set(df.columns.str.lower())
        if {"company_name", "days_since_2000", "date", "document", "url"}.issubset(cols):
            df_out = pd.DataFrame({
                "id": range(1, len(df) + 1),
                "title": df["Document"].astype(str),
                "summary": "",
                "content": df["Document"].astype(str),
                "published_at": pd.to_datetime(df["Date"], errors="coerce").dt.strftime("%Y-%m-%d"),
                "source": df["company_name"].astype(str),
                "url": df["URL"].astype(str)
            })
            return df_out
        else:
            return df
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        return None

df = load_csv()
if df is None:
    st.stop()


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
    ranked = []
    for it in items:
        title, content = it.get("title", ""), it.get("content", "")
        date, url = it.get("published_at", ""), it.get("url", "")
        k, r = legal_keyword_score(f"{title} {content}"), recency_score(date)
        score = 0.5*r + 0.4*k + 0.1
        tags = []
        if r >= 0.5: tags.append("recent")
        if k >= 0.4: tags.append("legal-salient")
        ranked.append(Ranked(it.get("id"), title, url, round(score, 4), tags))
    return sorted(ranked, key=lambda x: x.score, reverse=True)


def get_chroma():
    os.makedirs(CHROMA_DIR, exist_ok=True)
    return chromadb.PersistentClient(path=CHROMA_DIR, settings=Settings(allow_reset=False))

def build_collection(client, df):
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    col = client.get_or_create_collection(COLLECTION, embedding_function=EMBED_FN)
    docs, ids, metas = [], [], []
    for _, row in df.iterrows():
        txt = f"{row['title']}\n\n{row['summary']}\n\n{row['content']}"
        docs.append(txt)
        ids.append(str(row["id"]))
        metas.append({
            "id": row["id"],
            "title": row["title"],
            "url": row["url"],
            "published_at": row["published_at"],
            "source": row["source"]
        })
    col.add(documents=docs, metadatas=metas, ids=ids)

def load_collection(client):
    try:
        return client.get_collection(COLLECTION, embedding_function=EMBED_FN)
    except Exception:
        return None

def retrieve_topic(collection, query, k):
    res = collection.query(query_texts=[query], n_results=k, include=["documents","metadatas"])
    docs, metas = res.get("documents", [[]])[0], res.get("metadatas", [[]])[0]
    return [{"id": m.get("id"), "title": m.get("title"), "url": m.get("url",""),
             "published_at": m.get("published_at",""), "source": m.get("source",""),
             "content": d} for d, m in zip(docs, metas)]

def retrieve_broad(collection, k):
    seed = "legal, regulatory, enforcement, litigation, compliance, antitrust, investigation"
    return retrieve_topic(collection, seed, k=k)


def llm_explain(vendor, speed, rows, df_map):
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
            "tags": r.tags
        })
    sys = ("You are a legal-news analyst. Summarize why these top items matter "
           "(enforcement, recency, or impact).")
    prompt = f"Items:\n{json.dumps(items, indent=2)}\n\nExplain in 2–3 sentences."
    model = MODEL_MAP.get((vendor, speed))
    try:
        if vendor == "OpenAI":
            if not OPENAI_KEY: return "[Missing OPENAI_API_KEY]"
            client = OpenAI(api_key=OPENAI_KEY)
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": sys},
                          {"role": "user", "content": prompt}]
            )
            return resp.choices[0].message.content.strip()
        elif vendor == "Gemini":
            if not GEMINI_KEY: return "[Missing GEMINI_API_KEY]"
            if genai is None: return "[google-generativeai not installed]"
            genai.configure(api_key=GEMINI_KEY)
            gmodel = genai.GenerativeModel(model)
            r = gmodel.generate_content(sys + "\n\n" + prompt)
            return getattr(r, "text", "") or "[Empty response]"
        return "Explanation disabled."
    except Exception as e:
        return f"[LLM error: {e}]"


st.subheader("1) Build index")
client = get_chroma()
if st.button("Build / Rebuild Index"):
    with st.spinner("Indexing documents..."):
        build_collection(client, df)
    st.success("Vector index built successfully!")

collection = load_collection(client)
if not collection:
    st.warning("No collection found. Click 'Build / Rebuild Index' first.")

st.subheader("2) Ask a question")
mode = st.radio("Select:", ["Most interesting news", "News about a topic"], horizontal=True)
topic = ""
if mode == "News about a topic":
    topic = st.text_input("Enter a topic (e.g., GDPR, antitrust, mergers)")
top_k = st.slider("Top-K results", 3, 20, 8, 1)

if st.button("Run"):
    if not collection:
        st.error("Please build the index first.")
        st.stop()
    df_map = {r["id"]: r for _, r in df.iterrows()}
    if mode == "Most interesting news":
        hits = retrieve_broad(collection, k=max(top_k*4, 40))
        ranked = heuristic_rank(hits)[:top_k]
        st.markdown("### Most Interesting News")
        for i, r in enumerate(ranked, 1):
            meta = df_map.get(r.id, {})
            st.markdown(f"**{i}. {r.title}** — _{', '.join(r.tags)}_\n\n"
                        f"{meta.get('source','')} · {meta.get('published_at','')}  "
                        + (f"[link]({r.url})" if r.url else ""))
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
            st.markdown(f"**{i}. {r.title}** — _{', '.join(r.tags)}_\n\n"
                        f"{meta.get('source','')} · {meta.get('published_at','')}  "
                        + (f"[link]({r.url})" if r.url else ""))
        with st.expander("Why these items? (LLM)"):
            st.write(llm_explain(vendor, speed, ranked, df_map))


