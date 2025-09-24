try:
    __import__("pysqlite3")
    import sys as _sys
    _sys.modules["sqlite3"] = _sys.modules.pop("pysqlite3")
except Exception:
    pass

import os
import sys
import re
from io import StringIO
from pathlib import Path
import streamlit as st
from openai import OpenAI
import chromadb
from chromadb.config import Settings
from bs4 import BeautifulSoup
import requests 
import glob



st.title("HW4 — iSchool RAG Chatbot (Student Orgs)")

st.sidebar.header("Settings")
model_choice = st.sidebar.selectbox(
    "Choose LLM",
    ["gpt-5", "Mistral", "Gemini 2.5"],
    index=0,
)
show_sources_inline = st.sidebar.checkbox("Show sources inline in the answer", value=True)
top_k = st.sidebar.slider("Top-K retrieved chunks", 1, 10, 4, 1)




OPENAI_KEY = st.secrets.get("OPENAI_API_KEY")
GEMINI_KEY = st.secrets.get("GEMINI_API_KEY")
MISTRAL_KEY = st.secrets.get("MISTRAL_API_KEY")

if not OPENAI_KEY:
    st.error("Missing OPENAI_API_KEY in .streamlit/secrets.toml")
    st.stop()

if "openai_client" not in st.session_state:
    st.session_state.openai_client = OpenAI(api_key=OPENAI_KEY)

openai_client = st.session_state.openai_client


DATA_DIR = Path("data/su_orgs/su_orgs")  
candidates = [
    Path("data/su_orgs"),
    Path("data/su_orgs/su_orgs"),  # nested by mistake
    Path("su_orgs"),
]
for p in candidates:
    if p.exists() and any(p.glob("*.html")):
        DATA_DIR = p
        break

PERSIST_DIR = Path("data/.hw4_chroma")          
COLLECTION_NAME = "ischool_orgs_hw4"


def html_to_text_file(fp: Path) -> str:
    """
    Extract visible text from an HTML file.
    Uses the stdlib html parser for portability.
    """
    html = fp.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.extract()
    text = soup.get_text(separator=" ")
    
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_into_two_sentence_aware(text: str) -> tuple[str, str]:
    
    if not text:
        return "", ""
    mid = len(text) // 2
    window_start = max(0, mid - 600)
    window = text[window_start:mid]

    
    boundary = max(window.rfind("."), window.rfind("!"), window.rfind("?"))
    split_idx = (window_start + boundary + 1) if boundary != -1 else mid

    c1 = text[:split_idx].strip()
    c2 = text[split_idx:].strip()

    if not c1 or not c2:
        
        half = max(1, len(text) // 2)
        c1, c2 = text[:half].strip(), text[half:].strip()

    return c1, c2


def get_chroma_collection():
    PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(PERSIST_DIR), settings=Settings(allow_reset=False))
    names = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in names:
        return client.get_collection(COLLECTION_NAME)
    return client.create_collection(COLLECTION_NAME, metadata={"chunks_per_file": 2, "source": "su_orgs_html"})

def build_index():
    
    if not DATA_DIR.exists():
        st.error(f"Data folder not found: {DATA_DIR.resolve()}")
        return None

    html_files = sorted(DATA_DIR.glob("*.html"))
    if not html_files:
        st.warning(f"No HTML files in {DATA_DIR}. Unzip your dataset there.")
        return None

    collection = get_chroma_collection()

    ids, docs, metas = [], [], []
    batch_size = 200

    with st.spinner("Indexing HTML files..."):
        for fp in html_files:
            text = html_to_text_file(fp)
            c1, c2 = split_into_two_sentence_aware(text)

            
            ids.extend([f"{fp.name}::1", f"{fp.name}::2"])
            docs.extend([c1, c2])
            metas.extend([
                {"filename": fp.name, "chunk": 1},
                {"filename": fp.name, "chunk": 2},
            ])

            if len(ids) >= batch_size:
                _flush_batch(collection, ids, docs, metas)
                ids, docs, metas = [], [], []

        
        if ids:
            _flush_batch(collection, ids, docs, metas)

    st.success("Index built / updated.")
    return collection

def _flush_batch(collection, ids, docs, metas):
    
    vectors = []
    for t in docs:
        if not t:
            t = " "
        emb = openai_client.embeddings.create(model="text-embedding-3-small", input=t)
        vectors.append(emb.data[0].embedding)
    collection.add(ids=ids, documents=docs, metadatas=metas, embeddings=vectors)


def retrieve_chunks(collection, question: str, k: int):
    
    q_vec = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=question
    ).data[0].embedding

    
    res = collection.query(
        query_embeddings=[q_vec],
        n_results=k,
        include=["documents", "metadatas"]
    )
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    return docs, metas



def call_openai(messages):
    resp = openai_client.chat.completions.create(
        model="gpt-5",
        messages=messages,)
    return resp.choices[0].message.content.strip()

def call_gemini(messages):
    if not GEMINI_KEY:
        return "[Missing GEMINI_API_KEY]"
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_KEY)
    model = genai.GenerativeModel("gemini-2.5-pro")
    
    prompt = "\n".join(m["content"] for m in messages)
    resp = model.generate_content(prompt)
    return getattr(resp, "text", "") or "[Empty response]"

def call_mistral(messages):
    if not MISTRAL_KEY:
        return "[Missing MISTRAL_API_KEY]"
    headers = {"Authorization": f"Bearer {MISTRAL_KEY}", "Content-Type": "application/json"}
    payload = {"model": "mistral-small-2506", "messages": messages, "stream": False}
    r = requests.post("https://api.mistral.ai/v1/chat/completions", headers=headers, json=payload, timeout=60)
    data = r.json()
    if "choices" not in data:
        return f"[Mistral error: {data}]"
    return data["choices"][0]["message"]["content"].strip()

def ask_model(provider_name: str, messages):
    if provider_name == "gpt-5":
        return call_openai(messages)
    elif provider_name == "Gemini 2.5":
        return call_gemini(messages)
    else:
        return call_mistral(messages)


st.subheader("Step 1 — Build / Load Index")
col_a, col_b = st.columns([1,1])
with col_a:
    st.write(f"Data folder: `{DATA_DIR}`")
with col_b:
    st.write(f"Persist dir: `{PERSIST_DIR}`")

if "collection" not in st.session_state:
    st.session_state.collection = None

if st.button("🔧 Build / Update Index"):
    st.session_state.collection = build_index()

if st.session_state.collection is None and PERSIST_DIR.exists():
    try:
        client = chromadb.PersistentClient(path=str(PERSIST_DIR))
        st.session_state.collection = client.get_collection(COLLECTION_NAME)
    except Exception:
        pass

if st.session_state.collection:
    try:
        st.info(f"Collection ready • {st.session_state.collection.count()} chunks")
    except Exception:
        st.info("Collection ready")

# 
# Chat state

if "chat_log" not in st.session_state:
    st.session_state.chat_log = []  
if "qa_memory" not in st.session_state:
    st.session_state.qa_memory = [] 

# render history
st.subheader("Step 2 — Chat")
for m in st.session_state.chat_log:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])


user_q = st.chat_input("Ask about iSchool student organizations…")
if user_q:
    st.session_state.chat_log.append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.markdown(user_q)

    if not st.session_state.collection:
        st.error("Please build/load the index first (top of page).")
        st.stop()

    
    docs, metas = retrieve_chunks(st.session_state.collection, user_q, k=top_k)

    
    src_list = []
    for md in metas or []:
        fn = md.get("filename")
        ch = md.get("chunk")
        if fn:
            tag = f"{fn}#chunk{ch}"
            if tag not in src_list:
                src_list.append(tag)

    evidence = "\n\n---\n\n".join(docs) if docs else ""
    src_line = f"\n\nSources: {', '.join(src_list)}" if src_list else ""

    
    system_msg = (
        "You are a helpful assistant for Syracuse iSchool student organizations. "
        "Use the retrieved context to answer. If the context doesn't contain the answer, say so briefly."
    )
    
    mem_pairs = st.session_state.qa_memory[-5:]
    mem_block = ""
    if mem_pairs:
        mem_lines = [f"Q: {x['q']}\nA: {x['a']}" for x in mem_pairs]
        mem_block = "\n\nPrevious Q&A (last 5):\n" + "\n\n".join(mem_lines)

    user_block = f"Context:\n{evidence}\n\nQuestion: {user_q}\n"
    if show_sources_inline and src_line:
        user_block += src_line

    messages = [
        {"role": "system", "content": system_msg + (("\n\n" + mem_block) if mem_block else "")},
        {"role": "user", "content": user_block},
    ]

    
    answer = ask_model(model_choice, messages)

    
    with st.chat_message("assistant"):
        st.markdown(answer)
        with st.expander("Show retrieved sources"):
            if src_list:
                for s in src_list:
                    st.write("- " + s)
            else:
                st.write("No sources retrieved.")

    st.session_state.chat_log.append({"role": "assistant", "content": answer})
    st.session_state.qa_memory.append({"q": user_q, "a": answer})
    st.session_state.qa_memory = st.session_state.qa_memory[-5:]  
