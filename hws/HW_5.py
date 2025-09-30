try:
    __import__("pysqlite3")
    import sys as _sys
    _sys.modules["sqlite3"] = _sys.modules.pop("pysqlite3")
except Exception:
    pass

import re
from pathlib import Path
import streamlit as st
from openai import OpenAI
import chromadb
from chromadb.config import Settings

st.title("HW5 — Short-Term Memory RAG (iSchool Orgs)")

st.sidebar.header("Settings")
model_choice = st.sidebar.selectbox(
    "Choose LLM", ["gpt-5", "Mistral", "Gemini 2.5"], index=0
)
top_k = st.sidebar.slider("Top-K retrieved chunks", 1, 10, 4, 1)
memory_turns = st.sidebar.slider("Short-term memory (last N Q&A)", 0, 10, 5, 1)
show_sources_inline = st.sidebar.checkbox("Show sources inline in the answer", value=True)


OPENAI_KEY = st.secrets.get("OPENAI_API_KEY")
GEMINI_KEY = st.secrets.get("GEMINI_API_KEY")
MISTRAL_KEY = st.secrets.get("MISTRAL_API_KEY")

if not OPENAI_KEY:
    st.error("Missing OPENAI_API_KEY in .streamlit/secrets.toml")
    st.stop()

if "openai_client" not in st.session_state:
    st.session_state.openai_client = OpenAI(api_key=OPENAI_KEY)
openai_client = st.session_state.openai_client


DATA_DIR = Path("data/su_orgs")
PERSIST_DIR = Path("data/.chroma_hw")
COLLECTION_NAME = "ischool_orgs_hw4"

def get_chroma_collection():
    PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(PERSIST_DIR), settings=Settings(allow_reset=False))
    names = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in names:
        return client.get_collection(COLLECTION_NAME)
    return client.create_collection(COLLECTION_NAME, metadata={"chunks_per_file": 2, "source": "su_orgs_html"})


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

def get_relevant_org_info(query: str, k: int = 4):
    
    if not st.session_state.collection:
        return "", ""

    docs, metas = retrieve_chunks(st.session_state.collection, query, k)
    
    relevant_text = "\n\n---\n\n".join(docs) if docs else ""

    src_list = []
    for md in metas or []:
        fn = md.get("filename")
        ch = md.get("chunk")
        if fn:
            tag = f"{fn}#chunk{ch}"
            if tag not in src_list:
                src_list.append(tag)
    sources_line = f"Sources: {', '.join(src_list)}" if src_list else ""
    return relevant_text, sources_line


def call_openai(messages):
    resp = openai_client.chat.completions.create(model="gpt-5", messages=messages)
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
    import requests
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

if "collection" not in st.session_state:
    st.session_state.collection = None

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
else:
    st.warning("No Chroma collection found yet. Build it from the HW4 page first.")


if "chat_log" not in st.session_state:
    st.session_state.chat_log = []      # full chat UI log
if "qa_memory" not in st.session_state:
    st.session_state.qa_memory = []     # Q/A pairs for short-term memory

# Render history
st.subheader("Chat")
for m in st.session_state.chat_log:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

user_q = st.chat_input("Ask about iSchool student organizations…")
if user_q:
    st.session_state.chat_log.append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.markdown(user_q)

    if not st.session_state.collection:
        st.error("Please build/load the index on the HW4 page first.")
    else:
        
        relevant_text, sources_line = get_relevant_org_info(user_q, k=top_k)

        # Build short-term memory block
        mem_pairs = st.session_state.qa_memory[-memory_turns:] if memory_turns > 0 else []
        mem_block = ""
        if mem_pairs:
            mem_lines = [f"Q: {x['q']}\nA: {x['a']}" for x in mem_pairs]
            mem_block = "\n\nPrevious Q&A (last {}):\n".format(len(mem_pairs)) + "\n\n".join(mem_lines)

        # HW5: invoke LLM with results of vector search (NOT with raw prompt embeddings)
        system_msg = (
            "You are a helpful assistant for Syracuse iSchool student organizations. "
            "Use ONLY the provided CONTEXT to answer. If it's not in CONTEXT, say you don't know and suggest a related query."
        )

        context_block = f"CONTEXT:\n{relevant_text or '[no results]'}"
        if show_sources_inline and sources_line:
            context_block += f"\n\n{sources_line}"

        # Put memory into the system prompt per HW5 guidance 
        sys_full = system_msg + (("\n\n" + mem_block) if mem_block else "")

        messages = [
            {"role": "system", "content": sys_full},
            {"role": "user", "content": f"{context_block}\n\nUSER QUESTION: {user_q}"},
        ]

        answer = ask_model(model_choice, messages)

        with st.chat_message("assistant"):
            st.markdown(answer)
            with st.expander("Show retrieved context"):
                st.write(relevant_text if relevant_text else "No context retrieved.")
                if sources_line:
                    st.caption(sources_line)

        # Save to chat + memory
        st.session_state.chat_log.append({"role": "assistant", "content": answer})
        st.session_state.qa_memory.append({"q": user_q, "a": answer})
        # keep last N pairs only
        if memory_turns > 0:
            st.session_state.qa_memory = st.session_state.qa_memory[-memory_turns:]
        else:
            st.session_state.qa_memory = []
