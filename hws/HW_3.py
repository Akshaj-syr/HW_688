import streamlit as st
import requests
import json
from bs4 import BeautifulSoup
from openai import OpenAI
import google.generativeai as genai

st.set_page_config(page_title="HW3 Chatbot")

st.title("HW3 — Streaming Chatbot with URLs")


url1 = st.sidebar.text_input("Enter first URL")
url2 = st.sidebar.text_input("Enter second URL (optional)")

provider = st.sidebar.selectbox(
    "LLM provider",
    ["OpenAI", "Google (Gemini)", "Mistral"],
    index=0,
)

use_advanced = st.sidebar.checkbox("Use Advanced Model", value=False)

memory_type = st.sidebar.radio(
    "Conversation memory type",
    ["Buffer (6 Qs)", "Summary", "Buffer (2000 tokens)"],
    index=0,
)


openai_key = st.secrets.get("OPENAI_API_KEY")
gemini_key = st.secrets.get("GEMINI_API_KEY")
mistral_key = st.secrets.get("MISTRAL_API_KEY")


def read_url_content(u: str) -> str:
    try:
        r = requests.get(u, timeout=20)
        soup = BeautifulSoup(r.content, "html.parser")
        return soup.get_text(separator="\n", strip=True)
    except Exception as e:
        return f"[Error fetching {u}: {e}]"

def get_combined_context() -> str:
    text1, text2 = "", ""
    if url1:
        text1 = read_url_content(url1)
    if url2:
        text2 = read_url_content(url2)
    combined = text1 + "\n\n" + text2
    return combined[:20000]  


if "chat" not in st.session_state:
    st.session_state.chat = [
        {"role": "system", "content": "Explain things simply and clearly."}
    ]
if "summary" not in st.session_state:
    st.session_state.summary = ""

def apply_memory(msgs):
    if memory_type == "Buffer (6 Qs)":
        return msgs[-12:] 
    elif memory_type == "Summary":
        return [
            {"role": "system", "content": "Summary of conversation so far: " + st.session_state.summary}
        ] + msgs[-2:] 
    elif memory_type == "Buffer (2000 tokens)":
       
        total = ""
        result = []
        for m in reversed(msgs):
            if len(total) + len(m["content"]) > 8000:
                break
            result.insert(0, m)
            total += m["content"]
        return result
    return msgs


def stream_openai(messages, advanced):
    client = OpenAI(api_key=openai_key)
    model = "gpt-4o" if advanced else "gpt-4o-mini"
    with st.chat_message("assistant"):
        text = ""
        spot = st.empty()
        for chunk in client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
        ):
            part = chunk.choices[0].delta.content or ""
            text += part
            spot.write(text)
        return text

def stream_gemini(messages, advanced):
    genai.configure(api_key=gemini_key)
    model = "gemini-1.5-pro" if advanced else "gemini-1.5-flash"
    g = genai.GenerativeModel(model)
    full_prompt = "\n".join([m["content"] for m in messages])
    resp = g.generate_content(full_prompt)
    with st.chat_message("assistant"):
        spot = st.empty()
        for i in range(0, len(resp.text), 50):
            spot.write(resp.text[:i+50])
        return resp.text

def stream_mistral(messages, advanced):
    model = "mistral-large-latest" if advanced else "mistral-small-latest"
    headers = {"Authorization": f"Bearer {mistral_key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "stream": False}
    r = requests.post("https://api.mistral.ai/v1/chat/completions",
                      headers=headers, data=json.dumps(payload), timeout=60)
    data = r.json()
    if "choices" not in data:
        return f"Mistral error: {data.get('error', data)}"
    text = data["choices"][0]["message"]["content"]
    with st.chat_message("assistant"):
        spot = st.empty()
        for i in range(0, len(text), 50):
            spot.write(text[:i+50])
        return text


for m in st.session_state.chat:
    with st.chat_message(m["role"]):
        st.write(m["content"])

user_msg = st.chat_input("Type your question...")

if user_msg:

    st.session_state.chat.append({"role": "user", "content": user_msg})

    
    context = get_combined_context()
    if context:
        st.session_state.chat.append({"role": "system", "content": "Reference info:\n" + context})

    
    msgs = apply_memory(st.session_state.chat)

    
    if provider == "OpenAI":
        out = stream_openai(msgs, use_advanced)
    elif provider == "Google (Gemini)":
        out = stream_gemini(msgs, use_advanced)
    else:
        out = stream_mistral(msgs, use_advanced)

    
    st.session_state.chat.append({"role": "assistant", "content": out})

    
    if memory_type == "Summary":
        st.session_state.summary += " " + user_msg + " -> " + out
