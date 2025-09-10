import json
import requests
import streamlit as st
from bs4 import BeautifulSoup
from openai import OpenAI
import google.generativeai as genai

st.title("HW2 — URL Summarizer")

url = st.text_input("Enter a web page URL")

summary_style = st.sidebar.radio(
    "Summary format",
    ["100 words", "2 connecting paragraphs", "5 bullet points"],
    index=0,
)

output_language = st.sidebar.selectbox(
    "Output language",
    ["English", "French", "Spanish"],
    index=0,
)

provider = st.sidebar.selectbox(
    "LLM provider",
    ["OpenAI", "Google (Gemini)", "Mistral"],
    index=0,
)

use_advanced = st.sidebar.checkbox("Use Advanced Model", value=False)

def read_url_content(u: str) -> str:
    r = requests.get(u, timeout=20)
    soup = BeautifulSoup(r.content, "html.parser")
    return soup.get_text(separator="\n", strip=True)

def build_instructions(style: str, language: str) -> str:
    if style == "100 words":
        s = "Summarize the document in about 100 words (maximum 120)."
    elif style == "2 connecting paragraphs":
        s = "Summarize the document in exactly two connected paragraphs."
    else:
        s = "Summarize the document as exactly five concise bullet points."
    return f"{s} Respond in {language}. Be faithful to the source and avoid adding facts not present."

def summarize_openai(text: str, style: str, language: str, advanced: bool) -> str:
    key = st.secrets["OPENAI_API_KEY"]
    model = "gpt-4o" if advanced else "gpt-4o-mini"
    client = OpenAI(api_key=key)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You produce faithful, concise summaries."},
            {"role": "user", "content": f"{build_instructions(style, language)}\n\n----\n{text}\n----"},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content

def summarize_gemini(text: str, style: str, language: str, advanced: bool) -> str:
    key = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=key)
    model = "gemini-1.5-pro" if advanced else "gemini-1.5-flash"
    g = genai.GenerativeModel(
        model,
        system_instruction="You produce faithful, concise summaries."
    )
    prompt = f"{build_instructions(style, language)}\n\n----\n{text}\n----"
    resp = g.generate_content(prompt, generation_config={"temperature": 0.2})
    return resp.text

def summarize_mistral(text, style, language, advanced):
    key = st.secrets["MISTRAL_API_KEY"]
    model = "mistral-large-latest" if advanced else "mistral-small-latest"

    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You produce faithful, concise summaries."},
            {"role": "user", "content": f"{build_instructions(style, language)}\n\n----\n{text}\n----"},
        ],
        "temperature": 0.2,
    }

    r = requests.post("https://api.mistral.ai/v1/chat/completions",
                      headers=headers, data=json.dumps(payload), timeout=60)
    data = r.json()

    if "choices" not in data:
        return f"Mistral error: {data.get('error', data)}"

    return data["choices"][0]["message"]["content"]

if st.button("Summarize", type="primary") and url:
    page_text = read_url_content(url)
   
    page_text = page_text[:20000]
    if provider == "OpenAI":
        out = summarize_openai(page_text, summary_style, output_language, use_advanced)
    elif provider == "Google (Gemini)":
        out = summarize_gemini(page_text, summary_style, output_language, use_advanced)
    else:
        out = summarize_mistral(page_text, summary_style, output_language, use_advanced)

    st.subheader("Summary")
    st.write(out)
