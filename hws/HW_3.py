import streamlit as st
from openai import OpenAI
import os

st.set_page_config(page_title="Lab 3")

st.title("Lab 3: Chatbot")

# API key
api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("Missing API key")
    st.stop()

client = OpenAI(api_key=api_key)

# Conversation history
if "chat" not in st.session_state:
    st.session_state.chat = [
        {"role": "system", "content": "Explain things so a 10 year old can understand."}
    ]
if "waiting_more" not in st.session_state:
    st.session_state.waiting_more = False

# Show old messages
for m in st.session_state.chat:
    if m["role"] == "user":
        with st.chat_message("user"):
            st.write(m["content"])
    elif m["role"] == "assistant":
        with st.chat_message("assistant"):
            st.write(m["content"])

# User input
user_msg = st.chat_input("Type your question...")

def ask_model(prompt):
    st.session_state.chat.append({"role": "user", "content": prompt})
    msgs = st.session_state.chat[-5:]  
    with st.chat_message("assistant"):
        text = ""
        spot = st.empty()
        for chunk in client.chat.completions.create(
            model="gpt-4o-mini",
            messages=msgs,
            stream=True,
        ):
            part = chunk.choices[0].delta.content or ""
            text += part
            spot.write(text)
        st.session_state.chat.append({"role": "assistant", "content": text})
  
    st.session_state.chat.append({"role": "assistant", "content": "DO YOU WANT MORE INFO"})
    st.session_state.waiting_more = True

if user_msg:
    if st.session_state.waiting_more:
        if user_msg.lower().startswith("y"):
            ask_model("Yes, give me more info.")
        elif user_msg.lower().startswith("n"):
            st.session_state.chat.append({"role": "assistant", "content": "Okay! What new question can I help with?"})
            st.session_state.waiting_more = False
        else:
            ask_model(user_msg)
            st.session_state.waiting_more = False
    else:
        ask_model(user_msg)
