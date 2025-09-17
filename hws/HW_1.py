import streamlit as st
from openai import OpenAI
import fitz  


# Show title and description.
st.title("📄My Document question answering")
st.write(
    "Upload a document below and ask a question about it – GPT will answer! "
    "To use this app, you need to provide an OpenAI API key, which you can get "
    "[here](https://platform.openai.com/account/api-keys)."
)

# Ask user for their OpenAI API key via `st.text_input`.
openai_api_key = st.secrets["api_key"]
if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")
else:
    # Create an OpenAI client.
    client = OpenAI(api_key=openai_api_key)

    if "document" not in st.session_state:
        st.session_state["document"] = None
    if "uploaded_name" not in st.session_state:
        st.session_state["uploaded_name"] = None

    def read_pdf(file) -> str:
        data = file.getvalue()
        parts = []
        with fitz.open(stream=data, filetype="pdf") as doc:
            if getattr(doc, "needs_pass", False):
                raise ValueError("This PDF is password-protected.")
            for page in doc:
                parts.append(page.get_text("text"))
        text = "\n".join(parts).strip()
        if not text:
            # likely a scanned PDF without OCR
            raise ValueError("No selectable text found in PDF (might be a scanned PDF).")
        return text

    # Let the user upload a file via `st.file_uploader` +
    uploaded_file = st.file_uploader(
        "Upload a document (.txt or .pdf)",
        type=("txt", "pdf")
    )

    # If the file is removed from the UI, we must drop access to its data 
    if uploaded_file is None:
        if st.session_state["document"] is not None:
            st.session_state["document"] = None
            st.session_state["uploaded_name"] = None
    else:
        # Parse only when a new file is uploaded or the name changed
        if uploaded_file.name != st.session_state["uploaded_name"]:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            try:
                if file_extension == 'txt':
                    st.session_state["document"] = uploaded_file.getvalue().decode("utf-8", errors="ignore")
                elif file_extension == 'pdf':
                    st.session_state["document"] = read_pdf(uploaded_file)
                else:
                    st.error("Unsupported file type.")
                    st.stop()
                st.session_state["uploaded_name"] = uploaded_file.name
            except Exception as e:
                st.session_state["document"] = None
                st.session_state["uploaded_name"] = None
                st.error(f"Could not read file: {e}")
                st.stop()

    # Ask the user for a question via `st.text_area`.
    question = st.text_area(
        "Now ask a question about the document!",
        placeholder='e.g., "Is this course hard?"',
        disabled=(st.session_state["document"] is None),
    )

    # 3c: Try 4 different models and show the model name before each answer
    models = [
        "gpt-3.5-turbo",
        "gpt-4.1",
        "gpt-5-chat-latest",
        "gpt-5-nano",
    ]

    if st.session_state["document"] and question:
        # Build the prompt once, reusing for all models
        messages = [
            {
                "role": "user",
                "content": f"Here's a document:\n{st.session_state['document']}\n\n---\n\n{question}",
            }
        ]

        st.subheader("Answers")
        for m in models:
            st.markdown(f"**Model: {m}**")
            try:
                # Stream the response to the app
                stream = client.chat.completions.create(
                    model=m,
                    messages=messages,
                    stream=True,
                )
                # Extract and stream only the text deltas
                st.write_stream((chunk.choices[0].delta.content or "" for chunk in stream))
            except Exception as e:
                st.warning(f"{m} error: {e}")
            st.divider()
#comment21