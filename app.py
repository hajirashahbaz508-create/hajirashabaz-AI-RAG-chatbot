import streamlit as st
import ollama

import chatbot

st.set_page_config(page_title="Python RAG Chatbot", page_icon="🐍", layout="centered")
st.title("🐍 Python RAG Chatbot")
st.caption("Offline question answering using your notes and a local Ollama model")

# Check Ollama and both required models. This checks availability, not internet access.
try:
    listing = ollama.list()
    model_items = listing.get("models", []) if isinstance(listing, dict) else getattr(listing, "models", [])
    installed = {
        (item.get("name", "") if isinstance(item, dict) else getattr(item, "model", getattr(item, "name", "")))
        for item in model_items
    }
    missing = [name for name in (chatbot.EMBEDDING_MODEL, chatbot.LLM_MODEL)
               if not any(m == name or m.startswith(name + ":") for m in installed)]
    if missing:
        st.warning("Missing Ollama model(s): " + ", ".join(missing) + ". Run the ollama pull commands shown in the setup instructions.")
    else:
        st.success(f"Ollama ready · Chat model: {chatbot.LLM_MODEL}")
except Exception as exc:
    st.error("Ollama is not reachable. Start the Ollama app/service and try again. Details: " + str(exc))

with st.sidebar:
    st.subheader("Add notes")
    uploaded = st.file_uploader("Upload a PDF, TXT, or DOCX file", type=["pdf", "txt", "docx"])
    if st.button("Add file to knowledge base", disabled=uploaded is None, use_container_width=True):
        try:
            with st.spinner("Reading file and creating local embeddings..."):
                count = chatbot.add_uploaded_file(uploaded.name, uploaded.getvalue())
            st.success(f"Added {count} text chunks from {uploaded.name}.")
        except Exception as exc:
            st.error("Could not add this file: " + str(exc))
    st.caption("Adding a file updates the searchable notes. It does not retrain model weights.")
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        chatbot.reset_memory()
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask a question about your notes...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching notes and asking Ollama..."):
            answer = chatbot.ask_chatbot(question)
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
