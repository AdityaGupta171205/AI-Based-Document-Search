import streamlit as st
import os
import hashlib
from ingestion import load_documents
from indexing import build_vectorstore
from rag_pipeline import build_rag_chain
from utils.pdf_export import export_chat
from langchain_core.messages import HumanMessage, AIMessage

st.set_page_config(page_title="SmartDoc RAG", page_icon="🌐", layout="wide")

st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at top right, #1e293b, #0f172a);
    color: #f8fafc;
}
[data-testid="stSidebar"] {
    background-color: rgba(30, 41, 59, 0.7);
    backdrop-filter: blur(10px);
    border-right: 1px solid rgba(255, 255, 255, 0.1);
}
.stChatMessage {
    background-color: rgba(255, 255, 255, 0.03) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 20px !important;
    padding: 20px !important;
    margin-bottom: 15px !important;
}
[data-testid="stChatMessage"]:nth-child(even) { border-left: 5px solid #38bdf8 !important; }
[data-testid="stChatMessage"]:nth-child(odd) { border-left: 5px solid #818cf8 !important; }
.stButton>button {
    background: linear-gradient(90deg, #38bdf8 0%, #818cf8 100%);
    color: white;
    border-radius: 10px;
    border: none;
}
.stButton>button:hover {
    box-shadow: 0 0 10px #38bdf8, 0 0 20px #818cf8;
}
</style>
""", unsafe_allow_html=True)

DATA_DIR = "data"
CHROMA_BASE = "chroma_db"

for folder in [DATA_DIR, CHROMA_BASE]:
    if not os.path.exists(folder):
        os.makedirs(folder)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=70)
    st.header("SmartDoc Control")

    uploaded_files = st.file_uploader(
        "📂 Drop Knowledge Base",
        type=["pdf", "txt", "doc", "docx"],
        accept_multiple_files=True
    )

    st.divider()

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.session_state.pending_prompt = None
        st.rerun()

if uploaded_files:

    all_docs = []
    combined_hasher = hashlib.md5()

    for uploaded_file in uploaded_files:
        file_path = os.path.join(DATA_DIR, uploaded_file.name)

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        with open(file_path, "rb") as f:
            combined_hasher.update(f.read())

        docs = load_documents(file_path)
        all_docs.extend(docs)

    file_hash = combined_hasher.hexdigest()

    @st.cache_resource
    def initialize_rag(documents, file_hash):
        persist_path = os.path.join(CHROMA_BASE, file_hash)
        vs = build_vectorstore(documents, persist_dir=persist_path)
        return build_rag_chain(vs)

    (
        rag_answer,
        summarize_document,
        generate_notes,
        generate_quiz_func,
        extract_topics_func,
        generate_followups
    ) = initialize_rag(all_docs, file_hash)

    if(len(uploaded_files)<=1 and len(all_docs)<=1):
        st.sidebar.success(f"Linked: {len(all_docs)} Page Across {len(uploaded_files)} File")
    elif(len(uploaded_files)<=1 and len(all_docs)>1):
        st.sidebar.success(f"Linked: {len(all_docs)} Page Across {len(uploaded_files)} Files")
    elif(len(uploaded_files)>1 and len(all_docs)<=1):
        st.sidebar.success(f"Linked: {len(all_docs)} Pages Across {len(uploaded_files)} File")
    else:
        st.sidebar.success(f"Linked: {len(all_docs)} Pages Across {len(uploaded_files)} Files")

else:
    st.warning("Ready for input. Please upload a PDF to initialize the session.")
    st.stop()

st.title("⚡ SmartDoc RAG")
st.divider()

for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant" and "followups" in msg:
            col1, col2, col3 = st.columns(3)
            for i, suggestion in enumerate(msg["followups"][:3]):
                col = [col1, col2, col3][i]
                with col:
                    if st.button(
                        suggestion,
                        key=f"stored_followup_{idx}_{i}",
                        use_container_width=True
                    ):
                        st.session_state.pending_prompt = suggestion
                        st.rerun()

prompt = st.chat_input("Query the knowledge base...")

if st.session_state.pending_prompt:
    prompt = st.session_state.pending_prompt
    st.session_state.pending_prompt = None

if prompt:

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Decoding document vector..."):

            stream, sources = rag_answer(prompt, st.session_state.chat_history)

            response_container = st.empty()
            full_response = ""

            for chunk in stream:
                if chunk.content:
                    full_response += chunk.content
                    response_container.markdown(full_response + "▌")

            response_container.markdown(full_response)

            if sources:
                doc = sources[0]
                source_name = os.path.basename(doc.metadata.get('source', 'Document'))
                raw_page = doc.metadata.get('page', None)
                page_display = int(raw_page) + 1 if isinstance(raw_page, int) else "N/A"

                st.caption(f"Source: {source_name} | Page: {page_display}")


            followup_text = generate_followups(prompt, full_response)

            suggestions = []
            for line in followup_text.split("\n"):
                clean = line.strip().lstrip("-").strip()
                if not clean:
                    continue
                if clean.lower().startswith("here are"):
                    continue
                if ":" in clean and len(clean.split()) < 6:
                    continue
                suggestions.append(clean)


            if suggestions:
                st.markdown("### 💡 Suggested Follow-up Questions")
                col1, col2, col3 = st.columns(3)
                for i, suggestion in enumerate(suggestions[:3]):
                    col = [col1, col2, col3][i]
                    with col:
                        if st.button(
                            suggestion,
                            key=f"followup_{len(st.session_state.messages)}_{i}",
                            use_container_width=True
                        ):
                            st.session_state.pending_prompt = suggestion
                            st.rerun()

    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response,
        "followups": suggestions
    })

    st.session_state.chat_history.extend([
        HumanMessage(content=prompt),
        AIMessage(content=full_response)
    ])

with st.sidebar:
    st.divider()
    st.subheader("AI Tools")

    def run_stream_tool(func):
        stream, _ = func()
        response_container = st.empty()
        full_response = ""

        for chunk in stream:
            if chunk.content:
                full_response += chunk.content
                response_container.markdown(full_response + "▌")

        response_container.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        st.rerun()

    if st.button("📄 Generate Summary", use_container_width=True):
        run_stream_tool(summarize_document)

    if st.button("📝 Generate Study Notes", use_container_width=True):
        run_stream_tool(generate_notes)

    if st.button("🎯 Generate Quiz", use_container_width=True):
        run_stream_tool(generate_quiz_func)

    if st.button("📌 Extract Topics", use_container_width=True):
        run_stream_tool(extract_topics_func)

    st.divider()

    if st.button("📥 Export Chat to PDF", use_container_width=True):
        path = export_chat(st.session_state.messages)
        with open(path, "rb") as f:
            st.download_button(
                label="Download PDF",
                data=f,
                file_name="SmartDoc_Chat.pdf",
                mime="application/pdf"
            )
