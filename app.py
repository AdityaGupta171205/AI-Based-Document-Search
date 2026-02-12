import streamlit as st
import os
from ingestion import load_documents
from indexing import build_vectorstore
from rag_pipeline import build_rag_chain
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

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=70)
    st.header("SmartDoc Control")
    
    uploaded_file = st.file_uploader("📂 Drop Knowledge Base", type=["pdf", "txt", "doc", "docx"])
    
    st.divider()
    
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()
    
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if uploaded_file:
    file_path = os.path.join(DATA_DIR, uploaded_file.name)
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    @st.cache_resource
    def init_stable_rag(path, filename):
        unique_persist_dir = os.path.join(CHROMA_BASE, filename.replace(".pdf", "").replace(" ", "_"))
        
        docs = load_documents(path)
        vs = build_vectorstore(docs, persist_dir=unique_persist_dir, reindex=True)
        return build_rag_chain(vs), len(docs)

    try:
        rag_chain, doc_count = init_stable_rag(file_path, uploaded_file.name)
        st.sidebar.success(f"Linked: {doc_count} Elements")
    except Exception as e:
        st.error(f"Initialization Error: {e}")
        st.stop()
else:
    st.warning("Ready for input. Please upload a PDF to initialize the session.")
    st.stop()

st.title("⚡ SmartDoc RAG")
st.divider()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Query the knowledge base..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Decoding document vector..."):
            result = rag_chain(prompt, st.session_state.chat_history)
            st.markdown(result["answer"])
            
            if result["sources"]:
                st.markdown("---")
                doc = result["sources"][0]

                source_name = os.path.basename(doc.metadata.get('source', 'Document'))
                raw_page = doc.metadata.get('page', None)
                page_display = int(raw_page) + 1 if isinstance(raw_page, int) else "N/A"
                st.markdown(f"**Verified Source:** {source_name} | **Page:** {page_display}")
                
                with st.expander("Show Sources:"):
                    st.caption(doc.page_content)

    st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
    st.session_state.chat_history.extend([
        HumanMessage(content=prompt),
        AIMessage(content=result["answer"])
    ])