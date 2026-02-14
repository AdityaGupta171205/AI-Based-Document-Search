# ⚡ SmartDoc RAG

**SmartDoc RAG** is an intelligent, AI-powered document assistant that enables users to interact with their documents using natural language. Built using **Streamlit**, **LangChain**, **ChromaDB**, and **Groq (Llama 3.1)**, it leverages Retrieval-Augmented Generation (RAG) to deliver accurate, context-aware answers with verified source citations.

It functions as a research assistant, study companion, and document analysis tool in a clean, modern UI.

---

## 🚀 Features

### 📂 Multi-Document Upload
- Upload multiple PDF, TXT, or DOCX files.
- Automatically combines and indexes documents.
- Persistent vector storage using hash-based indexing.

### 🧠 Advanced RAG Pipeline
- Embedding-based retrieval using `sentence-transformers`.
- ChromaDB vector storage for semantic search.
- Similarity-based ranking for accurate context selection.

### 💬 Context-Aware Conversational Chat
- Maintains conversation history.
- Supports natural follow-up queries.
- Handles greetings and recap questions intelligently.

### 💡 Smart Follow-up Suggestions
- Automatically generates contextual follow-up questions.
- Clean 3-column suggestion layout.
- One-click interaction for seamless exploration.

### 📚 Verified Source Attribution
- Displays the most relevant document source.
- Shows filename and page number.
- Minimal, research-style citation format.

### 🛠️ AI Study Tools
From the sidebar, users can:
- 📄 Generate Document Summary
- 📝 Generate Study Notes
- 🎯 Generate Quiz Questions
- 📌 Extract Key Topics
- 📥 Export Chat to PDF

### ⚡ Streaming Responses
- Real-time token streaming for dynamic responses.
- Low latency inference using Groq’s LPU engine.

### 🎨 Modern UI
- Dark-themed, responsive interface.
- Clean chat-style layout.
- Minimal and distraction-free design.

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit  
- **LLM Framework:** LangChain (Modular Packages)  
- **Model:** Llama-3.1-8b-instant (via Groq)  
- **Embeddings:** sentence-transformers (all-MiniLM-L6-v2)  
- **Vector Database:** ChromaDB  
- **PDF Processing:** PyPDF  
- **Document Parsing:** docx2txt  
- **Environment Management:** python-dotenv  
- **Language:** Python 3.10+  

---

## 📂 Project Structure

```bash
AI-Based-Document-Search/
├── app.py            # Main Streamlit application
├── rag_pipeline.py   # RAG logic & AI tools
├── ingestion.py      # Document loading logic
├── indexing.py       # Vectorstore creation (ChromaDB)
├── utils/
│ └── pdf_export.py   # Chat export functionality
├── requirements.txt  # Project dependencies
├── .env              # Environment variables (API Keys)
├── data/             # Uploaded document storage
└── chroma_db/        # Persistent vector database
```

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/AdityaGupta171205/AI-Based-Document-Search.git](https://github.com/AdityaGupta171205/AI-Based-Document-Search.git)
cd AI-Based-Document-Search
```

### 2. Create a Virtual Environment
It is recommended to use a virtual environment to manage dependencies.

**Windows:**
```bash
python -m venv venv
.\venv\Scripts\activate
```

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
*If `requirements.txt` is missing, install the core packages manually:*
```bash
pip install streamlit langchain-core langchain-community langchain-text-splitters langchain-groq langchain-huggingface chromadb sentence-transformers pypdf docx2txt python-dotenv reportlab
```

### 4. Configure Environment Variables
Create a `.env` file in the root directory and add your Groq API key:
```env
GROQ_API_KEY=your_groq_api_key_here
```

---

## 🏃‍♂️ Usage

1️⃣ Run the application:

2️⃣ Open browser (default: `http://localhost:8501`)

3️⃣ Upload one or more documents via the sidebar.

4️⃣ Start chatting!

Example queries:
- “Summarize this document.”
- “What are the key topics discussed?”
- “Generate quiz questions from this document.”
- “What was my last question?”
- “Explain this in simple terms.”

---

## 🧠 How It Works (High-Level Architecture)

1. Documents are uploaded and parsed.
2. Text is split into chunks using recursive splitting.
3. Chunks are embedded using sentence-transformers.
4. Embeddings are stored in ChromaDB.
5. User query triggers similarity search.
6. Top relevant chunk is passed to Llama 3.1 via Groq.
7. Response is streamed with source citation.
8. Follow-up suggestions are generated dynamically.