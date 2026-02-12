# ⚡ SmartDoc RAG

**SmartDoc RAG** is an intelligent, AI-powered document assistant that allows users to chat with their PDF documents. Built using **Streamlit**, **LangChain**, and **Groq (Llama 3.1)**, it employs Retrieval-Augmented Generation (RAG) to provide accurate, context-aware answers with verified source citations.

## 🚀 Features

* **📄 Document Ingestion**: Upload PDF documents directly via the sidebar.
* **🧠 Advanced RAG Pipeline**: Uses vector embeddings (ChromaDB) to retrieve relevant document chunks.
* **💬 Context-Aware Chat**: Remembers conversation history for natural follow-up questions (e.g., "What was my last question?").
* **🔍 Source Citations**: Every answer includes the exact filename and page number of the source information.
* **⚡ High Performance**: Optimized for low latency using Groq's LPU inference engine.
* **🎨 Modern UI**: A clean, dark-themed interface built with Streamlit.

## 🛠️ Tech Stack

* **Frontend**: [Streamlit](https://streamlit.io/)
* **LLM Integration**: [LangChain](https://www.langchain.com/)
* **Model**: Llama-3.1-8b-instant (via [Groq](https://groq.com/))
* **Vector Database**: [ChromaDB](https://www.trychroma.com/)
* **Language**: Python 3.10+

## 📂 Project Structure

```bash
AI-Based-Document-Search/
├── app.py                 # Main Streamlit application
├── rag_pipeline.py        # RAG logic (History-aware retriever & QA chain)
├── ingestion.py           # Script to load and split documents
├── indexing.py            # Script to create/update vector store
├── requirements.txt       # Project dependencies
├── .env                   # Environment variables (API Keys)
├── data/                  # Folder for storing uploaded PDFs
└── chroma_db/             # Persistent vector database storage
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
pip install streamlit langchain langchain-groq langchain-community langchain-chroma pypdf python-dotenv
```

### 4. Configure Environment Variables
Create a `.env` file in the root directory and add your Groq API key:
```env
GROQ_API_KEY=your_groq_api_key_here
```

## 🏃‍♂️ Usage

1.  Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```
2.  Open your browser (usually `http://localhost:8501`).
3.  Upload a PDF document using the sidebar.
4.  Start chatting! Ask questions like "Summarize this document" or specific details found in the text.