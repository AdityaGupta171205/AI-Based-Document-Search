import os
import shutil

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata


def build_vectorstore(documents, persist_dir="chroma_db", reindex=True):
    if reindex and os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)

    documents = filter_complex_metadata(documents)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(documents)
    print(f"[Indexing] Created {len(chunks)} chunks")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )

    print("[Indexing] Documents indexed successfully")
    return vectorstore
