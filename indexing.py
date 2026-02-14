import os
import hashlib
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata


def get_file_hash(file_path):
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()


def build_vectorstore(documents, persist_dir):

    documents = filter_complex_metadata(documents)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    if os.path.exists(persist_dir):
        return Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings
        )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )

    vectorstore.persist()
    return vectorstore
