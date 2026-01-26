from langchain_community.document_loaders import (
    TextLoader, 
    PyPDFLoader, 
    Docx2txtLoader
)

def load_documents(file_path: str):
    file_extension = file_path.lower().split('.')[-1]
    
    if file_extension == 'txt':
        loader = TextLoader(file_path, encoding='utf-8')
    elif file_extension == 'pdf':
        loader = PyPDFLoader(file_path)
    elif file_extension in ['doc', 'docx']:
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError(f"Unsupported file format: .{file_extension}")

    documents = loader.load()
    print(f"[Ingestion] Successfully loaded {file_extension} file: {file_path}")
    return documents