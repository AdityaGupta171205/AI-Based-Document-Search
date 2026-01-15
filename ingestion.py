from langchain_community.document_loaders import UnstructuredPDFLoader
from tesseract_utils import configure_tesseract


def load_documents(pdf_path: str):
    configure_tesseract()

    loader = UnstructuredPDFLoader(
        pdf_path,
        mode="elements"
    )

    documents = loader.load()
    print(f"[Ingestion] Loaded {len(documents)} elements")
    return documents
