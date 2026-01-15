import pytesseract
from langchain_community.document_loaders import UnstructuredPDFLoader

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def load_documents(pdf_path: str):
    loader = UnstructuredPDFLoader(
        pdf_path,
        mode="elements"
    )
    documents = loader.load()
    print(f"[Ingestion] Loaded {len(documents)} elements")
    return documents
