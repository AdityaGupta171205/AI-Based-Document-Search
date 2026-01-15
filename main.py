from ingestion import load_documents
from indexing import build_vectorstore
from rag_pipeline import build_rag_chain

PDF_PATH = "data/sample1.pdf"

def main():
    documents = load_documents(PDF_PATH)
    vectorstore = build_vectorstore(documents, reindex=True)
    rag_chain = build_rag_chain(vectorstore)

    while True:
        query = input("\nAsk a question (or type 'exit'): ")
        if query.lower() == "exit":
            break

        result = rag_chain(query)


        print("\nAnswer:\n", result["answer"])

        print("\nSources:")
        for i, doc in enumerate(result["sources"], 1):
            page = doc.metadata.get("page", "N/A")
            source = doc.metadata.get("source", "Document")
            snippet = doc.page_content[:200].replace("\n", " ")
            print(f"[{i}] {source}, {snippet}...\n")

if __name__ == "__main__":
    main()
