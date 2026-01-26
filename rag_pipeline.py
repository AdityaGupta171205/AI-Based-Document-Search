import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()

def build_rag_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        groq_api_key=os.getenv("GROQ_API_KEY"),
    )

    contextual_prompt = ChatPromptTemplate.from_messages([
        ("system", """Answer the question using ONLY the provided context.

        Context:
        {context}

        Question:
        {question}
        """),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "Context:\n{context}\n\nQuestion: {question}"),
    ])

    def rag_answer(question: str, chat_history: list):
        docs = retriever.invoke(question)
        
        if docs:
            doc = docs[0]
            page = doc.metadata.get("page_number", "Unknown")
            source = os.path.basename(doc.metadata.get("source", "PDF"))
            context_string = f"--- SOURCE: {source} | PAGE: {page} ---\n{doc.page_content}"
        else:
            context_string = "No relevant context found."

        chain = contextual_prompt | llm
        response = chain.invoke({
            "context": context_string,
            "question": question,
            "chat_history": chat_history
        })

        return {
            "answer": response.content,
            "sources": docs
        }

    return rag_answer
