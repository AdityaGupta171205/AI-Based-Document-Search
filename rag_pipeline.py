import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


def build_rag_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        groq_api_key=os.getenv("GROQ_API_KEY"),
    )

    prompt = ChatPromptTemplate.from_template(
        """Answer the question using ONLY the provided context.

Context:
{context}

Question:
{question}
"""
    )

    def rag_answer(question: str):
        docs = retriever.invoke(question)[:2]

        context = "\n\n".join(doc.page_content for doc in docs)

        response = llm.invoke(
            prompt.format_messages(
                context=context,
                question=question
            )
        )

        return {
            "answer": response.content,
            "sources": docs
        }

    return rag_answer
