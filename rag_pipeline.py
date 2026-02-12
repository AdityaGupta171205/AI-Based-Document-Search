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

    rephrase_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant. Your main purpose is to answer questions strictly based on the provided Context.

        Guidelines:
        1. Greetings: If the user greets you (e.g., "Hi", "Hello", "How are you"), respond politely and professionally, then ask how you can help with the document.
        2. Memory: If the user asks about their past questions or the conversation history (e.g., "What was my last question?", "Recap our chat"), answer based on the Chat History provided below.
        3. Document Queries: For all other questions, answer ONLY using the "Context" below. Do not use outside knowledge.
        4. Unknowns: If the answer is not in the Context or Chat History, strictly state that you don't know.

        Context: {context}"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    def rag_answer(question: str, chat_history: list):
        if chat_history:
            rephrase_chain = rephrase_prompt | llm
            rephrased_response = rephrase_chain.invoke({
                "chat_history": chat_history, 
                "question": question
            })
            search_query = rephrased_response.content
        else:
            search_query = question

        docs = retriever.invoke(search_query)
        
        if docs:
            context_text = "\n\n".join([f"--- Source: {doc.metadata.get('source', 'PDF')} ---\n{doc.page_content}" for doc in docs])
        else:
            context_text = "No relevant context found."

        final_chain = qa_prompt | llm
        response = final_chain.invoke({
            "context": context_text,
            "question": question,
            "chat_history": chat_history
        })

        return {
            "answer": response.content,
            "sources": docs
        }

    return rag_answer
