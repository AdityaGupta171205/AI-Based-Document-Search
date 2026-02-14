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
        streaming=True,
        groq_api_key=os.getenv("GROQ_API_KEY"),
    )

    rephrase_prompt = ChatPromptTemplate.from_messages([
        ("system", "Rephrase the question to be standalone if needed."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant. Answer ONLY using the provided Context.

If the answer is not in the context, say you don't know.

Context:
{context}
"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    def rag_answer_stream(question: str, chat_history: list):

        if chat_history:
            rephrase_chain = rephrase_prompt | llm
            rephrased = rephrase_chain.invoke({
                "chat_history": chat_history,
                "question": question
            })
            search_query = rephrased.content
        else:
            search_query = question

        docs_with_scores = vectorstore.similarity_search_with_score(search_query, k=5)
        docs = []

        for doc, score in docs_with_scores:
            doc.metadata["score"] = score
            docs.append(doc)


        if docs:
            context_text = "\n\n".join([doc.page_content for doc in docs])
        else:
            context_text = "No relevant context found."

        final_chain = qa_prompt | llm

        stream = final_chain.stream({
            "context": context_text,
            "question": question,
            "chat_history": chat_history
        })

        return stream, docs

    def summarize_document():
        return rag_answer_stream(
            "Summarize the entire document in 10 structured bullet points.",
            []
        )

    def generate_notes():
        return rag_answer_stream(
            "Generate detailed structured study notes from this document.",
            []
        )

    def generate_quiz():
        return rag_answer_stream(
            """Generate:
1. 5 multiple choice questions (with 4 options each)
2. 3 short answer questions
3. 2 long answer questions

Based strictly on the document.""",
            []
        )

    def extract_topics():
        return rag_answer_stream(
            """Extract the key topics and subtopics from the document.
Present them in clean bullet format.""",
            []
        )

    def generate_followups(question, answer):
        prompt = f"""
            Generate exactly 3 follow-up questions based on the context below.

            Strict Rules:
            - Return ONLY 3 questions.
            - Do NOT include any introduction.
            - Do NOT include "Here are".
            - Do NOT number them.
            - Do NOT add explanations.
            - Each question must be on its own line.
            - Output only pure questions.

            User Question:
            {question}

            Assistant Answer:
            {answer}
                """
        response = llm.invoke(prompt)
        return response.content

    return (
        rag_answer_stream,
        summarize_document,
        generate_notes,
        generate_quiz,
        extract_topics,
        generate_followups
    )
