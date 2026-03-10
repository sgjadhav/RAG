
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os


def build_rag_chain(vectorstore):

    # Create retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k":4})

    # Gemini LLM
    llm = ChatGoogleGenerativeAI(
        model="models/gemini-2.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0
    )

    # Prompt template
    prompt = ChatPromptTemplate.from_template(
        """
        Answer the question based only on the context below.

        Context:
        {context}

        Question:
        {question}
        """
    )

    # RAG chain
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain
