import os
from typing import TypedDict, List
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph, END
from src.config.settings import settings

load_dotenv()
os.environ["GROQ_API_KEY"] = settings.GROQ_API_KEY

# 👇 State schema for LangGraph
class RAGState(TypedDict):
    question: str
    context: str
    docs: List
    generate: str

# ✅ Groq LLM
def get_llm():
    return ChatGroq(
        model=settings.LLM_MODEL,
        groq_api_key=os.environ["GROQ_API_KEY"]
    )

# ✅ HuggingFace embedding model
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

# ✅ Vector store from docs
def get_vectorstore(chunks, embeddings):
    return FAISS.from_documents(chunks, embeddings)

# ✅ LangGraph: RAG pipeline
def get_graph(vectorstore, llm):
    retriever = vectorstore.as_retriever()

    # Step 1: Retrieval node
    def retrieve_and_format(state: dict):
        question = state["question"]
        docs = retriever.invoke(question)  # use invoke() per langchain-core >= 0.1.46
        context = "\n".join([doc.page_content for doc in docs])
        return {
            "question": question,
            "context": context,
            "docs": docs
        }

    # Step 2: LLM Generation node
    prompt = PromptTemplate(
        input_variables=["question", "context"],
        template="""
        You are a helpful assistant. Use the following context to answer the question.

        {context}

        Question: {question}
        Answer:
        """
    )

    def generate(state: dict):
        prompt_str = prompt.format(**state)
        response = llm.invoke(prompt_str)
        return {"generate": response.content}

    # Build LangGraph
    builder = StateGraph(RAGState)
    builder.add_node("retrieve", retrieve_and_format)
    builder.add_node("generate", generate)
    builder.set_entry_point("retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", END)

    return builder.compile()
