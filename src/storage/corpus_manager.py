import os
from langchain_community.vectorstores import FAISS
from src.llm.model import get_embeddings
from src.config.settings import settings
from src.utils.logger import logger

def save_vectorstore(vectorstore, corpus_name: str):
    path = f"{settings.VECTOR_DB_DIR}/{corpus_name}"
    os.makedirs(path, exist_ok=True)
    vectorstore.save_local(path)
    logger.info(f"Saved vectorstore to {path}")

def load_vectorstore(corpus_name: str):
    path = f"{settings.VECTOR_DB_DIR}/{corpus_name}"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Vectorstore for corpus '{corpus_name}' not found.")
    return FAISS.load_local(path, get_embeddings(), allow_dangerous_deserialization=True)