
from src.dataloader.loader import load_documents
from src.textsplitter.splitter import split_documents
from src.llm.model import get_llm, get_embeddings, get_vectorstore, get_graph
from src.storage.corpus_manager import save_vectorstore, load_vectorstore
from src.utils.logger import logger

class RAGPipeline:
    def __init__(self):
        self.graph = None
        self.vectorstore = None
        self.corpus_name = None
        self.documents_loaded = False

    def initialize(self, folder_path: str, corpus_name: str):
        documents = load_documents(folder_path)
        if not documents:
            raise ValueError("No valid documents found in the provided folder.")

        chunks = split_documents(documents)
        embeddings = get_embeddings()
        self.vectorstore = get_vectorstore(chunks, embeddings)
        self.graph = get_graph(self.vectorstore, get_llm())
        self.corpus_name = corpus_name
        self.documents_loaded = True

        save_vectorstore(self.vectorstore, corpus_name)
        logger.info(f"Pipeline initialized for corpus '{corpus_name}'.")

    def load_existing_corpus(self, corpus_name):
        self.vectorstore = load_vectorstore(corpus_name)
        self.graph = get_graph(self.vectorstore, get_llm())
        self.corpus_name = corpus_name
        self.documents_loaded = True

    def ask(self, question: str):
        if not self.documents_loaded or self.graph is None:
            raise ValueError("System not ready. Load corpus first.")
        return self.graph.invoke({"question": question})

    def status(self):
        return {
            "current_corpus": self.corpus_name,
            "documents_loaded": self.documents_loaded,
            "vectorstore_ready": self.vectorstore is not None,
            "llm_ready": self.graph is not None
        }

    def reset(self):
        self.graph = None
        self.vectorstore = None
        self.documents_loaded = False
        self.corpus_name = None

