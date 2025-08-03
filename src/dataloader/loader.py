import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader, UnstructuredPowerPointLoader

def load_documents(folder_path: str):
    if not os.path.exists(folder_path):
        raise ValueError(f"Path '{folder_path}' does not exist.")

    docs = []
    for file in os.listdir(folder_path):
        ext = os.path.splitext(file)[-1]
        file_path = os.path.join(folder_path, file)
        if ext == ".txt":
            docs.extend(TextLoader(file_path).load())
        elif ext == ".pdf":
            docs.extend(PyPDFLoader(file_path).load())
        elif ext == ".docx":
            docs.extend(Docx2txtLoader(file_path).load())
        elif ext == ".pptx":
            docs.extend(UnstructuredPowerPointLoader(file_path).load())
    return docs