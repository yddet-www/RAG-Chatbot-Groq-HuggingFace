
from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Depends
from pydantic import BaseModel
import os
import shutil
from src.pipeline.rag_pipeline import RAGPipeline
from src.utils.auth import validate_api_key
from src.utils.chatlog import log_interaction
from typing import List
from fastapi import UploadFile, File

app = FastAPI(title="RAGStack AI (LangGraph Edition)")
rag_pipeline = RAGPipeline()

class Question(BaseModel):
    question: str

@app.post("/api/upload-docs/")
def upload_docs(
    corpus_name: str = Query(...),
    files: List[UploadFile] = File(...),
    api_key: str = Depends(validate_api_key)
):
    folder_path = f"temp_uploads/{corpus_name}"
    os.makedirs(folder_path, exist_ok=True)

    for file in files:
        file_path = os.path.join(folder_path, file.filename)
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

    try:
        rag_pipeline.initialize(folder_path, corpus_name)
        return {"message": f"Corpus '{corpus_name}' initialized with {len(files)} files."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ask/")
def ask_question(payload: Question, api_key: str = Depends(validate_api_key)):
    try:
        response = rag_pipeline.ask(payload.question)
        log_interaction(payload.question, response['generate'], rag_pipeline.corpus_name)
        return {
            "answer": response["generate"],
            "documents": [doc.metadata.get("source", "unknown") for doc in response.get("docs", [])]
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/load-corpus/{corpus_name}")
def load_existing(corpus_name: str, api_key: str = Depends(validate_api_key)):
    try:
        rag_pipeline.load_existing_corpus(corpus_name)
        return {"message": f"Corpus '{corpus_name}' loaded from disk."}
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/api/status")
def get_status():
    return rag_pipeline.status()

@app.post("/api/reset")
def reset_pipeline(api_key: str = Depends(validate_api_key)):
    rag_pipeline.reset()
    return {"message": "Pipeline reset."}

@app.on_event("startup")
async def startup_event():
    default_corpus = "source"
    load_existing(default_corpus, "iGEM-IIT")
