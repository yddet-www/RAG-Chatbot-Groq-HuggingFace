# AI Chatbot using Huggingface and Grok API

RAGStack AI is a production-grade, modular Retrieval-Augmented Generation (RAG) chatbot built with FastAPI, LangChain, LangGraph, Groq LLMs, and HuggingFace embeddings.

## Features

- FastAPI RESTful API with Swagger UI
- HuggingFace embedding models (no OpenAI dependency)
- Groq-powered LLMs (Gemma, Mixtral, LLaMA3)
- LangGraph pipeline for retrieval and generation
- Multi-document upload and corpus management
- Vector store persistence using FAISS
- Interaction logging (JSONL) with UTC timestamps
- API key-based access control

## Directory Structure

```
project-root/
├── src/
│   ├── api/                  # FastAPI app
│   ├── config/               # Settings loader
│   ├── dataloader/           # Document loaders
│   ├── llm/                  # Embeddings, LLM, LangGraph pipeline
│   ├── pipeline/             # RAGPipeline manager
│   ├── storage/              # FAISS save/load logic
│   ├── textsplitter/         # Document chunker
│   └── utils/                # Logger, time utils, auth
├── logs/                    # JSONL logs
├── vectorstores/            # Persisted FAISS indexes
├── temp_uploads/            # Uploaded document storage
└── .env                     # Environment variables
```

## Setup Instructions

### 1. Clone the repository

```bash
git clone <repo-url>
cd project-root
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_key
API_KEY=iGEM-IIT
```

### 4. Run the API server

```bash
uvicorn src.api.main:app --reload
```

Access Swagger docs at:
```
http://127.0.0.1:8000/docs
```

## API Endpoints

### Upload Documents
```
POST /upload-docs/?corpus_name=<name>
```
**Body:** multipart/form-data (key: `files`, multiple allowed)
**Headers:** `X-API-Key: supersecret`

### Ask a Question
```
POST /ask/
```
**Body:**
```json
{
  "question": "What is this document about?"
}
```
**Headers:** `X-API-Key: iGEM-IIT`

### Load Existing Corpus
```
GET /load-corpus/{corpus_name}
```

### Check Status
```
GET /status
```

### Reset Pipeline
```
POST /reset
```

## Logging
All interactions are logged in:
```
logs/{corpus_name}_chatlog.jsonl
```
Each entry includes timestamp (UTC), corpus, question, and answer.

## Notes
- Vector store is saved in `vectorstores/` with corpus names.
- Embeddings use HuggingFace models (e.g., `BAAI/bge-base-en`).
- Retrieval and generation logic is managed via LangGraph.

