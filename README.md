# Minimal RAG Backend (FastAPI + SQLite + Ollama)

No Docker, no Redis. Single process FastAPI app with SQLite for metadata and NumPy-based vector search.
Embeddings and LLM generation go through **Ollama** (local LLaMA).

## Prereqs
- Python 3.10+
- [Ollama](https://ollama.com) running locally (`ollama serve`) with models pulled:
  ```bash
  ollama pull llama3.1:8b
  ollama pull nomic-embed-text
  ```

## Install & Run
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --port 8000
# Swagger: http://localhost:8000/docs
```

## Quick test
```bash
curl -X POST 'http://localhost:8000/v1/ingest?sync=true'   -H 'Content-Type: application/json'   -d '{"text":"Абай Кунанбаев — казахский поэт и мыслитель."}'

curl -X POST 'http://localhost:8000/v1/query'   -H 'Content-Type: application/json'   -d '{"query":"Кто такой Абай Кунанбаев?"}'
```

## Notes
- Vectors stored as normalized float lists in SQLite (JSON). Retrieval uses cosine similarity in Python (NumPy).
- Swap to FAISS/Annoy later if needed.
