# Live-Learn RAG System with Gemini AI

Enhanced Retrieval-Augmented Generation (RAG) system with Google Gemini AI and user feedback learning capabilities.

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.8+
- Google AI API key ([получить здесь](https://aistudio.google.com/app/apikey))

### 2. Installation

```bash
# Клонируем репозиторий
git clone <repository-url>
cd rag-system

# Создаем виртуальное окружение
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Устанавливаем зависимости
pip install -r requirements.txt
```

### 3. Configuration

```bash
# Копируем пример конфигурации
cp .env.example .env

# Редактируем .env файл и добавляем ваш Google AI API key
# GOOGLE_API_KEY=your_actual_google_api_key_here
```

### 4. Run the Server

```bash
# Запускаем сервер разработки
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Или используем Python напрямую
python -m uvicorn app.main:app --reload
```

### 5. Test the System

```bash
# Запускаем тестовый скрипт
python test_rag_system.py
```

## 📋 API Endpoints

### Health Check
```http
GET /healthz
GET /healthz/detailed
```

### RAG Operations
```http
# Ask a question
POST /api/v1/feedback/ask
Content-Type: application/json

{
    "question": "What is artificial intelligence?",
    "session_id": "optional-session-id",
    "top_k": 6
}

# Ingest a document
POST /api/v1/ingest
Content-Type: application/json

{
    "text": "Your document content here...",
    "metadata": {"title": "Document Title", "source": "web"}
}

# Upload a file
POST /api/v1/ingest
Content-Type: multipart/form-data

file: your_document.txt
```

### Feedback System
```http
# Submit feedback
POST /api/v1/feedback/feedback
Content-Type: application/json

{
    "message_id": "message-id-from-ask-response",
    "question": "Original question",
    "model_answer": "Model's answer",
    "user_feedback": {
        "label": "incorrect",
        "correction_text": "The correct answer is...",
        "scope": "chunk",
        "reason": "The information is outdated"
    }
}
```

## 🏗️ Architecture

```
app/
├── main.py              # FastAPI application entry point
├── config.py            # Configuration management
├── database.py          # Database setup and connections
├── api/v1/              # API endpoints
│   ├── feedback.py      # Feedback and ask endpoints
│   └── rag.py          # Document ingestion and retrieval
├── models/              # Database models
│   ├── documents.py     # Document and chunk models
│   └── feedback.py      # Feedback system models
├── schemas/             # Pydantic schemas
│   ├── feedback.py      # Feedback API schemas
│   └── rag.py          # RAG API schemas  
├── services/            # Business logic
│   ├── embeddings.py    # Google AI & OpenAI embeddings
│   ├── llm.py          # Gemini LLM integration
│   ├── rag_pipeline.py  # Main RAG pipeline
│   └── feedback_handler.py # Feedback processing
├── utils/               # Utility functions
│   ├── text_processing.py # Text chunking and processing
│   └── vectors.py       # Vector operations
└── core/               # Core functionality
    └── exceptions.py   # Custom exceptions
```

## 🧪 Testing

### Automated Tests
```bash
# Запуск всех тестов
python test_rag_system.py

# Тест только базовой функциональности (без сервера)
python -c "import asyncio; from test_rag_system import test_basic_functionality; asyncio.run(test_basic_functionality())"
```

### Manual Testing

1. **Health Check**: `curl http://localhost:8000/healthz`
2. **Ask Question**: 
   ```bash
   curl -X POST "http://localhost:8000/api/v1/feedback/ask" \
        -H "Content-Type: application/json" \
        -d '{"question": "What is AI?", "session_id": "test"}'
   ```
3. **Ingest Document**:
   ```bash
   curl -X POST "http://localhost:8000/api/v1/ingest" \
        -H "Content-Type: application/json" \
        -d '{"text": "AI is artificial intelligence..."}'
   ```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GOOGLE_API_KEY` | Google AI API key | Required |
| `LLM_MODEL` | Gemini model name | `gemini-2.0-flash-exp` |
| `EMBEDDING_MODEL` | Embedding model | `text-embedding-004` |
| `CHUNK_SIZE` | Text chunk size in tokens | `400` |
| `CHUNK_OVERLAP` | Overlap between chunks | `40` |
| `DEFAULT_TOP_K` | Default retrieval count | `6` |
| `DEBUG` | Debug mode | `false` |

### Google AI Setup

1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Create a new API key
3. Add it to your `.env` file as `GOOGLE_API_KEY`

## 🐛 Troubleshooting

### Common Issues

1. **"Ask pipeline failed" error**:
   - Check if `GOOGLE_API_KEY` is set correctly
   - Verify the API key has proper permissions
   - Check logs for detailed error messages

2. **Import errors**:
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Check Python version (3.8+ required)

3. **Database issues**:
   - Delete `rag.db` to reset the database
   - Check file permissions in the project directory

4. **Empty responses**:
   - First ingest some documents using `/api/v1/ingest`
   - Check if documents are properly chunked and embedded

### Debug Mode

Set `DEBUG=true` in `.env` for detailed error messages and logs.

### Logs

Check the console output for detailed logs using Loguru formatting.

## 📖 Usage Examples

### 1. Basic Q&A

```python
import httpx
import asyncio

async def ask_question():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/api/v1/feedback/ask",
            json={
                "question": "What is machine learning?",
                "session_id": "demo-session",
                "top_k": 5
            }
        )
        return response.json()

result = asyncio.run(ask_question())
print(result['answer'])
```

### 2. Document Ingestion

```python
import httpx
import asyncio

async def ingest_document():
    document_text = """
    Machine Learning is a subset of artificial intelligence that provides 
    systems the ability to automatically learn and improve from experience 
    without being explicitly programmed.
    """
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/api/v1/ingest",
            json={
                "text": document_text,
                "metadata": {"title": "ML Introduction", "source": "manual"}
            }
        )
        return response.json()

result = asyncio.run(ingest_document())
print(f"Document {result['document_id']} ingested with {result['chunks']} chunks")
```

### 3. Feedback Submission

```python
import httpx
import asyncio

async def submit_feedback(message_id):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/api/v1/feedback/feedback",
            json={
                "message_id": message_id,
                "question": "What is machine learning?",
                "model_answer": "Previous answer from the model",
                "user_feedback": {
                    "label": "partially_correct",
                    "correction_text": "Also mention deep learning as a subset",
                    "scope": "chunk",
                    "reason": "Missing important information"
                }
            }
        )
        return response.json()

# Use message_id from ask response
result = asyncio.run(submit_feedback("your-message-id-here"))
print(f"Feedback processed: {result['status']}")
```

## 🔄 Development

### Code Structure

- **Services**: Core business logic (embeddings, LLM, pipeline)
- **Models**: SQLAlchemy database models
- **Schemas**: Pydantic input/output validation
- **API**: FastAPI route handlers
- **Utils**: Helper functions for text processing and vectors

### Adding New Features

1. Add models in `models/`
2. Create schemas in `schemas/`
3. Implement business logic in `services/`
4. Add API endpoints in `api/v1/`
5. Update tests in `test_rag_system.py`

## 📄 License

This project is licensed under the MIT License.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes and add tests
4. Submit a pull request

## 📞 Support

If you encounter issues:

1. Check the troubleshooting section above
2. Run the test script: `python test_rag_system.py`
3. Enable debug mode in `.env`
4. Check the logs for detailed error messages