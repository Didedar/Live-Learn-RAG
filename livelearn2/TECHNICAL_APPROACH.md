# LiveLearn RAG - Technical Approach

## Overall Architecture

LiveLearn RAG представляет собой современную систему Retrieval-Augmented Generation (RAG) с возможностью обучения на основе пользовательской обратной связи. Система построена на принципах микросервисной архитектуры с четким разделением ответственности между компонентами.

### Архитектурная диаграмма

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           LiveLearn RAG System                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐             │
│  │   Frontend      │    │   API Gateway   │    │   Monitoring    │             │
│  │   (HTML/JS)     │◄──►│   (FastAPI)     │◄──►│   (Logs/Metrics)│             │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘             │
│           │                       │                       │                    │
│           │                       │                       │                    │
│           ▼                       ▼                       ▼                    │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                        Core Services Layer                              │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │   │
│  │  │    RAG      │  │  Feedback   │  │  Learning   │  │   Trust     │    │   │
│  │  │  Pipeline   │  │  Handler    │  │   Engine    │  │   Scorer    │    │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘    │   │
│  │         │               │               │               │                │   │
│  │         ▼               ▼               ▼               ▼                │   │
│  │  ┌─────────────────────────────────────────────────────────────────┐    │   │
│  │  │              Hybrid Retrieval System                           │    │   │
│  │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │    │   │
│  │  │  │    Dense    │  │     BM25    │  │  Content    │            │    │   │
│  │  │  │  Retrieval  │  │   Search    │  │  Filter     │            │    │   │
│  │  │  └─────────────┘  └─────────────┘  └─────────────┘            │    │   │
│  │  └─────────────────────────────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│           │                       │                       │                    │
│           ▼                       ▼                       ▼                    │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                        Data Layer                                       │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │   │
│  │  │  SQLite     │  │  Vector     │  │  Feedback   │  │  Metadata   │    │   │
│  │  │  Database   │  │  Store      │  │   Events    │  │   Store     │    │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘    │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│           │                       │                       │                    │
│           ▼                       ▼                       ▼                    │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                        External Services                                │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                    │   │
│  │  │   Ollama    │  │  Embedding  │  │   File      │                    │   │
│  │  │   (LLM)     │  │   Models    │  │  Storage    │                    │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘                    │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Key Technologies Used

### 1. **Backend Framework**
- **FastAPI 0.104.1** - Современный веб-фреймворк для создания API
- **Uvicorn** - ASGI сервер для FastAPI
- **Pydantic 2.5.2** - Валидация данных и настройки

### 2. **Database & Storage**
- **SQLAlchemy 2.0.23** - ORM для работы с базой данных
- **SQLite** - Встроенная реляционная база данных
- **Alembic 1.12.1** - Миграции базы данных

### 3. **AI & Machine Learning**
- **Ollama** - Локальный сервер для запуска LLM моделей
- **Llama 3.2** - Основная языковая модель
- **Nomic Embed Text** - Модель для создания эмбеддингов
- **Scikit-learn 1.3.2** - ML библиотека для BM25 и TF-IDF
- **NumPy 1.24.4** - Численные вычисления
- **SciPy 1.11.4** - Научные вычисления

### 4. **Text Processing**
- **Tiktoken 0.5.2** - Токенизация текста
- **Custom Text Processing** - Собственные алгоритмы обработки

### 5. **Security & Validation**
- **SlowAPI 0.1.9** - Rate limiting
- **Custom Security Module** - Антиспам и валидация

### 6. **Monitoring & Logging**
- **Loguru 0.7.2** - Структурированное логирование
- **Custom Metrics** - KPI и метрики производительности

### 7. **Development & Testing**
- **Pytest 7.4.3** - Фреймворк тестирования
- **Pytest-asyncio** - Асинхронное тестирование
- **Black & isort** - Форматирование кода

## Main Modules/Components

### 1. **API Layer** (`app/api/v1/`)

#### **RAG API** (`rag.py`)
- **Назначение**: Базовые операции с документами и поиском
- **Эндпоинты**:
  - `POST /ingest` - Загрузка документов
  - `POST /query` - Поиск по базе знаний
  - `GET /documents` - Список документов
  - `DELETE /documents/{id}` - Удаление документов
  - `GET /stats` - Статистика системы

#### **Feedback API** (`feedback.py`)
- **Назначение**: Система обратной связи и обучения
- **Эндпоинты**:
  - `POST /ask` - Задать вопрос с получением ответа
  - `POST /feedback` - Предоставить обратную связь
  - `GET /feedback/history` - История фидбеков
  - `GET /feedback/stats` - Статистика фидбеков

#### **Hybrid RAG API** (`hybrid_rag.py`)
- **Назначение**: Гибридный поиск (Dense + BM25)
- **Особенности**: Комбинированный поиск для лучшей точности

#### **Strict RAG API** (`strict_rag.py`)
- **Назначение**: Строгое разделение контента
- **Особенности**: Защита от загрязнения базы знаний

### 2. **Core Services** (`app/services/`)

#### **RAG Pipeline** (`rag_pipeline.py`)
```python
class EnhancedRAGPipeline:
    """Основной RAG пайплайн с поддержкой фидбека"""
    
    async def ask(self, question: str, db: Session, session_id: str = None) -> Dict
    async def ingest_document(self, text: str, metadata: dict) -> Dict
    async def retrieve_with_feedback(self, query: str, db: Session) -> List[Chunk]
```

#### **Hybrid Retrieval** (`hybrid_rag_pipeline.py`)
```python
class HybridRAGPipeline:
    """Гибридный RAG с Dense + BM25 поиском"""
    
    def __init__(self, alpha: float = 0.6):  # α = 0.6 для оптимального баланса
        self.hybrid_retrieval = HybridRetrieval(
            embeddings_service=self.embeddings,
            bm25_service=self.bm25,
            alpha=alpha
        )
    
    async def hybrid_search(self, query: str, top_k: int = 4) -> List[RetrievalContext]
```

#### **Feedback Handler** (`enhanced_feedback_handler.py`)
```python
class EnhancedFeedbackHandler:
    """Обработка пользовательской обратной связи"""
    
    async def process_feedback(self, feedback_request: FeedbackRequest) -> FeedbackResponse
    async def apply_feedback_learning(self, feedback_event: FeedbackEvent) -> bool
    async def get_feedback_stats(self) -> Dict[str, Any]
```

#### **Learning Engine** (`learning_engine.py`)
```python
class LearningEngine:
    """Движок обучения на основе фидбека"""
    
    async def process_validated_feedback(self, feedback_event: FeedbackEvent) -> bool
    async def update_chunk_weights(self, chunk_id: int, weight_delta: float) -> bool
    async def create_feedback_chunk(self, correction_text: str) -> Chunk
```

#### **Trust Scorer** (`trust_scorer.py`)
```python
class TrustScorer:
    """Оценка доверия к пользовательскому фидбеку"""
    
    async def calculate_trust_score(self, feedback_event: FeedbackEvent) -> TrustScore
    async def update_user_reputation(self, user_id: str, feedback_quality: float) -> bool
```

#### **Ollama LLM** (`ollama_llm.py`)
```python
class OllamaLLM:
    """Интеграция с локальными Llama моделями через Ollama"""
    
    async def generate(self, prompt: str, system_prompt: str = None) -> str
    async def health_check(self) -> bool
    async def list_models(self) -> Dict[str, Any]
```

### 3. **Data Models** (`app/models/`)

#### **Documents** (`documents.py`)
```python
class Document(Base):
    """Метаданные документов"""
    id: int
    uri: str
    doc_metadata: dict
    created_at: datetime
    chunks: List[Chunk]

class Chunk(Base):
    """Текстовые фрагменты с эмбеддингами"""
    id: int
    document_id: int
    content: str
    embedding: List[float]
    source: str  # "original" или "user_feedback"
    version: int
```

#### **Feedback** (`feedback.py`)
```python
class MessageSession(Base):
    """Сессии вопрос-ответ"""
    id: str
    question: str
    answer: str
    contexts_used: List[dict]

class FeedbackEvent(Base):
    """События обратной связи"""
    id: str
    label: FeedbackLabel  # like, dislike, correct, incorrect
    rating: Optional[int]  # 1-5 звезд
    correction_text: Optional[str]
    is_spam: bool
    confidence_score: float
```

### 4. **Hybrid Retrieval System**

#### **Dense Retrieval**
- **Технология**: Векторные эмбеддинги через Nomic Embed Text
- **Алгоритм**: Косинусное сходство
- **Преимущества**: Семантический поиск, устойчивость к синонимам

#### **BM25 Search**
- **Технология**: Okapi BM25 через Scikit-learn
- **Параметры**: k1=1.5, b=0.75
- **Преимущества**: Точный поиск по ключевым словам, датам, номерам

#### **Hybrid Combination**
```python
# Формула гибридного поиска
hybrid_score = α * normalized_dense_score + (1-α) * normalized_bm25_score
# где α = 0.6 (оптимальное значение)
```

### 5. **Content Separation System**

#### **Source Tracking**
```python
class Chunk(Base):
    source: str = "original"  # "original" или "user_feedback"
    version: int = 1
```

#### **Strict Retrieval**
- **Метод**: `ask_with_feedback_separation()`
- **Параметр**: `include_feedback: bool = False`
- **Результат**: Четкое разделение оригинального и пользовательского контента

### 6. **Anti-Spam & Quality Control**

#### **Spam Detection**
```python
class UserSpamMetrics(Base):
    total_feedback_count: int
    spam_feedback_count: int
    feedback_rate_per_hour: float
    reputation_score: float  # 0.0 to 2.0
    is_trusted: bool
    is_blocked: bool
```

#### **Content Quality**
```python
class ContentQualityMetrics(Base):
    positive_feedback_count: int
    negative_feedback_count: int
    content_confidence: float  # 0.0 to 1.0
    factual_accuracy_score: float
    relevance_score: float
```

### 7. **Configuration & Settings** (`app/config.py`)

#### **Core Settings**
```python
class Settings(BaseSettings):
    # RAG Configuration
    chunk_size: int = 400
    chunk_overlap: int = 40
    default_top_k: int = 6
    
    # Ollama Configuration
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:latest"
    ollama_embedding_model: str = "nomic-embed-text:latest"
    
    # Hybrid Search
    use_hybrid_search: bool = True
    hybrid_alpha: float = 0.6
    retrieval_threshold: float = 0.4
    
    # Feedback System
    feedback_penalty_weight: float = -0.3
    feedback_boost_weight: float = 0.5
    spam_detection_threshold: float = 0.3
```

## Technical Implementation Details

### 1. **Hybrid Retrieval Algorithm**

```python
async def hybrid_search(self, query: str, top_k: int = 4) -> List[RetrievalContext]:
    """Гибридный поиск с нормализацией оценок"""
    
    # 1. Dense retrieval (семантический поиск)
    dense_results = await self.embeddings.search(query, top_k * 2)
    
    # 2. BM25 retrieval (ключевые слова)
    bm25_results = await self.bm25.search(query, top_k * 2)
    
    # 3. Нормализация оценок
    dense_scores = [r.score for r in dense_results]
    bm25_scores = [r.score for r in bm25_results]
    
    normalized_dense = self._normalize_scores(dense_scores)
    normalized_bm25 = self._normalize_scores(bm25_scores)
    
    # 4. Гибридная комбинация
    hybrid_results = []
    for i, (dense_r, bm25_r) in enumerate(zip(dense_results, bm25_results)):
        hybrid_score = (
            self.alpha * normalized_dense[i] + 
            (1 - self.alpha) * normalized_bm25[i]
        )
        
        hybrid_results.append(RetrievalContext(
            content=dense_r.content,
            doc_id=dense_r.doc_id,
            chunk_id=dense_r.chunk_id,
            score=hybrid_score,
            dense_score=dense_r.score,
            bm25_score=bm25_r.score,
            normalized_dense=normalized_dense[i],
            normalized_bm25=normalized_bm25[i],
            retrieval_method="hybrid"
        ))
    
    # 5. Сортировка и возврат топ-K результатов
    return sorted(hybrid_results, key=lambda x: x.score, reverse=True)[:top_k]
```

### 2. **Feedback Learning Process**

```python
async def process_feedback_learning(self, feedback_event: FeedbackEvent) -> bool:
    """Обработка обучения на основе фидбека"""
    
    # 1. Валидация фидбека
    if not await self._validate_feedback(feedback_event):
        return False
    
    # 2. Оценка доверия
    trust_score = await self.trust_scorer.calculate_trust_score(feedback_event)
    
    if trust_score.overall_score < self.trust_threshold:
        feedback_event.status = UpdateStatus.SPAM_FILTERED
        return False
    
    # 3. Применение изменений
    if feedback_event.label == FeedbackLabel.INCORRECT:
        # Создание нового чанка с исправлением
        await self._create_correction_chunk(feedback_event)
    elif feedback_event.label == FeedbackLabel.CORRECT:
        # Увеличение веса существующего чанка
        await self._boost_chunk_weight(feedback_event)
    
    # 4. Обновление метрик
    await self._update_quality_metrics(feedback_event)
    
    return True
```

### 3. **Content Separation Logic**

```python
async def ask_with_feedback_separation(
    self, 
    question: str, 
    include_feedback: bool = False
) -> Dict[str, Any]:
    """Вопрос с явным разделением контента"""
    
    # 1. Поиск только в оригинальном контенте
    original_contexts = await self._search_original_content(question)
    
    # 2. Опционально: поиск в пользовательском контенте
    feedback_contexts = []
    if include_feedback:
        feedback_contexts = await self._search_feedback_content(question)
    
    # 3. Генерация ответа с указанием источников
    answer = await self._generate_answer_with_sources(
        question, 
        original_contexts, 
        feedback_contexts
    )
    
    return {
        "answer": answer,
        "original_contexts": original_contexts,
        "feedback_contexts": feedback_contexts,
        "sources_separated": True
    }
```

## Performance Characteristics

### 1. **Retrieval Performance**
- **Dense Search**: ~50-100ms для 1000 чанков
- **BM25 Search**: ~20-50ms для 1000 чанков
- **Hybrid Combination**: ~70-150ms общее время

### 2. **Memory Usage**
- **Base System**: ~200-300MB
- **With Models**: ~2-4GB (зависит от размера модели)
- **Database**: ~10-50MB для 1000 документов

### 3. **Scalability**
- **Horizontal**: Поддержка множественных инстансов
- **Vertical**: Оптимизация для больших моделей
- **Database**: Индексы для быстрого поиска

## Security & Privacy

### 1. **Data Privacy**
- **Локальная обработка**: Все данные остаются на сервере
- **No Cloud Dependencies**: Полная независимость от облачных сервисов
- **Encryption**: Шифрование чувствительных данных

### 2. **Anti-Spam Protection**
- **Rate Limiting**: Ограничение частоты запросов
- **User Reputation**: Система репутации пользователей
- **Content Filtering**: Фильтрация некачественного контента

### 3. **Input Validation**
- **Pydantic Schemas**: Строгая валидация входных данных
- **File Upload Security**: Проверка типов и размеров файлов
- **SQL Injection Protection**: Параметризованные запросы

## Monitoring & Observability

### 1. **Logging**
- **Structured Logging**: JSON формат с контекстом
- **Log Levels**: DEBUG, INFO, WARNING, ERROR
- **Log Rotation**: Автоматическая ротация логов

### 2. **Metrics**
- **System Metrics**: CPU, Memory, Database
- **Business Metrics**: KPI обратной связи, качество ответов
- **Performance Metrics**: Время ответа, точность поиска

### 3. **Health Checks**
- **Basic Health**: `/healthz`
- **Detailed Health**: `/healthz/detailed`
- **Metrics Endpoint**: `/metrics`

## Deployment Architecture

### 1. **Development Environment**
```bash
# Локальная разработка
python -m app.main
uvicorn app.main:app --reload
```

### 2. **Production Environment**
```bash
# Продакшн развертывание
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker
```

### 3. **Docker Support**
```dockerfile
FROM python:3.9-slim
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "-m", "app.main"]
```

## Conclusion

LiveLearn RAG представляет собой современную, масштабируемую систему RAG с уникальными возможностями:

1. **Гибридный поиск** обеспечивает высокую точность для различных типов запросов
2. **Система обратной связи** позволяет непрерывно улучшать качество ответов
3. **Локальная обработка** гарантирует конфиденциальность данных
4. **Модульная архитектура** обеспечивает легкость расширения и поддержки
5. **Комплексная система безопасности** защищает от спама и некачественного контента

Система готова к использованию в продакшене и может быть адаптирована под специфические требования различных проектов.
