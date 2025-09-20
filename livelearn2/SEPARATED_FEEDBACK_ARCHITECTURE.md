# Архитектура разделенного фидбэка без протечек

## Обзор проблемы

В исходной системе фидбэк от пользователей смешивался с основными документами, что приводило к **протечкам** между разными намерениями запросов:

- Фидбэк по вопросу Q1 влиял на ответы для Q2 с похожей семантикой
- `correction_text` добавлялся как новые чанки в основной индекс
- Отсутствовала привязка фидбэка к конкретным намерениям пользователей

## Новая архитектура: Разделение хранилищ

### 1. Два изолированных хранилища

#### `docs_index` - Только оригинальные документы
```
✅ Содержит: Исходные документы и их чанки
❌ НЕ содержит: Фидбэк, коррекции, пользовательский контент
🔒 Источник: source = "original"
```

#### `feedback_store` - Изолированное хранение фидбэка
```
✅ Содержит: Intent-based фидбэк с метаданными
❌ НЕ содержит: Документы или их содержимое
🔑 Привязка: intent_key + evidence links
```

### 2. Система Intent Keys

#### Нормализация намерений
```python
def normalize_query(query: str) -> dict:
    # Удаление стоп-слов
    # Лемматизация
    # Извлечение сущностей
    # Сортировка токенов
    return {
        'normalized_text': 'машинное обучение? такое',
        'tokens': ['машинное', 'обучение', 'такое'],
        'entities': ['NUM:2023', 'PROPER:Python'],
        'intent_key': 'sha256_hash'
    }
```

#### Детерминированные ключи намерений
```
Q1: "Что такое машинное обучение?" 
    → intent_key: 2635f8c76444e463...

Q2: "Объясни машинное обучение"
    → intent_key: afb5f5adf1f9f6c0...

Q3: "Как приготовить пасту?"
    → intent_key: 0f35ade26fa211db...
```

### 3. Жёсткие фильтры (Gating)

#### Условия применения фидбэка
```python
def apply_feedback_filter(feedback, query, retrieved_docs):
    # Gate 1: Intent matching
    if feedback.intent_key != query.intent_key:
        if similarity(feedback.intent, query.intent) < TAU_INTENT:
            return False
    
    # Gate 2: Evidence overlap
    feedback_docs = {ev.doc_id for ev in feedback.evidence}
    retrieved_docs_ids = {doc.id for doc in retrieved_docs}
    if not (feedback_docs & retrieved_docs_ids):
        return False
    
    # Gate 3: User scope filtering
    if feedback.scope == 'local' and feedback.user_id != query.user_id:
        return False
    
    return True
```

### 4. Переранжирование без контаминации

```python
def rerank_with_feedback(docs, applicable_feedback):
    for doc in docs:
        for feedback in applicable_feedback:
            if doc.id in feedback.evidence_doc_ids:
                if feedback.label == 'reject':
                    doc.score += feedback.polarity * feedback.weight * 0.1
                elif feedback.label == 'prefer':
                    doc.score += feedback.polarity * feedback.weight * 0.1
                # ВАЖНО: correction_text НЕ попадает в контекст!
    
    return sorted(docs, key=lambda x: x.score, reverse=True)
```

## Новые модели данных

### IntentKey - Нормализованные намерения
```sql
CREATE TABLE intent_keys (
    id VARCHAR(64) PRIMARY KEY,  -- SHA-256 hash
    normalized_text TEXT NOT NULL,
    entities JSON DEFAULT '[]',
    tokens JSON DEFAULT '[]',
    embedding JSON NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### IntentFeedback - Изолированный фидбэк
```sql
CREATE TABLE intent_feedback (
    id VARCHAR(36) PRIMARY KEY,
    intent_key VARCHAR(64) NOT NULL,  -- Жёсткая привязка к намерению
    query_text TEXT NOT NULL,         -- Для отладки
    user_id VARCHAR(128),
    label ENUM('prefer', 'reject', 'fix', 'style'),
    polarity INTEGER,                 -- +1/-1
    weight FLOAT DEFAULT 1.0,         -- Доверие
    scope ENUM('local', 'cluster', 'global'),
    evidence JSON DEFAULT '[]',       -- Ссылки на docs
    notes TEXT,                       -- Объяснение пользователя
    correction_text TEXT,             -- НЕ попадает в контексты!
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## API Endpoints

### Новые endpoints с разделением

#### `/api/v1/separated/ask` - Запрос с изолированным фидбэком
```python
POST /api/v1/separated/ask
{
    "question": "Что такое машинное обучение?",
    "user_id": "user123",
    "top_k": 6
}

Response:
{
    "message_id": "uuid",
    "answer": "...",
    "contexts": [...],  # ТОЛЬКО из docs_index
    "feedback_applied_count": 2,
    "separation_integrity": "maintained"
}
```

#### `/api/v1/separated/feedback` - Изолированное хранение фидбэка
```python
POST /api/v1/separated/feedback
{
    "message_id": "uuid",
    "feedback_label": "reject",
    "target_doc_ids": [1, 2],
    "target_chunk_ids": [5, 6],
    "notes": "Информация устарела",
    "correction_text": "Актуальные данные: ..."
}

Response:
{
    "status": "stored_separately",
    "feedback_id": "uuid",
    "contamination_risk": "eliminated"
}
```

#### `/api/v1/separated/health/separation` - Проверка целостности
```python
GET /api/v1/separated/health/separation

Response:
{
    "separation_integrity": "HEALTHY",
    "feedback_chunks_in_docs_index": 0,    # Должно быть 0!
    "original_chunks": 1857,
    "feedback_entries": 5,
    "contamination_risk": "eliminated"
}
```

## Алгоритм работы

### 1. Обработка запроса
```
User Query → Intent Normalization → Intent Key Generation
     ↓
docs_index Retrieval (ONLY original content)
     ↓
Feedback Application (intent-based gating)
     ↓
Reranking (scores only, no content mixing)
     ↓
LLM Generation (ONLY original contexts)
```

### 2. Хранение фидбэка
```
User Feedback → Intent Key Lookup/Creation
     ↓
Evidence Linking (doc_ids, chunk_ids)
     ↓
Isolated Storage in feedback_store
     ↓
User Trust Score Update
```

### 3. Применение фидбэка
```
New Query → Intent Key Generation
     ↓
Feedback Retrieval (intent_key matching)
     ↓
Gating Filters (intent + evidence + user scope)
     ↓
Score Adjustments (NO content contamination)
```

## Гарантии архитектуры

### ✅ Что ГАРАНТИРОВАННО предотвращено

1. **Протечки между намерениями**
   - Фидбэк по ML не влияет на кулинарные вопросы
   - Жёсткое разделение через intent_key

2. **Контаминация контекстов**
   - `correction_text` НИКОГДА не попадает в LLM контексты
   - Только `source="original"` чанки в retrieval

3. **Семантические коллизии**
   - Похожие темы имеют разные intent_key
   - Фидбэк применяется только при точном/близком совпадении намерений

4. **Пользовательские протечки**
   - `scope="local"` ограничивает фидбэк одним пользователем
   - Изоляция между пользователями

### 🛡️ Механизмы защиты

1. **Архитектурное разделение**
   ```python
   # docs_index - только это разрешено
   chunk.source = "original"
   
   # feedback_store - изолированно
   feedback.intent_key = required_field
   ```

2. **Валидация на уровне кода**
   ```python
   def retrieve_from_docs_only():
       return chunks.filter(source="original")  # Жёсткий фильтр
   ```

3. **База данных constraints**
   ```sql
   -- Логическая проверка целостности
   CREATE VIEW separation_integrity_check AS
   SELECT 
       COUNT(CASE WHEN source = 'user_feedback' THEN 1 END) as contamination
   FROM chunks;
   ```

## Миграция и развёртывание

### 1. Миграция базы данных
```bash
# Применить новые таблицы
python manage.py migrate 006_separated_feedback_tables.sql

# Проверить целостность
curl /api/v1/separated/health/separation
```

### 2. Переход на новые endpoints
```python
# Старый способ (с протечками)
POST /api/v1/ask
POST /api/v1/feedback

# Новый способ (без протечек) 
POST /api/v1/separated/ask
POST /api/v1/separated/feedback
```

### 3. Проверка работоспособности
```bash
# Запуск демонстрации
python separated_feedback_demo.py

# Ожидаемый результат:
# ✅ docs_index чист от контаминации
# ✅ Протечек не обнаружено - фидбэк изолирован
# ✅ Гейтинг работает - фидбэк применяется только к релевантным намерениям
# 🏆 Оценка целостности: 100%
```

## Производительность и масштабирование

### Индексы для быстрого доступа
```sql
-- Быстрый поиск фидбэка по намерениям
INDEX idx_intent_feedback_intent_user (intent_key, user_id)
INDEX idx_intent_feedback_scope_intent (scope, intent_key)

-- Быстрая проверка целостности
INDEX idx_chunks_source (source)
```

### Кэширование intent_key
```python
# Кэш нормализованных намерений
@cache(ttl=3600)
def get_or_create_intent_key(query: str) -> str:
    return intent_processor.process_and_store_intent(query)
```

### TTL и очистка старого фидбэка
```sql
-- Автоматическая очистка старого фидбэка
DELETE FROM intent_feedback 
WHERE created_at < NOW() - INTERVAL 30 DAY
  AND scope = 'local';
```

## Мониторинг и метрики

### Ключевые метрики
```python
metrics = {
    "separation_integrity": "HEALTHY",  # HEALTHY | CONTAMINATED
    "feedback_contamination_count": 0,   # Должно быть 0
    "intent_coverage": 0.85,            # Доля запросов с фидбэком
    "gating_efficiency": 0.92,          # Доля корректно отфильтрованного фидбэка
    "user_feedback_quality": 0.78       # Средний trust_score
}
```

### Алерты
```yaml
alerts:
  - name: "Feedback Contamination Detected"
    condition: feedback_chunks_in_docs_index > 0
    severity: CRITICAL
    
  - name: "Intent Key Missing"
    condition: feedback_without_intent_key > 0
    severity: HIGH
    
  - name: "Low Separation Integrity"
    condition: separation_integrity != "HEALTHY"
    severity: CRITICAL
```

## Заключение

Новая архитектура **гарантированно исключает протечки** между разными намерениями пользователей за счёт:

1. **Строгого разделения хранилищ** - docs_index vs feedback_store
2. **Intent-based targeting** - фидбэк привязан к конкретным намерениям
3. **Жёстких фильтров** - многоуровневая проверка применимости фидбэка
4. **Изолированного переранжирования** - влияние только на порядок, не на содержание

Система обеспечивает **детерминированное и прозрачное** применение фидбэка без риска контаминации контекстов или протечек между разными типами запросов.


