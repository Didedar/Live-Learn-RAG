"""API endpoints for feedback system."""

from typing import Any, Dict
import inspect

from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger
from sqlalchemy.orm import Session

from ...core.exceptions import FeedbackError, ValidationError
from ...database import get_db
from ...schemas.feedback import (
    AskRequest, 
    AskResponse, 
    ContextInfo,
    FeedbackRequest,
    FeedbackResponse,
    UserFeedback
)
from ...services.feedback_handler import FeedbackHandler
from ...services.rag_pipeline import EnhancedRAGPipeline

router = APIRouter(prefix="/feedback", tags=["feedback"])

# Инициализация сервисов
rag_pipeline = EnhancedRAGPipeline()
feedback_handler = FeedbackHandler()

# ---------- утилиты ----------

def _is_coro_fn(fn) -> bool:
    return inspect.iscoroutinefunction(fn)

async def _call_maybe_async(fn, *args, **kwargs):
    """Вызвать метод, независимо sync/async."""
    if _is_coro_fn(fn):
        return await fn(*args, **kwargs)
    res = fn(*args, **kwargs)
    if inspect.isawaitable(res):
        return await res
    return res

def _pick_pipeline_method(pipeline: Any):
    """
    Найти метод ответа в пайплайне: ask/answer/query/generate/run.
    """
    for name in ("ask", "answer", "query", "generate", "run"):
        fn = getattr(pipeline, name, None)
        if callable(fn):
            return fn, name
    raise AttributeError(
        "Не найден метод ответа в EnhancedRAGPipeline "
        "(ожидались: ask/answer/query/generate/run)"
    )

def _normalize_context_item(x: Any) -> Dict[str, Any]:
    """Приводит элемент контекста к виду {text, metadata, score}."""
    if isinstance(x, dict):
        text = x.get("text") or x.get("content") or ""
        meta = x.get("metadata") or {}
        if not meta:
            meta = {k: v for k, v in x.items() if k in ("title", "source", "doc_id", "path")}
        return {"text": text, "metadata": meta, "score": x.get("score")}
    # объект с атрибутами
    text = getattr(x, "text", None) or getattr(x, "content", None) or ""
    score = getattr(x, "score", None)
    meta = getattr(x, "metadata", None)
    if meta is None:
        meta = {}
        for k in ("title", "source", "doc_id", "path"):
            v = getattr(x, k, None)
            if v is not None:
                meta[k] = v
    return {"text": text, "metadata": meta, "score": score}

def _extract_answer_and_contexts(result: Any) -> Dict[str, Any]:
    """
    Поддержка популярных форматов выхода из пайплайна:
    - dict: {answer|text|output, contexts|context|chunks|documents|sources}
    - tuple/list: (answer, contexts)
    - объект с атрибутами: .answer/.text + .contexts/.chunks/.documents/.sources
    """
    if isinstance(result, dict):
        answer = result.get("answer") or result.get("output") or result.get("text") or ""
        contexts = []
        for key in ("contexts", "context", "chunks", "documents", "sources", "evidence"):
            if key in result and result[key]:
                contexts = [_normalize_context_item(c) for c in result[key]]
                break
        
        # Добавляем message_id если есть
        message_id = result.get("message_id", "")
        
        return {"answer": answer, "contexts": contexts, "message_id": message_id}

    if isinstance(result, (list, tuple)) and result:
        answer = result[0] if len(result) > 0 else ""
        ctxs = result[1] if len(result) > 1 else []
        return {"answer": str(answer or ""), "contexts": [_normalize_context_item(c) for c in (ctxs or [])], "message_id": ""}

    answer = getattr(result, "answer", None) or getattr(result, "text", None) or ""
    ctx = (
        getattr(result, "contexts", None) or getattr(result, "context", None) or
        getattr(result, "chunks", None) or getattr(result, "documents", None) or
        getattr(result, "sources", None) or []
    )
    message_id = getattr(result, "message_id", "")
    
    return {"answer": str(answer or ""), "contexts": [_normalize_context_item(c) for c in (ctx or [])], "message_id": message_id}

# ---------- реальные эндпоинты ----------

@router.post("/ask", response_model=AskResponse, status_code=status.HTTP_200_OK)
async def ask_endpoint(payload: AskRequest, db: Session = Depends(get_db)):
    """
    Принять вопрос от фронта и вернуть ответ + контексты.
    Финальный путь: /api/v1/feedback/ask (см. include_router в app.main).
    """
    try:
        logger.info(f"Received ask request: {payload.question[:100]}...")
        
        fn, method_name = _pick_pipeline_method(rag_pipeline)
        sig = inspect.signature(fn)
        
        logger.info(f"Using pipeline method: {method_name}{sig}")

        kwargs = {}
        if "session_id" in sig.parameters:
            kwargs["session_id"] = getattr(payload, "session_id", None)
        if "top_k" in sig.parameters:
            kwargs["top_k"] = getattr(payload, "top_k", 6)
        if "db" in sig.parameters:
            kwargs["db"] = db

        if "question" in sig.parameters:
            raw = await _call_maybe_async(fn, question=payload.question, **kwargs)
        else:
            raw = await _call_maybe_async(fn, payload.question, **kwargs)

        logger.info(f"Pipeline returned: {type(raw)}")

        data = _extract_answer_and_contexts(raw)
        if not data.get("answer"):
            data["answer"] = "Ответ не найден. Уточните вопрос или загрузите знания."
        data.setdefault("contexts", [])
        
        # Обеспечиваем наличие message_id
        if not data.get("message_id"):
            import uuid
            data["message_id"] = str(uuid.uuid4())

        logger.info(f"Returning response with {len(data['contexts'])} contexts")
        return AskResponse(**data)

    except (ValidationError, FeedbackError) as e:
        logger.error(f"/feedback/ask validation error: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    except Exception as e:
        import traceback
        logger.exception(f"/feedback/ask failed: {e}")  # напечатает стектрейс
        # В DEV режиме вернём подробности в тело ответа:
        from ...config import settings
        if getattr(settings, "debug", False):
            raise HTTPException(
                status_code=500,
                detail={"error": "Ask pipeline failed",
                        "type": type(e).__name__,
                        "message": str(e),
                        "traceback": traceback.format_exc()}
            )
        # В PROD — как было:
        raise HTTPException(status_code=500, detail="Ask pipeline failed")

# Логирование при инициализации
try:
    fn, method_name = _pick_pipeline_method(rag_pipeline)
    sig = inspect.signature(fn)
    logger.info(f"Pipeline initialized with method: {method_name}{sig}")
except Exception as e:
    logger.error(f"Failed to initialize pipeline: {e}")


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    feedback_request: FeedbackRequest,
    db: Session = Depends(get_db)
):
    """
    Submit user feedback for a response.
    
    This endpoint allows users to provide feedback on RAG responses,
    which is used to improve future retrieval and generation.
    """
    try:
        logger.info(f"Received feedback for message: {feedback_request.message_id}")
        
        # Process feedback through handler
        result = await feedback_handler.process_feedback(db, feedback_request)
        
        return FeedbackResponse(**result)
        
    except ValueError as e:
        logger.error(f"Feedback validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Failed to process feedback: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process feedback"
        )


@router.get("/health")
async def feedback_health():
    """Health check for feedback service."""
    try:
        # Basic health check
        return {
            "status": "healthy",
            "service": "feedback",
            "pipeline_available": hasattr(rag_pipeline, 'ask'),
            "handler_available": hasattr(feedback_handler, 'process_feedback')
        }
    except Exception as e:
        logger.error(f"Feedback service health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }