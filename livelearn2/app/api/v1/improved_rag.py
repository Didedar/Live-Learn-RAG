"""
API endpoint для улучшенной RAG системы.
"""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ...database import get_db
from ...schemas.rag import QueryRequest, QueryResponse
from ...services.improved_rag_pipeline import ImprovedRAGPipeline
from ...core.security import optional_auth
from loguru import logger

router = APIRouter(prefix="/improved-rag", tags=["Improved RAG"])

# Глобальный экземпляр улучшенной RAG системы
improved_rag_pipeline = ImprovedRAGPipeline()


@router.post("/ask", response_model=QueryResponse)
async def ask_improved(
    request: QueryRequest,
    db: Session = Depends(get_db),
    user_id: Optional[str] = Depends(optional_auth)
):
    """
    Улучшенный RAG запрос с исправлениями на основе анализа ошибок.
    
    Основные улучшения:
    - Исправлены языковые проблемы (только русский язык)
    - Улучшены embeddings для лучшего поиска
    - Добавлена оценка уверенности (confidence scoring)
    - Оптимизированы веса гибридного поиска
    - Строгие правила отказа от неточных ответов
    """
    try:
        logger.info(f"Improved RAG query: {request.question[:100]}...")
        
        # Валидация запроса
        if not request.question or len(request.question.strip()) < 3:
            raise HTTPException(
                status_code=400, 
                detail="Вопрос слишком короткий. Минимум 3 символа."
            )
        
        if len(request.question) > 2000:
            raise HTTPException(
                status_code=400,
                detail="Вопрос слишком длинный. Максимум 2000 символов."
            )
        
        # Обработка запроса улучшенной системой
        result = await improved_rag_pipeline.ask(
            question=request.question.strip(),
            db=db,
            session_id=request.session_id,
            top_k=request.top_k or 4
        )
        
        # Формируем ответ
        response = QueryResponse(
            answer=result["answer"],
            contexts=result.get("contexts", []),
            message_id=result["message_id"],
            metadata={
                "retrieval_method": result.get("retrieval_method", "improved_hybrid"),
                "can_answer": result.get("can_answer", True),
                "confidence": result.get("confidence", 0.0),
                "max_score": result.get("max_score", 0.0),
                "improvements_applied": [
                    "language_consistency",
                    "enhanced_embeddings", 
                    "confidence_scoring",
                    "legal_optimization",
                    "strict_quality_control"
                ],
                "pipeline_version": "improved_v1.0"
            }
        )
        
        logger.info(f"Improved RAG response generated successfully")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in improved RAG endpoint: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при обработке запроса: {str(e)}"
        )


@router.get("/stats")
async def get_improved_stats(
    db: Session = Depends(get_db),
    user_id: Optional[str] = Depends(optional_auth)
):
    """
    Статистика улучшенной RAG системы.
    """
    try:
        stats = await improved_rag_pipeline.get_retrieval_stats(db)
        
        return {
            "status": "success",
            "pipeline_type": "improved_hybrid_rag",
            "version": "1.0",
            "improvements": {
                "language_fixes": "Строгие правила русского языка",
                "enhanced_embeddings": "Улучшенная семантическая логика",
                "confidence_scoring": "Оценка уверенности в ответах",
                "legal_optimization": "Оптимизация для юридических документов",
                "hybrid_search": "Улучшенные веса поиска (40% dense, 60% keyword)",
                "quality_control": "Автоматическая проверка качества ответов"
            },
            "statistics": stats
        }
        
    except Exception as e:
        logger.error(f"Error getting improved stats: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка получения статистики: {str(e)}"
        )


@router.post("/compare")
async def compare_with_original(
    request: QueryRequest,
    db: Session = Depends(get_db),
    user_id: Optional[str] = Depends(optional_auth)
):
    """
    Сравнение улучшенной системы с оригинальной.
    """
    try:
        from ...services.enhanced_rag_pipeline import EnhancedRAGPipeline
        
        logger.info(f"Comparing RAG systems for: {request.question[:100]}...")
        
        # Оригинальная система
        original_pipeline = EnhancedRAGPipeline()
        original_result = await original_pipeline.ask(
            question=request.question,
            db=db,
            session_id=request.session_id,
            top_k=request.top_k or 4
        )
        
        # Улучшенная система
        improved_result = await improved_rag_pipeline.ask(
            question=request.question,
            db=db,
            session_id=request.session_id,
            top_k=request.top_k or 4
        )
        
        return {
            "question": request.question,
            "original": {
                "answer": original_result["answer"],
                "can_answer": original_result.get("can_answer", True),
                "contexts_count": len(original_result.get("contexts", [])),
                "max_score": original_result.get("max_score", 0.0),
                "method": original_result.get("retrieval_method", "unknown")
            },
            "improved": {
                "answer": improved_result["answer"],
                "can_answer": improved_result.get("can_answer", True),
                "confidence": improved_result.get("confidence", 0.0),
                "contexts_count": len(improved_result.get("contexts", [])),
                "max_score": improved_result.get("max_score", 0.0),
                "method": improved_result.get("retrieval_method", "improved_hybrid")
            },
            "improvements_applied": [
                "Исправлены языковые проблемы",
                "Улучшен поиск релевантного контекста",
                "Добавлена оценка уверенности",
                "Оптимизированы веса поиска",
                "Строгий контроль качества"
            ]
        }
        
    except Exception as e:
        logger.error(f"Error in comparison: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при сравнении: {str(e)}"
        )

