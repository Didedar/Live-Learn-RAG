"""Enhanced feedback endpoints with learning capabilities."""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger
from sqlalchemy.orm import Session

from ...database import get_db
from ...services.enhanced_feedback_handler import EnhancedFeedbackHandler

router = APIRouter(prefix="/feedback", tags=["feedback-enhanced"])

# Инициализация сервисов
feedback_handler = EnhancedFeedbackHandler()


@router.get("/learning-stats")
async def get_learning_statistics(db: Session = Depends(get_db)):
    """
    Получить статистику обучения системы.
    
    Возвращает детальную информацию о:
    - Обработанных фидбэках
    - Примененных изменениях
    - Качестве фильтрации
    - Доверительных оценках
    """
    try:
        stats = await feedback_handler.get_feedback_statistics(db)
        return {
            "status": "success",
            "statistics": stats,
            "message": "Learning statistics retrieved successfully"
        }
    except Exception as e:
        logger.error(f"Failed to get learning statistics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve learning statistics"
        )


@router.post("/review/{feedback_id}")
async def review_queued_feedback(
    feedback_id: str,
    reviewer_decision: str,
    reviewer_id: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    Ручная проверка фидбэка, находящегося в очереди.
    
    Args:
        feedback_id: ID фидбэка для проверки
        reviewer_decision: approve/reject
        reviewer_id: ID проверяющего (опционально)
    """
    try:
        if reviewer_decision not in ["approve", "reject"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="reviewer_decision must be 'approve' or 'reject'"
            )
        
        result = await feedback_handler.review_queued_feedback(
            db=db,
            feedback_id=feedback_id,
            reviewer_decision=reviewer_decision,
            reviewer_id=reviewer_id
        )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to review feedback: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to review feedback"
        )


@router.get("/system-health")
async def get_system_health(db: Session = Depends(get_db)):
    """
    Проверить здоровье системы обучения.
    
    Возвращает информацию о:
    - Статусе компонентов системы
    - Качестве работы фильтров
    - Эффективности применения фидбэка
    """
    try:
        stats = await feedback_handler.get_feedback_statistics(db)
        
        # Анализируем здоровье системы
        system_health = stats.get("system_health", {})
        
        health_status = "healthy"
        issues = []
        
        # Проверяем коэффициент применения фидбэка
        application_rate = system_health.get("application_rate", 0)
        if application_rate < 0.1:
            health_status = "warning"
            issues.append("Low feedback application rate")
        elif application_rate < 0.05:
            health_status = "unhealthy"
            issues.append("Very low feedback application rate")
        
        # Проверяем коэффициент фильтрации
        filter_rate = system_health.get("filter_rate", 0)
        if filter_rate > 0.8:
            health_status = "warning"
            issues.append("High filter rate - may be too strict")
        
        # Проверяем очередь
        queue_rate = system_health.get("queue_rate", 0)
        if queue_rate > 0.5:
            health_status = "warning"
            issues.append("High queue rate - needs manual review")
        
        return {
            "status": health_status,
            "issues": issues,
            "metrics": system_health,
            "components": {
                "trust_scorer": "active",
                "content_filter": "active", 
                "learning_engine": "active"
            },
            "recommendations": _generate_recommendations(system_health)
        }
        
    except Exception as e:
        logger.error(f"Failed to check system health: {e}")
        return {
            "status": "error",
            "error": str(e),
            "components": {
                "trust_scorer": "unknown",
                "content_filter": "unknown",
                "learning_engine": "unknown"
            }
        }


def _generate_recommendations(health_metrics: dict) -> list:
    """Генерирует рекомендации на основе метрик здоровья."""
    recommendations = []
    
    application_rate = health_metrics.get("application_rate", 0)
    filter_rate = health_metrics.get("filter_rate", 0)
    queue_rate = health_metrics.get("queue_rate", 0)
    
    if application_rate < 0.1:
        recommendations.append(
            "Consider lowering trust score thresholds to apply more feedback"
        )
    
    if filter_rate > 0.7:
        recommendations.append(
            "Content filters may be too strict - review filter settings"
        )
    
    if queue_rate > 0.4:
        recommendations.append(
            "High queue rate detected - schedule manual review sessions"
        )
    
    if application_rate > 0.8 and filter_rate < 0.1:
        recommendations.append(
            "System is working well - consider increasing automation"
        )
    
    return recommendations
