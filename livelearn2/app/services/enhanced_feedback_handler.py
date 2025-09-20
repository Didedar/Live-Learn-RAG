"""Enhanced feedback handler with learning capabilities."""

import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any

from loguru import logger
from sqlalchemy.orm import Session

from ..config import settings
from ..models.feedback import (
    FeedbackEvent, FeedbackLabel, MessageSession, UpdateStatus
)
from ..schemas.feedback import FeedbackRequest
from .learning_engine import LearningEngine
from .trust_scorer import TrustScorer
from .content_filter import ContentFilter


class EnhancedFeedbackHandler:
    """Enhanced feedback handler with advanced learning capabilities."""
    
    def __init__(self):
        self.learning_engine = LearningEngine()
        self.trust_scorer = TrustScorer()
        self.content_filter = ContentFilter()
        
        logger.info("Enhanced feedback handler initialized")
    
    async def process_feedback(
        self,
        db: Session,
        feedback_request: FeedbackRequest,
        user_ip: str = "127.0.0.1"
    ) -> Dict[str, Any]:
        """
        Обрабатывает фидбэк пользователя с полной системой обучения.
        
        Процесс:
        1. Валидация запроса
        2. Извлечение контекста
        3. Обработка через систему обучения
        4. Логирование результатов
        
        Args:
            db: Сессия базы данных
            feedback_request: Запрос фидбэка
            
        Returns:
            Результат обработки с деталями обучения
        """
        try:
            logger.info(f"Processing enhanced feedback for message: {feedback_request.message_id}")
            
            # 1. Валидация запроса
            validation_result = await self._validate_feedback_request(db, feedback_request)
            if not validation_result["valid"]:
                return {
                    "status": "error",
                    "message": validation_result["error"],
                    "applied": False
                }
            
            message_session = validation_result["message_session"]
            
            # 2. Извлечение контекста и определение целевого чанка
            context_info = await self._extract_feedback_context(
                db, feedback_request, message_session
            )
            
            if not context_info["chunk_id"]:
                return {
                    "status": "error", 
                    "message": "Could not identify target chunk for feedback",
                    "applied": False
                }
            
            # 3. Обработка через систему обучения
            learning_result = await self.learning_engine.process_feedback_for_learning(
                db=db,
                feedback_content=feedback_request.user_feedback.correction_text or "User feedback",
                original_chunk_id=context_info["chunk_id"],
                user_id=getattr(feedback_request, 'user_id', None),
                message_id=feedback_request.message_id,
                feedback_label=feedback_request.user_feedback.label
            )
            
            # 4. Создание записи о фидбэке
            feedback_event = await self._create_feedback_event(
                db=db,
                feedback_request=feedback_request,
                message_session=message_session,
                learning_result=learning_result,
                context_info=context_info
            )
            
            db.commit()
            
            # 5. Формирование ответа
            response = {
                "status": learning_result["status"],
                "message": learning_result["message"],
                "applied": learning_result["applied"],
                "feedback_id": feedback_event.id,
                "learning_details": {
                    "trust_score": learning_result.get("trust_score"),
                    "confidence": learning_result.get("confidence"),
                    "strategy": learning_result.get("strategy"),
                    "changes": learning_result.get("changes")
                }
            }
            
            # Добавляем дополнительную информацию для пользователя
            if learning_result["applied"]:
                response["user_message"] = (
                    "Спасибо за ваш фидбэк! Он был проверен и применен к системе знаний. "
                    "Теперь система будет давать более точные ответы на похожие вопросы."
                )
            elif learning_result["status"] == "queued":
                response["user_message"] = (
                    "Спасибо за ваш фидбэк! Он отправлен на дополнительную проверку "
                    "и будет рассмотрен нашими экспертами."
                )
            else:
                response["user_message"] = (
                    "Спасибо за ваш фидбэк! К сожалению, он не прошел автоматическую проверку, "
                    "но мы учтем его при улучшении системы."
                )
            
            logger.info(
                f"Enhanced feedback processing completed: {learning_result['status']} "
                f"(applied: {learning_result['applied']})"
            )
            
            return response
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error in enhanced feedback processing: {e}")
            
            return {
                "status": "error",
                "message": f"Feedback processing failed: {str(e)}",
                "applied": False,
                "user_message": "Произошла ошибка при обработке вашего фидбэка. Попробуйте позже."
            }
    
    async def _validate_feedback_request(
        self,
        db: Session,
        feedback_request: FeedbackRequest
    ) -> Dict[str, Any]:
        """Валидирует запрос фидбэка."""
        
        # Проверяем существование сообщения
        message_session = db.query(MessageSession).filter(
            MessageSession.id == feedback_request.message_id
        ).first()
        
        if not message_session:
            return {
                "valid": False,
                "error": f"Message {feedback_request.message_id} not found"
            }
        
        # Проверяем наличие фидбэка
        if not feedback_request.user_feedback:
            return {
                "valid": False,
                "error": "No feedback provided"
            }
        
        # Проверяем тип фидбэка
        if feedback_request.user_feedback.label == FeedbackLabel.INCORRECT:
            if not feedback_request.user_feedback.correction_text:
                return {
                    "valid": False,
                    "error": "Correction text required for incorrect feedback"
                }
        
        return {
            "valid": True,
            "message_session": message_session
        }
    
    async def _extract_feedback_context(
        self,
        db: Session,
        feedback_request: FeedbackRequest,
        message_session: MessageSession
    ) -> Dict[str, Any]:
        """Извлекает контекст фидбэка для определения целевого чанка."""
        
        context_info = {
            "chunk_id": None,
            "document_id": None,
            "context_used": []
        }
        
        try:
            # Пытаемся извлечь информацию о контексте из сессии сообщения
            if hasattr(message_session, 'contexts_used') and message_session.contexts_used:
                contexts = message_session.contexts_used
                
                if isinstance(contexts, list) and contexts:
                    # Берем первый контекст как основной для фидбэка
                    primary_context = contexts[0]
                    
                    context_info.update({
                        "chunk_id": primary_context.get("chunk_id"),
                        "document_id": primary_context.get("doc_id"),
                        "context_used": contexts
                    })
            
            # Если есть целевая информация в фидбэке
            if (feedback_request.user_feedback.target and 
                feedback_request.user_feedback.target.chunk_id):
                
                context_info.update({
                    "chunk_id": feedback_request.user_feedback.target.chunk_id,
                    "document_id": feedback_request.user_feedback.target.doc_id
                })
            
            return context_info
            
        except Exception as e:
            logger.error(f"Error extracting feedback context: {e}")
            return context_info
    
    async def _create_feedback_event(
        self,
        db: Session,
        feedback_request: FeedbackRequest,
        message_session: MessageSession,
        learning_result: Dict[str, Any],
        context_info: Dict[str, Any]
    ) -> FeedbackEvent:
        """Создает запись о событии фидбэка."""
        
        # Определяем статус на основе результата обучения
        if learning_result["applied"]:
            status = UpdateStatus.APPLIED
        elif learning_result["status"] == "queued":
            status = UpdateStatus.QUEUED
        elif "filter" in learning_result.get("message", "").lower():
            status = UpdateStatus.CONTENT_FILTERED
        else:
            status = UpdateStatus.FAILED
        
        # Подготавливаем метаданные
        extra_data = {
            "feedback_label": feedback_request.user_feedback.label.value,
            "feedback_scope": feedback_request.user_feedback.scope.value,
            "learning_result": learning_result,
            "context_info": context_info,
            "original_question": message_session.question,
            "original_answer": message_session.answer[:500]  # Ограничиваем длину
        }
        
        # Добавляем информацию о целевом объекте
        if feedback_request.user_feedback.target:
            extra_data["target"] = {
                "doc_id": feedback_request.user_feedback.target.doc_id,
                "chunk_id": feedback_request.user_feedback.target.chunk_id
            }
        
        # Создаем событие
        feedback_event = FeedbackEvent(
            id=str(uuid.uuid4()),
            user_id=getattr(feedback_request, 'user_id', None),
            message_id=feedback_request.message_id,
            label=feedback_request.user_feedback.label,
            scope=feedback_request.user_feedback.scope,
            correction_text=(
                feedback_request.user_feedback.correction_text or 
                f"{feedback_request.user_feedback.label.value} feedback"
            ),
            status=status,
            extra_data=extra_data,
            created_at=datetime.utcnow()
        )
        
        db.add(feedback_event)
        db.flush()  # Получаем ID
        
        return feedback_event
    
    async def get_feedback_statistics(self, db: Session) -> Dict[str, Any]:
        """Получает статистику по фидбэку и обучению."""
        
        try:
            # Базовая статистика фидбэка
            total_feedback = db.query(FeedbackEvent).count()
            
            # Статистика по статусам
            status_stats = {}
            for status in UpdateStatus:
                count = db.query(FeedbackEvent).filter(
                    FeedbackEvent.status == status
                ).count()
                status_stats[status.value] = count
            
            # Статистика по типам фидбэка
            label_stats = {}
            for label in FeedbackLabel:
                count = db.query(FeedbackEvent).filter(
                    FeedbackEvent.label == label
                ).count()
                label_stats[label.value] = count
            
            # Статистика обучения из движка
            learning_stats = await self.learning_engine.get_learning_stats(db)
            
            # Возвращаем в формате, ожидаемом FeedbackStatsResponse
            return {
                "total_feedback_events": total_feedback,
                "feedback_by_label": label_stats,
                "pending_updates": status_stats.get("queued", 0),
                "applied_updates": status_stats.get("applied", 0),
                "failed_updates": status_stats.get("failed", 0),
                "reverted_updates": status_stats.get("reverted", 0),
                "spam_filtered": status_stats.get("spam_filtered", 0),
                "content_filtered": status_stats.get("content_filtered", 0),
                "avg_rating": 0.0,  # TODO: Вычислить средний рейтинг
                # Дополнительные данные для фронтенда
                "statistics": {
                    "feedback_statistics": {
                        "total_feedback": total_feedback,
                        "status_breakdown": status_stats,
                        "label_breakdown": label_stats
                    },
                    "learning_statistics": learning_stats,
                    "system_health": {
                        "application_rate": status_stats.get("applied", 0) / max(total_feedback, 1),
                        "filter_rate": (
                            status_stats.get("content_filtered", 0) + 
                            status_stats.get("spam_filtered", 0)
                        ) / max(total_feedback, 1),
                        "queue_rate": status_stats.get("queued", 0) / max(total_feedback, 1)
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting feedback statistics: {e}")
            return {
                "error": str(e),
                "feedback_statistics": {},
                "learning_statistics": {},
                "system_health": {}
            }
    
    async def review_queued_feedback(
        self,
        db: Session,
        feedback_id: str,
        reviewer_decision: str,
        reviewer_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Ручная проверка фидбэка, находящегося в очереди.
        
        Args:
            db: Сессия базы данных
            feedback_id: ID фидбэка для проверки
            reviewer_decision: Решение проверяющего (approve/reject)
            reviewer_id: ID проверяющего
            
        Returns:
            Результат проверки
        """
        try:
            # Находим фидбэк
            feedback_event = db.query(FeedbackEvent).filter(
                FeedbackEvent.id == feedback_id,
                FeedbackEvent.status == UpdateStatus.QUEUED
            ).first()
            
            if not feedback_event:
                return {
                    "status": "error",
                    "message": "Queued feedback not found"
                }
            
            if reviewer_decision.lower() == "approve":
                # Применяем фидбэк принудительно
                learning_result = await self.learning_engine.process_feedback_for_learning(
                    db=db,
                    feedback_content=feedback_event.correction_text,
                    original_chunk_id=feedback_event.extra_data.get("context_info", {}).get("chunk_id"),
                    user_id=feedback_event.user_id,
                    message_id=feedback_event.message_id,
                    feedback_label=feedback_event.label
                )
                
                # Обновляем статус
                feedback_event.status = UpdateStatus.APPLIED if learning_result["applied"] else UpdateStatus.FAILED
                feedback_event.extra_data["reviewer_decision"] = "approved"
                feedback_event.extra_data["reviewer_id"] = reviewer_id
                feedback_event.extra_data["review_date"] = datetime.utcnow().isoformat()
                
                db.commit()
                
                return {
                    "status": "approved",
                    "message": "Feedback approved and applied",
                    "learning_result": learning_result
                }
            
            else:
                # Отклоняем фидбэк
                feedback_event.status = UpdateStatus.FAILED
                feedback_event.extra_data["reviewer_decision"] = "rejected"
                feedback_event.extra_data["reviewer_id"] = reviewer_id
                feedback_event.extra_data["review_date"] = datetime.utcnow().isoformat()
                
                db.commit()
                
                return {
                    "status": "rejected",
                    "message": "Feedback rejected by reviewer"
                }
                
        except Exception as e:
            db.rollback()
            logger.error(f"Error in manual feedback review: {e}")
            return {
                "status": "error",
                "message": f"Review failed: {str(e)}"
            }