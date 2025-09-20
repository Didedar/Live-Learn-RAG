"""Learning engine for processing validated feedback and updating knowledge base."""

import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

from loguru import logger
from sqlalchemy.orm import Session

from ..config import settings
from ..models.documents import Chunk, Document
from ..models.feedback import (
    FeedbackEvent, FeedbackLabel, FeedbackScope, IndexMutation, UpdateStatus
)
from ..utils.text_processing import chunk_text
from .mock_embeddings import MockEmbeddings
from .trust_scorer import TrustScorer, TrustScore
from .content_filter import ContentFilter, FilterDecision, FilterResult


class LearningEngine:
    """Advanced learning engine that processes validated feedback."""
    
    def __init__(self):
        self.embeddings = MockEmbeddings()
        self.trust_scorer = TrustScorer()
        self.content_filter = ContentFilter()
        
        # Настройки обучения (более мягкие пороги)
        self.min_trust_score = 0.3  # Понижено с 0.6
        self.auto_apply_threshold = 0.7  # Понижено с 0.8
        self.review_threshold = 0.2  # Понижено с 0.5
        
        logger.info("Learning engine initialized")
    
    async def process_feedback_for_learning(
        self,
        db: Session,
        feedback_content: str,
        original_chunk_id: int,
        user_id: Optional[str] = None,
        message_id: Optional[str] = None,
        feedback_label: Optional[FeedbackLabel] = None
    ) -> Dict[str, Any]:
        """
        Обрабатывает фидбэк для обучения системы.
        
        Этапы:
        1. Фильтрация контента
        2. Оценка доверия
        3. Принятие решения об обновлении
        4. Применение изменений к векторному хранилищу
        
        Args:
            db: Сессия базы данных
            feedback_content: Содержание фидбэка
            original_chunk_id: ID оригинального чанка
            user_id: ID пользователя
            message_id: ID сообщения
            feedback_label: Тип фидбэка
            
        Returns:
            Результат обработки с деталями
        """
        try:
            logger.info(f"Processing feedback for learning: {feedback_content[:50]}...")
            
            # Получаем оригинальный чанк
            original_chunk = db.query(Chunk).filter(
                Chunk.id == original_chunk_id
            ).first()
            
            if not original_chunk:
                return {
                    "status": "error",
                    "message": "Original chunk not found",
                    "applied": False
                }
            
            # Этап 1: Фильтрация контента
            logger.info("Step 1: Content filtering")
            filter_decision = await self.content_filter.filter_content(
                feedback_content, user_id
            )
            
            if not filter_decision.is_safe:
                await self._log_feedback_event(
                    db, feedback_content, user_id, message_id,
                    UpdateStatus.CONTENT_FILTERED,
                    {"filter_decision": filter_decision.__dict__},
                    feedback_label
                )
                
                return {
                    "status": "rejected",
                    "message": f"Content filtered: {filter_decision.result.value}",
                    "reasons": filter_decision.reasons,
                    "applied": False,
                    "filter_decision": filter_decision.__dict__
                }
            
            # Этап 2: Оценка доверия
            logger.info("Step 2: Trust scoring")
            trust_score = await self.trust_scorer.calculate_trust_score(
                db=db,
                feedback_content=feedback_content,
                user_id=user_id,
                message_id=message_id,
                feedback_label=feedback_label,
                original_chunk=original_chunk
            )
            
            if not trust_score.is_trusted:
                await self._log_feedback_event(
                    db, feedback_content, user_id, message_id,
                    UpdateStatus.FAILED,
                    {
                        "trust_score": trust_score.__dict__,
                        "filter_decision": filter_decision.__dict__
                    },
                    feedback_label
                )
                
                return {
                    "status": "rejected",
                    "message": "Trust score too low",
                    "trust_score": trust_score.score,
                    "reasons": trust_score.reasons,
                    "applied": False
                }
            
            # Этап 3: Принятие решения об обновлении
            logger.info("Step 3: Decision making")
            update_decision = await self._make_update_decision(
                trust_score, filter_decision, feedback_label
            )
            
            # Этап 4: Применение изменений
            logger.info("Step 4: Applying changes")
            if update_decision["should_apply"]:
                application_result = await self._apply_feedback_to_knowledge_base(
                    db=db,
                    feedback_content=feedback_content,
                    original_chunk=original_chunk,
                    trust_score=trust_score,
                    feedback_label=feedback_label,
                    update_strategy=update_decision["strategy"]
                )
                
                # Логируем успешное применение
                await self._log_feedback_event(
                    db, feedback_content, user_id, message_id,
                    UpdateStatus.APPLIED,
                    {
                        "trust_score": trust_score.__dict__,
                        "filter_decision": filter_decision.__dict__,
                        "application_result": application_result
                    },
                    feedback_label
                )
                
                return {
                    "status": "applied",
                    "message": "Feedback successfully applied to knowledge base",
                    "trust_score": trust_score.score,
                    "confidence": trust_score.confidence,
                    "strategy": update_decision["strategy"],
                    "applied": True,
                    "changes": application_result
                }
            
            else:
                # Помещаем в очередь на ручную проверку
                await self._log_feedback_event(
                    db, feedback_content, user_id, message_id,
                    UpdateStatus.QUEUED,
                    {
                        "trust_score": trust_score.__dict__,
                        "filter_decision": filter_decision.__dict__,
                        "reason": update_decision["reason"]
                    },
                    feedback_label
                )
                
                return {
                    "status": "queued",
                    "message": "Feedback queued for manual review",
                    "trust_score": trust_score.score,
                    "confidence": trust_score.confidence,
                    "reason": update_decision["reason"],
                    "applied": False
                }
                
        except Exception as e:
            logger.error(f"Error in learning engine: {e}")
            
            # НЕ логируем здесь - будет rollback, и логирование тоже откатится
            
            return {
                "status": "error",
                "message": f"Learning engine error: {str(e)}",
                "applied": False
            }
    
    async def _make_update_decision(
        self,
        trust_score: TrustScore,
        filter_decision: FilterDecision,
        feedback_label: Optional[FeedbackLabel]
    ) -> Dict[str, Any]:
        """Принимает решение об обновлении знаний."""
        
        # Автоматическое применение для высоких оценок
        if (trust_score.score >= self.auto_apply_threshold and
            trust_score.confidence >= 0.7 and
            filter_decision.confidence >= 0.8):
            
            return {
                "should_apply": True,
                "strategy": "direct_update",
                "reason": "High trust and quality scores"
            }
        
        # Применение с осторожностью для средних оценок
        elif (trust_score.score >= self.min_trust_score and
              trust_score.confidence >= 0.5):
            
            # Определяем стратегию на основе типа фидбэка
            if feedback_label == FeedbackLabel.CORRECT:
                strategy = "enhancement"
            elif feedback_label == FeedbackLabel.INCORRECT:
                strategy = "correction"
            else:
                strategy = "supplemental"
            
            return {
                "should_apply": True,
                "strategy": strategy,
                "reason": "Sufficient trust score for careful application"
            }
        
        # Отправка на ручную проверку
        else:
            return {
                "should_apply": False,
                "strategy": None,
                "reason": f"Trust score {trust_score.score:.3f} below threshold {self.min_trust_score}"
            }
    
    async def _apply_feedback_to_knowledge_base(
        self,
        db: Session,
        feedback_content: str,
        original_chunk: Chunk,
        trust_score: TrustScore,
        feedback_label: Optional[FeedbackLabel],
        update_strategy: str
    ) -> Dict[str, Any]:
        """Применяет фидбэк к векторному хранилищу."""
        
        changes = {
            "chunks_created": 0,
            "chunks_updated": 0,
            "chunks_deprecated": 0,
            "new_chunk_ids": []
        }
        
        try:
            if update_strategy == "direct_update":
                # Прямое обновление существующего чанка
                await self._update_existing_chunk(
                    db, original_chunk, feedback_content, trust_score
                )
                changes["chunks_updated"] = 1
                
            elif update_strategy == "enhancement":
                # Создание дополнительного чанка с улучшенной информацией
                new_chunk = await self._create_enhanced_chunk(
                    db, original_chunk, feedback_content, trust_score
                )
                changes["chunks_created"] = 1
                changes["new_chunk_ids"].append(new_chunk.id)
                
            elif update_strategy == "correction":
                # Исправление с созданием нового чанка и депрекацией старого
                new_chunk = await self._create_corrected_chunk(
                    db, original_chunk, feedback_content, trust_score
                )
                await self._deprecate_chunk(db, original_chunk, trust_score)
                
                # Добавляем boost weight для нового чанка из фидбэка
                await self._add_feedback_boost(db, new_chunk.id, trust_score.score)
                
                changes["chunks_created"] = 1
                changes["chunks_deprecated"] = 1
                changes["new_chunk_ids"].append(new_chunk.id)
                
            elif update_strategy == "supplemental":
                # Создание дополнительного чанка без изменения оригинала
                new_chunk = await self._create_supplemental_chunk(
                    db, original_chunk, feedback_content, trust_score
                )
                
                # Добавляем boost weight для нового чанка из фидбэка
                await self._add_feedback_boost(db, new_chunk.id, trust_score.score)
                
                changes["chunks_created"] = 1
                changes["new_chunk_ids"].append(new_chunk.id)
            
            # НЕ делаем commit здесь - это сделает вызывающий метод
            logger.info(f"Applied feedback using strategy: {update_strategy}")
            
            return changes
            
        except Exception as e:
            logger.error(f"Error applying feedback to knowledge base: {e}")
            raise
    
    async def _update_existing_chunk(
        self,
        db: Session,
        original_chunk: Chunk,
        feedback_content: str,
        trust_score: TrustScore
    ):
        """Обновляет существующий чанк."""
        
        # Создаем улучшенную версию контента
        improved_content = await self._merge_content(
            original_chunk.content, feedback_content, trust_score.score
        )
        
        # Генерируем новое embedding
        new_embedding = await self.embeddings.embed_documents([improved_content])
        
        # Обновляем чанк
        original_chunk.content = improved_content
        original_chunk.embedding = new_embedding[0]
        original_chunk.version += 1
        
        # Обновляем версию и источник (метаданные храним в других полях)
        original_chunk.source = "user_feedback_updated"
        
        # Логируем метаданные обновления
        logger.info(f"Updated chunk {original_chunk.id} with feedback (trust_score: {trust_score.score})")
    
    async def _create_enhanced_chunk(
        self,
        db: Session,
        original_chunk: Chunk,
        feedback_content: str,
        trust_score: TrustScore
    ) -> Chunk:
        """Создает улучшенный чанк на основе фидбэка."""
        
        # Создаем расширенный контент
        enhanced_content = f"{original_chunk.content}\n\nДополнительная информация:\n{feedback_content}"
        
        # Генерируем embedding
        embedding = await self.embeddings.embed_documents([enhanced_content])
        
        # Создаем новый чанк
        new_chunk = Chunk(
            document_id=original_chunk.document_id,
            ordinal=original_chunk.ordinal + 0.1,  # Небольшое смещение
            content=enhanced_content,
            embedding=embedding[0],
            source="user_feedback_enhancement",
            version=1,
            metadata={
                "created_from_feedback": True,
                "original_chunk_id": original_chunk.id,
                "trust_score": trust_score.score,
                "enhancement_type": "user_feedback"
            }
        )
        
        db.add(new_chunk)
        db.flush()  # Получаем ID
        
        return new_chunk
    
    async def _create_corrected_chunk(
        self,
        db: Session,
        original_chunk: Chunk,
        feedback_content: str,
        trust_score: TrustScore
    ) -> Chunk:
        """Создает исправленный чанк."""
        
        # Используем фидбэк как исправленную версию
        corrected_content = feedback_content
        
        # Генерируем embedding
        embedding = await self.embeddings.embed_documents([corrected_content])
        
        # Создаем новый чанк
        new_chunk = Chunk(
            document_id=original_chunk.document_id,
            ordinal=original_chunk.ordinal,  # Заменяем оригинальный
            content=corrected_content,
            embedding=embedding[0],
            source="user_feedback_correction",
            version=1,
            metadata={
                "created_from_feedback": True,
                "original_chunk_id": original_chunk.id,
                "trust_score": trust_score.score,
                "correction_type": "user_feedback",
                "replaces_chunk": original_chunk.id
            }
        )
        
        db.add(new_chunk)
        db.flush()
        
        return new_chunk
    
    async def _create_supplemental_chunk(
        self,
        db: Session,
        original_chunk: Chunk,
        feedback_content: str,
        trust_score: TrustScore
    ) -> Chunk:
        """Создает дополнительный чанк."""
        
        # Генерируем embedding для фидбэка
        embedding = await self.embeddings.embed_documents([feedback_content])
        
        # Создаем дополнительный чанк
        new_chunk = Chunk(
            document_id=original_chunk.document_id,
            ordinal=original_chunk.ordinal + 0.5,
            content=feedback_content,
            embedding=embedding[0],
            source="user_feedback_supplement",
            version=1,
            metadata={
                "created_from_feedback": True,
                "related_chunk_id": original_chunk.id,
                "trust_score": trust_score.score,
                "supplement_type": "user_feedback"
            }
        )
        
        db.add(new_chunk)
        db.flush()
        
        return new_chunk
    
    async def _deprecate_chunk(
        self,
        db: Session,
        chunk: Chunk,
        trust_score: TrustScore
    ):
        """Помечает чанк как устаревший."""
        
        # Помечаем чанк как устаревший через source поле
        chunk.source = "deprecated_by_feedback"
        
        # Логируем информацию об устаревании
        logger.info(f"Deprecated chunk {chunk.id} due to user feedback (trust_score: {trust_score.score})")
    
    async def _add_feedback_boost(
        self,
        db: Session,
        chunk_id: int,
        trust_score: float
    ):
        """Добавляет boost weight для чанка из фидбэка."""
        from ..models.feedback import ChunkWeight
        
        # Вычисляем boost на основе trust score
        boost_weight = min(0.5, trust_score * 0.8)  # Максимум 0.5, пропорционально trust score
        
        # Создаем или обновляем ChunkWeight
        chunk_weight = db.query(ChunkWeight).filter(
            ChunkWeight.chunk_id == chunk_id
        ).first()
        
        if not chunk_weight:
            chunk_weight = ChunkWeight(
                chunk_id=chunk_id,
                penalty_weight=0.0,
                boost_weight=boost_weight,
                feedback_count=1,
                is_deprecated=False
            )
            db.add(chunk_weight)
        else:
            chunk_weight.boost_weight += boost_weight
            chunk_weight.feedback_count += 1
        
        db.flush()
        logger.info(f"Added boost weight {boost_weight:.3f} to chunk {chunk_id} (trust_score: {trust_score:.3f})")
    
    async def _merge_content(
        self,
        original_content: str,
        feedback_content: str,
        trust_score: float
    ) -> str:
        """Объединяет оригинальный контент с фидбэком."""
        
        # Простая стратегия объединения
        if trust_score >= 0.9:
            # Высокое доверие - заменяем полностью
            return feedback_content
        elif trust_score >= 0.7:
            # Среднее доверие - добавляем как дополнение
            return f"{original_content}\n\nОбновленная информация: {feedback_content}"
        else:
            # Низкое доверие - добавляем как примечание
            return f"{original_content}\n\nПримечание пользователя: {feedback_content}"
    
    async def _log_feedback_event(
        self,
        db: Session,
        feedback_content: str,
        user_id: Optional[str],
        message_id: Optional[str],
        status: UpdateStatus,
        metadata: Dict[str, Any],
        feedback_label: Optional[FeedbackLabel] = None
    ):
        """Логирует событие обработки фидбэка."""
        
        try:
            event = FeedbackEvent(
                id=str(uuid.uuid4()),
                user_id=user_id,
                message_id=message_id,
                label=feedback_label or FeedbackLabel.HELPFUL,  # Значение по умолчанию
                scope=FeedbackScope.CHUNK,  # Значение по умолчанию
                correction_text=feedback_content[:1000],  # Ограничиваем длину
                status=status,
                extra_data=metadata,
                created_at=datetime.utcnow()
            )
            
            db.add(event)
            db.flush()  # Получаем ID, но не коммитим
            
        except Exception as e:
            logger.error(f"Error logging feedback event: {e}")
            # НЕ делаем rollback здесь - это ответственность вызывающего кода
    
    async def get_learning_stats(self, db: Session) -> Dict[str, Any]:
        """Получает статистику обучения системы."""
        
        try:
            # Статистика по фидбэкам
            total_feedback = db.query(FeedbackEvent).count()
            applied_feedback = db.query(FeedbackEvent).filter(
                FeedbackEvent.status == UpdateStatus.APPLIED
            ).count()
            
            queued_feedback = db.query(FeedbackEvent).filter(
                FeedbackEvent.status == UpdateStatus.QUEUED
            ).count()
            
            filtered_feedback = db.query(FeedbackEvent).filter(
                FeedbackEvent.status.in_([
                    UpdateStatus.CONTENT_FILTERED,
                    UpdateStatus.SPAM_FILTERED
                ])
            ).count()
            
            # Статистика по чанкам
            feedback_chunks = db.query(Chunk).filter(
                Chunk.source.like('%feedback%')
            ).count()
            
            deprecated_chunks = db.query(Chunk).filter(
                Chunk.source == 'deprecated'
            ).count()
            
            return {
                "total_feedback_events": total_feedback,
                "applied_feedback": applied_feedback,
                "queued_feedback": queued_feedback,
                "filtered_feedback": filtered_feedback,
                "feedback_chunks_created": feedback_chunks,
                "deprecated_chunks": deprecated_chunks,
                "application_rate": applied_feedback / max(total_feedback, 1),
                "filter_rate": filtered_feedback / max(total_feedback, 1)
            }
            
        except Exception as e:
            logger.error(f"Error getting learning stats: {e}")
            return {
                "error": str(e),
                "total_feedback_events": 0,
                "applied_feedback": 0
            }
