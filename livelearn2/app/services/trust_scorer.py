"""Trust scoring system for feedback validation."""

import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from loguru import logger
from sqlalchemy.orm import Session

from ..models.feedback import FeedbackEvent, FeedbackLabel, MessageSession
from ..models.documents import Chunk


@dataclass
class TrustScore:
    """Trust score result with details."""
    score: float  # 0.0 - 1.0
    confidence: float  # 0.0 - 1.0
    reasons: List[str]
    is_trusted: bool
    metadata: Dict[str, Any]


class TrustScorer:
    """Advanced trust scoring system for feedback validation."""
    
    def __init__(self):
        self.min_trust_score = 0.3  # Минимальный порог доверия (понижен с 0.6)
        self.high_trust_threshold = 0.7  # Высокий уровень доверия (понижен с 0.8)
        
        # Веса для различных факторов (более сбалансированные)
        self.weights = {
            "user_history": 0.15,      # История пользователя (понижен)
            "content_quality": 0.35,   # Качество контента (повышен)
            "consistency": 0.25,       # Консистентность с существующими данными (повышен)
            "source_reliability": 0.15, # Надежность источника
            "temporal_factors": 0.10   # Временные факторы
        }
        
        logger.info("Trust scorer initialized")
    
    async def calculate_trust_score(
        self,
        db: Session,
        feedback_content: str,
        user_id: Optional[str] = None,
        message_id: Optional[str] = None,
        feedback_label: Optional[FeedbackLabel] = None,
        original_chunk: Optional[Chunk] = None
    ) -> TrustScore:
        """
        Вычисляет оценку доверия для фидбэка.
        
        Args:
            db: Сессия базы данных
            feedback_content: Содержание фидбэка
            user_id: ID пользователя
            message_id: ID сообщения
            feedback_label: Тип фидбэка
            original_chunk: Оригинальный чанк для сравнения
            
        Returns:
            TrustScore с оценкой и деталями
        """
        try:
            logger.info(f"Calculating trust score for feedback: {feedback_content[:50]}...")
            
            scores = {}
            reasons = []
            metadata = {}
            
            # 1. Оценка истории пользователя
            user_score, user_reasons = await self._evaluate_user_history(db, user_id)
            scores["user_history"] = user_score
            reasons.extend(user_reasons)
            metadata["user_feedback_count"] = await self._get_user_feedback_count(db, user_id)
            
            # 2. Оценка качества контента
            content_score, content_reasons = await self._evaluate_content_quality(feedback_content)
            scores["content_quality"] = content_score
            reasons.extend(content_reasons)
            metadata["content_length"] = len(feedback_content)
            
            # 3. Оценка консистентности
            consistency_score, consistency_reasons = await self._evaluate_consistency(
                db, feedback_content, original_chunk
            )
            scores["consistency"] = consistency_score
            reasons.extend(consistency_reasons)
            
            # 4. Оценка надежности источника
            source_score, source_reasons = await self._evaluate_source_reliability(
                db, message_id, feedback_label
            )
            scores["source_reliability"] = source_score
            reasons.extend(source_reasons)
            
            # 5. Временные факторы
            temporal_score, temporal_reasons = await self._evaluate_temporal_factors(
                db, user_id, message_id
            )
            scores["temporal_factors"] = temporal_score
            reasons.extend(temporal_reasons)
            
            # Вычисляем итоговый score
            final_score = sum(
                scores[factor] * self.weights[factor] 
                for factor in scores
            )
            
            # Вычисляем confidence
            confidence = self._calculate_confidence(scores)
            
            # Определяем доверие (более мягкие критерии)
            is_trusted = (
                final_score >= self.min_trust_score and 
                confidence >= 0.3  # Понижен с 0.5
            )
            
            metadata.update({
                "individual_scores": scores,
                "weights_used": self.weights,
                "threshold": self.min_trust_score
            })
            
            trust_score = TrustScore(
                score=final_score,
                confidence=confidence,
                reasons=reasons,
                is_trusted=is_trusted,
                metadata=metadata
            )
            
            logger.info(
                f"Trust score calculated: {final_score:.3f} "
                f"(confidence: {confidence:.3f}, trusted: {is_trusted})"
            )
            
            return trust_score
            
        except Exception as e:
            logger.error(f"Error calculating trust score: {e}")
            # Возвращаем более щедрую оценку при ошибке (benefit of doubt)
            return TrustScore(
                score=0.5,  # Увеличено с 0.3
                confidence=0.4,  # Увеличено с 0.1
                reasons=[f"Error in trust calculation, giving benefit of doubt: {str(e)}"],
                is_trusted=True,  # Изменено с False
                metadata={"error": str(e), "fallback_applied": True}
            )
    
    async def _evaluate_user_history(
        self, 
        db: Session, 
        user_id: Optional[str]
    ) -> Tuple[float, List[str]]:
        """Оценивает историю пользователя."""
        if not user_id:
            return 0.6, ["Anonymous user - neutral score (improved)"]  # Повышено с 0.5
        
        try:
            # Получаем историю фидбэков пользователя
            feedback_history = db.query(FeedbackEvent).filter(
                FeedbackEvent.user_id == user_id
            ).order_by(FeedbackEvent.created_at.desc()).limit(50).all()
            
            if not feedback_history:
                return 0.6, ["New user - benefit of doubt given"]  # Повышено с 0.4
            
            total_feedback = len(feedback_history)
            reasons = [f"User has {total_feedback} previous feedback entries"]
            
            # Анализируем качество предыдущих фидбэков
            quality_scores = []
            spam_count = 0
            helpful_count = 0
            
            for feedback in feedback_history:
                if feedback.status == "spam_filtered":
                    spam_count += 1
                elif feedback.label in [FeedbackLabel.HELPFUL, FeedbackLabel.CORRECT]:
                    helpful_count += 1
            
            # Базовый score на основе активности (более щедрые оценки)
            if total_feedback >= 15:  # Было 20
                base_score = 0.85  # Увеличено с 0.8
                reasons.append("Experienced user (15+ feedbacks)")
            elif total_feedback >= 8:  # Было 10
                base_score = 0.75  # Увеличено с 0.7
                reasons.append("Regular user (8+ feedbacks)")
            elif total_feedback >= 3:  # Было 5
                base_score = 0.7   # Увеличено с 0.6
                reasons.append("Active user (3+ feedbacks)")
            elif total_feedback >= 1:
                base_score = 0.65  # Новая категория
                reasons.append("User with some feedback experience")
            else:
                base_score = 0.6   # Увеличено с 0.5
                reasons.append("New user (benefit of doubt)")
            
            # Корректировка на основе качества (менее строгие штрафы)
            spam_ratio = spam_count / max(total_feedback, 1)
            helpful_ratio = helpful_count / max(total_feedback, 1)
            
            if spam_ratio > 0.5:  # Было 0.3
                base_score -= 0.3  # Уменьшено с 0.4
                reasons.append(f"High spam ratio: {spam_ratio:.2f}")
            elif spam_ratio > 0.2:  # Было 0.1
                base_score -= 0.1  # Уменьшено с 0.2
                reasons.append(f"Some spam detected: {spam_ratio:.2f}")
            
            if helpful_ratio > 0.6:  # Было 0.7
                base_score += 0.2
                reasons.append(f"High helpful ratio: {helpful_ratio:.2f}")
            elif helpful_ratio > 0.3:  # Было 0.4
                base_score += 0.15  # Увеличено с 0.1
                reasons.append(f"Good helpful ratio: {helpful_ratio:.2f}")
            elif helpful_ratio > 0.1:  # Новая категория
                base_score += 0.05
                reasons.append(f"Some helpful feedback: {helpful_ratio:.2f}")
            
            final_score = max(0.0, min(1.0, base_score))
            return final_score, reasons
            
        except Exception as e:
            logger.error(f"Error evaluating user history: {e}")
            return 0.6, [f"Error in user evaluation, assuming good intent: {str(e)}"]  # Увеличено с 0.3
    
    async def _evaluate_content_quality(
        self, 
        content: str
    ) -> Tuple[float, List[str]]:
        """Оценивает качество контента фидбэка."""
        reasons = []
        score = 0.6  # Более высокий базовый score (было 0.5)
        
        # Проверка длины (более мягкие критерии)
        length = len(content.strip())
        if length < 5:
            score -= 0.2  # Уменьшено с 0.3
            reasons.append("Very short feedback")
        elif length < 15:  # Было 30
            score -= 0.05  # Уменьшено с 0.1
            reasons.append("Short but acceptable feedback")
        elif length > 500:
            score += 0.15  # Увеличено с 0.1
            reasons.append("Detailed feedback")
        elif length > 50:  # Было 100
            score += 0.1   # Увеличено с 0.05
            reasons.append("Good length feedback")
        elif length >= 15:
            score += 0.05
            reasons.append("Adequate length")
        
        # Проверка на спам паттерны (более мягкие критерии)
        spam_patterns = [
            r"(.)\1{8,}",  # Повторяющиеся символы (было 4+, стало 8+)
            r"\b(test|тест)\b.*\b(test|тест)\b.*\b(test|тест)\b",  # Только 3+ повторения тестовых слов
            r"[!]{5,}|[?]{5,}",  # Множественная пунктуация (было 3+, стало 5+)
            r"^\s*[a-z]\s*$",  # Одна буква
        ]
        
        spam_detected = False
        for pattern in spam_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                score -= 0.2  # Уменьшено с 0.4
                reasons.append("Minor spam pattern detected")
                spam_detected = True
                break
        
        # Дополнительная проверка на очень короткий контент с повторами
        if not spam_detected and length < 10:
            words = content.strip().split()
            if len(set(words)) < len(words) / 2 and len(words) > 1:
                score -= 0.1
                reasons.append("Repetitive short content")
        
        # Проверка на конструктивность
        constructive_indicators = [
            r"потому что|because|так как",
            r"должно быть|should be|правильно",
            r"ошибка в том|error is|неправильно",
            r"лучше было бы|better would be|предлагаю",
        ]
        
        constructive_count = sum(
            1 for pattern in constructive_indicators 
            if re.search(pattern, content, re.IGNORECASE)
        )
        
        if constructive_count >= 2:
            score += 0.2
            reasons.append("Highly constructive feedback")
        elif constructive_count >= 1:
            score += 0.1
            reasons.append("Constructive feedback")
        
        # Проверка языка и грамматики (базовая)
        if re.search(r"[а-яё]", content, re.IGNORECASE):
            # Русский текст
            if len(re.findall(r"\b[а-яё]+\b", content, re.IGNORECASE)) >= 3:
                score += 0.05
                reasons.append("Proper Russian text")
        elif re.search(r"[a-z]", content, re.IGNORECASE):
            # Английский текст
            if len(re.findall(r"\b[a-z]+\b", content, re.IGNORECASE)) >= 3:
                score += 0.05
                reasons.append("Proper English text")
        
        final_score = max(0.0, min(1.0, score))
        return final_score, reasons
    
    async def _evaluate_consistency(
        self,
        db: Session,
        feedback_content: str,
        original_chunk: Optional[Chunk]
    ) -> Tuple[float, List[str]]:
        """Оценивает консистентность с существующими данными."""
        reasons = []
        score = 0.65  # Более высокий базовый score (было 0.5)
        
        if not original_chunk:
            return 0.7, ["No original chunk for comparison - assuming good intent"]  # Более щедрая оценка
        
        try:
            # Сравнение длины (более мягкие критерии)
            original_length = len(original_chunk.content)
            feedback_length = len(feedback_content)
            
            length_ratio = feedback_length / max(original_length, 1)
            
            if 0.1 <= length_ratio <= 5.0:  # Расширенный диапазон (было 0.3-3.0)
                score += 0.15  # Увеличено с 0.1
                reasons.append("Reasonable length compared to original")
            elif length_ratio < 0.05:  # Было 0.1
                score -= 0.1  # Уменьшено с 0.2
                reasons.append("Much shorter than original")
            elif length_ratio > 15:  # Было 10
                score -= 0.1  # Уменьшено с 0.2
                reasons.append("Much longer than original")
            
            # Проверка на схожесть тематики (улучшенная)
            original_words = set(re.findall(r"\b\w{2,}\b", original_chunk.content.lower()))  # Минимум 2 символа
            feedback_words = set(re.findall(r"\b\w{2,}\b", feedback_content.lower()))
            
            if original_words and feedback_words:
                overlap = len(original_words & feedback_words)
                overlap_ratio = overlap / max(len(original_words), 1)
                
                # Более детальная оценка перекрытия тем
                if overlap_ratio > 0.2:  # Было 0.3
                    score += 0.2  # Увеличено с 0.15
                    reasons.append(f"Good topical overlap: {overlap_ratio:.2f}")
                elif overlap_ratio > 0.05:  # Было 0.1
                    score += 0.1  # Увеличено с 0.05
                    reasons.append(f"Some topical overlap: {overlap_ratio:.2f}")
                elif overlap_ratio > 0:
                    score += 0.05  # Новая категория
                    reasons.append(f"Minimal topical overlap: {overlap_ratio:.2f}")
                else:
                    score -= 0.05  # Уменьшено с 0.1
                    reasons.append("No direct topical overlap - may be contextual")
            
            # Проверка на противоречия с другими чанками
            similar_chunks = db.query(Chunk).filter(
                Chunk.document_id == original_chunk.document_id
            ).limit(5).all()
            
            if similar_chunks:
                contradiction_score = await self._check_contradictions(
                    feedback_content, [chunk.content for chunk in similar_chunks]
                )
                score += contradiction_score * 0.2
                if contradiction_score > 0:
                    reasons.append("No major contradictions detected")
                elif contradiction_score < -0.5:
                    reasons.append("Potential contradictions detected")
            
            final_score = max(0.0, min(1.0, score))
            return final_score, reasons
            
        except Exception as e:
            logger.error(f"Error in consistency evaluation: {e}")
            return 0.6, [f"Error in consistency check, assuming reasonable feedback: {str(e)}"]  # Увеличено с 0.3
    
    async def _check_contradictions(
        self, 
        feedback: str, 
        existing_contents: List[str]
    ) -> float:
        """Проверяет на противоречия с существующим контентом."""
        # Упрощенная проверка на противоречия
        feedback_lower = feedback.lower()
        
        negative_indicators = ["не", "нет", "неправильно", "ошибка", "not", "no", "wrong", "error"]
        positive_indicators = ["да", "правильно", "верно", "точно", "yes", "correct", "right", "accurate"]
        
        feedback_negative = any(ind in feedback_lower for ind in negative_indicators)
        feedback_positive = any(ind in feedback_lower for ind in positive_indicators)
        
        contradiction_count = 0
        agreement_count = 0
        
        for content in existing_contents:
            content_lower = content.lower()
            content_negative = any(ind in content_lower for ind in negative_indicators)
            content_positive = any(ind in content_lower for ind in positive_indicators)
            
            # Простая логика противоречий
            if (feedback_negative and content_positive) or (feedback_positive and content_negative):
                contradiction_count += 1
            elif (feedback_negative and content_negative) or (feedback_positive and content_positive):
                agreement_count += 1
        
        if contradiction_count > agreement_count:
            return -0.5  # Много противоречий
        elif agreement_count > contradiction_count:
            return 0.3   # Согласованность
        else:
            return 0.0   # Нейтрально
    
    async def _evaluate_source_reliability(
        self,
        db: Session,
        message_id: Optional[str],
        feedback_label: Optional[FeedbackLabel]
    ) -> Tuple[float, List[str]]:
        """Оценивает надежность источника фидбэка."""
        reasons = []
        score = 0.5  # Базовый score
        
        # Оценка на основе типа фидбэка
        if feedback_label:
            if feedback_label in [FeedbackLabel.CORRECT, FeedbackLabel.HELPFUL]:
                score += 0.2
                reasons.append("Positive feedback type")
            elif feedback_label in [FeedbackLabel.INCORRECT, FeedbackLabel.NOT_HELPFUL]:
                score += 0.1
                reasons.append("Critical feedback type")
            elif feedback_label in [FeedbackLabel.LIKE, FeedbackLabel.DISLIKE]:
                score -= 0.1
                reasons.append("Simple like/dislike feedback")
        
        # Проверка контекста сообщения
        if message_id:
            try:
                message = db.query(MessageSession).filter(
                    MessageSession.id == message_id
                ).first()
                
                if message:
                    # Анализируем сложность исходного вопроса
                    question_length = len(message.question)
                    if question_length > 100:
                        score += 0.1
                        reasons.append("Complex question context")
                    elif question_length < 20:
                        score -= 0.05
                        reasons.append("Simple question context")
                    
                    # Анализируем качество исходного ответа
                    answer_length = len(message.answer)
                    if answer_length > 200:
                        score += 0.05
                        reasons.append("Detailed original answer")
                    
            except Exception as e:
                logger.error(f"Error checking message context: {e}")
                score -= 0.1
                reasons.append("Could not verify message context")
        
        final_score = max(0.0, min(1.0, score))
        return final_score, reasons
    
    async def _evaluate_temporal_factors(
        self,
        db: Session,
        user_id: Optional[str],
        message_id: Optional[str]
    ) -> Tuple[float, List[str]]:
        """Оценивает временные факторы."""
        reasons = []
        score = 0.7  # Более высокий базовый score (было 0.5)
        
        try:
            now = datetime.utcnow()
            
            # Проверяем время отклика на фидбэк (более мягкие критерии)
            if message_id:
                message = db.query(MessageSession).filter(
                    MessageSession.id == message_id
                ).first()
                
                if message and message.created_at:
                    time_diff = now - message.created_at
                    
                    if time_diff < timedelta(minutes=1):  # Было 5 минут
                        score -= 0.05  # Уменьшено с 0.1
                        reasons.append("Very quick feedback (potentially automated)")
                    elif time_diff < timedelta(hours=2):  # Расширено с 1 часа
                        score += 0.15  # Увеличено с 0.1
                        reasons.append("Timely feedback")
                    elif time_diff < timedelta(days=3):  # Расширено с 1 дня
                        score += 0.1   # Увеличено с 0.05
                        reasons.append("Reasonable timing")
                    elif time_diff < timedelta(days=7):  # Новая категория
                        score += 0.05
                        reasons.append("Delayed but acceptable feedback")
                    else:
                        score -= 0.02  # Уменьшено с 0.05
                        reasons.append("Very delayed feedback")
            
            # Проверяем частоту фидбэков пользователя (более толерантно)
            if user_id:
                recent_feedback = db.query(FeedbackEvent).filter(
                    FeedbackEvent.user_id == user_id,
                    FeedbackEvent.created_at >= now - timedelta(hours=1)
                ).count()
                
                if recent_feedback > 20:  # Было 10
                    score -= 0.2  # Уменьшено с 0.3
                    reasons.append("Very frequent feedback (potential spam)")
                elif recent_feedback > 10:  # Было 5
                    score -= 0.05  # Уменьшено с 0.1
                    reasons.append("High frequency feedback")
                elif recent_feedback >= 1:
                    score += 0.1  # Увеличено с 0.05
                    reasons.append("Active user providing feedback")
                else:
                    score += 0.05  # Новая категория
                    reasons.append("First feedback in recent period")
            
            final_score = max(0.0, min(1.0, score))
            return final_score, reasons
            
        except Exception as e:
            logger.error(f"Error in temporal evaluation: {e}")
            return 0.7, [f"Error in temporal analysis, assuming normal timing: {str(e)}"]  # Увеличено с 0.4
    
    def _calculate_confidence(self, scores: Dict[str, float]) -> float:
        """Вычисляет уверенность в оценке."""
        if not scores:
            return 0.0
        
        # Confidence основан на согласованности оценок
        score_values = list(scores.values())
        mean_score = sum(score_values) / len(score_values)
        
        # Вычисляем разброс
        variance = sum((score - mean_score) ** 2 for score in score_values) / len(score_values)
        std_dev = variance ** 0.5
        
        # Чем меньше разброс, тем выше confidence
        confidence = max(0.0, min(1.0, 1.0 - std_dev))
        
        # Бонус за количество факторов
        factor_bonus = min(0.2, len(scores) * 0.05)
        
        return min(1.0, confidence + factor_bonus)
    
    async def _get_user_feedback_count(
        self, 
        db: Session, 
        user_id: Optional[str]
    ) -> int:
        """Получает количество фидбэков пользователя."""
        if not user_id:
            return 0
        
        try:
            return db.query(FeedbackEvent).filter(
                FeedbackEvent.user_id == user_id
            ).count()
        except Exception:
            return 0
