"""Advanced content filtering system for feedback validation."""

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from loguru import logger


class FilterResult(str, Enum):
    """Content filter results."""
    APPROVED = "approved"
    REJECTED = "rejected" 
    NEEDS_REVIEW = "needs_review"
    SPAM = "spam"
    TOXIC = "toxic"
    LOW_QUALITY = "low_quality"


@dataclass
class FilterDecision:
    """Content filter decision with details."""
    result: FilterResult
    confidence: float  # 0.0 - 1.0
    reasons: List[str]
    metadata: Dict[str, any]
    is_safe: bool


class ContentFilter:
    """Advanced content filtering system."""
    
    def __init__(self):
        self.spam_patterns = [
            r"(.)\1{5,}",  # Повторяющиеся символы
            r"^\s*(test|тест)\s*$",  # Тестовые сообщения
            r"^\s*[a-z]\s*$",  # Одна буква
            r"[!]{4,}|[?]{4,}",  # Множественная пунктуация
            r"\b(\w+)\s+\1\s+\1\b",  # Повторяющиеся слова
        ]
        
        self.toxic_patterns = [
            r"\b(дурак|идиот|тупой)\b",
            r"\b(stupid|idiot|dumb)\b",
            r"\b(hate|ненавижу)\b",
        ]
        
        logger.info("Content filter initialized")
    
    async def filter_content(
        self,
        content: str,
        user_id: Optional[str] = None,
        context: Optional[Dict] = None
    ) -> FilterDecision:
        """Фильтрует контент и принимает решение о его качестве."""
        try:
            if not content or not content.strip():
                return FilterDecision(
                    result=FilterResult.REJECTED,
                    confidence=1.0,
                    reasons=["Empty content"],
                    metadata={"length": 0},
                    is_safe=False
                )
            
            content_clean = content.strip()
            reasons = []
            metadata = {"length": len(content_clean)}
            
            # Проверка на спам
            spam_score = self._check_spam(content_clean)
            if spam_score > 0.7:
                return FilterDecision(
                    result=FilterResult.SPAM,
                    confidence=spam_score,
                    reasons=["Spam patterns detected"],
                    metadata=metadata,
                    is_safe=False
                )
            
            # Проверка на токсичность
            toxic_score = self._check_toxicity(content_clean)
            if toxic_score > 0.6:
                return FilterDecision(
                    result=FilterResult.TOXIC,
                    confidence=toxic_score,
                    reasons=["Toxic content detected"],
                    metadata=metadata,
                    is_safe=False
                )
            
            # Проверка качества
            quality_score = self._check_quality(content_clean)
            if quality_score < 0.3:
                return FilterDecision(
                    result=FilterResult.LOW_QUALITY,
                    confidence=0.8,
                    reasons=["Low quality content"],
                    metadata=metadata,
                    is_safe=True
                )
            
            # Финальное решение
            if quality_score >= 0.7:
                result = FilterResult.APPROVED
                confidence = 0.9
                reasons = ["High quality content"]
            elif quality_score >= 0.5:
                result = FilterResult.APPROVED
                confidence = 0.7
                reasons = ["Good quality content"]
            else:
                result = FilterResult.NEEDS_REVIEW
                confidence = 0.5
                reasons = ["Content needs review"]
            
            return FilterDecision(
                result=result,
                confidence=confidence,
                reasons=reasons,
                metadata=metadata,
                is_safe=True
            )
            
        except Exception as e:
            logger.error(f"Error in content filtering: {e}")
            return FilterDecision(
                result=FilterResult.NEEDS_REVIEW,
                confidence=0.1,
                reasons=[f"Filter error: {str(e)}"],
                metadata={"error": str(e)},
                is_safe=False
            )
    
    def _check_spam(self, content: str) -> float:
        """Проверяет контент на спам."""
        score = 0.0
        for pattern in self.spam_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                score += 0.3
        return min(1.0, score)
    
    def _check_toxicity(self, content: str) -> float:
        """Проверяет контент на токсичность."""
        score = 0.0
        for pattern in self.toxic_patterns:
            if re.search(pattern, content.lower()):
                score += 0.4
        return min(1.0, score)
    
    def _check_quality(self, content: str) -> float:
        """Проверяет качество контента."""
        score = 0.5  # Базовый score
        
        # Длина
        if 20 <= len(content) <= 500:
            score += 0.2
        elif len(content) < 10:
            score -= 0.3
        
        # Конструктивные слова
        constructive = ["потому что", "предлагаю", "лучше", "исправить"]
        if any(word in content.lower() for word in constructive):
            score += 0.3
        
        return max(0.0, min(1.0, score))