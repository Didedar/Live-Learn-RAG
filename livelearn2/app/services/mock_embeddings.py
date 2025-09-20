"""Mock embedding service for testing without API keys."""

import asyncio
import hashlib
import re
from collections import Counter
from typing import List, Dict, Set

from loguru import logger
import numpy as np

from ..core.exceptions import EmbeddingError


class MockEmbeddings:
    """Mock embedding service that generates semantic-aware embeddings."""
    
    def __init__(self):
        self.dimension = 768  # Standard dimension
        
        # Ключевые слова для лучшего семантического поиска
        self.key_terms = {
            # Государственные услуги
            'регистрация': 10.0, 'услуга': 5.0, 'документ': 4.0, 'заявление': 4.0,
            'паспорт': 8.0, 'справка': 6.0, 'свидетельство': 6.0, 'лицензия': 6.0,
            'разрешение': 6.0, 'получить': 5.0, 'оформить': 5.0, 'подать': 4.0,
            
            # Места и организации
            'жительство': 8.0, 'место': 5.0, 'адрес': 5.0, 'прописка': 8.0,
            'egov': 7.0, 'портал': 5.0, 'цон': 6.0, 'министерство': 4.0,
            'полиция': 5.0, 'суд': 5.0, 'налоговая': 5.0,
            
            # Процедуры
            'онлайн': 6.0, 'электронный': 5.0, 'подписать': 4.0, 'оплатить': 4.0,
            'срок': 4.0, 'день': 3.0, 'стоимость': 4.0, 'бесплатно': 4.0,
            
            # Специфичные термины
            'ип': 7.0, 'предприниматель': 7.0, 'бизнес': 5.0, 'налог': 5.0,
            'несудимость': 8.0, 'криминальный': 6.0, 'судебный': 5.0,
            'брак': 6.0, 'развод': 6.0, 'семья': 4.0, 'ребенок': 4.0,
        }
        
        logger.info("Initialized Mock embeddings service with semantic awareness")
    
    def _extract_keywords(self, text: str) -> Dict[str, float]:
        """Извлекает ключевые слова из текста с весами."""
        text_lower = text.lower()
        
        # Убираем знаки препинания и разбиваем на слова
        words = re.findall(r'\b[а-яёa-z]+\b', text_lower)
        word_counts = Counter(words)
        
        # Считаем веса для ключевых слов
        keyword_weights = {}
        
        for word, count in word_counts.items():
            # Базовый вес из TF
            tf_weight = count / len(words) if words else 0
            
            # Бонус за ключевые слова
            key_bonus = 0
            for key_term, bonus in self.key_terms.items():
                if key_term in word:
                    key_bonus += bonus
            
            # Итоговый вес
            total_weight = tf_weight + key_bonus * 0.1
            if total_weight > 0:
                keyword_weights[word] = total_weight
        
        return keyword_weights
    
    def _text_to_embedding(self, text: str) -> List[float]:
        """Convert text to semantic-aware embedding."""
        # Извлекаем ключевые слова
        keywords = self._extract_keywords(text)
        
        # Создаем базовый embedding из хэша
        hash_obj = hashlib.md5(text.encode())
        hash_bytes = hash_obj.digest()
        base_numbers = [b / 255.0 for b in hash_bytes]
        
        # Расширяем до нужной размерности
        while len(base_numbers) < self.dimension:
            base_numbers.extend(base_numbers[:self.dimension - len(base_numbers)])
        base_numbers = base_numbers[:self.dimension]
        
        # Добавляем семантические компоненты на основе ключевых слов
        semantic_vector = [0.0] * self.dimension
        
        for word, weight in keywords.items():
            # Создаем вектор для каждого ключевого слова
            word_hash = hashlib.md5(word.encode()).digest()
            word_numbers = [b / 255.0 for b in word_hash]
            
            # Расширяем до размерности
            while len(word_numbers) < self.dimension:
                word_numbers.extend(word_numbers[:self.dimension - len(word_numbers)])
            word_numbers = word_numbers[:self.dimension]
            
            # Добавляем с весом
            for i in range(self.dimension):
                semantic_vector[i] += word_numbers[i] * weight
        
        # Комбинируем базовый и семантический векторы
        combined = []
        for i in range(self.dimension):
            combined.append(base_numbers[i] * 0.3 + semantic_vector[i] * 0.7)
        
        # Нормализуем
        norm = sum(x*x for x in combined) ** 0.5
        if norm > 0:
            combined = [x / norm for x in combined]
        
        return combined
    
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents."""
        if not texts:
            return []
        
        logger.info(f"Mock embedding {len(texts)} documents")
        
        embeddings = []
        for text in texts:
            embedding = self._text_to_embedding(text)
            embeddings.append(embedding)
            
        # Small delay to simulate API call
        await asyncio.sleep(0.1)
        
        return embeddings
    
    async def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        logger.debug(f"Mock embedding query: {text[:50]}...")
        embedding = self._text_to_embedding(text)
        await asyncio.sleep(0.05)
        return embedding
    
    async def health_check(self) -> bool:
        """Health check always passes for mock service."""
        return True
