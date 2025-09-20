"""Keyword-based search service for better retrieval without embeddings."""

import re
from typing import List, Dict, Any, Tuple
from collections import Counter
from loguru import logger
from sqlalchemy.orm import Session
from ..models.documents import Chunk


class KeywordSearchService:
    """Keyword-based search that works better than mock embeddings."""
    
    def __init__(self):
        # Ключевые термины для государственных услуг
        self.service_keywords = {
            # Регистрация и документы
            'регистрация': ['регистрация', 'регистрировать', 'зарегистрировать', 'прописка', 'прописать'],
            'жительство': ['жительство', 'место', 'адрес', 'проживание'],
            'паспорт': ['паспорт', 'удостоверение', 'личность'],
            'справка': ['справка', 'справку', 'документ'],
            'несудимость': ['несудимость', 'судимость', 'криминальный', 'судебный'],
            
            # Бизнес
            'ип': ['ип', 'предприниматель', 'индивидуальный', 'предпринимательство'],
            'бизнес': ['бизнес', 'деятельность', 'коммерческий'],
            
            # Процедуры
            'получить': ['получить', 'получение', 'оформить', 'оформление'],
            'подать': ['подать', 'подача', 'заявление', 'заявку'],
            'онлайн': ['онлайн', 'электронный', 'портал', 'egov', 'егов'],
        }
        
        # Синонимы для лучшего поиска
        self.synonyms = {
            'как': ['как', 'каким', 'способ', 'процедура'],
            'получить': ['получить', 'оформить', 'взять', 'сделать'],
            'нужно': ['нужно', 'необходимо', 'требуется', 'надо'],
            'документы': ['документы', 'документ', 'бумаги', 'справки'],
        }
        
        logger.info("Initialized keyword-based search service")
    
    def search_relevant_chunks(
        self, 
        db: Session, 
        query: str, 
        top_k: int = 8,
        min_score: float = 0.1
    ) -> List[Tuple[Chunk, float]]:
        """
        Поиск релевантных чанков на основе ключевых слов.
        
        Args:
            db: Database session
            query: Поисковый запрос
            top_k: Количество результатов
            min_score: Минимальный скор
            
        Returns:
            List of (chunk, score) tuples
        """
        logger.debug(f"Keyword search for: {query[:100]}...")
        
        # Получаем все оригинальные чанки
        chunks = db.query(Chunk).filter(
            Chunk.source == 'original'
        ).all()
        
        if not chunks:
            logger.warning("No original chunks found")
            return []
        
        # Извлекаем ключевые слова из запроса
        query_keywords = self._extract_keywords(query.lower())
        
        # Вычисляем скоры для каждого чанка
        scored_chunks = []
        
        for chunk in chunks:
            score = self._calculate_relevance_score(query_keywords, chunk.content.lower())
            
            if score >= min_score:
                scored_chunks.append((chunk, score))
        
        # Сортируем по скору
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        
        logger.info(f"Found {len(scored_chunks)} relevant chunks, returning top {top_k}")
        return scored_chunks[:top_k]
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Извлекает ключевые слова из текста."""
        # Убираем знаки препинания и разбиваем на слова
        words = re.findall(r'\b[а-яёa-z]+\b', text.lower())
        
        # Расширяем синонимами
        expanded_words = set(words)
        
        for word in words:
            # Добавляем синонимы
            for key, synonyms in self.synonyms.items():
                if word in synonyms:
                    expanded_words.update(synonyms)
            
            # Добавляем связанные термины
            for key, related in self.service_keywords.items():
                if word in related:
                    expanded_words.update(related)
        
        return list(expanded_words)
    
    def _calculate_relevance_score(self, query_keywords: List[str], content: str) -> float:
        """Вычисляет скор релевантности чанка."""
        if not query_keywords:
            return 0.0
        
        content_words = re.findall(r'\b[а-яёa-z]+\b', content.lower())
        content_counter = Counter(content_words)
        
        total_score = 0.0
        
        for keyword in query_keywords:
            # Точное совпадение
            if keyword in content_counter:
                # Вес зависит от частоты и важности слова
                weight = self._get_keyword_weight(keyword)
                frequency = content_counter[keyword]
                total_score += weight * min(frequency, 3)  # Ограничиваем влияние частоты
            
            # Частичное совпадение (подстрока)
            else:
                for content_word in content_words:
                    if keyword in content_word or content_word in keyword:
                        if len(keyword) > 3 and len(content_word) > 3:  # Избегаем коротких совпадений
                            similarity = self._string_similarity(keyword, content_word)
                            if similarity > 0.7:
                                weight = self._get_keyword_weight(keyword) * 0.5  # Меньший вес для частичных совпадений
                                total_score += weight * similarity
        
        # Нормализуем скор
        normalized_score = min(total_score / len(query_keywords), 1.0)
        
        return normalized_score
    
    def _get_keyword_weight(self, keyword: str) -> float:
        """Получает вес ключевого слова."""
        # Важные термины получают больший вес
        high_priority = ['регистрация', 'паспорт', 'справка', 'несудимость', 'ип', 'предприниматель']
        medium_priority = ['жительство', 'получить', 'оформить', 'документ', 'услуга']
        
        if keyword in high_priority:
            return 2.0
        elif keyword in medium_priority:
            return 1.5
        elif any(keyword in terms for terms in self.service_keywords.values()):
            return 1.2
        else:
            return 1.0
    
    def _string_similarity(self, s1: str, s2: str) -> float:
        """Простая мера схожести строк."""
        if not s1 or not s2:
            return 0.0
        
        # Простое измерение на основе общих символов
        common = set(s1) & set(s2)
        total = set(s1) | set(s2)
        
        if not total:
            return 0.0
        
        return len(common) / len(total)
    
    def test_search(self, db: Session, queries: List[str]) -> Dict[str, Any]:
        """Тестирует поиск с различными запросами."""
        results = {}
        
        for query in queries:
            chunks = self.search_relevant_chunks(db, query, top_k=3)
            
            results[query] = {
                'found_chunks': len(chunks),
                'top_results': [
                    {
                        'chunk_id': chunk.id,
                        'score': score,
                        'content_preview': chunk.content[:100] + '...'
                    }
                    for chunk, score in chunks[:3]
                ]
            }
        
        return results
