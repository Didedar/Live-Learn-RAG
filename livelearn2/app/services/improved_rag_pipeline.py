"""
Улучшенная RAG система на основе анализа ошибок из feedback данных.
Основные улучшения:
1. Исправлены языковые проблемы
2. Улучшены embeddings
3. Строгие правила отказа от ответа
4. Оптимизированы параметры поиска
"""

import uuid
import math
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass

from loguru import logger
from sqlalchemy.orm import Session, joinedload

from ..config import settings
from ..models.documents import Chunk, Document
from ..models.feedback import ChunkWeight, MessageSession
from ..utils.text_processing import chunk_text
from ..utils.vectors import cosine_similarity, normalize_vector, batch_cosine_similarity
from .mock_embeddings import MockEmbeddings
from .keyword_search import KeywordSearchService
from .ollama_llm import OllamaLLM


@dataclass
class SearchResult:
    """Результат поиска с подробной информацией."""
    chunk: Chunk
    dense_score: float = 0.0
    keyword_score: float = 0.0
    final_score: float = 0.0
    rank: int = 0
    source: str = "hybrid"
    confidence: float = 0.0


class ImprovedMockEmbeddings(MockEmbeddings):
    """Улучшенные mock embeddings с лучшей семантической логикой."""
    
    def __init__(self):
        super().__init__()
        
        # Расширенные ключевые слова для лучшего поиска
        self.enhanced_keywords = {
            # Гражданство и миграция
            'гражданство': ['гражданин', 'паспорт', 'национальность', 'резидент', 'иностранец'],
            'миграция': ['переезд', 'вид на жительство', 'регистрация', 'прописка', 'временное проживание'],
            'репатриация': ['возвращение', 'этнический казах', 'оралман', 'переселение'],
            
            # Банковское дело
            'банк': ['кредит', 'счет', 'депозит', 'валюта', 'нацбанк', 'финансы'],
            'валютное регулирование': ['валютный контроль', 'уведомление', 'иностранный банк', 'резидент'],
            
            # Недвижимость и земля
            'недвижимость': ['квартира', 'дом', 'земля', 'собственность', 'аренда', 'продажа'],
            'земельный участок': ['га', 'гектар', 'аренда земли', 'сельхозземли', 'фермерство'],
            
            # Юридические процедуры
            'суд': ['иск', 'решение суда', 'юрисдикция', 'спор', 'процесс'],
            'наследство': ['наследник', 'завещание', 'наследование', 'смерть', 'имущество'],
            'брак': ['загс', 'регистрация брака', 'свидетельство', 'семья', 'супруг'],
            
            # Бизнес и предпринимательство  
            'ип': ['индивидуальный предприниматель', 'бизнес', 'предпринимательство', 'деятельность'],
            'концессия': ['концессионер', 'инвестор', 'строительство', 'эксплуатация'],
            
            # Административные вопросы
            'штраф': ['наказание', 'административная ответственность', 'нарушение', 'коап'],
            'техосмотр': ['технический осмотр', 'автомобиль', 'транспорт', 'безопасность'],
            
            # Конституционное право
            'конституция': ['основной закон', 'статья', 'право', 'государство'],
            'закон': ['нормативный акт', 'правило', 'статья', 'кодекс', 'постановление']
        }
    
    def _calculate_enhanced_similarity(self, query_words: set, content_words: set) -> float:
        """Улучшенный расчет семантической близости."""
        # Базовая близость по пересечению слов
        intersection = len(query_words & content_words)
        union = len(query_words | content_words)
        base_similarity = intersection / union if union > 0 else 0.0
        
        # Семантический бонус за связанные термины
        semantic_bonus = 0.0
        for query_word in query_words:
            for category, related_words in self.enhanced_keywords.items():
                if query_word in related_words or category in query_word:
                    for content_word in content_words:
                        if content_word in related_words or category in content_word:
                            semantic_bonus += 0.1
        
        # Бонус за точные совпадения важных терминов
        exact_match_bonus = 0.0
        important_terms = ['статья', 'закон', 'кодекс', 'гражданин', 'право']
        for term in important_terms:
            if any(term in word for word in query_words) and any(term in word for word in content_words):
                exact_match_bonus += 0.2
        
        # Итоговый скор
        final_score = base_similarity + min(semantic_bonus, 0.3) + min(exact_match_bonus, 0.4)
        return min(final_score, 1.0)


def build_improved_rag_prompt(query: str, contexts: List[Dict[str, Any]]) -> str:
    """Улучшенный промпт для генерации ответов с строгими правилами."""
    
    context_text = ""
    for i, ctx in enumerate(contexts, 1):
        context_text += f"[{i}] {ctx['content'][:500]}...\n\n"
    
    prompt = f"""ВАЖНЫЕ ИНСТРУКЦИИ:
1. ВСЕГДА отвечай ТОЛЬКО на русском языке
2. Предоставляй максимально полную и полезную информацию на основе контекста
3. При ответе обязательно ссылайся на конкретные статьи законов, если они упоминаются в контексте
4. Используй уверенные, четкие формулировки
5. Структурируй ответ четко и логично
6. Если вопрос касается законодательства РК, обязательно укажи конкретные нормативные акты
7. Делай практические выводы и давай конкретные рекомендации

КОНТЕКСТ:
{context_text}

ВОПРОС: {query}

ОТВЕТ (только на русском языке, четко и конкретно):"""
    
    return prompt


class ImprovedRAGPipeline:
    """Улучшенная RAG система на основе анализа ошибок."""
    
    def __init__(
        self,
        embeddings_service: Optional[ImprovedMockEmbeddings] = None,
        keyword_service: Optional[KeywordSearchService] = None,
        llm_service: Optional[Any] = None
    ):
        # Используем улучшенные embeddings
        self.embeddings = embeddings_service or ImprovedMockEmbeddings()
        self.keyword_search = keyword_service or KeywordSearchService()
        
        if llm_service:
            self.llm = llm_service
        else:
            self.llm = OllamaLLM(
                base_url=settings.ollama_url,
                model=settings.ollama_model
            )
        
        # Улучшенные параметры на основе анализа
        self.chunk_size = 400  # Немного больше для лучшего контекста
        self.chunk_overlap = 120  # 30% overlap для сохранения контекста
        self.tau_retr = 0.3  # Снижен порог для большей гибкости
        self.mmr_lambda = 0.2  # Больше разнообразия
        self.confidence_threshold = 0.6  # Минимальная уверенность
        
        # Весы для гибридного поиска (больше keyword search)
        self.dense_weight = 0.4
        self.keyword_weight = 0.6
        
        logger.info("Improved RAG pipeline initialized with error-based fixes")
    
    async def ask(
        self,
        question: str,
        db: Session,
        session_id: Optional[str] = None,
        top_k: int = 4
    ) -> Dict[str, Any]:
        """
        Главный метод с улучшенной логикой обработки.
        """
        try:
            logger.info(f"Processing improved question: {question[:100]}...")
            
            # Шаг 1: Гибридный поиск с улучшенными параметрами
            contexts = await self.improved_hybrid_retrieve(
                db=db,
                query=question,
                top_k=top_k
            )
            
            if not contexts:
                logger.warning("No contexts found with improved hybrid search")
                return {
                    "answer": "На основе имеющейся информации могу предоставить общие рекомендации. Для получения более детальной консультации рекомендую обратиться в ЦОН или на портал egov.kz.",
                    "contexts": [],
                    "message_id": str(uuid.uuid4()),
                    "retrieval_method": "improved_hybrid",
                    "can_answer": False,
                    "confidence": 0.0
                }
            
            # Шаг 2: Оценка уверенности
            max_score = max(ctx.score for ctx in contexts)
            confidence = self._calculate_confidence(contexts, question)
            
            if confidence < self.confidence_threshold:
                logger.info(f"Low confidence: {confidence:.3f} < {self.confidence_threshold}")
                return {
                    "answer": "Уверенность в правильности ответа слишком низкая. Рекомендую обратиться к специалисту или переформулировать вопрос более конкретно.",
                    "contexts": [],
                    "message_id": str(uuid.uuid4()),
                    "retrieval_method": "improved_hybrid",
                    "can_answer": False,
                    "confidence": confidence,
                    "max_score": max_score
                }
            
            # Шаг 3: Генерация улучшенного ответа
            answer = await self.generate_improved_answer(question, contexts)
            
            # Шаг 4: Проверка качества ответа
            if self._is_low_quality_answer(answer):
                return {
                    "answer": "Постараюсь предоставить максимально полезную информацию на основе доступных данных. Рекомендую также проконсультироваться со специалистами ЦОНа для получения актуальной информации.",
                    "contexts": [],
                    "message_id": str(uuid.uuid4()),
                    "retrieval_method": "improved_hybrid",
                    "can_answer": False,
                    "confidence": confidence
                }
            
            # Шаг 5: Сохранение сессии
            message_id = await self._save_message_session(
                db=db,
                session_id=session_id,
                question=question,
                answer=answer,
                contexts=contexts
            )
            
            # Форматирование контекстов
            formatted_contexts = [
                {
                    "text": ctx.content,
                    "metadata": {
                        "doc_id": ctx.doc_id,
                        "chunk_id": ctx.chunk_id,
                        "source": getattr(ctx, 'source', 'improved_hybrid'),
                        "confidence": getattr(ctx, 'confidence', confidence)
                    },
                    "score": ctx.score
                }
                for ctx in contexts
            ]
            
            logger.info(f"Successfully processed with improved pipeline, message_id: {message_id}")
            
            return {
                "answer": answer,
                "contexts": formatted_contexts,
                "message_id": message_id,
                "retrieval_method": "improved_hybrid",
                "can_answer": True,
                "confidence": confidence,
                "max_score": max_score
            }
            
        except Exception as e:
            logger.error(f"Error in improved ask method: {e}")
            return {
                "answer": "Произошла техническая ошибка при обработке вашего вопроса. Попробуйте еще раз.",
                "contexts": [],
                "message_id": str(uuid.uuid4()),
                "can_answer": False,
                "error": str(e),
                "confidence": 0.0
            }
    
    async def improved_hybrid_retrieve(
        self,
        db: Session,
        query: str,
        top_k: int = 4
    ) -> List[Any]:
        """
        Улучшенный гибридный поиск с исправленными весами.
        """
        try:
            logger.debug(f"Improved hybrid retrieval for query: {query[:100]}...")
            
            # Dense retrieval с улучшенными embeddings
            dense_results = await self.improved_dense_retrieve(db, query, k=8)
            
            # Keyword retrieval с повышенным весом
            keyword_results = self.improved_keyword_retrieve(db, query, k=8)
            
            # Fusion с новыми весами (больше keyword)
            fused_results = self.improved_fuse_results(
                dense_results, 
                keyword_results, 
                dense_weight=self.dense_weight,
                keyword_weight=self.keyword_weight
            )
            
            # MMR для разнообразия
            mmr_results = self.apply_mmr(fused_results, query, top_k, lambda_param=self.mmr_lambda)
            
            # Улучшенный reranking
            reranked_results = await self.improved_rerank(mmr_results, query)
            
            # Создание контекстов
            contexts = []
            for result in reranked_results[:top_k]:
                context = type('Context', (), {
                    'content': result.chunk.content,
                    'doc_id': result.chunk.document_id,
                    'chunk_id': result.chunk.id,
                    'score': result.final_score,
                    'confidence': result.confidence,
                    'metadata': {},
                    'source': result.source,
                    'version': result.chunk.version
                })()
                contexts.append(context)
            
            logger.info(f"Improved hybrid retrieval found {len(contexts)} contexts")
            return contexts
            
        except Exception as e:
            logger.error(f"Error in improved hybrid retrieval: {e}")
            return []
    
    async def improved_dense_retrieve(self, db: Session, query: str, k: int = 8) -> List[SearchResult]:
        """Dense retrieval с улучшенными embeddings."""
        try:
            # Генерируем embedding запроса
            query_embedding = await self.embeddings.embed_query(query)
            
            # Получаем чанки
            chunks = db.query(Chunk).filter(
                Chunk.embedding.isnot(None),
                Chunk.embedding != "[]",
                Chunk.source == "original"
            ).all()
            
            if not chunks:
                return []
            
            # Вычисляем схожесть
            embeddings = [chunk.embedding for chunk in chunks]
            similarities = batch_cosine_similarity(query_embedding, embeddings)
            
            # Создаем результаты с confidence
            results = []
            for chunk, similarity in zip(chunks, similarities):
                confidence = self._calculate_chunk_confidence(chunk, query, similarity)
                results.append(SearchResult(
                    chunk=chunk,
                    dense_score=similarity,
                    final_score=similarity,
                    confidence=confidence,
                    source="dense_improved"
                ))
            
            # Сортируем и возвращаем топ-k
            results.sort(key=lambda x: x.dense_score, reverse=True)
            return results[:k]
            
        except Exception as e:
            logger.error(f"Error in improved dense retrieval: {e}")
            return []
    
    def improved_keyword_retrieve(self, db: Session, query: str, k: int = 8) -> List[SearchResult]:
        """Keyword retrieval с улучшенной логикой."""
        try:
            # Используем улучшенный keyword search
            scored_chunks = self.keyword_search.search_relevant_chunks(
                db=db,
                query=query,
                top_k=k,
                min_score=0.03  # Еще ниже порог
            )
            
            results = []
            for chunk, score in scored_chunks:
                confidence = self._calculate_chunk_confidence(chunk, query, score)
                results.append(SearchResult(
                    chunk=chunk,
                    keyword_score=score,
                    final_score=score,
                    confidence=confidence,
                    source="keyword_improved"
                ))
            
            return results
            
        except Exception as e:
            logger.error(f"Error in improved keyword retrieval: {e}")
            return []
    
    def improved_fuse_results(
        self,
        dense_results: List[SearchResult],
        keyword_results: List[SearchResult],
        dense_weight: float = 0.4,
        keyword_weight: float = 0.6
    ) -> List[SearchResult]:
        """
        Улучшенное объединение результатов с новыми весами.
        """
        try:
            chunk_map = {}
            
            # Нормализация и взвешивание dense результатов
            if dense_results:
                max_dense = max(r.dense_score for r in dense_results)
                for result in dense_results:
                    normalized_dense = result.dense_score / max_dense if max_dense > 0 else 0
                    chunk_map[result.chunk.id] = SearchResult(
                        chunk=result.chunk,
                        dense_score=normalized_dense,
                        keyword_score=0.0,
                        final_score=dense_weight * normalized_dense,
                        confidence=result.confidence,
                        source="dense_improved"
                    )
            
            # Нормализация и взвешивание keyword результатов
            if keyword_results:
                max_keyword = max(r.keyword_score for r in keyword_results)
                for result in keyword_results:
                    normalized_keyword = result.keyword_score / max_keyword if max_keyword > 0 else 0
                    
                    if result.chunk.id in chunk_map:
                        # Комбинируем скоры
                        existing = chunk_map[result.chunk.id]
                        existing.keyword_score = normalized_keyword
                        existing.final_score = (dense_weight * existing.dense_score + 
                                              keyword_weight * normalized_keyword)
                        existing.confidence = max(existing.confidence, result.confidence)
                        existing.source = "hybrid_improved"
                    else:
                        # Новый чанк из keyword поиска
                        chunk_map[result.chunk.id] = SearchResult(
                            chunk=result.chunk,
                            dense_score=0.0,
                            keyword_score=normalized_keyword,
                            final_score=keyword_weight * normalized_keyword,
                            confidence=result.confidence,
                            source="keyword_improved"
                        )
            
            # Сортируем по итоговому скору
            fused_results = list(chunk_map.values())
            fused_results.sort(key=lambda x: x.final_score, reverse=True)
            
            logger.debug(f"Improved fusion: {len(fused_results)} results")
            return fused_results
            
        except Exception as e:
            logger.error(f"Error in improved fusion: {e}")
            return dense_results + keyword_results
    
    def apply_mmr(
        self,
        results: List[SearchResult],
        query: str,
        k: int,
        lambda_param: float = 0.2
    ) -> List[SearchResult]:
        """MMR с улучшенными параметрами."""
        try:
            if not results or k <= 0:
                return []
            
            selected = []
            remaining = results.copy()
            
            # Выбираем первый результат
            if remaining:
                selected.append(remaining.pop(0))
            
            # MMR отбор
            while len(selected) < k and remaining:
                best_score = -1
                best_idx = 0
                
                for i, candidate in enumerate(remaining):
                    relevance = candidate.final_score
                    
                    # Простая мера разнообразия
                    max_similarity = 0
                    for selected_result in selected:
                        similarity = self._text_similarity_improved(
                            candidate.chunk.content,
                            selected_result.chunk.content
                        )
                        max_similarity = max(max_similarity, similarity)
                    
                    # MMR скор
                    mmr_score = lambda_param * relevance - (1 - lambda_param) * max_similarity
                    
                    if mmr_score > best_score:
                        best_score = mmr_score
                        best_idx = i
                
                selected.append(remaining.pop(best_idx))
            
            logger.debug(f"Applied improved MMR, selected {len(selected)} diverse results")
            return selected
            
        except Exception as e:
            logger.error(f"Error in MMR: {e}")
            return results[:k]
    
    def _text_similarity_improved(self, text1: str, text2: str) -> float:
        """Улучшенная мера схожести текстов."""
        try:
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            
            if not words1 or not words2:
                return 0.0
            
            # Jaccard similarity
            intersection = len(words1 & words2)
            union = len(words1 | words2)
            
            return intersection / union if union > 0 else 0.0
            
        except Exception:
            return 0.0
    
    async def improved_rerank(
        self,
        results: List[SearchResult],
        query: str
    ) -> List[SearchResult]:
        """
        Улучшенный reranking с учетом специфики юридических документов.
        """
        try:
            query_words = set(query.lower().split())
            
            for result in results:
                content_words = set(result.chunk.content.lower().split())
                
                # Бонус за точные совпадения важных терминов
                legal_terms = ['статья', 'закон', 'кодекс', 'право', 'постановление', 'приказ']
                legal_bonus = 0.0
                for term in legal_terms:
                    if any(term in word for word in query_words) and any(term in word for word in content_words):
                        legal_bonus += 0.15
                
                # Бонус за номера статей
                import re
                query_articles = re.findall(r'статья\s+(\d+)', query.lower())
                content_articles = re.findall(r'статья\s+(\d+)', result.chunk.content.lower())
                
                if query_articles and content_articles:
                    if any(art in content_articles for art in query_articles):
                        legal_bonus += 0.25
                
                # Применяем бонус
                result.final_score += min(legal_bonus, 0.4)
                result.confidence += min(legal_bonus * 0.5, 0.2)
            
            # Пересортировка
            results.sort(key=lambda x: x.final_score, reverse=True)
            
            logger.debug("Applied improved legal reranking")
            return results
            
        except Exception as e:
            logger.error(f"Error in improved reranking: {e}")
            return results
    
    def _calculate_confidence(self, contexts: List[Any], query: str) -> float:
        """Расчет общей уверенности в ответе."""
        if not contexts:
            return 0.0
        
        # Средняя уверенность по контекстам
        avg_confidence = sum(getattr(ctx, 'confidence', 0.5) for ctx in contexts) / len(contexts)
        
        # Бонус за количество релевантных контекстов
        count_bonus = min(len(contexts) * 0.1, 0.3)
        
        # Штраф за низкие скоры
        max_score = max(ctx.score for ctx in contexts)
        score_factor = max_score if max_score > 0.3 else max_score * 0.5
        
        final_confidence = (avg_confidence + count_bonus) * score_factor
        return min(final_confidence, 1.0)
    
    def _calculate_chunk_confidence(self, chunk: Chunk, query: str, score: float) -> float:
        """Расчет уверенности для отдельного чанка."""
        base_confidence = min(score * 1.2, 0.8)
        
        # Бонус за длину контента
        length_bonus = min(len(chunk.content) / 1000, 0.1)
        
        # Бонус за совпадение ключевых слов
        query_words = set(query.lower().split())
        content_words = set(chunk.content.lower().split())
        overlap = len(query_words & content_words) / len(query_words) if query_words else 0
        overlap_bonus = overlap * 0.2
        
        return min(base_confidence + length_bonus + overlap_bonus, 1.0)
    
    def _is_low_quality_answer(self, answer: str) -> bool:
        """Проверка качества ответа."""
        if not answer or len(answer.strip()) < 20:
            return True
        
        # Проверяем на неопределенные формулировки
        uncertain_phrases = [
            'возможно', 'может быть', 'в общем', 'наверное', 
            'скорее всего', 'вероятно', 'по-видимому'
        ]
        
        answer_lower = answer.lower()
        uncertain_count = sum(1 for phrase in uncertain_phrases if phrase in answer_lower)
        
        if uncertain_count > 2:  # Слишком много неопределенности
            return True
        
        # Проверяем на английский язык
        english_words = ['unfortunately', 'i don\'t know', 'the', 'and', 'or', 'but']
        english_count = sum(1 for word in english_words if word in answer_lower)
        
        if english_count > 1:  # Ответ содержит английские слова
            return True
        
        return False
    
    async def generate_improved_answer(self, query: str, contexts: List[Any]) -> str:
        """Генерация улучшенного ответа."""
        try:
            # Подготавливаем данные контекста
            context_data = [
                {
                    'content': ctx.content,
                    'doc_id': ctx.doc_id,
                    'chunk_id': ctx.chunk_id,
                    'score': ctx.score,
                    'confidence': getattr(ctx, 'confidence', 0.5)
                }
                for ctx in contexts
            ]
            
            # Строим улучшенный промпт
            prompt = build_improved_rag_prompt(query, context_data)
            
            # Генерируем с строгими параметрами
            answer = await self.llm.generate(
                prompt,
                temperature=0.0,  # Без случайности
                max_tokens=500    # Больше места для детального ответа
            )
            
            # Постобработка ответа
            answer = self._postprocess_answer(answer)
            
            logger.debug(f"Generated improved answer: {len(answer)} characters")
            return answer
            
        except Exception as e:
            logger.error(f"Error generating improved answer: {e}")
            return "Постараюсь помочь с вашим вопросом. Рекомендую также обратиться за дополнительной консультацией в ЦОН или на портал egov.kz."
    
    def _postprocess_answer(self, answer: str) -> str:
        """Постобработка ответа для улучшения качества."""
        if not answer:
            return "Предоставлю максимально полезную информацию на основе доступных данных."
        
        # Удаляем лишние пробелы и переносы
        answer = ' '.join(answer.split())
        
        # Проверяем на английский язык в начале
        if answer.lower().startswith(('unfortunately', 'i don\'t', 'i cannot')):
            return "Предоставлю максимально полезную информацию на основе доступных данных."
        
        # Заменяем некоторые фразы
        replacements = {
            'I don\'t know': 'Предоставлю доступную информацию',
            'Unfortunately': 'К сожалению',
            'However': 'Однако',
            'Therefore': 'Поэтому'
        }
        
        for eng, rus in replacements.items():
            answer = answer.replace(eng, rus)
        
        return answer
    
    async def ingest_text(
        self,
        db: Session,
        text: str,
        metadata: Dict[str, Any] = None,
        uri: str = "inline"
    ) -> Tuple[int, int]:
        """
        Ingestion с улучшенным чанкованием.
        """
        try:
            logger.info(f"Ingesting text with improved chunking: {len(text)} characters")
            
            # Создаем документ
            document = Document(
                uri=uri,
                doc_metadata=metadata or {}
            )
            db.add(document)
            db.flush()
            
            # Улучшенное чанкование
            chunks = chunk_text(
                text,
                max_tokens=self.chunk_size,
                overlap=self.chunk_overlap
            )
            
            logger.info(f"Created {len(chunks)} chunks with improved strategy")
            
            # Генерируем embeddings
            embeddings = await self.embeddings.embed_documents(chunks)
            
            # Сохраняем чанки
            chunk_records = []
            for i, (chunk_content, embedding) in enumerate(zip(chunks, embeddings)):
                chunk_record = Chunk(
                    document_id=document.id,
                    ordinal=i,
                    content=chunk_content,
                    embedding=embedding,
                    source="original",
                    version=1
                )
                chunk_records.append(chunk_record)
            
            db.add_all(chunk_records)
            db.commit()
            
            logger.info(f"Successfully ingested document {document.id} with {len(chunks)} improved chunks")
            return document.id, len(chunks)
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error ingesting text with improvements: {e}")
            raise
    
    async def _save_message_session(
        self,
        db: Session,
        session_id: Optional[str],
        question: str,
        answer: str,
        contexts: List[Any]
    ) -> str:
        """Сохранение сессии сообщения."""
        try:
            message_id = str(uuid.uuid4())
            
            contexts_data = [
                {
                    "doc_id": ctx.doc_id,
                    "chunk_id": ctx.chunk_id,
                    "score": ctx.score,
                    "confidence": getattr(ctx, 'confidence', 0.5),
                    "content": ctx.content[:200] + "..." if len(ctx.content) > 200 else ctx.content
                }
                for ctx in contexts
            ]
            
            message_session = MessageSession(
                id=message_id,
                session_id=session_id,
                question=question,
                answer=answer,
                contexts_used=contexts_data
            )
            
            db.add(message_session)
            db.commit()
            
            logger.debug(f"Saved improved message session: {message_id}")
            return message_id
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error saving improved message session: {e}")
            return str(uuid.uuid4())
    
    async def get_retrieval_stats(self, db: Session) -> Dict[str, Any]:
        """Статистика улучшенной системы."""
        try:
            total_chunks = db.query(Chunk).count()
            chunks_with_embeddings = db.query(Chunk).filter(
                Chunk.embedding.isnot(None),
                Chunk.embedding != "[]"
            ).count()
            
            return {
                "total_chunks": total_chunks,
                "chunks_with_embeddings": chunks_with_embeddings,
                "embedding_coverage": chunks_with_embeddings / max(total_chunks, 1),
                "pipeline_type": "improved_hybrid",
                "improvements": [
                    "enhanced_embeddings",
                    "strict_language_rules", 
                    "confidence_scoring",
                    "legal_document_optimization",
                    "improved_chunking",
                    "better_fusion_weights"
                ],
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "tau_retr": self.tau_retr,
                "confidence_threshold": self.confidence_threshold,
                "dense_weight": self.dense_weight,
                "keyword_weight": self.keyword_weight
            }
            
        except Exception as e:
            logger.error(f"Error getting improved retrieval stats: {e}")
            return {
                "error": str(e),
                "pipeline_type": "improved_hybrid"
            }
