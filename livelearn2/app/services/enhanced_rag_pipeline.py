"""Enhanced RAG pipeline with hybrid search and improved retrieval."""

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
from .ollama_llm import OllamaLLM, build_rag_prompt_for_ollama


@dataclass
class SearchResult:
    """Результат поиска с подробной информацией."""
    chunk: Chunk
    dense_score: float = 0.0
    keyword_score: float = 0.0
    final_score: float = 0.0
    rank: int = 0
    source: str = "hybrid"


class EnhancedRAGPipeline:
    """Enhanced RAG pipeline with hybrid search, reranking, and better chunking."""
    
    def __init__(
        self,
        embeddings_service: Optional[MockEmbeddings] = None,
        keyword_service: Optional[KeywordSearchService] = None,
        llm_service: Optional[Any] = None
    ):
        # Services
        self.embeddings = embeddings_service or MockEmbeddings()
        self.keyword_search = keyword_service or KeywordSearchService()
        
        if llm_service:
            self.llm = llm_service
        else:
            self.llm = OllamaLLM(
                base_url=settings.ollama_url,
                model=settings.ollama_model
            )
        
        # Improved parameters based on recommendations
        self.chunk_size = 350  # Optimal size 350-600 tokens
        self.chunk_overlap = 70  # 20-25% overlap
        self.tau_retr = 0.4  # Minimum retrieval threshold
        self.mmr_lambda = 0.15  # MMR diversity parameter
        
        logger.info("Enhanced RAG pipeline initialized with hybrid search")
    
    async def ask(
        self,
        question: str,
        db: Session,
        session_id: Optional[str] = None,
        top_k: int = 4  # Smaller k as recommended
    ) -> Dict[str, Any]:
        """
        Main ask method with enhanced retrieval and single answer logic.
        """
        try:
            logger.info(f"Processing enhanced question: {question[:100]}...")
            
            # Step 1: Hybrid retrieval
            contexts = await self.hybrid_retrieve(
                db=db,
                query=question,
                top_k=top_k
            )
            
            if not contexts:
                logger.warning("No contexts found with hybrid search")
                return {
                    "answer": "Предоставлю максимально полную информацию на основе имеющихся данных. Для получения дополнительных деталей рекомендую обратиться в ЦОН.",
                    "contexts": [],
                    "message_id": str(uuid.uuid4()),
                    "retrieval_method": "hybrid_search",
                    "can_answer": False
                }
            
            # Step 2: Apply answerability gate
            max_score = max(ctx.score for ctx in contexts)
            if max_score < self.tau_retr:
                logger.info(f"Failed answerability gate: {max_score:.3f} < {self.tau_retr}")
                return {
                    "answer": "Уверенность в ответе слишком низкая. Попробуйте задать более конкретный вопрос.",
                    "contexts": [],
                    "message_id": str(uuid.uuid4()),
                    "retrieval_method": "hybrid_search",
                    "can_answer": False,
                    "max_score": max_score
                }
            
            # Step 3: Generate answer using contexts
            answer = await self.generate_answer(question, contexts)
            
            # Step 4: Save message session
            message_id = await self._save_message_session(
                db=db,
                session_id=session_id,
                question=question,
                answer=answer,
                contexts=contexts
            )
            
            # Format contexts for response
            formatted_contexts = [
                {
                    "text": ctx.content,
                    "metadata": {
                        "doc_id": ctx.doc_id,
                        "chunk_id": ctx.chunk_id,
                        "source": getattr(ctx, 'source', 'hybrid'),
                        "version": getattr(ctx, 'version', 1),
                        "retrieval_method": "hybrid_search"
                    },
                    "score": ctx.score
                }
                for ctx in contexts
            ]
            
            logger.info(f"Successfully processed with hybrid search, message_id: {message_id}")
            
            return {
                "answer": answer,
                "contexts": formatted_contexts,
                "message_id": message_id,
                "retrieval_method": "hybrid_search",
                "can_answer": True,
                "max_score": max_score
            }
            
        except Exception as e:
            logger.error(f"Error in enhanced ask method: {e}")
            return {
                "answer": "Произошла ошибка при обработке вашего вопроса.",
                "contexts": [],
                "message_id": str(uuid.uuid4()),
                "can_answer": False,
                "error": str(e)
            }
    
    async def hybrid_retrieve(
        self,
        db: Session,
        query: str,
        top_k: int = 4
    ) -> List[Any]:
        """
        Hybrid retrieval combining dense and keyword search.
        """
        try:
            logger.debug(f"Hybrid retrieval for query: {query[:100]}...")
            
            # Step 1: Dense retrieval with embeddings
            dense_results = await self.dense_retrieve(db, query, k=10)
            
            # Step 2: Keyword/BM25 retrieval
            keyword_results = self.keyword_retrieve(db, query, k=10)
            
            # Step 3: Fusion of results
            fused_results = self.fuse_results(dense_results, keyword_results, alpha=0.6)
            
            # Step 4: Apply MMR for diversity
            mmr_results = self.apply_mmr(fused_results, query, top_k, lambda_param=self.mmr_lambda)
            
            # Step 5: Simple reranking (placeholder for cross-encoder)
            reranked_results = await self.simple_rerank(mmr_results, query)
            
            # Create context objects
            contexts = []
            for result in reranked_results[:top_k]:
                context = type('Context', (), {
                    'content': result.chunk.content,
                    'doc_id': result.chunk.document_id,
                    'chunk_id': result.chunk.id,
                    'score': result.final_score,
                    'metadata': {},
                    'source': result.source,
                    'version': result.chunk.version
                })()
                contexts.append(context)
            
            logger.info(f"Hybrid retrieval found {len(contexts)} contexts")
            return contexts
            
        except Exception as e:
            logger.error(f"Error in hybrid retrieval: {e}")
            return []
    
    async def dense_retrieve(self, db: Session, query: str, k: int = 10) -> List[SearchResult]:
        """Dense retrieval using embeddings."""
        try:
            # Generate query embedding
            query_embedding = await self.embeddings.embed_query(query)
            
            # Get all chunks with embeddings
            chunks = db.query(Chunk).filter(
                Chunk.embedding.isnot(None),
                Chunk.embedding != "[]",
                Chunk.source == "original"
            ).all()
            
            if not chunks:
                return []
            
            # Calculate similarities
            embeddings = [chunk.embedding for chunk in chunks]
            similarities = batch_cosine_similarity(query_embedding, embeddings)
            
            # Create search results
            results = []
            for chunk, similarity in zip(chunks, similarities):
                results.append(SearchResult(
                    chunk=chunk,
                    dense_score=similarity,
                    final_score=similarity,
                    source="dense"
                ))
            
            # Sort and return top-k
            results.sort(key=lambda x: x.dense_score, reverse=True)
            return results[:k]
            
        except Exception as e:
            logger.error(f"Error in dense retrieval: {e}")
            return []
    
    def keyword_retrieve(self, db: Session, query: str, k: int = 10) -> List[SearchResult]:
        """Keyword/BM25 retrieval."""
        try:
            # Use keyword search service
            scored_chunks = self.keyword_search.search_relevant_chunks(
                db=db,
                query=query,
                top_k=k,
                min_score=0.05  # Lower threshold for keyword search
            )
            
            # Convert to SearchResult objects
            results = []
            for chunk, score in scored_chunks:
                results.append(SearchResult(
                    chunk=chunk,
                    keyword_score=score,
                    final_score=score,
                    source="keyword"
                ))
            
            return results
            
        except Exception as e:
            logger.error(f"Error in keyword retrieval: {e}")
            return []
    
    def fuse_results(
        self,
        dense_results: List[SearchResult],
        keyword_results: List[SearchResult],
        alpha: float = 0.6
    ) -> List[SearchResult]:
        """
        Fuse dense and keyword results using weighted combination.
        
        Args:
            dense_results: Results from dense retrieval
            keyword_results: Results from keyword retrieval
            alpha: Weight for dense scores (0.6 recommended)
        """
        try:
            # Create a map of chunk_id -> SearchResult for fusion
            chunk_map = {}
            
            # Normalize scores to 0-1 range
            if dense_results:
                max_dense = max(r.dense_score for r in dense_results)
                for result in dense_results:
                    normalized_dense = result.dense_score / max_dense if max_dense > 0 else 0
                    chunk_map[result.chunk.id] = SearchResult(
                        chunk=result.chunk,
                        dense_score=normalized_dense,
                        keyword_score=0.0,
                        final_score=alpha * normalized_dense,
                        source="dense"
                    )
            
            if keyword_results:
                max_keyword = max(r.keyword_score for r in keyword_results)
                for result in keyword_results:
                    normalized_keyword = result.keyword_score / max_keyword if max_keyword > 0 else 0
                    
                    if result.chunk.id in chunk_map:
                        # Combine scores
                        existing = chunk_map[result.chunk.id]
                        existing.keyword_score = normalized_keyword
                        existing.final_score = alpha * existing.dense_score + (1 - alpha) * normalized_keyword
                        existing.source = "hybrid"
                    else:
                        # New chunk from keyword search
                        chunk_map[result.chunk.id] = SearchResult(
                            chunk=result.chunk,
                            dense_score=0.0,
                            keyword_score=normalized_keyword,
                            final_score=(1 - alpha) * normalized_keyword,
                            source="keyword"
                        )
            
            # Convert to list and sort
            fused_results = list(chunk_map.values())
            fused_results.sort(key=lambda x: x.final_score, reverse=True)
            
            logger.debug(f"Fused {len(fused_results)} results from dense and keyword search")
            return fused_results
            
        except Exception as e:
            logger.error(f"Error in result fusion: {e}")
            return dense_results + keyword_results  # Fallback
    
    def apply_mmr(
        self,
        results: List[SearchResult],
        query: str,
        k: int,
        lambda_param: float = 0.15
    ) -> List[SearchResult]:
        """
        Apply Maximal Marginal Relevance for diversity.
        """
        try:
            if not results or k <= 0:
                return []
            
            selected = []
            remaining = results.copy()
            
            # Select first result (highest score)
            if remaining:
                selected.append(remaining.pop(0))
            
            # Select remaining results with MMR
            while len(selected) < k and remaining:
                best_score = -1
                best_idx = 0
                
                for i, candidate in enumerate(remaining):
                    # Relevance score
                    relevance = candidate.final_score
                    
                    # Diversity score (simple text similarity)
                    max_similarity = 0
                    for selected_result in selected:
                        similarity = self.text_similarity(
                            candidate.chunk.content,
                            selected_result.chunk.content
                        )
                        max_similarity = max(max_similarity, similarity)
                    
                    # MMR score
                    mmr_score = lambda_param * relevance - (1 - lambda_param) * max_similarity
                    
                    if mmr_score > best_score:
                        best_score = mmr_score
                        best_idx = i
                
                selected.append(remaining.pop(best_idx))
            
            logger.debug(f"Applied MMR, selected {len(selected)} diverse results")
            return selected
            
        except Exception as e:
            logger.error(f"Error in MMR: {e}")
            return results[:k]  # Fallback
    
    def text_similarity(self, text1: str, text2: str) -> float:
        """Simple text similarity for MMR."""
        try:
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            
            if not words1 or not words2:
                return 0.0
            
            intersection = len(words1 & words2)
            union = len(words1 | words2)
            
            return intersection / union if union > 0 else 0.0
            
        except Exception:
            return 0.0
    
    async def simple_rerank(
        self,
        results: List[SearchResult],
        query: str
    ) -> List[SearchResult]:
        """
        Simple reranking (placeholder for cross-encoder).
        """
        try:
            # For now, just boost results that have query terms in content
            query_words = set(query.lower().split())
            
            for result in results:
                content_words = set(result.chunk.content.lower().split())
                overlap = len(query_words & content_words)
                
                # Small boost for direct query term overlap
                if overlap > 0:
                    boost = min(0.1 * overlap / len(query_words), 0.2)
                    result.final_score += boost
            
            # Re-sort by final score
            results.sort(key=lambda x: x.final_score, reverse=True)
            
            logger.debug("Applied simple reranking")
            return results
            
        except Exception as e:
            logger.error(f"Error in reranking: {e}")
            return results
    
    async def ingest_text(
        self,
        db: Session,
        text: str,
        metadata: Dict[str, Any] = None,
        uri: str = "inline"
    ) -> Tuple[int, int]:
        """
        Ingest text with improved chunking strategy.
        """
        try:
            logger.info(f"Ingesting text: {len(text)} characters")
            
            # Create document record
            document = Document(
                uri=uri,
                doc_metadata=metadata or {}
            )
            db.add(document)
            db.flush()
            
            # Improved chunking with headers and overlap
            chunks = chunk_text(
                text,
                max_tokens=self.chunk_size,
                overlap=self.chunk_overlap
            )
            
            logger.info(f"Created {len(chunks)} chunks with improved strategy")
            
            # Generate embeddings
            embeddings = await self.embeddings.embed_documents(chunks)
            
            # Save chunks
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
            
            logger.info(f"Successfully ingested document {document.id} with {len(chunks)} chunks")
            return document.id, len(chunks)
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error ingesting text: {e}")
            raise
    
    async def generate_answer(self, query: str, contexts: List[Any]) -> str:
        """Generate answer using LLM with strict parameters."""
        try:
            # Prepare context data
            context_data = [
                {
                    'content': ctx.content,
                    'doc_id': ctx.doc_id,
                    'chunk_id': ctx.chunk_id,
                    'score': ctx.score,
                    'metadata': getattr(ctx, 'metadata', {})
                }
                for ctx in contexts
            ]
            
            # Build prompt for Ollama with strict instructions
            prompt = build_rag_prompt_for_ollama(query, context_data)
            
            # Generate with strict parameters
            answer = await self.llm.generate(
                prompt,
                temperature=0.0,  # No randomness
                max_tokens=400    # Limit response length
            )
            
            logger.debug(f"Generated answer: {len(answer)} characters")
            return answer
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return "Постараюсь предоставить полезную информацию. Для получения точных данных рекомендую обратиться в ЦОН или на egov.kz."
    
    async def _save_message_session(
        self,
        db: Session,
        session_id: Optional[str],
        question: str,
        answer: str,
        contexts: List[Any]
    ) -> str:
        """Save message session for feedback tracking."""
        try:
            message_id = str(uuid.uuid4())
            
            # Format contexts for storage
            contexts_data = [
                {
                    "doc_id": ctx.doc_id,
                    "chunk_id": ctx.chunk_id,
                    "score": ctx.score,
                    "content": ctx.content[:200] + "..." if len(ctx.content) > 200 else ctx.content
                }
                for ctx in contexts
            ]
            
            # Create message session
            message_session = MessageSession(
                id=message_id,
                session_id=session_id,
                question=question,
                answer=answer,
                contexts_used=contexts_data
            )
            
            db.add(message_session)
            db.commit()
            
            logger.debug(f"Saved message session: {message_id}")
            return message_id
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error saving message session: {e}")
            return str(uuid.uuid4())
    
    async def get_retrieval_stats(self, db: Session) -> Dict[str, Any]:
        """Get enhanced retrieval statistics."""
        try:
            total_chunks = db.query(Chunk).count()
            chunks_with_embeddings = db.query(Chunk).filter(
                Chunk.embedding.isnot(None),
                Chunk.embedding != "[]"
            ).count()
            
            chunks_with_feedback = db.query(ChunkWeight).count()
            
            return {
                "total_chunks": total_chunks,
                "chunks_with_embeddings": chunks_with_embeddings,
                "chunks_with_feedback": chunks_with_feedback,
                "embedding_coverage": chunks_with_embeddings / max(total_chunks, 1),
                "pipeline_type": "enhanced_hybrid",
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "tau_retr": self.tau_retr,
                "mmr_lambda": self.mmr_lambda,
                "features": [
                    "hybrid_search",
                    "mmr_diversity",
                    "simple_reranking",
                    "answerability_gate"
                ]
            }
            
        except Exception as e:
            logger.error(f"Error getting retrieval stats: {e}")
            return {
                "total_chunks": 0,
                "chunks_with_embeddings": 0,
                "chunks_with_feedback": 0,
                "embedding_coverage": 0.0,
                "pipeline_type": "enhanced_hybrid",
                "error": str(e)
            }

