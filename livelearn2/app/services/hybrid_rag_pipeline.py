"""Hybrid RAG pipeline with Dense + BM25 retrieval."""

import uuid
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass

from loguru import logger
from sqlalchemy.orm import Session, joinedload

from ..config import settings
from ..models.documents import Chunk, Document
from ..models.feedback import ChunkWeight, MessageSession
from ..utils.text_processing import chunk_text
from .hybrid_retrieval import HybridRetrieval, HybridResult
from .mock_embeddings import MockEmbeddings
from .bm25_search import BM25Search
from .ollama_llm import OllamaLLM, build_rag_prompt_for_ollama


@dataclass
class RetrievalContext:
    """Enhanced context with hybrid retrieval metadata."""
    content: str
    doc_id: int
    chunk_id: int
    score: float
    dense_score: float = 0.0
    bm25_score: float = 0.0
    normalized_dense: float = 0.0
    normalized_bm25: float = 0.0
    matched_terms: set = None
    retrieval_method: str = "hybrid"
    metadata: Dict[str, Any] = None
    source: str = "original"
    version: int = 1
    
    def __post_init__(self):
        if self.matched_terms is None:
            self.matched_terms = set()
        if self.metadata is None:
            self.metadata = {}


class HybridRAGPipeline:
    """
    Advanced RAG pipeline with hybrid retrieval (Dense + BM25).
    
    Key features:
    - Hybrid retrieval combining dense embeddings and BM25
    - Precise formula: score = α · z(dense) + (1-α) · z(bm25) where α ≈ 0.6
    - Better handling of factual queries (dates, names, numbers, IINs, laws)
    - Feedback-aware learning
    - Answerability gate for quality control
    """
    
    def __init__(
        self,
        embeddings_service: Optional[MockEmbeddings] = None,
        bm25_service: Optional[BM25Search] = None,
        llm_service: Optional[Any] = None,
        alpha: float = 0.6,  # Dense weight as recommended
        tau_retr: float = 0.4,  # Answerability threshold
        max_contexts: int = 4  # Optimal context count
    ):
        """
        Initialize hybrid RAG pipeline.
        
        Args:
            embeddings_service: Dense embedding service
            bm25_service: BM25 search service  
            llm_service: Language model service
            alpha: Weight for dense scores (0.6 recommended)
            tau_retr: Minimum retrieval threshold
            max_contexts: Maximum number of contexts to use
        """
        # Initialize services
        self.embeddings = embeddings_service or MockEmbeddings()
        self.bm25 = bm25_service or BM25Search(k1=1.5, b=0.75)
        
        if llm_service:
            self.llm = llm_service
        else:
            self.llm = OllamaLLM(
                base_url=settings.ollama_url,
                model=settings.ollama_model
            )
        
        # Initialize hybrid retrieval
        self.hybrid_retrieval = HybridRetrieval(
            embeddings_service=self.embeddings,
            bm25_service=self.bm25,
            alpha=alpha
        )
        
        # Pipeline parameters
        self.alpha = alpha
        self.tau_retr = tau_retr
        self.max_contexts = max_contexts
        
        # Improved chunking parameters
        self.chunk_size = 350  # Optimal size for factual content
        self.chunk_overlap = 70  # 20% overlap
        
        logger.info(
            f"Initialized Hybrid RAG pipeline with α={alpha:.2f}, "
            f"τ_retr={tau_retr:.2f}, max_contexts={max_contexts}"
        )
    
    async def ask(
        self,
        question: str,
        db: Session,
        session_id: Optional[str] = None,
        top_k: int = None,
        explain: bool = False
    ) -> Dict[str, Any]:
        """
        Main ask method with hybrid retrieval.
        
        Args:
            question: User question
            db: Database session
            session_id: Optional session ID
            top_k: Number of contexts (defaults to max_contexts)
            explain: Whether to include retrieval explanations
            
        Returns:
            Dictionary with answer and metadata
        """
        if top_k is None:
            top_k = self.max_contexts
        
        try:
            logger.info(f"Hybrid RAG processing: {question[:100]}...")
            
            # Step 1: Hybrid retrieval
            hybrid_results = await self.hybrid_retrieval.hybrid_search(
                db=db,
                query=question,
                top_k=top_k,
                dense_k=15,  # Retrieve more candidates for fusion
                bm25_k=15
            )
            
            if not hybrid_results:
                logger.warning("No results from hybrid retrieval")
                return self._create_no_answer_response(
                    "Не найдено релевантной информации для ответа на ваш вопрос.",
                    explain=explain
                )
            
            # Step 2: Apply answerability gate
            max_score = max(result.final_score for result in hybrid_results)
            if max_score < self.tau_retr:
                logger.info(f"Failed answerability gate: {max_score:.3f} < {self.tau_retr}")
                return self._create_no_answer_response(
                    f"Уверенность в ответе слишком низкая ({max_score:.2f}). "
                    f"Попробуйте задать более конкретный вопрос.",
                    max_score=max_score,
                    explain=explain
                )
            
            # Step 3: Apply feedback weights if available
            enhanced_results = await self._apply_feedback_weights(db, hybrid_results)
            
            # Step 4: Convert to contexts
            contexts = self._hybrid_results_to_contexts(enhanced_results)
            
            # Step 5: Generate answer
            answer = await self.generate_answer(question, contexts)
            
            # Step 6: Save message session
            message_id = await self._save_message_session(
                db=db,
                session_id=session_id,
                question=question,
                answer=answer,
                contexts=contexts
            )
            
            # Step 7: Prepare response
            response = {
                "answer": answer,
                "contexts": self._format_contexts_for_response(contexts),
                "message_id": message_id,
                "retrieval_method": "hybrid_dense_bm25",
                "can_answer": True,
                "max_score": max_score,
                "retrieval_stats": {
                    "total_candidates": len(hybrid_results),
                    "used_contexts": len(contexts),
                    "alpha": self.alpha,
                    "answerability_threshold": self.tau_retr
                }
            }
            
            # Add explanations if requested
            if explain:
                response["retrieval_explanation"] = self._create_retrieval_explanation(
                    question, hybrid_results, contexts
                )
            
            logger.info(f"Hybrid RAG completed successfully, message_id: {message_id}")
            return response
            
        except Exception as e:
            logger.error(f"Error in hybrid RAG pipeline: {e}")
            return self._create_error_response(str(e), explain=explain)
    
    async def ingest_text(
        self,
        db: Session,
        text: str,
        metadata: Dict[str, Any] = None,
        uri: str = "inline",
        rebuild_bm25: bool = True
    ) -> Tuple[int, int]:
        """
        Ingest text with hybrid indexing.
        
        Args:
            db: Database session
            text: Text content
            metadata: Document metadata
            uri: Document URI
            rebuild_bm25: Whether to rebuild BM25 index after ingestion
            
        Returns:
            Tuple of (document_id, chunk_count)
        """
        try:
            logger.info(f"Ingesting text for hybrid RAG: {len(text)} characters")
            
            # Create document
            document = Document(
                uri=uri,
                doc_metadata=metadata or {}
            )
            db.add(document)
            db.flush()
            
            # Improved chunking
            chunks = chunk_text(
                text,
                max_tokens=self.chunk_size,
                overlap=self.chunk_overlap
            )
            
            logger.info(f"Created {len(chunks)} chunks with optimized parameters")
            
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
            
            # Rebuild BM25 index to include new content
            if rebuild_bm25:
                logger.info("Rebuilding BM25 index after ingestion...")
                self.bm25.build_index(db)
            
            logger.info(
                f"Successfully ingested document {document.id} with {len(chunks)} chunks "
                f"(BM25 {'rebuilt' if rebuild_bm25 else 'not rebuilt'})"
            )
            
            return document.id, len(chunks)
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error in hybrid ingestion: {e}")
            raise
    
    async def _apply_feedback_weights(
        self,
        db: Session,
        hybrid_results: List[HybridResult]
    ) -> List[HybridResult]:
        """Apply feedback weights to hybrid results."""
        try:
            # Get feedback weights for chunks
            chunk_ids = [result.chunk.id for result in hybrid_results]
            
            weights_query = db.query(ChunkWeight).filter(
                ChunkWeight.chunk_id.in_(chunk_ids)
            ).all()
            
            weights_map = {w.chunk_id: w for w in weights_query}
            
            # Apply feedback adjustments
            for result in hybrid_results:
                weight = weights_map.get(result.chunk.id)
                
                if weight:
                    # Skip deprecated chunks
                    if weight.is_deprecated:
                        result.final_score = 0.0
                        continue
                    
                    # Apply boost/penalty
                    boost = weight.boost_weight + weight.penalty_weight
                    result.final_score = max(0.0, min(1.0, result.final_score + boost))
            
            # Re-sort after feedback application
            hybrid_results.sort(key=lambda x: x.final_score, reverse=True)
            
            # Filter out deprecated chunks
            hybrid_results = [r for r in hybrid_results if r.final_score > 0]
            
            logger.debug(f"Applied feedback to {len(weights_map)} chunks")
            return hybrid_results
            
        except Exception as e:
            logger.error(f"Error applying feedback weights: {e}")
            return hybrid_results
    
    def _hybrid_results_to_contexts(self, hybrid_results: List[HybridResult]) -> List[RetrievalContext]:
        """Convert hybrid results to retrieval contexts."""
        contexts = []
        
        for result in hybrid_results:
            context = RetrievalContext(
                content=result.chunk.content,
                doc_id=result.chunk.document_id,
                chunk_id=result.chunk.id,
                score=result.final_score,
                dense_score=result.dense_score,
                bm25_score=result.bm25_score,
                normalized_dense=result.normalized_dense,
                normalized_bm25=result.normalized_bm25,
                matched_terms=result.matched_terms,
                retrieval_method=result.retrieval_method,
                source=result.chunk.source,
                version=result.chunk.version
            )
            contexts.append(context)
        
        return contexts
    
    def _format_contexts_for_response(self, contexts: List[RetrievalContext]) -> List[Dict[str, Any]]:
        """Format contexts for API response."""
        return [
            {
                "text": ctx.content,
                "metadata": {
                    "doc_id": ctx.doc_id,
                    "chunk_id": ctx.chunk_id,
                    "source": ctx.source,
                    "version": ctx.version,
                    "retrieval_method": ctx.retrieval_method,
                    "dense_score": ctx.dense_score,
                    "bm25_score": ctx.bm25_score,
                    "matched_terms": list(ctx.matched_terms)
                },
                "score": ctx.score
            }
            for ctx in contexts
        ]
    
    def _create_no_answer_response(
        self,
        message: str,
        max_score: float = 0.0,
        explain: bool = False
    ) -> Dict[str, Any]:
        """Create response for cases where no answer can be provided."""
        response = {
            "answer": message,
            "contexts": [],
            "message_id": str(uuid.uuid4()),
            "retrieval_method": "hybrid_dense_bm25",
            "can_answer": False,
            "max_score": max_score
        }
        
        if explain:
            response["retrieval_explanation"] = {
                "reason": "no_answer",
                "details": f"Maximum retrieval score {max_score:.3f} below threshold {self.tau_retr:.3f}"
            }
        
        return response
    
    def _create_error_response(self, error_message: str, explain: bool = False) -> Dict[str, Any]:
        """Create error response."""
        response = {
            "answer": "Произошла ошибка при обработке вашего запроса.",
            "contexts": [],
            "message_id": str(uuid.uuid4()),
            "retrieval_method": "hybrid_dense_bm25",
            "can_answer": False,
            "error": error_message
        }
        
        if explain:
            response["retrieval_explanation"] = {
                "reason": "error",
                "details": error_message
            }
        
        return response
    
    def _create_retrieval_explanation(
        self,
        question: str,
        hybrid_results: List[HybridResult],
        contexts: List[RetrievalContext]
    ) -> Dict[str, Any]:
        """Create detailed retrieval explanation."""
        return {
            "query": question,
            "hybrid_parameters": {
                "alpha": self.alpha,
                "dense_weight": self.alpha,
                "bm25_weight": 1 - self.alpha,
                "answerability_threshold": self.tau_retr
            },
            "retrieval_stats": {
                "total_candidates": len(hybrid_results),
                "used_contexts": len(contexts),
                "max_score": max(r.final_score for r in hybrid_results) if hybrid_results else 0
            },
            "top_results": [
                self.hybrid_retrieval.explain_hybrid_score(result)
                for result in hybrid_results[:3]
            ],
            "bm25_index_stats": self.bm25.get_index_statistics()
        }
    
    async def generate_answer(self, query: str, contexts: List[RetrievalContext]) -> str:
        """Generate answer using LLM with hybrid context information."""
        try:
            # Prepare context data with hybrid metadata
            context_data = []
            
            for ctx in contexts:
                context_info = {
                    'content': ctx.content,
                    'doc_id': ctx.doc_id,
                    'chunk_id': ctx.chunk_id,
                    'score': ctx.score,
                    'retrieval_method': ctx.retrieval_method,
                    'metadata': ctx.metadata
                }
                
                # Add hybrid-specific metadata
                if ctx.matched_terms:
                    context_info['matched_terms'] = list(ctx.matched_terms)
                
                context_data.append(context_info)
            
            # Build enhanced prompt
            prompt = build_rag_prompt_for_ollama(query, context_data)
            
            # Generate answer with strict parameters
            answer = await self.llm.generate(
                prompt,
                temperature=0.0,  # No randomness for factual queries
                max_tokens=400
            )
            
            logger.debug(f"Generated answer: {len(answer)} characters")
            return answer
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return "Предоставлю максимально полезную информацию. Для получения дополнительных деталей рекомендую обратиться в ЦОН или на egov.kz."
    
    async def _save_message_session(
        self,
        db: Session,
        session_id: Optional[str],
        question: str,
        answer: str,
        contexts: List[RetrievalContext]
    ) -> str:
        """Save message session with hybrid metadata."""
        try:
            message_id = str(uuid.uuid4())
            
            # Format contexts with hybrid information
            contexts_data = []
            for ctx in contexts:
                context_data = {
                    "doc_id": ctx.doc_id,
                    "chunk_id": ctx.chunk_id,
                    "score": ctx.score,
                    "dense_score": ctx.dense_score,
                    "bm25_score": ctx.bm25_score,
                    "retrieval_method": ctx.retrieval_method,
                    "matched_terms": list(ctx.matched_terms),
                    "content": ctx.content[:200] + "..." if len(ctx.content) > 200 else ctx.content
                }
                contexts_data.append(context_data)
            
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
            
            logger.debug(f"Saved hybrid message session: {message_id}")
            return message_id
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error saving message session: {e}")
            return str(uuid.uuid4())
    
    async def get_pipeline_stats(self, db: Session) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics."""
        try:
            # Basic chunk statistics
            total_chunks = db.query(Chunk).count()
            chunks_with_embeddings = db.query(Chunk).filter(
                Chunk.embedding.isnot(None),
                Chunk.embedding != "[]"
            ).count()
            
            chunks_with_feedback = db.query(ChunkWeight).count()
            
            return {
                "pipeline_type": "hybrid_dense_bm25",
                "parameters": {
                    "alpha": self.alpha,
                    "tau_retr": self.tau_retr,
                    "max_contexts": self.max_contexts,
                    "chunk_size": self.chunk_size,
                    "chunk_overlap": self.chunk_overlap
                },
                "chunk_statistics": {
                    "total_chunks": total_chunks,
                    "chunks_with_embeddings": chunks_with_embeddings,
                    "chunks_with_feedback": chunks_with_feedback,
                    "embedding_coverage": chunks_with_embeddings / max(total_chunks, 1)
                },
                "hybrid_retrieval_stats": self.hybrid_retrieval.get_statistics(),
                "features": [
                    "hybrid_dense_bm25_retrieval",
                    "normalized_score_fusion",
                    "answerability_gate",
                    "feedback_learning",
                    "factual_query_optimization"
                ]
            }
            
        except Exception as e:
            logger.error(f"Error getting pipeline stats: {e}")
            return {
                "pipeline_type": "hybrid_dense_bm25",
                "error": str(e)
            }
