"""Strict RAG pipeline with single-answer-or-refuse architecture."""

import json
import uuid
from typing import List, Optional, Tuple, Dict, Any
from pydantic import BaseModel, Field

from loguru import logger
from sqlalchemy.orm import Session, joinedload

from ..config import settings
from ..models.documents import Chunk, Document
from ..models.feedback import MessageSession
from ..utils.text_processing import chunk_text
from ..utils.vectors import cosine_similarity, normalize_vector, batch_cosine_similarity
from .keyword_search import KeywordSearchService
from .ollama_llm import OllamaLLM


class LlmResponse(BaseModel):
    """Structured response from LLM."""
    can_answer: bool
    final_answer: str = Field(default="")
    evidence_ids: List[str] = Field(default_factory=list)


class StrictRAGPipeline:
    """RAG pipeline with single-answer-or-refuse architecture."""
    
    # Thresholds for answerability gate
    TAU_RETR = 0.40  # Minimum retrieval score threshold
    TAU_NLI = 0.65   # Minimum NLI entailment threshold (placeholder for now)
    
    def __init__(
        self,
        search_service: Optional[KeywordSearchService] = None,
        llm_service: Optional[Any] = None
    ):
        # Use keyword search for better retrieval
        self.search_service = search_service or KeywordSearchService()
        logger.info("Using keyword-based search for better retrieval")
        
        # Always use Ollama LLM with strict parameters
        if llm_service:
            self.llm = llm_service
        else:
            self.llm = OllamaLLM(
                base_url=settings.ollama_url,
                model=settings.ollama_model
            )
            # Override default parameters for strict generation
            self.llm.temperature = 0.0  # No randomness
            self.llm.top_p = 0.1        # Very focused sampling
            logger.info(f"Using Ollama LLM: {settings.ollama_model} (strict mode)")
        
        logger.info("Strict RAG pipeline initialized with single-answer-or-refuse architecture")
    
    async def ask(
        self,
        question: str,
        db: Session,
        session_id: Optional[str] = None,
        top_k: int = 4  # Small k as recommended
    ) -> Dict[str, Any]:
        """
        Main ask method with strict single-answer logic.
        
        Args:
            question: User question
            db: Database session
            session_id: Optional session ID for tracking
            top_k: Number of top contexts (4-6 recommended)
            
        Returns:
            Dictionary with single answer or refusal
        """
        try:
            logger.info(f"Processing strict question: {question[:100]}...")
            
            # Step 1: Build context with conflict filtering
            contexts, can_answer = await self.build_context(db, question, top_k)
            
            if not can_answer or not contexts:
                logger.info("Gate decision: Cannot answer - insufficient context")
                return {
                    "answer": "Предоставлю всю доступную информацию по вашему вопросу. Для получения дополнительных деталей рекомендую обратиться в ЦОН.",
                    "contexts": [],
                    "message_id": str(uuid.uuid4()),
                    "can_answer": False,
                    "reason": "insufficient_context"
                }
            
            # Step 2: Generate structured answer with strict format
            llm_response = await self.ask_llm_strict(question, contexts)
            
            if not llm_response.can_answer or not llm_response.final_answer.strip():
                logger.info("LLM decision: Cannot answer - no reliable information")
                return {
                    "answer": "Предоставлю всю доступную информацию по вашему вопросу. Для получения дополнительных деталей рекомендую обратиться в ЦОН.",
                    "contexts": [],
                    "message_id": str(uuid.uuid4()),
                    "can_answer": False,
                    "reason": "llm_refused"
                }
            
            # Step 3: Save message session
            message_id = await self._save_message_session(
                db=db,
                session_id=session_id,
                question=question,
                answer=llm_response.final_answer,
                contexts=contexts
            )
            
            # Format contexts for response
            formatted_contexts = [
                {
                    "text": ctx.content,
                    "metadata": {
                        "doc_id": ctx.doc_id,
                        "chunk_id": ctx.chunk_id,
                        "source": getattr(ctx, 'source', 'original'),
                        "version": getattr(ctx, 'version', 1),
                        "evidence_used": str(ctx.chunk_id) in llm_response.evidence_ids
                    },
                    "score": ctx.score
                }
                for ctx in contexts
            ]
            
            logger.info(f"Successfully provided single answer, message_id: {message_id}")
            
            return {
                "answer": llm_response.final_answer.strip(),
                "contexts": formatted_contexts,
                "message_id": message_id,
                "can_answer": True,
                "evidence_count": len(llm_response.evidence_ids)
            }
            
        except Exception as e:
            logger.error(f"Error in strict ask method: {e}")
            return {
                "answer": "Произошла ошибка при обработке вашего вопроса.",
                "contexts": [],
                "message_id": str(uuid.uuid4()),
                "can_answer": False,
                "reason": "system_error"
            }
    
    async def build_context(
        self,
        db: Session,
        query: str,
        top_k: int = 4
    ) -> Tuple[List[Any], bool]:
        """
        Build context with strict filtering and conflict detection.
        
        Args:
            db: Database session
            query: Search query
            top_k: Number of results to return
            
        Returns:
            Tuple of (contexts, can_answer_flag)
        """
        try:
            logger.debug(f"Building strict context for query: {query[:100]}...")
            
            # Step 1: Retrieve candidates with small k
            candidates = await self.retrieve_candidates(db, query, k=8)  # Get more for filtering
            
            if not candidates:
                logger.warning("No candidates found")
                return [], False
            
            # Step 2: Rerank and take top candidates
            reranked = await self.rerank_candidates(query, candidates, topk=min(4, len(candidates)))
            
            # Step 3: Filter conflicts - if top-2 have conflicts, keep only top-1
            filtered = self.filter_conflicts(reranked)
            
            # Step 4: Apply answerability gate
            max_score = max(ctx.score for ctx in filtered) if filtered else 0.0
            
            # Check retrieval score threshold
            if max_score < self.TAU_RETR:
                logger.info(f"Failed retrieval threshold: {max_score:.3f} < {self.TAU_RETR}")
                return [], False
            
            # TODO: Add NLI entailment check here when available
            # For now, we rely on retrieval score only
            nli_passed = True  # Placeholder
            
            if not nli_passed:
                logger.info("Failed NLI entailment threshold")
                return [], False
            
            # Take final top-k
            final_contexts = filtered[:top_k]
            
            logger.info(f"Built context with {len(final_contexts)} chunks, max_score: {max_score:.3f}")
            return final_contexts, True
            
        except Exception as e:
            logger.error(f"Error building context: {e}")
            return [], False
    
    async def retrieve_candidates(
        self,
        db: Session,
        query: str,
        k: int = 8,
        score_threshold: float = 0.1
    ) -> List[Any]:
        """
        Retrieve candidate chunks using keyword search.
        
        Args:
            db: Database session
            query: Search query
            k: Number of candidates to retrieve
            score_threshold: Minimum relevance score
            
        Returns:
            List of candidate context objects
        """
        try:
            # Use keyword search instead of embeddings
            scored_chunks = self.search_service.search_relevant_chunks(
                db=db,
                query=query,
                top_k=k,
                min_score=score_threshold
            )
            
            if not scored_chunks:
                logger.warning("No relevant chunks found with keyword search")
                return []
            
            # Create context objects
            contexts = []
            for chunk, score in scored_chunks:
                context = type('Context', (), {
                    'content': chunk.content,
                    'doc_id': chunk.document_id,
                    'chunk_id': chunk.id,
                    'score': score,
                    'metadata': {},  # Simplified for keyword search
                    'source': chunk.source,
                    'version': chunk.version
                })()
                contexts.append(context)
            
            logger.debug(f"Retrieved {len(contexts)} candidates with keyword search")
            return contexts
            
        except Exception as e:
            logger.error(f"Error retrieving candidates: {e}")
            return []
    
    async def rerank_candidates(
        self,
        query: str,
        candidates: List[Any],
        topk: int = 2
    ) -> List[Any]:
        """
        Rerank candidates using cross-encoder logic.
        For now, this is a placeholder that returns top candidates by score.
        
        Args:
            query: Search query
            candidates: Candidate contexts
            topk: Number of top results to return
            
        Returns:
            Reranked contexts
        """
        # TODO: Implement actual cross-encoder reranking (monoT5/bge-reranker)
        # For now, just return top candidates by similarity score
        
        sorted_candidates = sorted(candidates, key=lambda x: x.score, reverse=True)
        return sorted_candidates[:topk]
    
    def filter_conflicts(self, contexts: List[Any]) -> List[Any]:
        """
        Filter out conflicting contexts.
        If top-2 have conflicts, keep only top-1.
        
        Args:
            contexts: List of context objects
            
        Returns:
            Filtered contexts
        """
        if len(contexts) < 2:
            return contexts
        
        # Check for conflicts between top-2
        if self.has_conflict(contexts[0].content, contexts[1].content):
            logger.info("Conflict detected between top-2 contexts, keeping only top-1")
            return [contexts[0]]  # Keep only the highest scored
        
        return contexts
    
    def has_conflict(self, text1: str, text2: str) -> bool:
        """
        Simple heuristic check for conflicts between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            True if texts seem to conflict
        """
        # TODO: Implement more sophisticated conflict detection
        # For now, simple heuristics:
        
        # Check for contradictory keywords
        contradictory_pairs = [
            ("да", "нет"), ("yes", "no"),
            ("можно", "нельзя"), ("can", "cannot"),
            ("разрешено", "запрещено"), ("allowed", "forbidden"),
            ("есть", "нет"), ("have", "don't have"),
            ("включает", "не включает"), ("includes", "excludes")
        ]
        
        text1_lower = text1.lower()
        text2_lower = text2.lower()
        
        for word1, word2 in contradictory_pairs:
            if word1 in text1_lower and word2 in text2_lower:
                return True
            if word2 in text1_lower and word1 in text2_lower:
                return True
        
        return False
    
    def nli_entailment(self, query: str, passage: str) -> float:
        """
        Placeholder for NLI entailment check.
        
        Args:
            query: User query
            passage: Context passage
            
        Returns:
            Entailment score (0-1)
        """
        # TODO: Implement actual NLI cross-encoder
        # For now, return a score based on simple heuristics
        
        query_words = set(query.lower().split())
        passage_words = set(passage.lower().split())
        
        # Simple word overlap as proxy for entailment
        overlap = len(query_words.intersection(passage_words))
        return min(1.0, overlap / max(len(query_words), 1))
    
    async def ask_llm_strict(self, question: str, contexts: List[Any]) -> LlmResponse:
        """
        Ask LLM with strict structured output format.
        
        Args:
            question: User question
            contexts: Retrieved contexts
            
        Returns:
            Structured LLM response
        """
        try:
            # Build context string with IDs
            context_parts = []
            for ctx in contexts:
                context_parts.append(f"[{ctx.chunk_id}]\n{ctx.content}")
            
            context_text = "\n\n".join(context_parts)
            
            # Build strict prompt
            prompt = f"""Ты — ассистент RAG. Отвечай ровно на вопрос, используя ТОЛЬКО приведённые фрагменты.
Требования:
- Если ответа НЕТ в контексте, установи can_answer=false и не придумывай.
- Если ответ ЕСТЬ, дай единый, недвусмысленный итог без вариантов, без "или".
- Никаких списков, никаких пояснений — только один итоговый ответ.
- Следуй схеме JSON и НИЧЕГО лишнего.

Вопрос: "{question}"

Контекст (фрагменты с id):
{context_text}

Выводи строго:
{{"can_answer": <true|false>, "final_answer": "<строка>", "evidence_ids": ["<id1>", "..."]}}"""
            
            # Generate with strict parameters
            raw_response = await self.llm.generate(
                prompt, 
                temperature=0.0,  # No randomness
                max_tokens=1000   # Limit response length
            )
            
            # Parse JSON response
            try:
                # Try to extract JSON from response
                json_start = raw_response.find('{')
                json_end = raw_response.rfind('}') + 1
                
                if json_start >= 0 and json_end > json_start:
                    json_str = raw_response[json_start:json_end]
                    data = json.loads(json_str)
                    
                    return LlmResponse(
                        can_answer=data.get("can_answer", False),
                        final_answer=data.get("final_answer", ""),
                        evidence_ids=data.get("evidence_ids", [])
                    )
                else:
                    logger.warning("No valid JSON found in LLM response")
                    return LlmResponse(can_answer=False)
                    
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse LLM JSON response: {e}")
                logger.debug(f"Raw response: {raw_response}")
                return LlmResponse(can_answer=False)
            
        except Exception as e:
            logger.error(f"Error in strict LLM generation: {e}")
            return LlmResponse(can_answer=False)
    
    async def ingest_text(
        self,
        db: Session,
        text: str,
        metadata: Dict[str, Any] = None,
        uri: str = "inline"
    ) -> Tuple[int, int]:
        """
        Ingest text content into the knowledge base.
        
        Args:
            db: Database session
            text: Text content to ingest
            metadata: Document metadata
            uri: Document URI/identifier
            
        Returns:
            Tuple of (document_id, chunk_count)
        """
        try:
            logger.info(f"Ingesting text: {len(text)} characters")
            
            # Create document record
            document = Document(
                uri=uri,
                doc_metadata=metadata or {}
            )
            db.add(document)
            db.flush()  # Get document ID
            
            # Chunk the text
            chunks = chunk_text(
                text,
                max_tokens=settings.chunk_size,
                overlap=settings.chunk_overlap
            )
            
            logger.info(f"Created {len(chunks)} chunks")
            
            # Generate embeddings for chunks
            embeddings = await self.embeddings.embed_documents(chunks)
            
            # Save chunks with embeddings
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
    
    async def _save_message_session(
        self,
        db: Session,
        session_id: Optional[str],
        question: str,
        answer: str,
        contexts: List[Any]
    ) -> str:
        """Save message session for tracking."""
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
            return str(uuid.uuid4())  # Return a fallback ID
    
    async def get_retrieval_stats(self, db: Session) -> Dict[str, Any]:
        """Get retrieval statistics."""
        try:
            total_chunks = db.query(Chunk).count()
            original_chunks = db.query(Chunk).filter(
                Chunk.source == "original"
            ).count()
            chunks_with_embeddings = db.query(Chunk).filter(
                Chunk.embedding.isnot(None),
                Chunk.embedding != "[]",
                Chunk.source == "original"
            ).count()
            
            return {
                "total_chunks": total_chunks,
                "original_chunks": original_chunks,
                "chunks_with_embeddings": chunks_with_embeddings,
                "embedding_coverage": chunks_with_embeddings / max(original_chunks, 1),
                "tau_retr": self.TAU_RETR,
                "tau_nli": self.TAU_NLI,
                "pipeline_mode": "strict_single_answer"
            }
            
        except Exception as e:
            logger.error(f"Error getting retrieval stats: {e}")
            return {
                "total_chunks": 0,
                "original_chunks": 0,
                "chunks_with_embeddings": 0,
                "embedding_coverage": 0.0,
                "tau_retr": self.TAU_RETR,
                "tau_nli": self.TAU_NLI,
                "pipeline_mode": "strict_single_answer"
            }
