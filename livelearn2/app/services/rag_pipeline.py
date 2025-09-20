"""Enhanced RAG pipeline with feedback-aware retrieval."""

import uuid
from typing import List, Optional, Tuple, Dict, Any

from loguru import logger
from sqlalchemy.orm import Session, joinedload

from ..config import settings
from ..models.documents import Chunk, Document
from ..models.feedback import ChunkWeight, MessageSession
from ..schemas.feedback import ContextInfo
from ..utils.text_processing import chunk_text
from ..utils.vectors import cosine_similarity, normalize_vector, batch_cosine_similarity
from .mock_embeddings import MockEmbeddings
from .ollama_llm import OllamaLLM, build_rag_prompt_for_ollama


class EnhancedRAGPipeline:
    """RAG pipeline with feedback-aware retrieval and learning capabilities."""
    
    def __init__(
        self,
        embeddings_service: Optional[MockEmbeddings] = None,
        llm_service: Optional[Any] = None
    ):
        # Always use mock embeddings (local)
        self.embeddings = embeddings_service or MockEmbeddings()
        logger.info("Using mock embeddings (local)")
        
        # Always use Ollama LLM
        if llm_service:
            self.llm = llm_service
        else:
            self.llm = OllamaLLM(
                base_url=settings.ollama_url,
                model=settings.ollama_model
            )
            logger.info(f"Using Ollama LLM: {settings.ollama_model}")
            
        self.use_ollama = True  # Always use Ollama
        logger.info("Enhanced RAG pipeline initialized (Ollama-only mode)")
    
    async def ask(
        self,
        question: str,
        db: Session,
        session_id: Optional[str] = None,
        top_k: int = 6
    ) -> Dict[str, Any]:
        """
        Main ask method that handles the complete RAG flow.
        
        Args:
            question: User question
            db: Database session
            session_id: Optional session ID for tracking
            top_k: Number of top contexts to retrieve
            
        Returns:
            Dictionary with answer and contexts
        """
        try:
            logger.info(f"Processing question: {question[:100]}...")
            
            # Retrieve relevant contexts
            contexts = await self.retrieve_with_feedback(
                db=db,
                query=question,
                top_k=top_k
            )
            
            if not contexts:
                logger.warning("No contexts found for question")
                return {
                    "answer": "Предоставлю максимально полную информацию на основе имеющихся данных. Для получения дополнительных деталей рекомендую обратиться в ЦОН или на egov.kz.",
                    "contexts": [],
                    "message_id": str(uuid.uuid4())
                }
            
            # Generate answer using contexts
            answer = await self.generate_answer(question, contexts)
            
            # Save message session
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
                        "source": getattr(ctx, 'source', 'original'),
                        "version": getattr(ctx, 'version', 1)
                    },
                    "score": ctx.score
                }
                for ctx in contexts
            ]
            
            logger.info(f"Successfully processed question, message_id: {message_id}")
            
            return {
                "answer": answer,
                "contexts": formatted_contexts,
                "message_id": message_id
            }
            
        except Exception as e:
            logger.error(f"Error in ask method: {e}")
            raise
    
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
    
    async def retrieve_with_feedback(
        self,
        db: Session,
        query: str,
        top_k: int = 6,
        score_threshold: float = 0.1
    ) -> List[Any]:
        """
        Retrieve relevant chunks with feedback-aware scoring.
        
        Args:
            db: Database session
            query: Search query
            top_k: Number of results to return
            score_threshold: Minimum similarity score
            
        Returns:
            List of context objects with scores
        """
        try:
            logger.debug(f"Retrieving contexts for query: {query[:100]}...")
            
            # Generate query embedding
            query_embedding = await self.embeddings.embed_query(query)
            
            # Get all chunks with embeddings and weights
            chunks_query = (
                db.query(Chunk)
                .outerjoin(ChunkWeight, Chunk.id == ChunkWeight.chunk_id)
                .filter(
                    Chunk.embedding.isnot(None),
                    Chunk.embedding != "[]"
                )
                .options(joinedload(Chunk.document))
            )
            
            chunks = chunks_query.all()
            
            if not chunks:
                logger.warning("No chunks with embeddings found")
                return []
            
            logger.debug(f"Found {len(chunks)} chunks with embeddings")
            
            # Calculate similarities
            embeddings = [chunk.embedding for chunk in chunks]
            similarities = batch_cosine_similarity(query_embedding, embeddings)
            
            # Apply feedback weights
            weighted_scores = []
            for i, (chunk, similarity) in enumerate(zip(chunks, similarities)):
                weight = getattr(chunk, 'weight', None)
                
                # Skip deprecated chunks
                if weight and weight.is_deprecated:
                    continue
                
                # Apply feedback weights
                final_score = similarity
                if weight:
                    final_score += weight.boost_weight
                    final_score += weight.penalty_weight
                
                # Clamp score
                final_score = max(0.0, min(1.0, final_score))
                
                if final_score >= score_threshold:
                    weighted_scores.append((chunk, final_score, i))
            
            # Sort by score and take top-k
            weighted_scores.sort(key=lambda x: x[1], reverse=True)
            top_results = weighted_scores[:top_k]
            
            # Create context objects
            contexts = []
            for chunk, score, _ in top_results:
                # Create a simple context object
                context = type('Context', (), {
                    'content': chunk.content,
                    'doc_id': chunk.document.id,
                    'chunk_id': chunk.id,
                    'score': score,
                    'metadata': chunk.document.doc_metadata,
                    'source': chunk.source,
                    'version': chunk.version
                })()
                contexts.append(context)
            
            logger.info(f"Retrieved {len(contexts)} relevant contexts")
            return contexts
            
        except Exception as e:
            logger.error(f"Error in retrieve_with_feedback: {e}")
            raise
    
    async def generate_answer(self, query: str, contexts: List[Any]) -> str:
        """
        Generate answer using LLM based on query and contexts.
        
        Args:
            query: User question
            contexts: Retrieved contexts
            
        Returns:
            Generated answer
        """
        try:
            logger.debug("Generating answer using LLM")
            
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
            
            # Build prompt for Ollama
            prompt = build_rag_prompt_for_ollama(query, context_data)
            answer = await self.llm.generate(prompt, temperature=0.1)
            
            logger.debug(f"Generated answer: {len(answer)} characters")
            return answer
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return "Предоставлю всю доступную информацию по вашему вопросу. Рекомендую также обратиться в ЦОН для получения актуальных данных."
    
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
            return str(uuid.uuid4())  # Return a fallback ID
    
    async def get_retrieval_stats(self, db: Session) -> Dict[str, Any]:
        """Get retrieval statistics."""
        try:
            total_chunks = db.query(Chunk).count()
            chunks_with_embeddings = db.query(Chunk).filter(
                Chunk.embedding.isnot(None),
                Chunk.embedding != "[]"
            ).count()
            
            chunks_with_feedback = db.query(ChunkWeight).count()
            deprecated_chunks = db.query(ChunkWeight).filter(
                ChunkWeight.is_deprecated == True
            ).count()
            
            return {
                "total_chunks": total_chunks,
                "chunks_with_embeddings": chunks_with_embeddings,
                "chunks_with_feedback": chunks_with_feedback,
                "deprecated_chunks": deprecated_chunks,
                "feedback_coverage": chunks_with_feedback / max(total_chunks, 1)
            }
            
        except Exception as e:
            logger.error(f"Error getting retrieval stats: {e}")
            return {
                "total_chunks": 0,
                "chunks_with_embeddings": 0,
                "chunks_with_feedback": 0,
                "deprecated_chunks": 0,
                "feedback_coverage": 0.0
            }