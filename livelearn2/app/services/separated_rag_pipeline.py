"""Separated RAG pipeline with intent-based feedback isolation."""

import uuid
from typing import List, Optional, Tuple, Dict, Any

from loguru import logger
from sqlalchemy.orm import Session, joinedload

from ..config import settings
from ..models.documents import Chunk, Document
from ..models.feedback import MessageSession
from ..models.intent_feedback import IntentFeedback
from ..schemas.feedback import ContextInfo
from ..utils.text_processing import chunk_text
from ..utils.vectors import cosine_similarity, normalize_vector, batch_cosine_similarity
from .mock_embeddings import MockEmbeddings
from .ollama_llm import OllamaLLM, build_rag_prompt_for_ollama
from .intent_processor import IntentProcessor
from .separated_feedback_handler import SeparatedFeedbackHandler


class SeparatedRAGPipeline:
    """RAG pipeline with strict feedback-document separation."""
    
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
        
        # Initialize intent-based components
        self.intent_processor = IntentProcessor(self.embeddings)
        self.feedback_handler = SeparatedFeedbackHandler(
            self.embeddings, 
            self.intent_processor
        )
        
        logger.info("Separated RAG pipeline initialized with intent-based feedback")
    
    async def ask(
        self,
        question: str,
        db: Session,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        top_k: int = 6
    ) -> Dict[str, Any]:
        """
        Main ask method with separated feedback application.
        
        Args:
            question: User question
            db: Database session
            session_id: Optional session ID for tracking
            user_id: User identifier for personalized feedback
            top_k: Number of top contexts to retrieve
            
        Returns:
            Dictionary with answer and contexts
        """
        try:
            logger.info(f"Processing question: {question[:100]}...")
            
            # Step 1: Retrieve from docs_index ONLY (no feedback contamination)
            raw_contexts = await self.retrieve_from_docs_only(
                db=db,
                query=question,
                top_k=top_k * 2  # Get more candidates for feedback filtering
            )
            
            if not raw_contexts:
                logger.warning("No contexts found for question")
                return {
                    "answer": "I couldn't find any relevant information to answer your question. Please try rephrasing your query or ensure the knowledge base contains relevant documents.",
                    "contexts": [],
                    "message_id": str(uuid.uuid4())
                }
            
            # Step 2: Apply intent-based feedback for reranking
            reranked_contexts, applied_feedback_ids = await self.feedback_handler.apply_feedback_to_query(
                db=db,
                query_text=question,
                retrieved_docs=[
                    {
                        'doc_id': ctx.doc_id,
                        'chunk_id': ctx.chunk_id,
                        'score': ctx.score,
                        'content': ctx.content,
                        'metadata': getattr(ctx, 'metadata', {})
                    }
                    for ctx in raw_contexts
                ],
                user_id=user_id
            )
            
            # Step 3: Take top-k after feedback application
            final_contexts = reranked_contexts[:top_k]
            
            # Convert back to context objects
            context_objects = []
            for ctx_dict in final_contexts:
                context = type('Context', (), {
                    'content': ctx_dict['content'],
                    'doc_id': ctx_dict['doc_id'],
                    'chunk_id': ctx_dict['chunk_id'],
                    'score': ctx_dict['score'],
                    'metadata': ctx_dict.get('metadata', {}),
                    'source': 'original',  # Only original docs, never feedback
                    'version': 1
                })()
                context_objects.append(context)
            
            # Step 4: Generate answer using ONLY original document contexts
            answer = await self.generate_answer(question, context_objects)
            
            # Step 5: Save message session
            message_id = await self._save_message_session(
                db=db,
                session_id=session_id,
                question=question,
                answer=answer,
                contexts=context_objects,
                applied_feedback_ids=applied_feedback_ids
            )
            
            # Format contexts for response
            formatted_contexts = [
                {
                    "text": ctx.content,
                    "metadata": {
                        "doc_id": ctx.doc_id,
                        "chunk_id": ctx.chunk_id,
                        "source": ctx.source,
                        "version": ctx.version,
                        "feedback_applied": len(applied_feedback_ids) > 0
                    },
                    "score": ctx.score
                }
                for ctx in context_objects
            ]
            
            logger.info(f"Successfully processed question, message_id: {message_id}, feedback_applied: {len(applied_feedback_ids)}")
            
            return {
                "answer": answer,
                "contexts": formatted_contexts,
                "message_id": message_id,
                "feedback_applied_count": len(applied_feedback_ids)
            }
            
        except Exception as e:
            logger.error(f"Error in separated ask method: {e}")
            raise
    
    async def retrieve_from_docs_only(
        self,
        db: Session,
        query: str,
        top_k: int = 12,
        score_threshold: float = 0.1
    ) -> List[Any]:
        """
        Retrieve ONLY from original documents (docs_index).
        NO feedback contamination at this stage.
        
        Args:
            db: Database session
            query: Search query
            top_k: Number of results to return
            score_threshold: Minimum similarity score
            
        Returns:
            List of context objects with scores
        """
        try:
            logger.debug(f"Retrieving from docs_index only: {query[:100]}...")
            
            # Generate query embedding
            query_embedding = await self.embeddings.embed_query(query)
            
            # Get ONLY original document chunks (filter out feedback-generated content)
            chunks_query = (
                db.query(Chunk)
                .filter(
                    Chunk.embedding.isnot(None),
                    Chunk.embedding != "[]",
                    Chunk.source == "original"  # CRITICAL: Only original docs
                )
                .options(joinedload(Chunk.document))
            )
            
            chunks = chunks_query.all()
            
            if not chunks:
                logger.warning("No original chunks with embeddings found")
                return []
            
            logger.debug(f"Found {len(chunks)} original chunks with embeddings")
            
            # Calculate similarities
            embeddings = [chunk.embedding for chunk in chunks]
            similarities = batch_cosine_similarity(query_embedding, embeddings)
            
            # Create scored results WITHOUT any feedback influence
            scored_results = []
            for chunk, similarity in zip(chunks, similarities):
                if similarity >= score_threshold:
                    scored_results.append((chunk, similarity))
            
            # Sort by pure similarity score
            scored_results.sort(key=lambda x: x[1], reverse=True)
            top_results = scored_results[:top_k]
            
            # Create context objects
            contexts = []
            for chunk, score in top_results:
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
            
            logger.info(f"Retrieved {len(contexts)} pure document contexts")
            return contexts
            
        except Exception as e:
            logger.error(f"Error in docs-only retrieval: {e}")
            raise
    
    async def ingest_text(
        self,
        db: Session,
        text: str,
        metadata: Dict[str, Any] = None,
        uri: str = "inline"
    ) -> Tuple[int, int]:
        """
        Ingest text content into docs_index ONLY.
        
        Args:
            db: Database session
            text: Text content to ingest
            metadata: Document metadata
            uri: Document URI/identifier
            
        Returns:
            Tuple of (document_id, chunk_count)
        """
        try:
            logger.info(f"Ingesting text into docs_index: {len(text)} characters")
            
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
            
            # Save chunks with embeddings - ALWAYS marked as "original"
            chunk_records = []
            for i, (chunk_content, embedding) in enumerate(zip(chunks, embeddings)):
                chunk_record = Chunk(
                    document_id=document.id,
                    ordinal=i,
                    content=chunk_content,
                    embedding=embedding,
                    source="original",  # CRITICAL: Never "user_feedback"
                    version=1
                )
                chunk_records.append(chunk_record)
            
            db.add_all(chunk_records)
            db.commit()
            
            logger.info(f"Successfully ingested document {document.id} with {len(chunks)} original chunks")
            
            return document.id, len(chunks)
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error ingesting text: {e}")
            raise
    
    async def store_user_feedback(
        self,
        db: Session,
        message_id: str,
        feedback_label: str,
        target_doc_ids: List[int],
        target_chunk_ids: List[int],
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        notes: Optional[str] = None,
        correction_text: Optional[str] = None
    ) -> str:
        """
        Store user feedback in separated feedback_store.
        
        Args:
            db: Database session
            message_id: Original message ID
            feedback_label: Type of feedback (prefer, reject, fix, style)
            target_doc_ids: Document IDs this feedback applies to
            target_chunk_ids: Chunk IDs this feedback applies to
            user_id: User identifier
            session_id: Session identifier
            notes: User's explanation
            correction_text: Correction text for FIX feedback
            
        Returns:
            Feedback ID
        """
        try:
            # Get original message to extract query
            message = db.query(MessageSession).filter(
                MessageSession.id == message_id
            ).first()
            
            if not message:
                raise ValueError(f"Message {message_id} not found")
            
            # Map string labels to enum
            from ..models.intent_feedback import FeedbackLabel
            label_mapping = {
                'prefer': FeedbackLabel.PREFER,
                'reject': FeedbackLabel.REJECT,
                'fix': FeedbackLabel.FIX,
                'style': FeedbackLabel.STYLE
            }
            
            feedback_enum = label_mapping.get(feedback_label.lower())
            if not feedback_enum:
                raise ValueError(f"Invalid feedback label: {feedback_label}")
            
            # Store in separated feedback system
            feedback_id = await self.feedback_handler.store_feedback(
                db=db,
                message_id=message_id,
                query_text=message.question,
                feedback_label=feedback_enum,
                evidence_docs=target_doc_ids,
                evidence_chunks=target_chunk_ids,
                user_id=user_id,
                session_id=session_id,
                notes=notes,
                correction_text=correction_text
            )
            
            logger.info(f"Stored feedback {feedback_id} in separated feedback_store")
            return feedback_id
            
        except Exception as e:
            logger.error(f"Error storing user feedback: {e}")
            raise
    
    async def generate_answer(self, query: str, contexts: List[Any]) -> str:
        """
        Generate answer using LLM based on query and ORIGINAL contexts only.
        
        Args:
            query: User question
            contexts: Retrieved contexts (ONLY from docs_index)
            
        Returns:
            Generated answer
        """
        try:
            logger.debug("Generating answer using LLM with original contexts only")
            
            # Prepare context data - ensure no feedback contamination
            context_data = []
            for ctx in contexts:
                # Verify this is original content
                if getattr(ctx, 'source', 'original') != 'original':
                    logger.warning(f"Skipping non-original context: {ctx.source}")
                    continue
                
                context_data.append({
                    'content': ctx.content,
                    'doc_id': ctx.doc_id,
                    'chunk_id': ctx.chunk_id,
                    'score': ctx.score,
                    'metadata': getattr(ctx, 'metadata', {})
                })
            
            if not context_data:
                return "I apologize, but I couldn't find reliable source information to answer your question."
            
            # Build prompt for Ollama
            prompt = build_rag_prompt_for_ollama(query, context_data)
            answer = await self.llm.generate(prompt, temperature=0.1)
            
            logger.debug(f"Generated answer: {len(answer)} characters from {len(context_data)} original contexts")
            return answer
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return "I apologize, but I couldn't generate a proper response due to an error."
    
    async def _save_message_session(
        self,
        db: Session,
        session_id: Optional[str],
        question: str,
        answer: str,
        contexts: List[Any],
        applied_feedback_ids: List[str] = None
    ) -> str:
        """Save message session with feedback tracking."""
        try:
            message_id = str(uuid.uuid4())
            
            # Format contexts for storage
            contexts_data = [
                {
                    "doc_id": ctx.doc_id,
                    "chunk_id": ctx.chunk_id,
                    "score": ctx.score,
                    "content": ctx.content[:200] + "..." if len(ctx.content) > 200 else ctx.content,
                    "source": getattr(ctx, 'source', 'original')
                }
                for ctx in contexts
            ]
            
            # Create message session with feedback metadata
            message_session = MessageSession(
                id=message_id,
                session_id=session_id,
                question=question,
                answer=answer,
                contexts_used=contexts_data
            )
            
            # Add feedback tracking if available
            if applied_feedback_ids:
                if hasattr(message_session, 'extra_data'):
                    message_session.extra_data = {'applied_feedback_ids': applied_feedback_ids}
            
            db.add(message_session)
            db.commit()
            
            logger.debug(f"Saved message session: {message_id}")
            return message_id
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error saving message session: {e}")
            return str(uuid.uuid4())  # Return a fallback ID
    
    async def get_feedback_stats(self, db: Session) -> Dict[str, Any]:
        """Get feedback system statistics."""
        try:
            # Count feedback entries
            total_feedback = db.query(IntentFeedback).count()
            
            # Count by scope
            local_feedback = db.query(IntentFeedback).filter(
                IntentFeedback.scope == 'local'
            ).count()
            
            global_feedback = db.query(IntentFeedback).filter(
                IntentFeedback.scope == 'global'
            ).count()
            
            # Count verified feedback
            verified_feedback = db.query(IntentFeedback).filter(
                IntentFeedback.is_verified == True
            ).count()
            
            # Count spam
            spam_feedback = db.query(IntentFeedback).filter(
                IntentFeedback.is_spam == True
            ).count()
            
            return {
                "total_feedback_entries": total_feedback,
                "local_scope": local_feedback,
                "global_scope": global_feedback,
                "verified_feedback": verified_feedback,
                "spam_filtered": spam_feedback,
                "separation_integrity": "maintained"  # Always true in this architecture
            }
            
        except Exception as e:
            logger.error(f"Error getting feedback stats: {e}")
            return {
                "total_feedback_entries": 0,
                "local_scope": 0,
                "global_scope": 0,
                "verified_feedback": 0,
                "spam_filtered": 0,
                "separation_integrity": "error"
            }


