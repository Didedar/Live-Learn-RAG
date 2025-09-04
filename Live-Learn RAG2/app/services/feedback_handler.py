"""Feedback handling service for learning from user corrections."""

import uuid
from datetime import datetime
from typing import List, Optional, Tuple, Dict, Any

from loguru import logger
from sqlalchemy.orm import Session

from ..config import settings
from ..models.documents import Chunk, Document
from ..models.feedback import (
    ChunkWeight,
    FeedbackEvent,
    FeedbackLabel,
    FeedbackScope,
    IndexMutation,
    MessageSession,
    UpdateStatus,
)
from ..schemas.feedback import FeedbackRequest, UpdateInfo, UserFeedback
from ..utils.text_processing import chunk_text
from .embeddings import GoogleEmbeddings


class FeedbackHandler:
    """Handle user feedback and update knowledge base."""
    
    def __init__(self, embeddings_service: Optional[GoogleEmbeddings] = None):
        self.embeddings = embeddings_service or GoogleEmbeddings()
        logger.info("Feedback handler initialized")
    
    async def process_feedback(
        self,
        db: Session,
        feedback_request: FeedbackRequest
    ) -> Dict[str, Any]:
        """
        Process user feedback and update knowledge base.
        
        Args:
            db: Database session
            feedback_request: Feedback request data
            
        Returns:
            Processing result with update information
        """
        try:
            logger.info(f"Processing feedback for message: {feedback_request.message_id}")
            
            # Verify message exists
            message_session = db.query(MessageSession).filter(
                MessageSession.id == feedback_request.message_id
            ).first()
            
            if not message_session:
                raise ValueError(f"Message {feedback_request.message_id} not found")
            
            # Create feedback event
            feedback_event = await self._create_feedback_event(
                db, feedback_request, message_session
            )
            
            # Process the feedback based on type
            update_ids = await self._apply_feedback(db, feedback_event)
            
            db.commit()
            
            logger.info(f"Processed feedback successfully, created {len(update_ids)} updates")
            
            return {
                "status": "applied" if update_ids else "queued",
                "feedback_id": feedback_event.id,
                "update_ids": update_ids,
                "message": f"Feedback processed with {len(update_ids)} updates"
            }
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error processing feedback: {e}")
            raise
    
    async def _create_feedback_event(
        self,
        db: Session,
        feedback_request: FeedbackRequest,
        message_session: MessageSession
    ) -> FeedbackEvent:
        """Create feedback event record."""
        try:
            user_feedback = feedback_request.user_feedback
            
            feedback_event = FeedbackEvent(
                id=str(uuid.uuid4()),
                message_id=message_session.id,
                label=user_feedback.label,
                correction_text=user_feedback.correction_text,
                scope=user_feedback.scope,
                reason=user_feedback.reason,
                target_doc_id=user_feedback.target.doc_id if user_feedback.target else None,
                target_chunk_id=user_feedback.target.chunk_id if user_feedback.target else None,
                user_id=None  # Could be extracted from request context
            )
            
            db.add(feedback_event)
            db.flush()  # Get the ID
            
            logger.debug(f"Created feedback event: {feedback_event.id}")
            return feedback_event
            
        except Exception as e:
            logger.error(f"Error creating feedback event: {e}")
            raise
    
    async def _apply_feedback(
        self,
        db: Session,
        feedback_event: FeedbackEvent
    ) -> List[str]:
        """Apply feedback to update the knowledge base."""
        try:
            update_ids = []
            
            if feedback_event.label == FeedbackLabel.INCORRECT:
                # Handle incorrect feedback
                if feedback_event.scope == FeedbackScope.CHUNK:
                    update_ids.extend(
                        await self._handle_incorrect_chunk_feedback(db, feedback_event)
                    )
                elif feedback_event.scope == FeedbackScope.DOCUMENT:
                    update_ids.extend(
                        await self._handle_incorrect_document_feedback(db, feedback_event)
                    )
                
                # If correction text is provided, potentially add new content
                if feedback_event.correction_text:
                    update_ids.extend(
                        await self._add_correction_content(db, feedback_event)
                    )
            
            elif feedback_event.label == FeedbackLabel.PARTIALLY_CORRECT:
                # Handle partially correct feedback with adjustments
                update_ids.extend(
                    await self._handle_partial_feedback(db, feedback_event)
                )
            
            elif feedback_event.label == FeedbackLabel.CORRECT:
                # Boost the relevant chunks
                update_ids.extend(
                    await self._handle_correct_feedback(db, feedback_event)
                )
            
            return update_ids
            
        except Exception as e:
            logger.error(f"Error applying feedback: {e}")
            return []
    
    async def _handle_incorrect_chunk_feedback(
        self,
        db: Session,
        feedback_event: FeedbackEvent
    ) -> List[str]:
        """Handle feedback indicating a chunk is incorrect."""
        try:
            update_ids = []
            
            if feedback_event.target_chunk_id:
                # Apply penalty to specific chunk
                update_id = await self._apply_chunk_penalty(
                    db, feedback_event, feedback_event.target_chunk_id
                )
                update_ids.append(update_id)
            else:
                # Apply penalty to all chunks used in the message
                message_session = db.query(MessageSession).filter(
                    MessageSession.id == feedback_event.message_id
                ).first()
                
                if message_session and message_session.contexts_used:
                    for context in message_session.contexts_used:
                        chunk_id = context.get('chunk_id')
                        if chunk_id:
                            update_id = await self._apply_chunk_penalty(
                                db, feedback_event, chunk_id
                            )
                            update_ids.append(update_id)
            
            return update_ids
            
        except Exception as e:
            logger.error(f"Error handling incorrect chunk feedback: {e}")
            return []
    
    async def _handle_incorrect_document_feedback(
        self,
        db: Session,
        feedback_event: FeedbackEvent
    ) -> List[str]:
        """Handle feedback indicating a document is incorrect."""
        try:
            update_ids = []
            
            if feedback_event.target_doc_id:
                # Apply penalty to all chunks in the document
                chunks = db.query(Chunk).filter(
                    Chunk.document_id == feedback_event.target_doc_id
                ).all()
                
                for chunk in chunks:
                    update_id = await self._apply_chunk_penalty(
                        db, feedback_event, chunk.id
                    )
                    update_ids.append(update_id)
            
            return update_ids
            
        except Exception as e:
            logger.error(f"Error handling incorrect document feedback: {e}")
            return []
    
    async def _handle_partial_feedback(
        self,
        db: Session,
        feedback_event: FeedbackEvent
    ) -> List[str]:
        """Handle partially correct feedback."""
        try:
            update_ids = []
            
            # Apply smaller penalty for partial incorrectness
            penalty_weight = settings.feedback_penalty_weight * 0.5
            
            if feedback_event.target_chunk_id:
                update_id = await self._apply_chunk_weight(
                    db, feedback_event, feedback_event.target_chunk_id, 
                    penalty=penalty_weight
                )
                update_ids.append(update_id)
            
            return update_ids
            
        except Exception as e:
            logger.error(f"Error handling partial feedback: {e}")
            return []
    
    async def _handle_correct_feedback(
        self,
        db: Session,
        feedback_event: FeedbackEvent
    ) -> List[str]:
        """Handle correct feedback by boosting relevant chunks."""
        try:
            update_ids = []
            
            if feedback_event.target_chunk_id:
                update_id = await self._apply_chunk_boost(
                    db, feedback_event, feedback_event.target_chunk_id
                )
                update_ids.append(update_id)
            else:
                # Boost all chunks used in the message
                message_session = db.query(MessageSession).filter(
                    MessageSession.id == feedback_event.message_id
                ).first()
                
                if message_session and message_session.contexts_used:
                    for context in message_session.contexts_used:
                        chunk_id = context.get('chunk_id')
                        if chunk_id:
                            update_id = await self._apply_chunk_boost(
                                db, feedback_event, chunk_id
                            )
                            update_ids.append(update_id)
            
            return update_ids
            
        except Exception as e:
            logger.error(f"Error handling correct feedback: {e}")
            return []
    
    async def _apply_chunk_penalty(
        self,
        db: Session,
        feedback_event: FeedbackEvent,
        chunk_id: int
    ) -> str:
        """Apply penalty weight to a chunk."""
        return await self._apply_chunk_weight(
            db, feedback_event, chunk_id,
            penalty=settings.feedback_penalty_weight
        )
    
    async def _apply_chunk_boost(
        self,
        db: Session,
        feedback_event: FeedbackEvent,
        chunk_id: int
    ) -> str:
        """Apply boost weight to a chunk."""
        return await self._apply_chunk_weight(
            db, feedback_event, chunk_id,
            boost=settings.feedback_boost_weight
        )
    
    async def _apply_chunk_weight(
        self,
        db: Session,
        feedback_event: FeedbackEvent,
        chunk_id: int,
        penalty: float = 0.0,
        boost: float = 0.0
    ) -> str:
        """Apply weight adjustment to a chunk."""
        try:
            # Get or create chunk weight
            chunk_weight = db.query(ChunkWeight).filter(
                ChunkWeight.chunk_id == chunk_id
            ).first()
            
            if not chunk_weight:
                chunk_weight = ChunkWeight(
                    chunk_id=chunk_id,
                    penalty_weight=0.0,
                    boost_weight=0.0,
                    feedback_count=0
                )
                db.add(chunk_weight)
            
            # Apply adjustments
            chunk_weight.penalty_weight += penalty
            chunk_weight.boost_weight += boost
            chunk_weight.feedback_count += 1
            chunk_weight.last_updated = datetime.utcnow()
            
            # Create index mutation record
            mutation = IndexMutation(
                id=str(uuid.uuid4()),
                feedback_event_id=feedback_event.id,
                operation="rerank_bias",
                status=UpdateStatus.APPLIED,
                affected_chunk_id=chunk_id,
                operation_data={
                    "penalty_adjustment": penalty,
                    "boost_adjustment": boost,
                    "new_penalty_weight": chunk_weight.penalty_weight,
                    "new_boost_weight": chunk_weight.boost_weight
                },
                applied_at=datetime.utcnow()
            )
            
            db.add(mutation)
            db.flush()
            
            logger.debug(f"Applied weight adjustment to chunk {chunk_id}")
            return mutation.id
            
        except Exception as e:
            logger.error(f"Error applying chunk weight: {e}")
            raise
    
    async def _add_correction_content(
        self,
        db: Session,
        feedback_event: FeedbackEvent
    ) -> List[str]:
        """Add correction content as new chunks."""
        try:
            if not feedback_event.correction_text:
                return []
            
            update_ids = []
            
            # Create a new document for the correction
            correction_doc = Document(
                uri=f"correction_{feedback_event.id}",
                doc_metadata={
                    "type": "user_correction",
                    "original_message_id": feedback_event.message_id,
                    "feedback_event_id": feedback_event.id,
                    "created_from": "user_feedback"
                }
            )
            
            db.add(correction_doc)
            db.flush()
            
            # Chunk the correction text
            chunks = chunk_text(
                feedback_event.correction_text,
                max_tokens=settings.chunk_size,
                overlap=settings.chunk_overlap
            )
            
            # Generate embeddings
            embeddings = await self.embeddings.embed_documents(chunks)
            
            # Create chunk records
            for i, (chunk_text, embedding) in enumerate(zip(chunks, embeddings)):
                chunk = Chunk(
                    document_id=correction_doc.id,
                    ordinal=i,
                    content=chunk_text,
                    embedding=embedding,
                    source="user_feedback",
                    version=1
                )
                db.add(chunk)
                db.flush()
                
                # Create index mutation
                mutation = IndexMutation(
                    id=str(uuid.uuid4()),
                    feedback_event_id=feedback_event.id,
                    operation="upsert",
                    status=UpdateStatus.APPLIED,
                    affected_doc_id=correction_doc.id,
                    affected_chunk_id=chunk.id,
                    operation_data={
                        "chunk_content": chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text,
                        "source": "user_feedback"
                    },
                    applied_at=datetime.utcnow()
                )
                
                db.add(mutation)
                db.flush()
                update_ids.append(mutation.id)
            
            logger.info(f"Added {len(chunks)} correction chunks from feedback")
            return update_ids
            
        except Exception as e:
            logger.error(f"Error adding correction content: {e}")
            return []
    
    async def revert_update(
        self,
        db: Session,
        update_id: str
    ) -> bool:
        """Revert a feedback update."""
        try:
            mutation = db.query(IndexMutation).filter(
                IndexMutation.id == update_id
            ).first()
            
            if not mutation:
                logger.warning(f"Update {update_id} not found")
                return False
            
            if mutation.status == UpdateStatus.REVERTED:
                logger.warning(f"Update {update_id} already reverted")
                return False
            
            # Revert based on operation type
            if mutation.operation == "rerank_bias":
                await self._revert_weight_adjustment(db, mutation)
            elif mutation.operation == "upsert":
                await self._revert_content_addition(db, mutation)
            elif mutation.operation == "tombstone":
                await self._revert_content_removal(db, mutation)
            
            # Mark as reverted
            mutation.status = UpdateStatus.REVERTED
            db.commit()
            
            logger.info(f"Successfully reverted update {update_id}")
            return True
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error reverting update {update_id}: {e}")
            return False
    
    async def _revert_weight_adjustment(
        self,
        db: Session,
        mutation: IndexMutation
    ) -> None:
        """Revert weight adjustments."""
        if not mutation.affected_chunk_id:
            return
        
        chunk_weight = db.query(ChunkWeight).filter(
            ChunkWeight.chunk_id == mutation.affected_chunk_id
        ).first()
        
        if chunk_weight:
            operation_data = mutation.operation_data
            penalty_adj = operation_data.get("penalty_adjustment", 0.0)
            boost_adj = operation_data.get("boost_adjustment", 0.0)
            
            # Reverse the adjustments
            chunk_weight.penalty_weight -= penalty_adj
            chunk_weight.boost_weight -= boost_adj
            chunk_weight.feedback_count = max(0, chunk_weight.feedback_count - 1)
            chunk_weight.last_updated = datetime.utcnow()
    
    async def _revert_content_addition(
        self,
        db: Session,
        mutation: IndexMutation
    ) -> None:
        """Revert content addition by marking chunks as deprecated."""
        if mutation.affected_chunk_id:
            chunk = db.query(Chunk).filter(
                Chunk.id == mutation.affected_chunk_id
            ).first()
            
            if chunk:
                # Mark chunk as deprecated instead of deleting
                chunk_weight = db.query(ChunkWeight).filter(
                    ChunkWeight.chunk_id == chunk.id
                ).first()
                
                if not chunk_weight:
                    chunk_weight = ChunkWeight(chunk_id=chunk.id)
                    db.add(chunk_weight)
                
                chunk_weight.is_deprecated = True
                chunk_weight.last_updated = datetime.utcnow()
    
    async def _revert_content_removal(
        self,
        db: Session,
        mutation: IndexMutation
    ) -> None:
        """Revert content removal by un-deprecating chunks."""
        if mutation.affected_chunk_id:
            chunk_weight = db.query(ChunkWeight).filter(
                ChunkWeight.chunk_id == mutation.affected_chunk_id
            ).first()
            
            if chunk_weight:
                chunk_weight.is_deprecated = False
                chunk_weight.last_updated = datetime.utcnow()