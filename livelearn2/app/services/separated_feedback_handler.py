"""Separated feedback handler with intent-based gating."""

import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Set, Tuple

from loguru import logger
from sqlalchemy.orm import Session

from ..config import settings
from ..models.documents import Chunk, Document
from ..models.intent_feedback import (
    IntentFeedback, 
    FeedbackLabel, 
    FeedbackScope,
    FeedbackApplication,
    UserFeedbackMetrics
)
from ..models.feedback import MessageSession
from .intent_processor import IntentProcessor
from .mock_embeddings import MockEmbeddings


class SeparatedFeedbackHandler:
    """Handle feedback with strict intent-based separation."""
    
    def __init__(
        self, 
        embeddings_service: Optional[MockEmbeddings] = None,
        intent_processor: Optional[IntentProcessor] = None
    ):
        self.embeddings = embeddings_service or MockEmbeddings()
        self.intent_processor = intent_processor or IntentProcessor(self.embeddings)
        
        # Configuration
        self.intent_similarity_threshold = 0.8
        self.max_feedback_age_days = 30
        self.min_trust_score = 0.5
        
        logger.info("Separated feedback handler initialized")
    
    async def store_feedback(
        self,
        db: Session,
        message_id: str,
        query_text: str,
        feedback_label: FeedbackLabel,
        evidence_docs: List[int],
        evidence_chunks: List[int],
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        notes: Optional[str] = None,
        correction_text: Optional[str] = None
    ) -> str:
        """
        Store feedback with intent-based targeting.
        
        Args:
            db: Database session
            message_id: Original message ID
            query_text: User's original query
            feedback_label: Type of feedback
            evidence_docs: Document IDs this feedback applies to
            evidence_chunks: Chunk IDs this feedback applies to
            user_id: User identifier
            session_id: Session identifier
            notes: User's explanation
            correction_text: Correction for FIX feedback
            
        Returns:
            Feedback ID
        """
        try:
            logger.info(f"Storing feedback for query: {query_text[:100]}...")
            
            # Process intent
            intent_key = await self.intent_processor.process_and_store_intent(db, query_text)
            
            # Check user trust score
            trust_score = await self._get_user_trust_score(db, user_id)
            if trust_score < self.min_trust_score:
                logger.warning(f"User {user_id} has low trust score: {trust_score}")
            
            # Determine polarity
            polarity = 1 if feedback_label in [FeedbackLabel.PREFER, FeedbackLabel.FIX] else -1
            
            # Build evidence structure
            evidence = []
            for doc_id in evidence_docs:
                evidence.append({
                    "doc_id": doc_id,
                    "chunk_id": None,
                    "type": "document"
                })
            
            for chunk_id in evidence_chunks:
                # Get chunk's document ID
                chunk = db.query(Chunk).filter(Chunk.id == chunk_id).first()
                if chunk:
                    evidence.append({
                        "doc_id": chunk.document_id,
                        "chunk_id": chunk_id,
                        "type": "chunk"
                    })
            
            # Create feedback record
            feedback = IntentFeedback(
                intent_key=intent_key,
                query_text=query_text,
                user_id=user_id,
                session_id=session_id,
                message_id=message_id,
                label=feedback_label,
                polarity=polarity,
                weight=trust_score,
                scope=FeedbackScope.LOCAL,  # Start as local
                evidence=evidence,
                notes=notes,
                correction_text=correction_text,
                confidence_score=trust_score
            )
            
            db.add(feedback)
            db.flush()
            
            # Update user metrics
            await self._update_user_metrics(db, user_id)
            
            db.commit()
            
            logger.info(f"Stored feedback {feedback.id} for intent {intent_key[:16]}...")
            return feedback.id
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error storing feedback: {e}")
            raise
    
    async def apply_feedback_to_query(
        self,
        db: Session,
        query_text: str,
        retrieved_docs: List[Dict[str, Any]],
        user_id: Optional[str] = None
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Apply relevant feedback to rerank query results.
        
        Args:
            db: Database session
            query_text: User's query
            retrieved_docs: Initial retrieval results
            user_id: User identifier
            
        Returns:
            Tuple of (reranked_docs, applied_feedback_ids)
        """
        try:
            logger.debug(f"Applying feedback to query: {query_text[:100]}...")
            
            # Get query intent
            query_intent_key = await self.intent_processor.process_and_store_intent(db, query_text)
            
            # Find applicable feedback with gating
            applicable_feedback = await self._get_applicable_feedback(
                db, query_intent_key, retrieved_docs, user_id
            )
            
            if not applicable_feedback:
                logger.debug("No applicable feedback found")
                return retrieved_docs, []
            
            logger.info(f"Found {len(applicable_feedback)} applicable feedback entries")
            
            # Apply feedback to rerank documents
            reranked_docs = await self._rerank_with_feedback(
                retrieved_docs, applicable_feedback
            )
            
            # Log application
            applied_feedback_ids = [fb.id for fb in applicable_feedback]
            await self._log_feedback_application(
                db, query_intent_key, query_text, user_id,
                retrieved_docs, reranked_docs, applied_feedback_ids
            )
            
            logger.debug(f"Applied {len(applicable_feedback)} feedback entries")
            return reranked_docs, applied_feedback_ids
            
        except Exception as e:
            logger.error(f"Error applying feedback: {e}")
            return retrieved_docs, []
    
    async def _get_applicable_feedback(
        self,
        db: Session,
        query_intent_key: str,
        retrieved_docs: List[Dict[str, Any]],
        user_id: Optional[str] = None
    ) -> List[IntentFeedback]:
        """Get feedback that passes gating conditions."""
        try:
            # Extract document IDs from retrieval results
            retrieved_doc_ids = set()
            for doc in retrieved_docs:
                if 'doc_id' in doc:
                    retrieved_doc_ids.add(doc['doc_id'])
                elif 'metadata' in doc and 'doc_id' in doc['metadata']:
                    retrieved_doc_ids.add(doc['metadata']['doc_id'])
            
            # Base query for feedback
            feedback_query = db.query(IntentFeedback).filter(
                IntentFeedback.is_spam == False,
                IntentFeedback.created_at >= datetime.utcnow() - timedelta(days=self.max_feedback_age_days)
            )
            
            # Gate 1: Intent matching (strict)
            exact_match_feedback = feedback_query.filter(
                IntentFeedback.intent_key == query_intent_key
            ).all()
            
            # Gate 2: Find similar intents if no exact match
            similar_intent_feedback = []
            if not exact_match_feedback:
                similar_intent_keys = await self.intent_processor.find_similar_intents(
                    db, query_intent_key, self.intent_similarity_threshold
                )
                
                if similar_intent_keys:
                    similar_intent_feedback = feedback_query.filter(
                        IntentFeedback.intent_key.in_(similar_intent_keys)
                    ).all()
            
            # Combine feedback sources
            all_candidate_feedback = exact_match_feedback + similar_intent_feedback
            
            # Gate 3: Evidence overlap filter
            applicable_feedback = []
            for feedback in all_candidate_feedback:
                if self._has_evidence_overlap(feedback.evidence, retrieved_doc_ids):
                    # Gate 4: User and scope filtering
                    if self._passes_user_scope_filter(feedback, user_id):
                        applicable_feedback.append(feedback)
            
            logger.debug(f"Filtered to {len(applicable_feedback)} applicable feedback entries")
            return applicable_feedback
            
        except Exception as e:
            logger.error(f"Error getting applicable feedback: {e}")
            return []
    
    def _has_evidence_overlap(
        self, 
        evidence: List[Dict[str, Any]], 
        retrieved_doc_ids: Set[int]
    ) -> bool:
        """Check if feedback evidence overlaps with retrieved documents."""
        try:
            evidence_doc_ids = set()
            for item in evidence:
                if 'doc_id' in item:
                    evidence_doc_ids.add(item['doc_id'])
            
            # Must have at least one overlapping document
            overlap = evidence_doc_ids & retrieved_doc_ids
            return len(overlap) > 0
            
        except Exception as e:
            logger.error(f"Error checking evidence overlap: {e}")
            return False
    
    def _passes_user_scope_filter(
        self, 
        feedback: IntentFeedback, 
        user_id: Optional[str]
    ) -> bool:
        """Check if feedback passes user and scope filters."""
        try:
            # Global scope - always applies
            if feedback.scope == FeedbackScope.GLOBAL:
                return True
            
            # Local scope - only for same user
            if feedback.scope == FeedbackScope.LOCAL:
                return feedback.user_id == user_id
            
            # Cluster scope - applies to similar users/contexts
            if feedback.scope == FeedbackScope.CLUSTER:
                # For now, treat as global (can be enhanced with user clustering)
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking user scope filter: {e}")
            return False
    
    async def _rerank_with_feedback(
        self,
        docs: List[Dict[str, Any]],
        feedback: List[IntentFeedback]
    ) -> List[Dict[str, Any]]:
        """Rerank documents based on feedback."""
        try:
            # Create doc_id to index mapping
            doc_positions = {}
            for i, doc in enumerate(docs):
                doc_id = doc.get('doc_id') or doc.get('metadata', {}).get('doc_id')
                if doc_id:
                    doc_positions[doc_id] = i
            
            # Apply feedback adjustments
            score_adjustments = {}
            rejected_docs = set()
            
            for fb in feedback:
                for evidence_item in fb.evidence:
                    doc_id = evidence_item.get('doc_id')
                    if doc_id not in doc_positions:
                        continue
                    
                    # Calculate adjustment based on feedback
                    adjustment = fb.polarity * fb.weight * 0.1  # Scale factor
                    
                    if fb.label == FeedbackLabel.REJECT:
                        # Strong negative signal - consider blacklisting
                        if fb.weight > 0.8:  # High confidence rejection
                            rejected_docs.add(doc_id)
                        else:
                            score_adjustments[doc_id] = score_adjustments.get(doc_id, 0) + adjustment
                    elif fb.label == FeedbackLabel.PREFER:
                        # Positive boost
                        score_adjustments[doc_id] = score_adjustments.get(doc_id, 0) + adjustment
                    elif fb.label == FeedbackLabel.FIX:
                        # Neutral - use for validation trigger
                        pass  # Could trigger re-validation logic
            
            # Apply adjustments to documents
            adjusted_docs = []
            for doc in docs:
                doc_id = doc.get('doc_id') or doc.get('metadata', {}).get('doc_id')
                
                # Skip rejected documents
                if doc_id in rejected_docs:
                    logger.debug(f"Rejecting document {doc_id} based on feedback")
                    continue
                
                # Apply score adjustment
                if doc_id in score_adjustments:
                    original_score = doc.get('score', 0.0)
                    adjusted_score = max(0.0, min(1.0, original_score + score_adjustments[doc_id]))
                    doc = {**doc, 'score': adjusted_score}
                    logger.debug(f"Adjusted doc {doc_id} score: {original_score:.3f} -> {adjusted_score:.3f}")
                
                adjusted_docs.append(doc)
            
            # Re-sort by adjusted scores
            adjusted_docs.sort(key=lambda x: x.get('score', 0.0), reverse=True)
            
            return adjusted_docs
            
        except Exception as e:
            logger.error(f"Error reranking with feedback: {e}")
            return docs
    
    async def _get_user_trust_score(
        self, 
        db: Session, 
        user_id: Optional[str]
    ) -> float:
        """Get user's trust score."""
        try:
            if not user_id:
                return 1.0  # Default trust for anonymous users
            
            metrics = db.query(UserFeedbackMetrics).filter(
                UserFeedbackMetrics.user_id == user_id
            ).first()
            
            if not metrics:
                return 1.0  # Default trust for new users
            
            return metrics.trust_score
            
        except Exception as e:
            logger.error(f"Error getting user trust score: {e}")
            return 1.0
    
    async def _update_user_metrics(
        self, 
        db: Session, 
        user_id: Optional[str]
    ) -> None:
        """Update user feedback metrics."""
        try:
            if not user_id:
                return
            
            metrics = db.query(UserFeedbackMetrics).filter(
                UserFeedbackMetrics.user_id == user_id
            ).first()
            
            if not metrics:
                metrics = UserFeedbackMetrics(
                    user_id=user_id,
                    total_feedback_count=0,
                    trust_score=1.0
                )
                db.add(metrics)
            
            metrics.total_feedback_count += 1
            metrics.last_feedback_time = datetime.utcnow()
            metrics.updated_at = datetime.utcnow()
            
            # Simple trust score calculation (can be enhanced)
            if metrics.total_feedback_count > 5:
                verification_ratio = metrics.verified_feedback_count / metrics.total_feedback_count
                spam_ratio = metrics.spam_feedback_count / metrics.total_feedback_count
                
                # Adjust trust score based on verification and spam ratios
                metrics.trust_score = min(2.0, max(0.0, 1.0 + verification_ratio - spam_ratio))
            
        except Exception as e:
            logger.error(f"Error updating user metrics: {e}")
    
    async def _log_feedback_application(
        self,
        db: Session,
        query_intent_key: str,
        query_text: str,
        user_id: Optional[str],
        original_docs: List[Dict[str, Any]],
        reranked_docs: List[Dict[str, Any]],
        applied_feedback_ids: List[str]
    ) -> None:
        """Log feedback application for analysis."""
        try:
            # Extract scores for logging
            original_scores = {}
            adjusted_scores = {}
            rejected_docs = []
            
            original_doc_ids = set()
            for doc in original_docs:
                doc_id = doc.get('doc_id') or doc.get('metadata', {}).get('doc_id')
                if doc_id:
                    original_scores[str(doc_id)] = doc.get('score', 0.0)
                    original_doc_ids.add(doc_id)
            
            reranked_doc_ids = set()
            for doc in reranked_docs:
                doc_id = doc.get('doc_id') or doc.get('metadata', {}).get('doc_id')
                if doc_id:
                    adjusted_scores[str(doc_id)] = doc.get('score', 0.0)
                    reranked_doc_ids.add(doc_id)
            
            # Find rejected documents
            rejected_docs = list(original_doc_ids - reranked_doc_ids)
            
            # Create application log
            application = FeedbackApplication(
                query_intent_key=query_intent_key,
                query_text=query_text,
                user_id=user_id,
                applied_feedback_ids=applied_feedback_ids,
                original_doc_scores=original_scores,
                adjusted_doc_scores=adjusted_scores,
                rejected_doc_ids=rejected_docs
            )
            
            db.add(application)
            db.commit()
            
        except Exception as e:
            logger.error(f"Error logging feedback application: {e}")
    
    async def promote_feedback_scope(
        self,
        db: Session,
        feedback_id: str,
        new_scope: FeedbackScope,
        verification_count: int = 1
    ) -> bool:
        """Promote feedback scope after verification."""
        try:
            feedback = db.query(IntentFeedback).filter(
                IntentFeedback.id == feedback_id
            ).first()
            
            if not feedback:
                return False
            
            # Update scope and verification
            feedback.scope = new_scope
            feedback.verification_count += verification_count
            feedback.is_verified = True
            
            # Increase weight for verified feedback
            feedback.weight = min(2.0, feedback.weight * 1.2)
            
            db.commit()
            
            logger.info(f"Promoted feedback {feedback_id} to {new_scope}")
            return True
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error promoting feedback scope: {e}")
            return False


