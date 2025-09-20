"""Separated RAG API endpoints with intent-based feedback."""

from typing import Dict, Any, Optional, List
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ...database import get_db
from ...services.separated_rag_pipeline import SeparatedRAGPipeline
from ...models.intent_feedback import FeedbackLabel
from loguru import logger

router = APIRouter()

# Global pipeline instance
pipeline = None


def get_pipeline():
    """Get or create the separated RAG pipeline."""
    global pipeline
    if pipeline is None:
        pipeline = SeparatedRAGPipeline()
    return pipeline


# Request/Response models
class SeparatedAskRequest(BaseModel):
    """Request for asking a question with separated feedback."""
    question: str = Field(..., min_length=1, max_length=2000)
    session_id: Optional[str] = Field(None, max_length=128)
    user_id: Optional[str] = Field(None, max_length=128)
    top_k: int = Field(6, ge=1, le=20)


class SeparatedAskResponse(BaseModel):
    """Response from separated ask endpoint."""
    message_id: str
    answer: str
    contexts: List[Dict[str, Any]]
    session_id: Optional[str] = None
    feedback_applied_count: int = 0
    separation_integrity: str = "maintained"


class SeparatedFeedbackRequest(BaseModel):
    """Request for providing separated feedback."""
    message_id: str = Field(..., description="ID from /ask response")
    feedback_label: str = Field(..., description="prefer, reject, fix, or style")
    target_doc_ids: List[int] = Field(default_factory=list, description="Document IDs this feedback applies to")
    target_chunk_ids: List[int] = Field(default_factory=list, description="Chunk IDs this feedback applies to")
    user_id: Optional[str] = Field(None, max_length=128)
    session_id: Optional[str] = Field(None, max_length=128)
    notes: Optional[str] = Field(None, max_length=1000, description="User's explanation")
    correction_text: Optional[str] = Field(None, max_length=5000, description="Correction text for FIX feedback")


class SeparatedFeedbackResponse(BaseModel):
    """Response from separated feedback endpoint."""
    status: str = "stored_separately"
    feedback_id: str
    intent_key: Optional[str] = None
    message: str = "Feedback stored in separated feedback_store"
    contamination_risk: str = "eliminated"


class IngestRequest(BaseModel):
    """Request for ingesting content into docs_index only."""
    text: str = Field(..., min_length=1)
    metadata: Optional[Dict[str, Any]] = None
    uri: str = Field("inline", max_length=500)


class IngestResponse(BaseModel):
    """Response from ingest endpoint."""
    document_id: int
    chunk_count: int
    storage_location: str = "docs_index_only"
    message: str


class FeedbackStatsResponse(BaseModel):
    """Separated feedback system statistics."""
    total_feedback_entries: int
    local_scope: int
    global_scope: int
    verified_feedback: int
    spam_filtered: int
    separation_integrity: str
    architecture: str = "intent_based_separation"


@router.post("/ask", response_model=SeparatedAskResponse)
async def ask_with_separated_feedback(
    request: SeparatedAskRequest,
    db: Session = Depends(get_db)
):
    """
    Ask a question with separated feedback application.
    
    This endpoint:
    1. Retrieves ONLY from docs_index (original documents)
    2. Applies intent-based feedback for reranking
    3. Never contaminates context with feedback content
    """
    try:
        logger.info(f"Separated ask request: {request.question[:100]}...")
        
        rag_pipeline = get_pipeline()
        
        result = await rag_pipeline.ask(
            question=request.question,
            db=db,
            session_id=request.session_id,
            user_id=request.user_id,
            top_k=request.top_k
        )
        
        return SeparatedAskResponse(
            message_id=result["message_id"],
            answer=result["answer"],
            contexts=result["contexts"],
            session_id=request.session_id,
            feedback_applied_count=result.get("feedback_applied_count", 0)
        )
        
    except Exception as e:
        logger.error(f"Error in separated ask: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing question: {str(e)}"
        )


@router.post("/feedback", response_model=SeparatedFeedbackResponse)
async def provide_separated_feedback(
    request: SeparatedFeedbackRequest,
    db: Session = Depends(get_db)
):
    """
    Provide feedback that will be stored separately from documents.
    
    This endpoint:
    1. Stores feedback in separated feedback_store
    2. Links feedback to intent keys, not document content
    3. Prevents feedback from contaminating docs_index
    """
    try:
        logger.info(f"Separated feedback request for message: {request.message_id}")
        
        # Validate feedback label
        valid_labels = ['prefer', 'reject', 'fix', 'style']
        if request.feedback_label.lower() not in valid_labels:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid feedback label. Must be one of: {valid_labels}"
            )
        
        # Validate targets
        if not request.target_doc_ids and not request.target_chunk_ids:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least one target_doc_id or target_chunk_id must be provided"
            )
        
        rag_pipeline = get_pipeline()
        
        feedback_id = await rag_pipeline.store_user_feedback(
            db=db,
            message_id=request.message_id,
            feedback_label=request.feedback_label,
            target_doc_ids=request.target_doc_ids,
            target_chunk_ids=request.target_chunk_ids,
            user_id=request.user_id,
            session_id=request.session_id,
            notes=request.notes,
            correction_text=request.correction_text
        )
        
        return SeparatedFeedbackResponse(
            feedback_id=feedback_id,
            message=f"Feedback stored separately with intent-based targeting. No contamination of docs_index."
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error storing separated feedback: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error storing feedback: {str(e)}"
        )


@router.post("/ingest", response_model=IngestResponse)
async def ingest_into_docs_index_only(
    request: IngestRequest,
    db: Session = Depends(get_db)
):
    """
    Ingest content into docs_index only.
    
    This endpoint:
    1. Stores content ONLY in docs_index
    2. Never mixes with feedback content
    3. Maintains strict separation
    """
    try:
        logger.info(f"Ingesting content into docs_index: {len(request.text)} characters")
        
        rag_pipeline = get_pipeline()
        
        document_id, chunk_count = await rag_pipeline.ingest_text(
            db=db,
            text=request.text,
            metadata=request.metadata,
            uri=request.uri
        )
        
        return IngestResponse(
            document_id=document_id,
            chunk_count=chunk_count,
            message=f"Content ingested into docs_index only. {chunk_count} chunks created."
        )
        
    except Exception as e:
        logger.error(f"Error ingesting content: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error ingesting content: {str(e)}"
        )


@router.get("/feedback/stats", response_model=FeedbackStatsResponse)
async def get_separated_feedback_stats(db: Session = Depends(get_db)):
    """
    Get statistics about the separated feedback system.
    
    This endpoint provides insights into:
    1. Feedback storage separation
    2. Intent-based targeting effectiveness
    3. System integrity metrics
    """
    try:
        rag_pipeline = get_pipeline()
        
        stats = await rag_pipeline.get_feedback_stats(db)
        
        return FeedbackStatsResponse(**stats)
        
    except Exception as e:
        logger.error(f"Error getting feedback stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting feedback stats: {str(e)}"
        )


@router.get("/health/separation")
async def check_separation_integrity(db: Session = Depends(get_db)):
    """
    Health check for separation integrity.
    
    This endpoint verifies:
    1. No feedback content in docs_index
    2. No document content in feedback_store
    3. Proper intent-based gating
    """
    try:
        from ...models.documents import Chunk
        from ...models.intent_feedback import IntentFeedback
        
        # Check for contamination in docs_index
        feedback_chunks = db.query(Chunk).filter(
            Chunk.source == "user_feedback"
        ).count()
        
        # Check feedback store integrity
        total_feedback = db.query(IntentFeedback).count()
        
        # Verify only original chunks are in docs_index
        original_chunks = db.query(Chunk).filter(
            Chunk.source == "original"
        ).count()
        
        total_chunks = db.query(Chunk).count()
        
        integrity_status = "HEALTHY" if feedback_chunks == 0 else "CONTAMINATED"
        
        return {
            "separation_integrity": integrity_status,
            "feedback_chunks_in_docs_index": feedback_chunks,
            "original_chunks": original_chunks,
            "total_chunks": total_chunks,
            "feedback_entries": total_feedback,
            "architecture": "intent_based_separation",
            "contamination_risk": "eliminated" if feedback_chunks == 0 else "detected"
        }
        
    except Exception as e:
        logger.error(f"Error checking separation integrity: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error checking integrity: {str(e)}"
        )


@router.post("/admin/promote_feedback")
async def promote_feedback_scope(
    feedback_id: str,
    new_scope: str,
    verification_count: int = 1,
    db: Session = Depends(get_db)
):
    """
    Promote feedback scope (admin endpoint).
    
    This endpoint allows administrators to:
    1. Promote local feedback to cluster/global scope
    2. Verify feedback quality
    3. Manage feedback propagation
    """
    try:
        valid_scopes = ['local', 'cluster', 'global']
        if new_scope not in valid_scopes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid scope. Must be one of: {valid_scopes}"
            )
        
        rag_pipeline = get_pipeline()
        
        from ...models.intent_feedback import FeedbackScope
        scope_mapping = {
            'local': FeedbackScope.LOCAL,
            'cluster': FeedbackScope.CLUSTER,
            'global': FeedbackScope.GLOBAL
        }
        
        success = await rag_pipeline.feedback_handler.promote_feedback_scope(
            db=db,
            feedback_id=feedback_id,
            new_scope=scope_mapping[new_scope],
            verification_count=verification_count
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Feedback not found or promotion failed"
            )
        
        return {
            "status": "promoted",
            "feedback_id": feedback_id,
            "new_scope": new_scope,
            "verification_count": verification_count,
            "message": f"Feedback promoted to {new_scope} scope"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error promoting feedback: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error promoting feedback: {str(e)}"
        )


