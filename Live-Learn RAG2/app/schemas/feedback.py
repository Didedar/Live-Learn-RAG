"""Pydantic schemas for feedback API."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator

from ..models.feedback import FeedbackLabel, FeedbackScope, UpdateStatus


class ContextInfo(BaseModel):
    """Context information from RAG retrieval."""
    doc_id: int
    chunk_id: int
    score: float
    content: str = Field(..., description="Chunk content")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AskRequest(BaseModel):
    """Request for asking a question."""
    question: str = Field(..., min_length=1, max_length=2000)
    session_id: Optional[str] = Field(None, max_length=128)
    top_k: int = Field(6, ge=1, le=20)


class AskResponse(BaseModel):
    """Response from ask endpoint."""
    message_id: str
    answer: str
    contexts: List[ContextInfo]
    session_id: Optional[str] = None


class FeedbackTarget(BaseModel):
    """Target information for feedback."""
    doc_id: Optional[int] = None
    chunk_id: Optional[int] = None

    @validator("chunk_id")
    def validate_chunk_requires_doc(cls, v, values):
        """If chunk_id is provided, doc_id should also be provided."""
        if v is not None and values.get("doc_id") is None:
            raise ValueError("doc_id is required when chunk_id is provided")
        return v


class UserFeedback(BaseModel):
    """User feedback details."""
    label: FeedbackLabel
    correction_text: Optional[str] = Field(None, max_length=5000)
    scope: FeedbackScope = FeedbackScope.CHUNK
    target: Optional[FeedbackTarget] = None
    reason: Optional[str] = Field(None, max_length=1000)

    @validator("correction_text")
    def validate_correction_text(cls, v, values):
        """Correction text is required for incorrect feedback."""
        label = values.get("label")
        if label == FeedbackLabel.INCORRECT and not v:
            raise ValueError("correction_text is required for incorrect feedback")
        return v


class FeedbackRequest(BaseModel):
    """Request for providing feedback."""
    message_id: str = Field(..., description="ID from /ask response")
    question: str = Field(..., description="Original question")
    model_answer: str = Field(..., description="Answer from RAG")
    user_feedback: UserFeedback


class UpdateInfo(BaseModel):
    """Information about knowledge base updates."""
    update_id: str
    operation: str
    status: UpdateStatus
    affected_doc_id: Optional[int] = None
    affected_chunk_id: Optional[int] = None


class FeedbackResponse(BaseModel):
    """Response from feedback endpoint."""
    status: str = Field(..., description="queued or applied")
    feedback_id: str
    update_ids: List[str] = Field(default_factory=list)
    message: Optional[str] = None


class RevertRequest(BaseModel):
    """Request to revert an update."""
    update_id: str


class RevertResponse(BaseModel):
    """Response from revert operation."""
    status: str
    reverted_update_id: str
    message: Optional[str] = None


class FeedbackHistoryResponse(BaseModel):
    """Response for feedback history."""
    feedback_id: str
    message_id: str
    question: str
    label: FeedbackLabel
    scope: FeedbackScope
    correction_text: Optional[str]
    target_doc_id: Optional[int]
    target_chunk_id: Optional[int]
    created_at: datetime
    updates: List[UpdateInfo]


class FeedbackStatsResponse(BaseModel):
    """Feedback statistics."""
    total_feedback_events: int
    feedback_by_label: Dict[str, int]
    pending_updates: int
    applied_updates: int
    failed_updates: int
    reverted_updates: int