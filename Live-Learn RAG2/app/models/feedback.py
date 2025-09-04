"""Database models for feedback system."""

import uuid
from datetime import datetime
from enum import Enum
from typing import Optional

from sqlalchemy import JSON, DateTime, Enum as SQLEnum, Float, ForeignKey, Integer, String, Text, func, Boolean
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..database import Base


class FeedbackLabel(str, Enum):
    """Feedback labels for user corrections."""
    CORRECT = "correct"
    PARTIALLY_CORRECT = "partially_correct"
    INCORRECT = "incorrect"


class FeedbackScope(str, Enum):
    """Scope of feedback application."""
    CHUNK = "chunk"
    DOCUMENT = "doc"
    GLOBAL = "global"


class UpdateStatus(str, Enum):
    """Status of knowledge base updates."""
    QUEUED = "queued"
    APPLIED = "applied"
    FAILED = "failed"
    REVERTED = "reverted"


class MessageSession(Base):
    """Session tracking for question-answer pairs."""
    __tablename__ = "message_sessions"
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id: Mapped[Optional[str]] = mapped_column(String(128), index=True)
    question: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text)
    contexts_used: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    
    # Relationship to feedback events
    feedback_events: Mapped[list["FeedbackEvent"]] = relationship(
        "FeedbackEvent", back_populates="message_session", cascade="all, delete-orphan"
    )


class FeedbackEvent(Base):
    """User feedback events for learning system."""
    __tablename__ = "feedback_events"
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    message_id: Mapped[str] = mapped_column(String(36), ForeignKey("message_sessions.id"), index=True)  # ИСПРАВЛЕНО: добавлен ForeignKey
    
    # Feedback details
    label: Mapped[FeedbackLabel] = mapped_column(SQLEnum(FeedbackLabel))
    correction_text: Mapped[Optional[str]] = mapped_column(Text)
    scope: Mapped[FeedbackScope] = mapped_column(SQLEnum(FeedbackScope))
    reason: Mapped[Optional[str]] = mapped_column(Text)
    
    # Target information
    target_doc_id: Mapped[Optional[int]] = mapped_column(Integer)
    target_chunk_id: Mapped[Optional[int]] = mapped_column(Integer)
    
    # Metadata
    user_id: Mapped[Optional[str]] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    message_session: Mapped[MessageSession] = relationship("MessageSession", back_populates="feedback_events")
    index_mutations: Mapped[list["IndexMutation"]] = relationship(
        "IndexMutation", back_populates="feedback_event", cascade="all, delete-orphan"
    )


class IndexMutation(Base):
    """Track changes to the knowledge base index."""
    __tablename__ = "index_mutations"
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    feedback_event_id: Mapped[str] = mapped_column(String(36), ForeignKey("feedback_events.id"), index=True)  # ИСПРАВЛЕНО: добавлен ForeignKey
    
    # Mutation details
    operation: Mapped[str] = mapped_column(String(50))  # "tombstone", "upsert", "rerank_bias"
    status: Mapped[UpdateStatus] = mapped_column(SQLEnum(UpdateStatus), default=UpdateStatus.QUEUED)
    
    # Target information
    affected_doc_id: Mapped[Optional[int]] = mapped_column(Integer)
    affected_chunk_id: Mapped[Optional[int]] = mapped_column(Integer)
    
    # Operation data
    operation_data: Mapped[dict] = mapped_column(JSON, default=dict)
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    
    # Timing
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    applied_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    
    # Relationships
    feedback_event: Mapped[FeedbackEvent] = relationship("FeedbackEvent", back_populates="index_mutations")


class ChunkWeight(Base):
    """Feedback-based weights for chunks."""
    __tablename__ = "chunk_weights"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    chunk_id: Mapped[int] = mapped_column(Integer, ForeignKey("chunks.id"), unique=True, index=True)  # ИСПРАВЛЕНО: добавлен ForeignKey
    
    # Weight modifiers
    penalty_weight: Mapped[float] = mapped_column(Float, default=0.0)
    boost_weight: Mapped[float] = mapped_column(Float, default=0.0)
    is_deprecated: Mapped[bool] = mapped_column(Boolean, default=False)  # ИСПРАВЛЕНО: добавлен Boolean тип
    
    # Metadata
    feedback_count: Mapped[int] = mapped_column(Integer, default=0)
    last_updated: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    
    # Back reference to chunk
    chunk: Mapped["Chunk"] = relationship("Chunk", back_populates="weight")  # type: ignore