"""Intent-based feedback models for separated storage architecture."""

import uuid
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any

from sqlalchemy import JSON, DateTime, Enum as SQLEnum, Float, ForeignKey, Integer, String, Text, func, Boolean, Index
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..database import Base


class FeedbackLabel(str, Enum):
    """Feedback labels for user corrections."""
    PREFER = "prefer"      # Предпочесть этот результат
    REJECT = "reject"      # Отклонить этот результат
    FIX = "fix"           # Исправление с объяснением
    STYLE = "style"       # Стилистическое предпочтение


class FeedbackScope(str, Enum):
    """Scope of feedback application."""
    LOCAL = "local"       # Только для этого пользователя
    CLUSTER = "cluster"   # Для похожих намерений
    GLOBAL = "global"     # Глобально


class IntentKey(Base):
    """Normalized intent keys for feedback targeting."""
    __tablename__ = "intent_keys"
    
    id: Mapped[str] = mapped_column(String(64), primary_key=True)  # SHA-256 hash
    normalized_text: Mapped[str] = mapped_column(Text, index=True)
    entities: Mapped[List[str]] = mapped_column(JSON, default=list)  # Extracted entities
    tokens: Mapped[List[str]] = mapped_column(JSON, default=list)   # Normalized tokens
    embedding: Mapped[List[float]] = mapped_column(JSON)            # Intent embedding
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    feedback_entries: Mapped[List["IntentFeedback"]] = relationship(
        "IntentFeedback", back_populates="intent_key_obj"
    )


class IntentFeedback(Base):
    """Intent-specific feedback storage (separate from docs)."""
    __tablename__ = "intent_feedback"
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Intent binding
    intent_key: Mapped[str] = mapped_column(String(64), ForeignKey("intent_keys.id"), index=True)
    query_text: Mapped[str] = mapped_column(Text)  # Original query for debugging
    
    # User and session info
    user_id: Mapped[Optional[str]] = mapped_column(String(128), index=True)
    session_id: Mapped[Optional[str]] = mapped_column(String(128), index=True)
    message_id: Mapped[str] = mapped_column(String(36), index=True)  # Link to original message
    
    # Feedback details
    label: Mapped[FeedbackLabel] = mapped_column(SQLEnum(FeedbackLabel))
    polarity: Mapped[int] = mapped_column(Integer)  # +1 for positive, -1 for negative
    weight: Mapped[float] = mapped_column(Float, default=1.0)  # Trust/confidence weight
    scope: Mapped[FeedbackScope] = mapped_column(SQLEnum(FeedbackScope), default=FeedbackScope.LOCAL)
    
    # Evidence linking (what docs/chunks this applies to)
    evidence: Mapped[List[Dict[str, Any]]] = mapped_column(JSON, default=list)  # [{"doc_id": 45, "chunk_id": 3, "offsets": [...]}]
    
    # Feedback content
    notes: Mapped[Optional[str]] = mapped_column(Text)  # User's explanation
    correction_text: Mapped[Optional[str]] = mapped_column(Text)  # For FIX label
    
    # Quality control
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False)
    verification_count: Mapped[int] = mapped_column(Integer, default=0)  # How many users confirmed
    is_spam: Mapped[bool] = mapped_column(Boolean, default=False)
    confidence_score: Mapped[float] = mapped_column(Float, default=1.0)
    
    # TTL and decay
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))  # For TTL
    decay_factor: Mapped[float] = mapped_column(Float, default=1.0)  # Reduces over time
    
    # Relationships
    intent_key_obj: Mapped[IntentKey] = relationship("IntentKey", back_populates="feedback_entries")
    
    # Indexes for performance
    __table_args__ = (
        Index('ix_intent_feedback_user_intent', 'user_id', 'intent_key'),
        Index('ix_intent_feedback_scope_intent', 'scope', 'intent_key'),
        Index('ix_intent_feedback_created', 'created_at'),
    )


class FeedbackApplication(Base):
    """Track when and how feedback was applied to queries."""
    __tablename__ = "feedback_applications"
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Query info
    query_intent_key: Mapped[str] = mapped_column(String(64), index=True)
    query_text: Mapped[str] = mapped_column(Text)
    user_id: Mapped[Optional[str]] = mapped_column(String(128), index=True)
    
    # Applied feedback
    applied_feedback_ids: Mapped[List[str]] = mapped_column(JSON, default=list)
    
    # Results
    original_doc_scores: Mapped[Dict[str, float]] = mapped_column(JSON, default=dict)  # Before feedback
    adjusted_doc_scores: Mapped[Dict[str, float]] = mapped_column(JSON, default=dict)  # After feedback
    rejected_doc_ids: Mapped[List[int]] = mapped_column(JSON, default=list)  # Blacklisted docs
    
    # Metadata
    applied_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    
    # Indexes
    __table_args__ = (
        Index('ix_feedback_applications_intent_user', 'query_intent_key', 'user_id'),
        Index('ix_feedback_applications_applied_at', 'applied_at'),
    )


class FeedbackCluster(Base):
    """Clusters of similar intents for feedback propagation."""
    __tablename__ = "feedback_clusters"
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Cluster info
    cluster_name: Mapped[Optional[str]] = mapped_column(String(255))
    center_embedding: Mapped[List[float]] = mapped_column(JSON)  # Cluster centroid
    threshold: Mapped[float] = mapped_column(Float, default=0.8)  # Similarity threshold
    
    # Member intents
    intent_keys: Mapped[List[str]] = mapped_column(JSON, default=list)
    
    # Statistics
    member_count: Mapped[int] = mapped_column(Integer, default=0)
    feedback_count: Mapped[int] = mapped_column(Integer, default=0)
    
    # Metadata
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class UserFeedbackMetrics(Base):
    """Track user feedback quality and reputation."""
    __tablename__ = "user_feedback_metrics"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    
    # Quality metrics
    total_feedback_count: Mapped[int] = mapped_column(Integer, default=0)
    verified_feedback_count: Mapped[int] = mapped_column(Integer, default=0)
    spam_feedback_count: Mapped[int] = mapped_column(Integer, default=0)
    
    # Reputation
    trust_score: Mapped[float] = mapped_column(Float, default=1.0)  # 0.0 to 2.0
    is_trusted: Mapped[bool] = mapped_column(Boolean, default=False)
    is_blocked: Mapped[bool] = mapped_column(Boolean, default=False)
    
    # Rate limiting
    last_feedback_time: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    feedback_rate_per_hour: Mapped[float] = mapped_column(Float, default=0.0)
    
    # Metadata
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


