"""Database models for documents and chunks."""

from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, Integer, JSON, String, Text, func, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..database import Base


class Document(Base):
    """Document model for storing ingested content metadata."""
    __tablename__ = "documents"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    uri: Mapped[str] = mapped_column(String(512), default="inline")
    doc_metadata: Mapped[dict] = mapped_column(JSON, default=dict)  # metadata -> doc_metadata для избежания конфликтов
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    chunks: Mapped[list["Chunk"]] = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")


class Chunk(Base):
    """Chunk model for storing text chunks with embeddings."""
    __tablename__ = "chunks"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_id: Mapped[int] = mapped_column(Integer, ForeignKey("documents.id"), index=True)  # ИСПРАВЛЕНО: добавлен ForeignKey
    ordinal: Mapped[int] = mapped_column(Integer, index=True)
    content: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list] = mapped_column(JSON, default=list)
    
    # Source tracking for feedback
    source: Mapped[str] = mapped_column(String(50), default="original")  # "original", "user_feedback"
    version: Mapped[int] = mapped_column(Integer, default=1)
    
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships - ИСПРАВЛЕНО: правильная настройка relationship
    document: Mapped[Document] = relationship("Document", back_populates="chunks")