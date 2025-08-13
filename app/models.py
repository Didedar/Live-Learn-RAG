from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import Integer, String, Text, JSON, DateTime, func
from .db import Base

class Document(Base):
    __tablename__ = "documents"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    uri: Mapped[str] = mapped_column(String(512), default="inline")
    metadata: Mapped[dict] = mapped_column(JSON, default={})
    created_at: Mapped = mapped_column(DateTime(timezone=True), server_default=func.now())

class Chunk(Base):
    __tablename__ = "chunks"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_id: Mapped[int] = mapped_column(Integer, index=True)
    ordinal: Mapped[int] = mapped_column(Integer, index=True)
    content: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list] = mapped_column(JSON, default=[])
    created_at: Mapped = mapped_column(DateTime(timezone=True), server_default=func.now())
