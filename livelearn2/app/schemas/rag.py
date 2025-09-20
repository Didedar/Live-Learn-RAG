"""Pydantic schemas for RAG API endpoints."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator


class IngestRequest(BaseModel):
    """Request for document ingestion."""
    text: Optional[str] = Field(None, description="Document text content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")
    sync: bool = Field(True, description="Process synchronously")

    @validator("text")
    def validate_text(cls, v):
        """Validate text content."""
        if v is not None and not v.strip():
            raise ValueError("Text content cannot be empty")
        return v

    @validator("metadata")
    def validate_metadata(cls, v):
        """Validate metadata structure."""
        if not isinstance(v, dict):
            raise ValueError("Metadata must be a dictionary")
        
        # Check for reasonable metadata size
        import json
        if len(json.dumps(v)) > 10000:  # 10KB limit
            raise ValueError("Metadata too large")
        
        return v


class IngestResponse(BaseModel):
    """Response from document ingestion."""
    document_id: int = Field(..., description="ID of the created document")
    chunks: int = Field(..., description="Number of chunks created")
    sync: bool = Field(..., description="Whether processing was synchronous")


class QueryRequest(BaseModel):
    """Request for knowledge base query."""
    query: str = Field(..., min_length=1, max_length=2000, description="Search query")
    top_k: int = Field(6, ge=1, le=20, description="Number of results to return")

    @validator("query")
    def validate_query(cls, v):
        """Validate query content."""
        if not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()


class Citation(BaseModel):
    """Citation information for retrieved chunks."""
    document_id: int = Field(..., description="Source document ID")
    chunk_id: int = Field(..., description="Source chunk ID")
    score: float = Field(..., description="Relevance score")
    text: str = Field(..., description="Chunk content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Chunk metadata")


class QueryResponse(BaseModel):
    """Response from knowledge base query."""
    answer: str = Field(..., description="Generated answer")
    citations: List[Citation] = Field(default_factory=list, description="Source citations")


class DocumentInfo(BaseModel):
    """Document information for listing."""
    id: int = Field(..., description="Document ID")
    uri: str = Field(..., description="Document URI")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")
    chunk_count: int = Field(..., description="Number of chunks")
    created_at: str = Field(..., description="Creation timestamp")


class DocumentListResponse(BaseModel):
    """Response for document listing."""
    documents: List[DocumentInfo] = Field(..., description="List of documents")
    total: int = Field(..., description="Total document count")
    limit: int = Field(..., description="Results limit")
    offset: int = Field(..., description="Results offset")
    has_more: bool = Field(..., description="Whether more results exist")


class DeleteResponse(BaseModel):
    """Response from document deletion."""
    message: str = Field(..., description="Deletion confirmation")
    deleted_chunks: int = Field(..., description="Number of deleted chunks")


class RAGStats(BaseModel):
    """RAG system statistics."""
    documents: int = Field(..., description="Total documents")
    chunks: int = Field(..., description="Total chunks")
    chunks_with_embeddings: int = Field(..., description="Chunks with embeddings")
    embedding_coverage: float = Field(..., description="Embedding coverage ratio")
    avg_chunks_per_document: float = Field(..., description="Average chunks per document")
    total_chunks: int = Field(..., description="Total chunks (alias)")
    chunks_with_feedback: int = Field(..., description="Chunks with feedback")
    deprecated_chunks: int = Field(..., description="Deprecated chunks")
    feedback_coverage: float = Field(..., description="Feedback coverage ratio")