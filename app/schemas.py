from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

class IngestRequest(BaseModel):
    text: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    sync: bool = True

class IngestResponse(BaseModel):
    document_id: int
    chunks: int
    sync: bool

class QueryRequest(BaseModel):
    query: str
    top_k: int = 6

class Citation(BaseModel):
    document_id: int
    chunk_id: int
    score: float
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class QueryResponse(BaseModel):
    answer: str
    citations: List[Citation]
