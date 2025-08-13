from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
from .db import get_db, Base, engine
from .schemas import IngestRequest, IngestResponse, QueryRequest, QueryResponse, Citation
from .services_pipeline import ingest_text, rag_answer

router = APIRouter(prefix="/v1")

Base.metadata.create_all(bind=engine)

@router.post("/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest, db: Session = Depends(get_db), file: UploadFile | None = File(default=None)):
    if not req.text and not file:
        raise HTTPException(status_code=400, detail="Provide 'text' or upload 'file'.")
    content = req.text
    if file:
        content = (await file.read()).decode("utf-8", errors="ignore")
    doc_id, chunks = ingest_text(db, text=content, metadata=req.metadata)
    return IngestResponse(document_id=doc_id, chunks=chunks, sync=True)

@router.post("/query", response_model=QueryResponse)
def query(req: QueryRequest, db: Session = Depends(get_db)):
    answer, ctx = rag_answer(db, query=req.query, top_k=req.top_k)
    return QueryResponse(
        answer=answer,
        citations=[Citation(**c) for c in ctx]
    )
