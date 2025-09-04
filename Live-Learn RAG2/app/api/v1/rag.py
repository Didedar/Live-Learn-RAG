"""API endpoints for basic RAG functionality."""

from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from loguru import logger
from sqlalchemy.orm import Session

from ...core.exceptions import IngestionError, RetrievalError, ValidationError
from ...core.security import optional_auth, sanitize_input, validate_file_upload
from ...database import get_db
from ...schemas.rag import IngestRequest, IngestResponse, QueryRequest, QueryResponse, Citation
from ...services.rag_pipeline import EnhancedRAGPipeline

router = APIRouter(tags=["rag"])

# Initialize RAG pipeline
rag_pipeline = EnhancedRAGPipeline()


@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    request: IngestRequest = None,
    file: Optional[UploadFile] = File(None),
    db: Session = Depends(get_db),
    user_id: Optional[str] = Depends(optional_auth)
):
    """
    Ingest a document into the knowledge base.
    
    This endpoint supports both text input and file upload.
    The document will be chunked and embedded for later retrieval.
    """
    try:
        # Validate input
        if not request and not file:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Either 'text' in request body or file upload is required"
            )
        
        # Get content from request or file
        content = ""
        source_uri = "inline"
        
        if file:
            # Validate uploaded file
            file_content = await file.read()
            
            if not validate_file_upload(file.filename, file_content):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid file format or content"
                )
            
            # Decode file content
            try:
                content = file_content.decode('utf-8', errors='ignore')
                source_uri = file.filename
            except Exception as e:
                logger.error(f"Failed to decode file {file.filename}: {e}")
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="Could not decode file content as UTF-8"
                )
        
        elif request and request.text:
            content = request.text
            source_uri = "api_input"
        
        # Validate and sanitize content
        if not content or not content.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Document content cannot be empty"
            )
        
        try:
            content = sanitize_input(content, max_length=1000000)  # 1MB limit
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Content validation failed: {str(e)}"
            )
        
        # Prepare metadata
        metadata = {}
        if request and request.metadata:
            metadata = request.metadata
        
        # Add system metadata
        metadata.update({
            "ingested_by": user_id or "anonymous",
            "source_type": "file" if file else "api",
            "original_filename": file.filename if file else None
        })
        
        # Ingest the document
        try:
            doc_id, chunk_count = await rag_pipeline.ingest_text(
                db=db,
                text=content,
                metadata=metadata,
                uri=source_uri
            )
            
            logger.info(
                "Document ingested successfully",
                doc_id=doc_id,
                chunk_count=chunk_count,
                source_uri=source_uri,
                user_id=user_id
            )
            
            return IngestResponse(
                document_id=doc_id,
                chunks=chunk_count,
                sync=True
            )
            
        except Exception as e:
            logger.error(f"Document ingestion failed: {e}")
            raise IngestionError(f"Failed to ingest document: {str(e)}")
    
    except HTTPException:
        raise
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except IngestionError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error in document ingestion: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during document ingestion"
        )


@router.post("/query", response_model=QueryResponse)
async def query_knowledge_base(
    request: QueryRequest,
    db: Session = Depends(get_db),
    user_id: Optional[str] = Depends(optional_auth)
):
    """
    Query the knowledge base for relevant information.
    
    This is the basic query endpoint without session tracking.
    For advanced features with feedback support, use /feedback/ask instead.
    """
    try:
        # Validate query
        if not request.query or not request.query.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Query cannot be empty"
            )
        
        # Sanitize query
        try:
            clean_query = sanitize_input(request.query.strip(), max_length=2000)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Query validation failed: {str(e)}"
            )
        
        # Validate top_k
        if request.top_k < 1 or request.top_k > 20:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="top_k must be between 1 and 20"
            )
        
        # Perform retrieval and generation
        try:
            # Retrieve relevant chunks
            contexts = await rag_pipeline.retrieve_with_feedback(
                db=db,
                query=clean_query,
                top_k=request.top_k
            )
            
            if not contexts:
                logger.warning(
                    "No relevant contexts found",
                    query=clean_query[:100],
                    user_id=user_id
                )
                return QueryResponse(
                    answer="I couldn't find any relevant information to answer your question. Please try rephrasing your query or ensure the knowledge base contains relevant documents.",
                    citations=[]
                )
            
            # Generate answer
            answer = await rag_pipeline.generate_answer(clean_query, contexts)
            
            # Convert contexts to citations
            citations = [
                Citation(
                    document_id=ctx.doc_id,
                    chunk_id=ctx.chunk_id,
                    score=ctx.score,
                    text=ctx.content,
                    metadata=ctx.metadata
                )
                for ctx in contexts
            ]
            
            logger.info(
                "Query processed successfully",
                query_length=len(clean_query),
                contexts_found=len(contexts),
                answer_length=len(answer),
                user_id=user_id
            )
            
            return QueryResponse(
                answer=answer,
                citations=citations
            )
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            raise RetrievalError(f"Failed to process query: {str(e)}")
    
    except HTTPException:
        raise
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except RetrievalError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error in query processing: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during query processing"
        )


@router.get("/documents")
async def list_documents(
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db),
    user_id: Optional[str] = Depends(optional_auth)
):
    """
    List documents in the knowledge base.
    
    Returns metadata about ingested documents for management purposes.
    """
    try:
        from ...models.documents import Document
        
        # Validate parameters
        if limit < 1 or limit > 100:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="limit must be between 1 and 100"
            )
        
        if offset < 0:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="offset must be non-negative"
            )
        
        # Query documents
        documents = (
            db.query(Document)
            .order_by(Document.created_at.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )
        
        # Get total count
        total_count = db.query(Document).count()
        
        # Format response
        doc_list = []
        for doc in documents:
            chunk_count = len(doc.chunks) if doc.chunks else 0
            
            doc_list.append({
                "id": doc.id,
                "uri": doc.uri,
                "metadata": doc.metadata,
                "chunk_count": chunk_count,
                "created_at": doc.created_at.isoformat()
            })
        
        return {
            "documents": doc_list,
            "total": total_count,
            "limit": limit,
            "offset": offset,
            "has_more": (offset + limit) < total_count
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list documents: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve document list"
        )


@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: int,
    db: Session = Depends(get_db),
    user_id: Optional[str] = Depends(optional_auth)
):
    """
    Delete a document and all its chunks from the knowledge base.
    
    This operation cannot be undone.
    """
    try:
        from ...models.documents import Document
        
        # Find the document
        document = db.query(Document).filter(Document.id == document_id).first()
        
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {document_id} not found"
            )
        
        # Store info for logging
        chunk_count = len(document.chunks) if document.chunks else 0
        uri = document.uri
        
        # Delete the document (cascades to chunks)
        db.delete(document)
        db.commit()
        
        logger.info(
            "Document deleted",
            document_id=document_id,
            uri=uri,
            chunk_count=chunk_count,
            user_id=user_id
        )
        
        return {
            "message": f"Document {document_id} deleted successfully",
            "deleted_chunks": chunk_count
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to delete document {document_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete document"
        )


@router.get("/stats")
async def get_rag_stats(
    db: Session = Depends(get_db),
    user_id: Optional[str] = Depends(optional_auth)
):
    """
    Get statistics about the RAG system.
    
    Returns counts and performance metrics.
    """
    try:
        # Get basic stats
        stats = await rag_pipeline.get_retrieval_stats(db)
        
        # Add additional database stats
        from ...models.documents import Document, Chunk
        
        document_count = db.query(Document).count()
        chunk_count = db.query(Chunk).count()
        
        # Get embedding statistics
        chunks_with_embeddings = db.query(Chunk).filter(
            Chunk.embedding.isnot(None),
            Chunk.embedding != "[]"
        ).count()
        
        return {
            "documents": document_count,
            "chunks": chunk_count,
            "chunks_with_embeddings": chunks_with_embeddings,
            "embedding_coverage": chunks_with_embeddings / max(chunk_count, 1),
            "avg_chunks_per_document": chunk_count / max(document_count, 1),
            **stats
        }
        
    except Exception as e:
        logger.error(f"Failed to get RAG stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve statistics"
        )