"""Hybrid RAG API endpoints with Dense + BM25 retrieval."""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from loguru import logger

from ...database import get_db
from ...schemas.rag import QueryRequest, QueryResponse
from ...services.hybrid_rag_pipeline import HybridRAGPipeline
from ...services.mock_embeddings import MockEmbeddings
from ...services.bm25_search import BM25Search


router = APIRouter(tags=["Hybrid RAG"])

# Global pipeline instance
_pipeline = None


def get_hybrid_pipeline() -> HybridRAGPipeline:
    """Get or create hybrid RAG pipeline instance."""
    global _pipeline
    
    if _pipeline is None:
        logger.info("Initializing Hybrid RAG pipeline...")
        
        # Initialize services
        embeddings = MockEmbeddings()
        bm25 = BM25Search(k1=1.5, b=0.75, epsilon=0.25)
        
        # Create pipeline with optimized parameters
        _pipeline = HybridRAGPipeline(
            embeddings_service=embeddings,
            bm25_service=bm25,
            alpha=0.6,  # 60% dense, 40% BM25 as recommended
            tau_retr=0.4,  # Answerability threshold
            max_contexts=4  # Optimal context count
        )
        
        logger.info("Hybrid RAG pipeline initialized successfully")
    
    return _pipeline


@router.post("/ask", response_model=QueryResponse)
async def ask_hybrid(
    request: QueryRequest,
    db: Session = Depends(get_db),
    pipeline: HybridRAGPipeline = Depends(get_hybrid_pipeline)
) -> QueryResponse:
    """
    Ask a question using hybrid retrieval (Dense + BM25).
    
    This endpoint combines:
    - Dense embeddings for semantic similarity
    - BM25 for exact keyword matching (dates, names, numbers, IINs, laws)
    - Normalized score fusion: score = α·z(dense) + (1-α)·z(bm25)
    - Answerability gate for quality control
    """
    try:
        logger.info(f"Hybrid RAG request: {request.query[:100]}...")
        
        # Process question with hybrid pipeline
        result = await pipeline.ask(
            question=request.query,
            db=db,
            session_id=None,  # QueryRequest doesn't have session_id
            top_k=request.top_k,
            explain=False  # Can be made configurable
        )
        
        # Convert contexts to citations
        from ...schemas.rag import Citation
        citations = []
        for ctx in result["contexts"]:
            citation = Citation(
                document_id=ctx["metadata"]["doc_id"],
                chunk_id=ctx["metadata"]["chunk_id"],
                score=ctx["score"],
                text=ctx["text"],
                metadata=ctx["metadata"]
            )
            citations.append(citation)
        
        # Convert to QueryResponse format
        response = QueryResponse(
            answer=result["answer"],
            citations=citations
        )
        
        logger.info(f"Hybrid RAG response generated: {result['message_id']}")
        return response
        
    except Exception as e:
        logger.error(f"Error in hybrid RAG: {e}")
        raise HTTPException(status_code=500, detail=f"Hybrid RAG error: {str(e)}")


@router.post("/ask/explain", response_model=dict)
async def ask_hybrid_with_explanation(
    request: QueryRequest,
    db: Session = Depends(get_db),
    pipeline: HybridRAGPipeline = Depends(get_hybrid_pipeline)
) -> dict:
    """
    Ask a question with detailed retrieval explanations.
    
    Returns detailed information about:
    - Hybrid score calculation
    - Dense vs BM25 contributions
    - Matched terms and normalization
    - Retrieval statistics
    """
    try:
        logger.info(f"Hybrid RAG explain request: {request.query[:100]}...")
        
        # Process with explanations
        result = await pipeline.ask(
            question=request.query,
            db=db,
            session_id=None,
            top_k=request.top_k,
            explain=True
        )
        
        logger.info(f"Hybrid RAG explanation generated: {result['message_id']}")
        return result
        
    except Exception as e:
        logger.error(f"Error in hybrid RAG explanation: {e}")
        raise HTTPException(status_code=500, detail=f"Hybrid RAG explanation error: {str(e)}")


@router.get("/search/{query}", response_model=dict)
async def hybrid_search(
    query: str,
    top_k: int = 5,
    db: Session = Depends(get_db),
    pipeline: HybridRAGPipeline = Depends(get_hybrid_pipeline)
) -> dict:
    """
    Perform hybrid search without answer generation.
    
    Returns raw retrieval results with detailed scoring information.
    """
    try:
        logger.info(f"Hybrid search request: {query[:100]}...")
        
        # Perform search with explanation
        result = await pipeline.hybrid_retrieval.search_with_explanation(
            db=db,
            query=query,
            top_k=top_k
        )
        
        logger.info(f"Hybrid search completed: {result['total_results']} results")
        return result
        
    except Exception as e:
        logger.error(f"Error in hybrid search: {e}")
        raise HTTPException(status_code=500, detail=f"Hybrid search error: {str(e)}")


from pydantic import BaseModel

class HybridIngestRequest(BaseModel):
    text: str
    uri: str = "inline"
    metadata: Optional[dict] = None
    rebuild_bm25: bool = True

@router.post("/ingest", response_model=dict)
async def ingest_text_hybrid(
    request: HybridIngestRequest,
    db: Session = Depends(get_db),
    pipeline: HybridRAGPipeline = Depends(get_hybrid_pipeline)
) -> dict:
    """
    Ingest text into the hybrid knowledge base.
    
    This will:
    - Create optimized chunks (350 tokens, 70 token overlap)
    - Generate dense embeddings
    - Rebuild BM25 index for keyword search
    """
    try:
        logger.info(f"Hybrid ingestion request: {len(request.text)} characters")
        
        doc_id, chunk_count = await pipeline.ingest_text(
            db=db,
            text=request.text,
            metadata=request.metadata or {},
            uri=request.uri,
            rebuild_bm25=request.rebuild_bm25
        )
        
        # Get updated statistics
        stats = await pipeline.get_pipeline_stats(db)
        
        result = {
            "document_id": doc_id,
            "chunk_count": chunk_count,
            "message": f"Successfully ingested {chunk_count} chunks",
            "bm25_rebuilt": request.rebuild_bm25,
            "pipeline_stats": stats
        }
        
        logger.info(f"Hybrid ingestion completed: doc_id={doc_id}, chunks={chunk_count}")
        return result
        
    except Exception as e:
        logger.error(f"Error in hybrid ingestion: {e}")
        raise HTTPException(status_code=500, detail=f"Hybrid ingestion error: {str(e)}")


@router.get("/stats", response_model=dict)
async def get_hybrid_stats(
    db: Session = Depends(get_db),
    pipeline: HybridRAGPipeline = Depends(get_hybrid_pipeline)
) -> dict:
    """
    Get comprehensive hybrid RAG pipeline statistics.
    
    Returns information about:
    - Pipeline parameters (α, τ_retr, etc.)
    - Chunk and embedding statistics
    - BM25 index statistics
    - Feature list
    """
    try:
        logger.info("Getting hybrid RAG statistics...")
        
        stats = await pipeline.get_pipeline_stats(db)
        
        logger.info("Hybrid RAG statistics retrieved")
        return stats
        
    except Exception as e:
        logger.error(f"Error getting hybrid stats: {e}")
        raise HTTPException(status_code=500, detail=f"Hybrid stats error: {str(e)}")


@router.post("/rebuild-bm25", response_model=dict)
async def rebuild_bm25_index(
    db: Session = Depends(get_db),
    pipeline: HybridRAGPipeline = Depends(get_hybrid_pipeline)
) -> dict:
    """
    Manually rebuild the BM25 index.
    
    Useful after bulk data ingestion or when BM25 performance degrades.
    """
    try:
        logger.info("Rebuilding BM25 index...")
        
        # Rebuild BM25 index
        pipeline.bm25.build_index(db)
        
        # Get updated statistics
        bm25_stats = pipeline.bm25.get_index_statistics()
        
        result = {
            "message": "BM25 index rebuilt successfully",
            "bm25_statistics": bm25_stats
        }
        
        logger.info("BM25 index rebuilt successfully")
        return result
        
    except Exception as e:
        logger.error(f"Error rebuilding BM25 index: {e}")
        raise HTTPException(status_code=500, detail=f"BM25 rebuild error: {str(e)}")


@router.get("/test-retrieval/{query}", response_model=dict)
async def test_retrieval_methods(
    query: str,
    top_k: int = 5,
    db: Session = Depends(get_db),
    pipeline: HybridRAGPipeline = Depends(get_hybrid_pipeline)
) -> dict:
    """
    Test and compare different retrieval methods.
    
    Returns results from:
    - Dense retrieval only
    - BM25 retrieval only  
    - Hybrid retrieval (combined)
    """
    try:
        logger.info(f"Testing retrieval methods for: {query[:100]}...")
        
        # Dense retrieval
        dense_results = await pipeline.hybrid_retrieval.dense_retrieve(
            db=db, query=query, top_k=top_k
        )
        
        # BM25 retrieval
        bm25_results = pipeline.hybrid_retrieval.bm25_retrieve(
            db=db, query=query, top_k=top_k
        )
        
        # Hybrid retrieval
        hybrid_results = await pipeline.hybrid_retrieval.hybrid_search(
            db=db, query=query, top_k=top_k
        )
        
        result = {
            "query": query,
            "dense_results": {
                "count": len(dense_results),
                "results": [
                    {
                        "chunk_id": chunk.id,
                        "score": score,
                        "content_preview": chunk.content[:100] + "..."
                    }
                    for chunk, score in dense_results[:3]
                ]
            },
            "bm25_results": {
                "count": len(bm25_results),
                "results": [
                    {
                        "chunk_id": chunk.id,
                        "score": score,
                        "matched_terms": list(matched_terms),
                        "content_preview": chunk.content[:100] + "..."
                    }
                    for chunk, score, matched_terms in bm25_results[:3]
                ]
            },
            "hybrid_results": {
                "count": len(hybrid_results),
                "results": [
                    {
                        "chunk_id": result.chunk.id,
                        "final_score": result.final_score,
                        "dense_score": result.normalized_dense,
                        "bm25_score": result.normalized_bm25,
                        "method": result.retrieval_method,
                        "matched_terms": list(result.matched_terms),
                        "content_preview": result.chunk.content[:100] + "..."
                    }
                    for result in hybrid_results[:3]
                ]
            },
            "comparison": {
                "alpha": pipeline.alpha,
                "dense_weight": pipeline.alpha,
                "bm25_weight": 1 - pipeline.alpha
            }
        }
        
        logger.info("Retrieval method comparison completed")
        return result
        
    except Exception as e:
        logger.error(f"Error in retrieval comparison: {e}")
        raise HTTPException(status_code=500, detail=f"Retrieval comparison error: {str(e)}")
