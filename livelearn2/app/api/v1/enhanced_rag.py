"""Enhanced RAG API endpoints with hybrid search and improved retrieval."""

from typing import Optional, List, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, status, Query
from loguru import logger
from sqlalchemy.orm import Session

from ...core.security import optional_auth, sanitize_input
from ...database import get_db
from ...schemas.rag import QueryRequest, QueryResponse, Citation
from ...services.enhanced_rag_pipeline import EnhancedRAGPipeline
from ...services.ollama_embeddings import get_embedding_service
from ...services.keyword_search import KeywordSearchService

router = APIRouter(tags=["enhanced-rag"])

# Initialize enhanced RAG pipeline
embedding_service = get_embedding_service()
keyword_service = KeywordSearchService()
enhanced_rag_pipeline = EnhancedRAGPipeline(
    embeddings_service=embedding_service,
    keyword_service=keyword_service
)


@router.post("/query", response_model=QueryResponse)
async def enhanced_query(
    request: QueryRequest,
    db: Session = Depends(get_db),
    user_id: Optional[str] = Depends(optional_auth)
):
    """
    Enhanced query endpoint with hybrid search and improved retrieval.
    
    Features:
    - Hybrid search (dense + keyword/BM25)
    - MMR for diversity
    - Simple reranking
    - Answerability gate
    - Improved chunking strategy
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
        if request.top_k < 1 or request.top_k > 10:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="top_k must be between 1 and 10 for enhanced RAG"
            )
        
        # Process with enhanced RAG
        try:
            result = await enhanced_rag_pipeline.ask(
                question=clean_query,
                db=db,
                session_id=None,  # No session tracking for this endpoint
                top_k=request.top_k
            )
            
            if not result.get("can_answer", False):
                logger.info(
                    "Enhanced RAG refused to answer",
                    query=clean_query[:100],
                    reason=result.get("reason", "unknown"),
                    user_id=user_id
                )
                return QueryResponse(
                    answer=result["answer"],
                    citations=[]
                )
            
            # Convert contexts to citations
            citations = []
            for ctx in result.get("contexts", []):
                citations.append(Citation(
                    document_id=ctx["metadata"]["doc_id"],
                    chunk_id=ctx["metadata"]["chunk_id"],
                    score=ctx["score"],
                    text=ctx["text"],
                    metadata={
                        **ctx["metadata"],
                        "retrieval_method": result.get("retrieval_method", "enhanced"),
                        "max_score": result.get("max_score", 0.0)
                    }
                ))
            
            logger.info(
                "Enhanced query processed successfully",
                query_length=len(clean_query),
                contexts_found=len(citations),
                answer_length=len(result["answer"]),
                max_score=result.get("max_score", 0.0),
                user_id=user_id
            )
            
            return QueryResponse(
                answer=result["answer"],
                citations=citations
            )
            
        except Exception as e:
            logger.error(f"Enhanced query processing failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to process enhanced query: {str(e)}"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in enhanced query processing: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during query processing"
        )


@router.get("/debug/hybrid-retrieve")
async def debug_hybrid_retrieve(
    q: str,
    k: int = Query(default=8, ge=1, le=20),
    alpha: float = Query(default=0.6, ge=0.0, le=1.0),
    mmr_lambda: float = Query(default=0.15, ge=0.0, le=1.0),
    db: Session = Depends(get_db),
    user_id: Optional[str] = Depends(optional_auth)
):
    """
    Debug endpoint для гибридного поиска.
    
    Показывает результаты dense, keyword и hybrid поиска отдельно.
    """
    try:
        if not q or not q.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Query parameter 'q' is required"
            )
        
        clean_query = q.strip()[:500]
        logger.info(f"Debug hybrid retrieve for query: {clean_query[:100]}...")
        
        # Get dense results
        dense_results = await enhanced_rag_pipeline.dense_retrieve(db, clean_query, k=k)
        
        # Get keyword results
        keyword_results = enhanced_rag_pipeline.keyword_retrieve(db, clean_query, k=k)
        
        # Fuse results
        fused_results = enhanced_rag_pipeline.fuse_results(
            dense_results, keyword_results, alpha=alpha
        )
        
        # Apply MMR
        mmr_results = enhanced_rag_pipeline.apply_mmr(
            fused_results, clean_query, k=min(k, 6), lambda_param=mmr_lambda
        )
        
        # Format results
        def format_search_results(results, result_type: str):
            return [
                {
                    "rank": i + 1,
                    "chunk_id": result.chunk.id,
                    "doc_id": result.chunk.document_id,
                    "dense_score": round(result.dense_score, 4),
                    "keyword_score": round(result.keyword_score, 4),
                    "final_score": round(result.final_score, 4),
                    "content_preview": result.chunk.content[:150] + "..." if len(result.chunk.content) > 150 else result.chunk.content,
                    "source": result.source,
                    "result_type": result_type
                }
                for i, result in enumerate(results)
            ]
        
        return {
            "query": clean_query,
            "parameters": {
                "k": k,
                "alpha": alpha,
                "mmr_lambda": mmr_lambda,
                "embedding_service": embedding_service.get_current_service()
            },
            "results": {
                "dense_only": format_search_results(dense_results[:k], "dense"),
                "keyword_only": format_search_results(keyword_results[:k], "keyword"),
                "fused": format_search_results(fused_results[:k], "fused"),
                "mmr_final": format_search_results(mmr_results, "mmr")
            },
            "diagnostics": {
                "dense_count": len(dense_results),
                "keyword_count": len(keyword_results),
                "fused_count": len(fused_results),
                "mmr_final_count": len(mmr_results),
                "embedding_service_active": embedding_service.get_current_service()
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Debug hybrid retrieve failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Debug hybrid retrieval failed: {str(e)}"
        )


@router.get("/stats")
async def get_enhanced_stats(
    db: Session = Depends(get_db),
    user_id: Optional[str] = Depends(optional_auth)
):
    """
    Get enhanced RAG statistics.
    """
    try:
        # Get enhanced pipeline stats
        stats = await enhanced_rag_pipeline.get_retrieval_stats(db)
        
        # Get embedding service info
        embedding_info = {
            "current_service": embedding_service.get_current_service(),
            "service_healthy": await embedding_service.health_check()
        }
        
        # Add embedding-specific info if available
        if hasattr(embedding_service, 'ollama_embeddings') and embedding_service.ollama_embeddings:
            ollama_info = await embedding_service.ollama_embeddings.get_model_info()
            embedding_info.update(ollama_info)
        
        return {
            **stats,
            "embedding_service": embedding_info,
            "pipeline_features": [
                "hybrid_search",
                "mmr_diversity", 
                "answerability_gate",
                "improved_chunking",
                "ollama_embeddings" if "ollama" in embedding_service.get_current_service() else "mock_embeddings"
            ]
        }
        
    except Exception as e:
        logger.error(f"Failed to get enhanced stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve enhanced statistics"
        )


@router.post("/test-search")
async def test_search_methods(
    queries: List[str],
    db: Session = Depends(get_db),
    user_id: Optional[str] = Depends(optional_auth)
):
    """
    Test different search methods with multiple queries.
    
    Useful for comparing search quality across methods.
    """
    try:
        if not queries or len(queries) > 10:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Provide 1-10 test queries"
            )
        
        results = {}
        
        for query in queries:
            clean_query = query.strip()[:500]
            if not clean_query:
                continue
            
            try:
                # Test keyword search
                keyword_results = keyword_service.search_relevant_chunks(
                    db=db, query=clean_query, top_k=3
                )
                
                # Test dense search
                dense_results = await enhanced_rag_pipeline.dense_retrieve(
                    db, clean_query, k=3
                )
                
                # Test hybrid search
                hybrid_contexts = await enhanced_rag_pipeline.hybrid_retrieve(
                    db, clean_query, top_k=3
                )
                
                results[clean_query] = {
                    "keyword_search": {
                        "count": len(keyword_results),
                        "top_scores": [round(score, 3) for _, score in keyword_results[:3]]
                    },
                    "dense_search": {
                        "count": len(dense_results),
                        "top_scores": [round(r.dense_score, 3) for r in dense_results[:3]]
                    },
                    "hybrid_search": {
                        "count": len(hybrid_contexts),
                        "top_scores": [round(ctx.score, 3) for ctx in hybrid_contexts[:3]]
                    }
                }
                
            except Exception as e:
                results[clean_query] = {"error": str(e)}
        
        return {
            "test_queries": queries,
            "results": results,
            "embedding_service": embedding_service.get_current_service()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Search method testing failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search testing failed: {str(e)}"
        )

