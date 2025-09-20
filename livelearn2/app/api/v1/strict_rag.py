"""API endpoints for strict RAG with single-answer-or-refuse architecture."""

from typing import Optional, List, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger
from sqlalchemy.orm import Session

from ...core.exceptions import RetrievalError, ValidationError
from ...core.security import optional_auth, sanitize_input
from ...database import get_db
from ...schemas.rag import QueryRequest, QueryResponse, Citation
from ...services.strict_rag_pipeline import StrictRAGPipeline

router = APIRouter(tags=["strict-rag"])

# Initialize strict RAG pipeline
strict_rag_pipeline = StrictRAGPipeline()


@router.post("/ask", response_model=Dict[str, Any])
async def ask_strict(
    request: QueryRequest,
    db: Session = Depends(get_db),
    user_id: Optional[str] = Depends(optional_auth)
):
    """
    Строгий RAG endpoint с архитектурой "один ответ или отказ".
    
    Особенности:
    - Малый k (4-6) для ретривера
    - Фильтр противоречий
    - Gate для проверки ответимости
    - Строгое структурированное генерирование
    - Только один четкий ответ или честный отказ
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
        
        # Validate and clamp top_k for strict mode (4-6 recommended)
        top_k = min(max(request.top_k, 1), 6)  # Clamp to 1-6 range
        if top_k != request.top_k:
            logger.info(f"Clamped top_k from {request.top_k} to {top_k} for strict mode")
        
        # Process query with strict RAG
        try:
            result = await strict_rag_pipeline.ask(
                question=clean_query,
                db=db,
                session_id=None,  # No session tracking in basic endpoint
                top_k=top_k
            )
            
            logger.info(
                "Strict query processed",
                query_length=len(clean_query),
                can_answer=result["can_answer"],
                contexts_used=len(result.get("contexts", [])),
                user_id=user_id
            )
            
            # Return structured response
            return {
                "answer": result["answer"],
                "can_answer": result["can_answer"],
                "message_id": result["message_id"],
                "contexts": result.get("contexts", []),
                "metadata": {
                    "pipeline_mode": "strict_single_answer",
                    "top_k_used": top_k,
                    "reason": result.get("reason"),
                    "evidence_count": result.get("evidence_count", 0)
                }
            }
            
        except Exception as e:
            logger.error(f"Strict query processing failed: {e}")
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
        logger.error(f"Unexpected error in strict query processing: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during query processing"
        )


@router.post("/ask-with-session")
async def ask_strict_with_session(
    request: Dict[str, Any],
    db: Session = Depends(get_db),
    user_id: Optional[str] = Depends(optional_auth)
):
    """
    Строгий RAG с отслеживанием сессии для фидбека.
    
    Принимает:
    - query: вопрос пользователя
    - session_id: идентификатор сессии (опционально)
    - top_k: количество контекстов (1-6, по умолчанию 4)
    """
    try:
        # Extract parameters
        query = request.get("query", "").strip()
        session_id = request.get("session_id")
        top_k = min(max(request.get("top_k", 4), 1), 6)
        
        if not query:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Query cannot be empty"
            )
        
        # Sanitize query
        clean_query = sanitize_input(query, max_length=2000)
        
        # Process with session tracking
        result = await strict_rag_pipeline.ask(
            question=clean_query,
            db=db,
            session_id=session_id,
            top_k=top_k
        )
        
        logger.info(
            "Strict query with session processed",
            session_id=session_id,
            message_id=result["message_id"],
            can_answer=result["can_answer"],
            user_id=user_id
        )
        
        return {
            "answer": result["answer"],
            "can_answer": result["can_answer"],
            "message_id": result["message_id"],
            "session_id": session_id,
            "contexts": result.get("contexts", []),
            "metadata": {
                "pipeline_mode": "strict_single_answer_with_session",
                "top_k_used": top_k,
                "reason": result.get("reason"),
                "evidence_count": result.get("evidence_count", 0)
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in session-based strict query: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process query with session"
        )


@router.get("/config")
async def get_strict_config(
    user_id: Optional[str] = Depends(optional_auth)
):
    """
    Получить конфигурацию строгого RAG pipeline.
    """
    return {
        "pipeline_mode": "strict_single_answer",
        "tau_retrieval": strict_rag_pipeline.TAU_RETR,
        "tau_nli": strict_rag_pipeline.TAU_NLI,
        "recommended_top_k": {
            "min": 1,
            "max": 6,
            "default": 4,
            "description": "Small k to avoid conflicting context"
        },
        "features": {
            "conflict_filtering": True,
            "answerability_gate": True,
            "structured_output": True,
            "single_answer_only": True,
            "honest_refusal": True
        },
        "llm_parameters": {
            "temperature": 0.0,
            "top_p": 0.1,
            "format": "structured_json"
        }
    }


@router.post("/tune-thresholds")
async def tune_thresholds(
    request: Dict[str, float],
    db: Session = Depends(get_db),
    user_id: Optional[str] = Depends(optional_auth)
):
    """
    Настроить пороги для gate ответимости.
    
    Принимает:
    - tau_retr: порог для retrieval score (0.1-0.8)
    - tau_nli: порог для NLI entailment (0.5-0.9)
    """
    try:
        tau_retr = request.get("tau_retr")
        tau_nli = request.get("tau_nli")
        
        # Validate thresholds
        if tau_retr is not None:
            if not (0.1 <= tau_retr <= 0.8):
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="tau_retr must be between 0.1 and 0.8"
                )
            strict_rag_pipeline.TAU_RETR = tau_retr
        
        if tau_nli is not None:
            if not (0.5 <= tau_nli <= 0.9):
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="tau_nli must be between 0.5 and 0.9"
                )
            strict_rag_pipeline.TAU_NLI = tau_nli
        
        logger.info(
            "Strict RAG thresholds updated",
            tau_retr=strict_rag_pipeline.TAU_RETR,
            tau_nli=strict_rag_pipeline.TAU_NLI,
            user_id=user_id
        )
        
        return {
            "message": "Thresholds updated successfully",
            "tau_retr": strict_rag_pipeline.TAU_RETR,
            "tau_nli": strict_rag_pipeline.TAU_NLI
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating thresholds: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update thresholds"
        )


@router.get("/stats")
async def get_strict_stats(
    db: Session = Depends(get_db),
    user_id: Optional[str] = Depends(optional_auth)
):
    """
    Получить статистику строгого RAG pipeline.
    """
    try:
        stats = await strict_rag_pipeline.get_retrieval_stats(db)
        
        return {
            "pipeline_stats": stats,
            "current_config": {
                "tau_retr": strict_rag_pipeline.TAU_RETR,
                "tau_nli": strict_rag_pipeline.TAU_NLI,
                "pipeline_mode": "strict_single_answer"
            },
            "recommendations": {
                "tau_retr_range": "0.35-0.45 for most cases",
                "tau_nli_range": "0.6-0.7 for reliable answers",
                "top_k_range": "4-6 to minimize conflicts",
                "conflict_filtering": "enabled by default"
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get strict RAG stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve statistics"
        )


@router.post("/test-query")
async def test_strict_query(
    request: Dict[str, Any],
    db: Session = Depends(get_db),
    user_id: Optional[str] = Depends(optional_auth)
):
    """
    Тестовый endpoint для отладки строгого RAG.
    
    Возвращает детальную информацию о процессе принятия решений.
    """
    try:
        query = request.get("query", "").strip()
        top_k = min(max(request.get("top_k", 4), 1), 6)
        
        if not query:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Query cannot be empty"
            )
        
        clean_query = sanitize_input(query, max_length=2000)
        
        # Get detailed pipeline execution
        contexts, can_answer = await strict_rag_pipeline.build_context(db, clean_query, top_k)
        
        # Get candidates before filtering
        candidates = await strict_rag_pipeline.retrieve_candidates(db, clean_query, k=8)
        
        debug_info = {
            "query": clean_query,
            "top_k_requested": top_k,
            "retrieval_phase": {
                "candidates_found": len(candidates),
                "candidate_scores": [c.score for c in candidates[:5]] if candidates else [],
                "max_candidate_score": max([c.score for c in candidates]) if candidates else 0.0
            },
            "filtering_phase": {
                "contexts_after_filter": len(contexts),
                "final_scores": [c.score for c in contexts] if contexts else [],
                "max_final_score": max([c.score for c in contexts]) if contexts else 0.0
            },
            "gate_decision": {
                "can_answer": can_answer,
                "tau_retr": strict_rag_pipeline.TAU_RETR,
                "tau_nli": strict_rag_pipeline.TAU_NLI,
                "passed_retrieval_threshold": (max([c.score for c in contexts]) if contexts else 0.0) >= strict_rag_pipeline.TAU_RETR
            }
        }
        
        # If can answer, also test LLM response
        if can_answer and contexts:
            llm_response = await strict_rag_pipeline.ask_llm_strict(clean_query, contexts)
            debug_info["llm_phase"] = {
                "llm_can_answer": llm_response.can_answer,
                "final_answer_length": len(llm_response.final_answer) if llm_response.final_answer else 0,
                "evidence_ids": llm_response.evidence_ids
            }
        
        return {
            "debug_info": debug_info,
            "recommendation": "Adjust tau_retr/tau_nli if needed based on gate_decision results"
        }
        
    except Exception as e:
        logger.error(f"Error in test query: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Test query failed: {str(e)}"
        )
