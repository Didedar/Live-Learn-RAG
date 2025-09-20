"""Main FastAPI application with Ollama - Local-only version."""

import asyncio
import os
import time as time_module
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, Request, Response, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from fastapi import APIRouter
from pydantic import BaseModel

API_ROOT = os.getenv("API_ROOT", "/api")

# Исправлено: не импортируем settings сразу
from .core.exceptions import RAGException, get_status_code
from .database import init_db, check_db_health, optimize_database

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

def get_settings():
    """Ленивый импорт настроек."""
    try:
        from .config import settings
        return settings
    except Exception as e:
        logger.error(f"Failed to load settings: {e}")
        # Возвращаем базовые настройки как fallback
        class FallbackSettings:
            app_name = "Live-Learn RAG"
            app_env = "dev"  
            debug = False
            pass  # No longer need Google API key
        
        return FallbackSettings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    settings = get_settings()
    
    logger.info("Starting RAG application", version="2.0.0", env=settings.app_env)
    
    try:
        # Initialize database
        init_db()
        
        # Always check Ollama service (local-only mode)
        await check_ollama_service()
        
        # Optimize database
        if settings.app_env == "prod":
            optimize_database()
        
        logger.info("Application startup completed successfully")
        
    except Exception as e:
        logger.error(f"Application startup failed: {e}")
        # Не останавливаем приложение, если только настройки не загружены
        if "GOOGLE_API_KEY" not in str(e):
            raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down RAG application")

# Create FastAPI application
app = FastAPI(
    title="Live-Learn RAG Backend",
    description="Enhanced RAG system with Ollama (Local AI)",
    version="2.0.0",
    lifespan=lifespan
)

# Add rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Add middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure properly for production
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

@app.middleware("http")
async def performance_monitoring_middleware(request: Request, call_next):
    """Monitor request performance."""
    start_time = time_module.time()
    
    response = await call_next(request)
    
    process_time = time_module.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    
    # Log slow requests
    if process_time > 5.0:  # 5 seconds
        logger.warning(
            "Slow request detected",
            path=request.url.path,
            method=request.method,
            duration=process_time
        )
    
    return response

# Fallback schemas for emergency fallback endpoint
class AskReq(BaseModel):
    question: str
    session_id: str = None
    top_k: int = 6

# Import and include routers with error handling
routers_loaded = []

# Load RAG router
try:
    from .api.v1.rag import router as rag_router
    app.include_router(rag_router, prefix=f"{API_ROOT}/v1")
    routers_loaded.append("rag")
    logger.info(f"RAG router loaded at {API_ROOT}/v1")
except Exception as e:
    logger.error(f"Failed to load RAG router: {e}")

# Load Feedback router
try:
    from .api.v1.feedback import router as feedback_router
    app.include_router(feedback_router, prefix=f"{API_ROOT}/v1")
    routers_loaded.append("feedback")
    logger.info(f"Feedback router loaded at {API_ROOT}/v1")
except Exception as e:
    logger.error(f"Failed to load Feedback router: {e}")

# Load Separated RAG router
try:
    from .api.v1.separated_rag import router as separated_rag_router
    app.include_router(separated_rag_router, prefix=f"{API_ROOT}/v1/separated")
    routers_loaded.append("separated-rag")
    logger.info(f"Separated RAG router loaded at {API_ROOT}/v1/separated")
except Exception as e:
    logger.error(f"Failed to load Separated RAG router: {e}")

# Load Strict RAG router
try:
    from .api.v1.strict_rag import router as strict_rag_router
    app.include_router(strict_rag_router, prefix=f"{API_ROOT}/v1/strict")
    routers_loaded.append("strict-rag")
    logger.info(f"Strict RAG router loaded at {API_ROOT}/v1/strict")
except Exception as e:
    logger.error(f"Failed to load Strict RAG router: {e}")

# Load Enhanced RAG router
try:
    from .api.v1.enhanced_rag import router as enhanced_rag_router
    app.include_router(enhanced_rag_router, prefix=f"{API_ROOT}/v1/enhanced")
    routers_loaded.append("enhanced-rag")
    logger.info(f"Enhanced RAG router loaded at {API_ROOT}/v1/enhanced")
except Exception as e:
    logger.error(f"Failed to load Enhanced RAG router: {e}")

# Load Improved RAG router (с исправлениями на основе feedback анализа)
try:
    from .api.v1.improved_rag import router as improved_rag_router
    app.include_router(improved_rag_router, prefix=f"{API_ROOT}/v1/improved")
    routers_loaded.append("improved-rag")
    logger.info(f"Improved RAG router loaded at {API_ROOT}/v1/improved")
except Exception as e:
    logger.error(f"Failed to load Improved RAG router: {e}")

# Load Hybrid RAG router (Dense + BM25 retrieval)
try:
    from .api.v1.hybrid_rag import router as hybrid_rag_router
    app.include_router(hybrid_rag_router, prefix=f"{API_ROOT}/v1/hybrid")
    routers_loaded.append("hybrid-rag")
    logger.info(f"Hybrid RAG router loaded at {API_ROOT}/v1/hybrid")
except Exception as e:
    logger.error(f"Failed to load Hybrid RAG router: {e}")

# Emergency fallback endpoint if feedback router failed to load
if "feedback" not in routers_loaded:
    @app.post(f"{API_ROOT}/v1/feedback/ask")
    async def feedback_ask_fallback(req: AskReq):
        """Emergency fallback for feedback/ask endpoint."""
        logger.warning("Using emergency fallback for feedback/ask")
        
        try:
            # Try to use RAG pipeline directly if available
            from .services.rag_pipeline import EnhancedRAGPipeline
            from .database import get_db
            
            pipeline = EnhancedRAGPipeline()
            
            # Simple database session
            db_gen = get_db()
            db = next(db_gen)
            
            try:
                result = await pipeline.ask(
                    question=req.question,
                    db=db,
                    session_id=req.session_id,
                    top_k=req.top_k
                )
                return result
            finally:
                try:
                    next(db_gen)  # Close the generator
                except StopIteration:
                    pass
                    
        except Exception as e:
            logger.error(f"Emergency fallback also failed: {e}")
            return {
                "answer": f"Система временно недоступна. Ошибка: {str(e)}",
                "contexts": [],
                "message_id": "fallback-error"
            }

@app.exception_handler(RAGException)
async def rag_exception_handler(request: Request, exc: RAGException):
    """Handle custom RAG exceptions."""
    status_code = get_status_code(exc)
    
    logger.error(
        "RAG exception occurred",
        exception_type=type(exc).__name__,
        message=exc.message,
        path=request.url.path,
        method=request.method
    )
    
    return JSONResponse(
        status_code=status_code,
        content=exc.to_dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    settings = get_settings()
    
    logger.error(
        "Unexpected exception occurred",
        exception_type=type(exc).__name__,
        message=str(exc),
        path=request.url.path,
        method=request.method
    )
    
    if settings.debug:
        import traceback
        return JSONResponse(
            status_code=500,
            content={
                "error": "InternalServerError",
                "message": str(exc),
                "type": type(exc).__name__,
                "traceback": traceback.format_exc()
            }
        )
    else:
        return JSONResponse(
            status_code=500,
            content={
                "error": "InternalServerError",
                "message": "An unexpected error occurred"
            }
        )

# Health check endpoints
@app.get("/healthz")
async def health_check():
    """Basic health check."""
    settings = get_settings()
    
    return {
        "status": "ok",
        "env": settings.app_env,
        "timestamp": time_module.time(),
        "version": "2.0.0",
        "ai_configured": settings.is_ollama_configured(),
        "routers_loaded": routers_loaded
    }

@app.get("/healthz/detailed")
async def detailed_health_check():
    """Detailed health check with dependencies."""
    settings = get_settings()
    
    checks = {
        "app": {"status": "ok", "routers_loaded": routers_loaded},
        "database": check_db_health(),
    }
    
    # Check AI services - only Ollama (local-only mode)
    try:
        checks["ollama"] = await check_ollama_health()
    except Exception as e:
        checks["ollama"] = {"status": "error", "error": str(e)}
    checks["gemini"] = {"status": "disabled", "message": "Gemini removed - using Ollama only"}
    
    # Determine overall status
    overall_status = "ok"
    for service, check in checks.items():
        if check.get("status") not in ["ok", "healthy", "not_configured", "disabled"]:
            overall_status = "degraded"
            break
    
    return {
        "status": overall_status,
        "timestamp": time_module.time(),
        "checks": checks
    }

@app.get("/metrics")
async def metrics():
    """Application metrics endpoint."""
    try:
        from .database import get_db_session
        from .models.documents import Document, Chunk
        
        with get_db_session() as db:
            document_count = db.query(Document).count()
            chunk_count = db.query(Chunk).count()
            
            return {
                "documents_total": document_count,
                "chunks_total": chunk_count,
                "avg_chunks_per_document": chunk_count / max(document_count, 1),
                "routers_loaded": routers_loaded
            }
            
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        return {"error": "Failed to retrieve metrics"}

# Removed check_external_services - not needed for Ollama-only mode


async def check_ollama_service():
    """Check Ollama service connectivity on startup."""
    logger.info("Checking Ollama service")
    
    try:
        from .services.ollama_llm import OllamaLLM
        
        settings = get_settings()
        ollama = OllamaLLM(
            base_url=settings.ollama_url,
            model=settings.ollama_model
        )
        
        if await ollama.health_check():
            logger.info("✅ Ollama service is healthy")
        else:
            logger.warning("⚠️ Ollama service health check failed")
            
    except Exception as e:
        logger.warning(f"Could not check Ollama service: {e}")

async def check_ollama_health() -> Dict[str, Any]:
    """Check Ollama service health."""
    try:
        from .services.ollama_llm import OllamaLLM
        
        settings = get_settings()
        ollama = OllamaLLM(
            base_url=settings.ollama_url,
            model=settings.ollama_model
        )
        
        # Get model info
        model_info = await ollama.list_models()
        
        if await ollama.health_check():
            return {
                "status": "ok", 
                "model": settings.ollama_model,
                "url": settings.ollama_url,
                "model_available": model_info.get("model_available", False),
                "available_models": model_info.get("available_models", [])
            }
        else:
            return {
                "status": "unhealthy", 
                "message": "Health check failed",
                "url": settings.ollama_url,
                "model": settings.ollama_model
            }
            
    except Exception as e:
        logger.error(f"Ollama health check failed: {e}")
        return {"status": "error", "error": str(e)}


# Removed check_gemini_health - not needed for Ollama-only mode

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    settings = get_settings()
    
    # Always using Ollama now
    status_message = "✅ Ready (Ollama)"
    
    if not routers_loaded:
        status_message = "⚠️ Some routers failed to load"
    
    # Always using Ollama now
    description = "Enhanced RAG system with local Llama via Ollama"
    ai_model = settings.ollama_model
    ai_features = [
        "Document ingestion and chunking",
        "Hybrid retrieval (Dense + BM25)",
        "Vector similarity search", 
        "Keyword/factual search (BM25)",
        "Local Llama-powered answer generation",
        "User feedback learning system",
        "Performance monitoring"
    ]
    
    return {
        "name": "Live-Learn RAG Backend",
        "version": "2.0.0",
        "description": description,
        "ai_model": ai_model,
        "llm_provider": "Ollama",
        "routers_loaded": routers_loaded,
        "features": ai_features,
        "endpoints": {
            "health": "/healthz",
            "docs": "/docs",
            "api": f"{API_ROOT}/v1"
        },
        "status": status_message
    }

if __name__ == "__main__":
    import uvicorn
    
    settings = get_settings()
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_config=None
    )