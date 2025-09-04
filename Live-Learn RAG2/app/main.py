"""Main FastAPI application with Gemini - Fixed version."""

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
            google_api_key = os.getenv("GOOGLE_API_KEY")
        
        return FallbackSettings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    settings = get_settings()
    
    logger.info("Starting RAG application", version="2.0.0", env=settings.app_env)
    
    try:
        # Initialize database
        init_db()
        
        # Check external services only if API key is available
        if hasattr(settings, 'google_api_key') and settings.google_api_key:
            await check_external_services()
        else:
            logger.warning("Google API key not configured, skipping service checks")
        
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
    description="Enhanced RAG system with Gemini AI",
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
        "ai_configured": bool(getattr(settings, 'google_api_key', None)),
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
    
    # Only check AI services if API key is configured
    if hasattr(settings, 'google_api_key') and settings.google_api_key:
        try:
            checks["gemini"] = await check_gemini_health()
        except Exception as e:
            checks["gemini"] = {"status": "error", "error": str(e)}
    else:
        checks["gemini"] = {"status": "not_configured", "message": "GOOGLE_API_KEY not set"}
    
    # Determine overall status
    overall_status = "ok"
    for service, check in checks.items():
        if check.get("status") not in ["ok", "healthy", "not_configured"]:
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

async def check_external_services():
    """Check external service connectivity on startup."""
    logger.info("Checking external services")
    
    try:
        gemini_status = await check_gemini_health()
        if gemini_status["status"] != "ok":
            logger.warning("Gemini service check failed", status=gemini_status)
    except Exception as e:
        logger.warning(f"Could not check Gemini service: {e}")

async def check_gemini_health() -> Dict[str, Any]:
    """Check Gemini service health."""
    try:
        settings = get_settings()
        
        if not hasattr(settings, 'google_api_key') or not settings.google_api_key:
            return {"status": "not_configured", "message": "Google API key not configured"}
        
        # Test Google AI
        import google.generativeai as genai
        genai.configure(api_key=settings.google_api_key)
        
        # Quick test
        model = genai.GenerativeModel(settings.llm_model)
        response = model.generate_content("Say OK")
        
        if response and response.text:
            return {"status": "ok", "model": settings.llm_model}
        else:
            return {"status": "error", "message": "No response from Gemini"}
            
    except Exception as e:
        logger.error(f"Gemini health check failed: {e}")
        return {"status": "unhealthy", "error": str(e)}

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    settings = get_settings()
    
    status_message = "✅ Ready"
    if not getattr(settings, 'google_api_key', None):
        status_message = "⚠️ Needs Google API Key"
    elif not routers_loaded:
        status_message = "⚠️ Some routers failed to load"
    
    return {
        "name": "Live-Learn RAG Backend",
        "version": "2.0.0",
        "description": "Enhanced RAG system with Gemini AI",
        "ai_model": getattr(settings, 'llm_model', 'gemini-2.0-flash-exp'),
        "routers_loaded": routers_loaded,
        "features": [
            "Document ingestion and chunking",
            "Vector similarity search",
            "Gemini-powered answer generation",
            "User feedback learning system",
            "Performance monitoring"
        ],
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