"""Logging configuration for the RAG application."""

import sys
import time as time_module  # Ð˜Ð¡ÐŸÐ ÐÐ’Ð›Ð•ÐÐž: Ð¿ÐµÑ€ÐµÐ¸Ð¼ÐµÐ½Ð¾Ð²Ð°Ñ‚ÑŒ Ð¸Ð¼Ð¿Ð¾Ñ€Ñ‚
from pathlib import Path
from typing import Dict, Any

from loguru import logger

from ..config import settings


def setup_logging():
    """Setup structured logging with loguru."""
    
    # Remove default handler
    logger.remove()
    
    # Console handler with colors for development
    if settings.app_env == "dev":
        logger.add(
            sys.stdout,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                   "<level>{level: <8}</level> | "
                   "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
                   "<level>{message}</level>",
            level="DEBUG" if settings.debug else "INFO",
            colorize=True,
            backtrace=True,
            diagnose=True
        )
    else:
        # Production: JSON format for structured logging
        logger.add(
            sys.stdout,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
            level="INFO",
            serialize=True,  # JSON output
            backtrace=False,
            diagnose=False
        )
    
    # File handler for persistent logs
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logger.add(
        log_dir / "app.log",
        rotation="100 MB",
        retention="30 days",
        compression="gz",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
        level="INFO",
        backtrace=True,
        diagnose=True
    )
    
    # Error-only log file
    logger.add(
        log_dir / "errors.log",
        rotation="50 MB",
        retention="90 days",
        compression="gz",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message} | {extra}",
        level="ERROR",
        backtrace=True,
        diagnose=True,
        filter=lambda record: record["level"].name in ["ERROR", "CRITICAL"]
    )
    
    # Set log levels for third-party libraries
    import logging
    
    # Reduce noise from httpx and other libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    
    logger.info("Logging configured", env=settings.app_env, debug=settings.debug)


def get_logger(name: str):
    """Get a logger instance with the given name."""
    return logger.bind(component=name)


class LoggingMiddleware:
    """Custom middleware for request/response logging."""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        # Extract request info
        method = scope["method"]
        path = scope["path"]
        query_string = scope.get("query_string", b"").decode()
        client_ip = scope.get("client", ["unknown", None])[0]
        
        # Log request
        request_logger = logger.bind(
            method=method,
            path=path,
            query_string=query_string,
            client_ip=client_ip,
            request_id=None  # TODO: Add request ID generation
        )
        
        start_time = None
        status_code = None
        
        # Capture response info
        async def send_wrapper(message):
            nonlocal status_code, start_time
            
            if message["type"] == "http.response.start":
                status_code = message["status"]
                start_time = time_module.time()  # Ð˜Ð¡ÐŸÐ ÐÐ’Ð›Ð•ÐÐž: Ð¸ÑÐ¿Ð¾Ð»ÑŒÐ·Ð¾Ð²Ð°Ñ‚ÑŒ time_module
                
            elif message["type"] == "http.response.body" and not message.get("more_body", False):
                # Log response
                if start_time:
                    duration = time_module.time() - start_time  # Ð˜Ð¡ÐŸÐ ÐÐ’Ð›Ð•ÐÐž: Ð¸ÑÐ¿Ð¾Ð»ÑŒÐ·Ð¾Ð²Ð°Ñ‚ÑŒ time_module
                    
                    log_level = "INFO"
                    if status_code >= 500:
                        log_level = "ERROR"
                    elif status_code >= 400:
                        log_level = "WARNING"
                    
                    request_logger.log(
                        log_level,
                        "Request completed",
                        status_code=status_code,
                        duration_ms=round(duration * 1000, 2)
                    )
            
            await send(message)
        
        await self.app(scope, receive, send_wrapper)


def log_function_call(func_name: str, **kwargs):
    """Decorator for logging function calls."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            func_logger = logger.bind(function=func_name)
            
            # Log function entry
            func_logger.debug(f"Entering {func_name}", **kwargs)
            
            try:
                result = func(*args, **kwargs)
                func_logger.debug(f"Exiting {func_name} successfully")
                return result
            except Exception as e:
                func_logger.error(f"Error in {func_name}: {e}", exception=str(e))
                raise
                
        return wrapper
    return decorator


def log_async_function_call(func_name: str, **kwargs):
    """Decorator for logging async function calls."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            func_logger = logger.bind(function=func_name)
            
            # Log function entry
            func_logger.debug(f"Entering async {func_name}", **kwargs)
            
            try:
                result = await func(*args, **kwargs)
                func_logger.debug(f"Exiting async {func_name} successfully")
                return result
            except Exception as e:
                func_logger.error(f"Error in async {func_name}: {e}", exception=str(e))
                raise
                
        return wrapper
    return decorator


def sanitize_log_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize sensitive data from logs."""
    
    sensitive_keys = {
        'password', 'token', 'key', 'secret', 'auth', 'credential',
        'api_key', 'access_token', 'refresh_token', 'jwt'
    }
    
    sanitized = {}
    
    for key, value in data.items():
        key_lower = key.lower()
        
        # Check if key contains sensitive information
        if any(sensitive in key_lower for sensitive in sensitive_keys):
            sanitized[key] = "***REDACTED***"
        elif isinstance(value, dict):
            sanitized[key] = sanitize_log_data(value)
        elif isinstance(value, str) and len(value) > 1000:
            # Truncate very long strings
            sanitized[key] = value[:1000] + "... (truncated)"
        else:
            sanitized[key] = value
    
    return sanitized


def log_performance(operation: str, **context):
    """Context manager for performance logging."""
    class PerformanceLogger:
        def __init__(self, operation: str, **context):
            self.operation = operation
            self.context = context
            self.start_time = None
            self.logger = logger.bind(operation=operation, **context)
        
        def __enter__(self):
            self.start_time = time_module.time()  # Ð˜Ð¡ÐŸÐ ÐÐ’Ð›Ð•ÐÐž: Ð¸ÑÐ¿Ð¾Ð»ÑŒÐ·Ð¾Ð²Ð°Ñ‚ÑŒ time_module
            self.logger.debug(f"Starting {self.operation}")
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            duration = time_module.time() - self.start_time  # Ð˜Ð¡ÐŸÐ ÐÐ’Ð›Ð•ÐÐž: Ð¸ÑÐ¿Ð¾Ð»ÑŒÐ·Ð¾Ð²Ð°Ñ‚ÑŒ time_module
            
            if exc_type:
                self.logger.error(
                    f"Failed {self.operation}",
                    duration_ms=round(duration * 1000, 2),
                    error=str(exc_val)
                )
            else:
                self.logger.info(
                    f"Completed {self.operation}",
                    duration_ms=round(duration * 1000, 2)
                )
    
    return PerformanceLogger(operation, **context)


# Configure logging on import
setup_logging()