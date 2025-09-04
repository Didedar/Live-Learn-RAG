"""Security utilities for the RAG system."""

import re
from typing import Optional

from fastapi import HTTPException, Header, status
from loguru import logger

from ..config import settings


async def optional_auth(x_api_key: Optional[str] = Header(None)) -> Optional[str]:
    """
    Optional API key authentication.
    
    Args:
        x_api_key: API key from header
        
    Returns:
        User ID or None if no authentication required
        
    Raises:
        HTTPException: If API key is required but invalid
    """
    # If no API key is configured, skip authentication
    if not settings.api_key:
        return None
    
    # If API key is configured but not provided
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required"
        )
    
    # Validate API key
    if x_api_key != settings.api_key:
        logger.warning(f"Invalid API key attempt: {x_api_key[:10]}...")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    
    # Return a user ID (in real system, this would be extracted from the key)
    return "authenticated_user"


def sanitize_input(text: str, max_length: int = 10000) -> str:
    """
    Sanitize user input text.
    
    Args:
        text: Input text to sanitize
        max_length: Maximum allowed length
        
    Returns:
        Sanitized text
        
    Raises:
        ValueError: If text is too long or contains forbidden content
    """
    if not isinstance(text, str):
        raise ValueError("Input must be a string")
    
    # Check length
    if len(text) > max_length:
        raise ValueError(f"Text too long: {len(text)} > {max_length}")
    
    # Remove control characters but keep basic formatting
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    # Check for suspicious patterns (basic XSS/injection prevention)
    suspicious_patterns = [
        r'<script[^>]*>',
        r'javascript:',
        r'on\w+\s*=',
        r'data:.*base64',
    ]
    
    for pattern in suspicious_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            logger.warning(f"Suspicious pattern detected: {pattern}")
            # Don't raise error, just log for now
            break
    
    return text


def validate_file_upload(filename: Optional[str], content: bytes) -> bool:
    """
    Validate uploaded file.
    
    Args:
        filename: Original filename
        content: File content bytes
        
    Returns:
        True if file is valid, False otherwise
    """
    if not filename:
        logger.warning("File upload without filename")
        return False
    
    # Check file size (10MB limit)
    max_size = 10 * 1024 * 1024  # 10MB
    if len(content) > max_size:
        logger.warning(f"File too large: {len(content)} bytes")
        return False
    
    # Check file extension
    allowed_extensions = {
        '.txt', '.md', '.rtf', 
        '.pdf', '.doc', '.docx',
        '.html', '.htm', '.xml',
        '.csv', '.json', '.yaml', '.yml'
    }
    
    file_ext = None
    if '.' in filename:
        file_ext = '.' + filename.rsplit('.', 1)[1].lower()
    
    if file_ext not in allowed_extensions:
        logger.warning(f"Unsupported file extension: {file_ext}")
        return False
    
    # Basic content validation
    if len(content) == 0:
        logger.warning("Empty file uploaded")
        return False
    
    # Check for potentially malicious content
    try:
        # Try to decode as UTF-8
        text_content = content.decode('utf-8', errors='ignore')
        
        # Check for suspicious patterns in text files
        if file_ext in {'.txt', '.md', '.html', '.htm', '.xml', '.csv', '.json'}:
            suspicious_patterns = [
                b'<script',
                b'javascript:',
                b'<?php',
                b'<%',
                b'exec(',
                b'system(',
                b'shell_exec(',
            ]
            
            content_lower = content.lower()
            for pattern in suspicious_patterns:
                if pattern in content_lower:
                    logger.warning(f"Suspicious content pattern: {pattern}")
                    return False
    
    except Exception as e:
        logger.warning(f"Error validating file content: {e}")
        return False
    
    return True


def rate_limit_key(request) -> str:
    """
    Generate rate limiting key for requests.
    
    Args:
        request: FastAPI request object
        
    Returns:
        Rate limiting key string
    """
    # Use client IP as default
    client_ip = request.client.host if request.client else "unknown"
    
    # If authenticated, use user ID
    auth_header = request.headers.get(settings.api_key_header)
    if auth_header and auth_header == settings.api_key:
        return f"user:authenticated_user"
    
    return f"ip:{client_ip}"


def mask_sensitive_data(data: str, mask_char: str = "*", visible_chars: int = 4) -> str:
    """
    Mask sensitive data for logging.
    
    Args:
        data: Sensitive data to mask
        mask_char: Character to use for masking
        visible_chars: Number of characters to show at start/end
        
    Returns:
        Masked string
    """
    if not data or len(data) <= visible_chars * 2:
        return mask_char * len(data) if data else ""
    
    start = data[:visible_chars]
    end = data[-visible_chars:]
    middle = mask_char * (len(data) - visible_chars * 2)
    
    return f"{start}{middle}{end}"


def validate_session_id(session_id: Optional[str]) -> bool:
    """
    Validate session ID format.
    
    Args:
        session_id: Session ID to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not session_id:
        return True  # Optional field
    
    # Check length
    if len(session_id) > 128:
        return False
    
    # Check format (alphanumeric, hyphens, underscores only)
    if not re.match(r'^[a-zA-Z0-9_-]+$', session_id):
        return False
    
    return True


def validate_metadata(metadata: dict) -> bool:
    """
    Validate metadata dictionary.
    
    Args:
        metadata: Metadata to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not isinstance(metadata, dict):
        return False
    
    # Check size (JSON serialized)
    try:
        import json
        serialized = json.dumps(metadata)
        if len(serialized) > 10000:  # 10KB limit
            return False
    except (TypeError, ValueError):
        return False
    
    # Check for suspicious keys/values
    suspicious_keys = ['__proto__', 'constructor', 'prototype']
    
    def check_dict(d):
        for key, value in d.items():
            if not isinstance(key, str):
                return False
            if key.lower() in suspicious_keys:
                return False
            if isinstance(value, dict):
                if not check_dict(value):
                    return False
            elif isinstance(value, str) and len(value) > 1000:
                return False
        return True
    
    return check_dict(metadata)