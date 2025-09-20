"""Text processing utilities for RAG system."""

import re
from typing import List, Optional

import tiktoken
from loguru import logger


def chunk_text(
    text: str,
    max_tokens: int = 350,  # Optimal size 350-600 tokens
    overlap: int = 70,      # 20-25% overlap
    model: str = "gpt-4o-mini",
    preserve_headers: bool = True
) -> List[str]:
    """
    Split text into overlapping chunks with improved strategy.
    
    Args:
        text: Input text to chunk
        max_tokens: Maximum tokens per chunk (350-600 recommended)
        overlap: Number of overlapping tokens between chunks (20-25% of max_tokens)
        model: Model name for tokenizer (fallback to cl100k_base)
        preserve_headers: Whether to include headers/titles in chunk beginnings
        
    Returns:
        List of text chunks with improved boundaries
    """
    try:
        if not text or not text.strip():
            logger.warning("Empty text provided for chunking")
            return []
        
        # Clean text first
        cleaned_text = clean_text_for_chunking(text)
        
        # Get tokenizer
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            logger.warning(f"Model {model} not found, using cl100k_base encoding")
            encoding = tiktoken.get_encoding("cl100k_base")
        
        # Tokenize the text
        tokens = encoding.encode(cleaned_text)
        total_tokens = len(tokens)
        
        if total_tokens <= max_tokens:
            # Text is small enough, return as single chunk
            return [cleaned_text]
        
        chunks = []
        start = 0
        
        # Extract document structure for headers
        headers = extract_headers(cleaned_text) if preserve_headers else []
        
        while start < total_tokens:
            # Calculate end position
            end = min(start + max_tokens, total_tokens)
            
            # Extract chunk tokens
            chunk_tokens = tokens[start:end]
            
            # Decode back to text
            chunk_text = encoding.decode(chunk_tokens)
            
            # Try to break at better boundaries (sentences, paragraphs)
            if end < total_tokens:  # Not the last chunk
                chunk_text = improve_chunk_boundary(chunk_text, cleaned_text, start, end, encoding)
            
            # Add relevant headers to chunk beginning
            if preserve_headers and headers:
                chunk_text = add_context_headers(chunk_text, headers, cleaned_text)
            
            # Clean up the chunk
            chunk_text = chunk_text.strip()
            
            if chunk_text and len(chunk_text) > 20:  # Minimum chunk size
                chunks.append(chunk_text)
            
            # Move start position (with overlap)
            if end >= total_tokens:
                break
            
            # Calculate better overlap position
            overlap_start = max(start, end - overlap)
            start = overlap_start
            
            # Ensure we don't get stuck
            if start <= end - max_tokens + 10:
                start = end - overlap
        
        logger.info(
            "Enhanced text chunked",
            original_tokens=total_tokens,
            num_chunks=len(chunks),
            max_tokens=max_tokens,
            overlap=overlap,
            avg_chunk_size=sum(len(encoding.encode(chunk)) for chunk in chunks) // len(chunks) if chunks else 0
        )
        
        return chunks
        
    except Exception as e:
        logger.error(f"Failed to chunk text: {e}")
        # Fallback to simple splitting
        return simple_chunk_text(text, max_chars=max_tokens * 4)  # Rough estimate


def clean_text_for_chunking(text: str) -> str:
    """Clean text specifically for better chunking."""
    try:
        # Remove excessive whitespace but preserve structure
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)  # Max 2 consecutive newlines
        text = re.sub(r' +', ' ', text)  # Single spaces
        
        # Fix common OCR issues
        text = re.sub(r'([a-zа-я])-\s*\n\s*([a-zа-я])', r'\1\2', text)  # Fix hyphenated words
        text = re.sub(r'([.!?])\s*\n\s*([A-ZА-Я])', r'\1 \2', text)  # Fix sentence breaks
        
        # Normalize quotes and dashes
        text = re.sub(r'["""]', '"', text)
        text = re.sub(r'[–—]', '-', text)
        
        return text.strip()
    except Exception:
        return text


def extract_headers(text: str) -> List[str]:
    """Extract potential headers from text."""
    try:
        headers = []
        
        # Look for lines that might be headers
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Heuristics for headers
            is_header = False
            
            # All caps and short
            if line.isupper() and len(line) < 100:
                is_header = True
            
            # Starts with numbers (1., 1.1, etc.)
            if re.match(r'^\d+\.(\d+\.)*\s+', line):
                is_header = True
            
            # Ends with colon and is short
            if line.endswith(':') and len(line) < 80:
                is_header = True
            
            # Common header words
            header_words = ['глава', 'раздел', 'статья', 'пункт', 'chapter', 'section', 'article']
            if any(word in line.lower() for word in header_words):
                is_header = True
            
            if is_header:
                headers.append(line)
        
        return headers[:10]  # Limit to prevent spam
    except Exception:
        return []


def improve_chunk_boundary(chunk_text: str, full_text: str, start_pos: int, end_pos: int, encoding) -> str:
    """Improve chunk boundary to break at better positions."""
    try:
        # Try to break at sentence boundaries
        sentences = re.split(r'([.!?]+\s+)', chunk_text)
        
        if len(sentences) > 3:  # If we have multiple sentences
            # Remove the last incomplete sentence
            sentences = sentences[:-2] if len(sentences) > 2 else sentences
            improved_chunk = ''.join(sentences).strip()
            
            if len(improved_chunk) > len(chunk_text) * 0.7:  # Don't cut too much
                return improved_chunk
        
        # Try to break at paragraph boundaries
        paragraphs = chunk_text.split('\n\n')
        if len(paragraphs) > 1:
            # Remove last paragraph if it's incomplete
            improved_chunk = '\n\n'.join(paragraphs[:-1]).strip()
            
            if len(improved_chunk) > len(chunk_text) * 0.7:
                return improved_chunk
        
        return chunk_text
    except Exception:
        return chunk_text


def add_context_headers(chunk_text: str, headers: List[str], full_text: str) -> str:
    """Add relevant headers to provide context."""
    try:
        # Find which headers are relevant to this chunk
        relevant_headers = []
        
        for header in headers:
            # If header appears before this chunk in full text
            header_pos = full_text.find(header)
            chunk_pos = full_text.find(chunk_text[:100])  # First part of chunk
            
            if 0 <= header_pos < chunk_pos:
                relevant_headers.append(header)
        
        # Add the most recent relevant headers
        if relevant_headers:
            context = ' > '.join(relevant_headers[-2:])  # Last 2 headers
            return f"[Контекст: {context}]\n\n{chunk_text}"
        
        return chunk_text
    except Exception:
        return chunk_text


def simple_chunk_text(
    text: str,
    max_chars: int = 1600,
    overlap_chars: int = 160
) -> List[str]:
    """
    Simple character-based text chunking as fallback.
    
    Args:
        text: Input text
        max_chars: Maximum characters per chunk
        overlap_chars: Overlapping characters
        
    Returns:
        List of text chunks
    """
    try:
        if not text or not text.strip():
            return []
        
        text = text.strip()
        
        if len(text) <= max_chars:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + max_chars, len(text))
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings near the break point
                search_start = max(start, end - 200)
                search_text = text[search_start:end + 100]
                
                # Find sentence boundaries
                sentence_ends = []
                for match in re.finditer(r'[.!?]+\s+', search_text):
                    pos = search_start + match.end()
                    if pos > start + max_chars // 2:  # Don't make chunks too small
                        sentence_ends.append(pos)
                
                if sentence_ends:
                    end = sentence_ends[0]
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            if end >= len(text):
                break
            
            start = end - overlap_chars
        
        return chunks
        
    except Exception as e:
        logger.error(f"Failed simple text chunking: {e}")
        return [text]  # Return original text as fallback


def clean_text(text: str) -> str:
    """
    Clean and normalize text for processing.
    
    Args:
        text: Input text
        
    Returns:
        Cleaned text
    """
    try:
        if not text:
            return ""
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove excessive punctuation
        text = re.sub(r'[.]{3,}', '...', text)
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        
        # Clean up quotes
        text = re.sub(r'["""]', '"', text)
        text = re.sub(r"[â€˜â€™']", '"', text)
        
        # Remove control characters but keep basic formatting
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        
        # Strip and ensure single spaces
        text = text.strip()
        text = re.sub(r' +', ' ', text)
        
        return text
        
    except Exception as e:
        logger.error(f"Failed to clean text: {e}")
        return text


def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """
    Extract potential keywords from text.
    
    Args:
        text: Input text
        max_keywords: Maximum number of keywords to return
        
    Returns:
        List of keywords
    """
    try:
        if not text:
            return []
        
        # Simple keyword extraction (could be enhanced with NLP)
        text = clean_text(text.lower())
        
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'under', 'between', 'among',
            'this', 'that', 'these', 'those', 'i', 'me', 'my', 'myself', 'we',
            'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself',
            'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
            'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs',
            'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these',
            'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'will',
            'would', 'should', 'could', 'can', 'may', 'might', 'must', 'shall'
        }
        
        # Extract words
        words = re.findall(r'\b[a-zA-Z]+\b', text)
        
        # Filter out stop words and short words
        keywords = [
            word for word in words
            if len(word) > 2 and word.lower() not in stop_words
        ]
        
        # Count frequency
        word_freq = {}
        for word in keywords:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency and take top keywords
        sorted_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        return [word for word, freq in sorted_keywords[:max_keywords]]
        
    except Exception as e:
        logger.error(f"Failed to extract keywords: {e}")
        return []


def detect_language(text: str) -> str:
    """
    Simple language detection based on character patterns.
    
    Args:
        text: Input text
        
    Returns:
        Language code (en, ru, kk, etc.) or 'unknown'
    """
    try:
        if not text:
            return 'unknown'
        
        # Simple heuristics based on character sets
        text_sample = text[:500].lower()
        
        # Check for Cyrillic characters
        cyrillic_count = len(re.findall(r'[Ð°-ÑÑ‘]', text_sample))
        
        # Check for Kazakh specific characters
        kazakh_chars = len(re.findall(r'[Ó™Ò“Ò›Ò£Ó©Ò±Ò¯Ò»Ñ–]', text_sample))
        
        # Check for Latin characters
        latin_count = len(re.findall(r'[a-z]', text_sample))
        
        total_chars = cyrillic_count + latin_count + kazakh_chars
        
        if total_chars == 0:
            return 'unknown'
        
        # Determine language
        cyrillic_ratio = cyrillic_count / total_chars
        kazakh_ratio = kazakh_chars / total_chars
        latin_ratio = latin_count / total_chars
        
        if kazakh_ratio > 0.05:  # Presence of Kazakh-specific characters
            return 'kk'
        elif cyrillic_ratio > 0.3:
            return 'ru'
        elif latin_ratio > 0.5:
            return 'en'
        else:
            return 'unknown'
            
    except Exception as e:
        logger.error(f"Failed to detect language: {e}")
        return 'unknown'


def truncate_text(text: str, max_length: int = 1000, ellipsis: str = "...") -> str:
    """
    Truncate text to maximum length with ellipsis.
    
    Args:
        text: Input text
        max_length: Maximum length
        ellipsis: Ellipsis string
        
    Returns:
        Truncated text
    """
    try:
        if not text or len(text) <= max_length:
            return text
        
        # Try to break at word boundary
        truncated = text[:max_length - len(ellipsis)]
        
        # Find last space
        last_space = truncated.rfind(' ')
        
        if last_space > max_length * 0.8:  # Don't cut too much
            truncated = truncated[:last_space]
        
        return truncated + ellipsis
        
    except Exception as e:
        logger.error(f"Failed to truncate text: {e}")
        return text


def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """
    Count tokens in text using tiktoken.
    
    Args:
        text: Input text
        model: Model name for tokenizer
        
    Returns:
        Number of tokens
    """
    try:
        if not text:
            return 0
        
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        
        tokens = encoding.encode(text)
        return len(tokens)
        
    except Exception as e:
        logger.error(f"Failed to count tokens: {e}")
        # Rough estimate: 1 token â‰ˆ 4 characters
        return len(text) // 4


def validate_text_input(text: str, min_length: int = 1, max_length: int = 100000) -> bool:
    """
    Validate text input for processing.
    
    Args:
        text: Input text
        min_length: Minimum text length
        max_length: Maximum text length
        
    Returns:
        True if valid, False otherwise
    """
    try:
        if not isinstance(text, str):
            return False
        
        text = text.strip()
        
        if len(text) < min_length or len(text) > max_length:
            return False
        
        # Check for reasonable character content
        printable_chars = sum(1 for c in text if c.isprintable() or c.isspace())
        
        if printable_chars / len(text) < 0.8:  # At least 80% printable
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Text validation failed: {e}")
        return False