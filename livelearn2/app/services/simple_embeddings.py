"""Simple local embeddings using sentence-transformers."""

import asyncio
from typing import List
from loguru import logger

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers not available, falling back to mock embeddings")

from .mock_embeddings import MockEmbeddings


class SimpleEmbeddings:
    """Local embeddings using sentence-transformers."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize with a lightweight sentence-transformer model.
        
        Args:
            model_name: HuggingFace model name (default: all-MiniLM-L6-v2, ~80MB)
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.warning("Falling back to MockEmbeddings")
            self.fallback = MockEmbeddings()
            self.model = None
            self.dimension = 768
            return
        
        try:
            logger.info(f"Loading sentence-transformer model: {model_name}")
            self.model = SentenceTransformer(model_name)
            self.dimension = self.model.get_sentence_embedding_dimension()
            self.fallback = None
            logger.info(f"Loaded model with dimension: {self.dimension}")
        except Exception as e:
            logger.error(f"Failed to load sentence-transformer: {e}")
            logger.warning("Falling back to MockEmbeddings")
            self.fallback = MockEmbeddings()
            self.model = None
            self.dimension = 768
    
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents."""
        if self.fallback:
            return await self.fallback.embed_documents(texts)
        
        if not texts:
            return []
        
        logger.info(f"Embedding {len(texts)} documents with sentence-transformer")
        
        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None, 
                lambda: self.model.encode(texts, convert_to_numpy=True)
            )
            
            # Convert to list of lists
            return [embedding.tolist() for embedding in embeddings]
            
        except Exception as e:
            logger.error(f"Error in sentence-transformer embedding: {e}")
            # Fallback to mock if sentence-transformer fails
            if not self.fallback:
                self.fallback = MockEmbeddings()
            return await self.fallback.embed_documents(texts)
    
    async def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        if self.fallback:
            return await self.fallback.embed_query(text)
        
        logger.debug(f"Embedding query with sentence-transformer: {text[:50]}...")
        
        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                None, 
                lambda: self.model.encode([text], convert_to_numpy=True)
            )
            
            return embedding[0].tolist()
            
        except Exception as e:
            logger.error(f"Error in sentence-transformer query embedding: {e}")
            # Fallback to mock if sentence-transformer fails
            if not self.fallback:
                self.fallback = MockEmbeddings()
            return await self.fallback.embed_query(text)
    
    async def health_check(self) -> bool:
        """Health check."""
        if self.fallback:
            return await self.fallback.health_check()
        
        try:
            # Test embedding
            test_embedding = await self.embed_query("test")
            return len(test_embedding) == self.dimension
        except Exception:
            return False
