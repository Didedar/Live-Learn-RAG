"""Embedding service with Google AI and OpenAI support."""

import asyncio
from typing import List, Optional

import google.generativeai as genai
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from ..config import settings
from ..core.exceptions import EmbeddingError
from ..utils.vectors import normalize_vector


class GoogleEmbeddings:
    """Google AI embeddings service."""
    
    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or settings.google_api_key
        self.model = model or settings.embedding_model
        
        # Configure Google AI
        genai.configure(api_key=self.api_key)
        
        logger.info(f"Initialized Google embeddings with model: {self.model}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception)
    )
    async def _embed_single(self, text: str, task_type: str = "retrieval_document") -> List[float]:
        """Embed a single text with retry logic."""
        try:
            clean_text = text.strip()
            if not clean_text:
                logger.warning("Empty text provided for embedding")
                return [0.0] * 768  # Google embeddings are typically 768-dimensional
            
            # Use Google AI embedding
            result = genai.embed_content(
                model=f"models/{self.model}",
                content=clean_text,
                task_type=task_type
            )
            
            if not result or 'embedding' not in result:
                raise EmbeddingError(f"No embedding in response: {result}")
            
            embedding = result['embedding']
            
            if not isinstance(embedding, list) or not embedding:
                raise EmbeddingError(f"Invalid embedding format: {type(embedding)}")
            
            # Normalize the vector
            normalized_embedding = normalize_vector(embedding)
            
            logger.debug(
                "Generated Google embedding",
                text_length=len(clean_text),
                embedding_dim=len(normalized_embedding),
                model=self.model
            )
            
            return normalized_embedding
            
        except Exception as e:
            error_msg = f"Google embedding error: {str(e)}"
            logger.error(error_msg)
            raise EmbeddingError(error_msg) from e
    
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple documents.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            logger.warning("Empty text list provided for embedding")
            return []
        
        logger.info(f"Embedding {len(texts)} documents with Google AI")
        
        try:
            embeddings = []
            
            # Process in batches to avoid rate limits
            batch_size = 10
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                # Process batch with limited concurrency
                semaphore = asyncio.Semaphore(3)  # Max 3 concurrent requests
                
                async def embed_with_semaphore(text: str) -> List[float]:
                    async with semaphore:
                        return await self._embed_single(text, "retrieval_document")
                
                batch_embeddings = await asyncio.gather(
                    *[embed_with_semaphore(text) for text in batch],
                    return_exceptions=True
                )
                
                # Handle results
                for j, result in enumerate(batch_embeddings):
                    if isinstance(result, Exception):
                        logger.error(f"Failed to embed document {i + j}: {result}")
                        embeddings.append([0.0] * 768)  # Fallback zero vector
                    else:
                        embeddings.append(result)
                
                # Rate limiting delay
                if i + batch_size < len(texts):
                    await asyncio.sleep(0.5)  # 500ms delay between batches
            
            logger.info(f"Successfully embedded {len(embeddings)} documents")
            return embeddings
            
        except Exception as e:
            error_msg = f"Failed to embed documents with Google AI: {str(e)}"
            logger.error(error_msg)
            raise EmbeddingError(error_msg) from e
    
    async def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text.
        
        Args:
            text: Query text to embed
            
        Returns:
            Embedding vector
        """
        logger.debug(f"Embedding query with Google AI: {text[:100]}...")
        
        try:
            embedding = await self._embed_single(text, "retrieval_query")
            logger.debug(f"Query embedded successfully, dimension: {len(embedding)}")
            return embedding
            
        except Exception as e:
            error_msg = f"Failed to embed query with Google AI: {str(e)}"
            logger.error(error_msg)
            raise EmbeddingError(error_msg) from e
    
    async def health_check(self) -> bool:
        """Check if the embedding service is healthy."""
        try:
            test_embedding = await self._embed_single("Health check test")
            logger.info("Google AI embedding service health check passed")
            return True
        except Exception as e:
            logger.error(f"Google AI embedding service health check failed: {e}")
            return False


class OpenAIEmbeddings:
    """OpenAI embeddings service as alternative."""
    
    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or settings.openai_api_key
        self.model = model or settings.openai_embedding_model
        
        if not self.api_key:
            raise EmbeddingError("OpenAI API key required")
        
        logger.info(f"Initialized OpenAI embeddings with model: {self.model}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def _embed_single(self, text: str) -> List[float]:
        """Embed single text with OpenAI."""
        try:
            from openai import AsyncOpenAI
            
            client = AsyncOpenAI(api_key=self.api_key)
            
            response = await client.embeddings.create(
                input=text.strip(),
                model=self.model
            )
            
            embedding = response.data[0].embedding
            return normalize_vector(embedding)
            
        except Exception as e:
            error_msg = f"OpenAI embedding error: {str(e)}"
            logger.error(error_msg)
            raise EmbeddingError(error_msg) from e
    
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents with OpenAI."""
        if not texts:
            return []
        
        logger.info(f"Embedding {len(texts)} documents with OpenAI")
        
        try:
            embeddings = []
            
            # OpenAI allows batch processing
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=self.api_key)
            
            response = await client.embeddings.create(
                input=texts,
                model=self.model
            )
            
            for data in response.data:
                embedding = normalize_vector(data.embedding)
                embeddings.append(embedding)
            
            logger.info(f"Successfully embedded {len(embeddings)} documents with OpenAI")
            return embeddings
            
        except Exception as e:
            error_msg = f"Failed to embed documents with OpenAI: {str(e)}"
            logger.error(error_msg)
            raise EmbeddingError(error_msg) from e
    
    async def embed_query(self, text: str) -> List[float]:
        """Embed query with OpenAI."""
        return await self._embed_single(text)
    
    async def health_check(self) -> bool:
        """Health check for OpenAI embeddings."""
        try:
            await self._embed_single("Health check test")
            logger.info("OpenAI embedding service health check passed")
            return True
        except Exception as e:
            logger.error(f"OpenAI embedding service health check failed: {e}")
            return False


# Factory function to get the appropriate embedding service
def get_embedding_service():
    """Get the configured embedding service."""
    if settings.use_openai_embeddings:
        return OpenAIEmbeddings()
    else:
        return GoogleEmbeddings()


# Global embedding service instance
embedding_service = get_embedding_service()