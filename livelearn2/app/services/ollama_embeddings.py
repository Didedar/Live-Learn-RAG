"""Ollama embeddings service for better semantic search."""

import asyncio
from typing import List, Optional

import httpx
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import settings
from ..core.exceptions import EmbeddingError
from ..utils.vectors import normalize_vector


class OllamaEmbeddings:
    """Ollama embeddings service using nomic-embed-text or similar models."""
    
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "nomic-embed-text:latest"
    ):
        """
        Initialize Ollama embeddings service.
        
        Args:
            base_url: Ollama server URL
            model: Embedding model name (e.g., 'nomic-embed-text', 'bge-m3')
        """
        self.base_url = base_url.rstrip('/')
        self.model_name = model
        self.dimension = 768  # Standard dimension for most models
        self.timeout = 60.0
        
        logger.info(f"Initialized Ollama embeddings with model: {self.model_name}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def _embed_single(self, text: str) -> List[float]:
        """Embed a single text with retry logic."""
        try:
            clean_text = text.strip()
            if not clean_text:
                logger.warning("Empty text provided for embedding")
                return [0.0] * self.dimension
            
            # Prepare request for Ollama embeddings API
            request_data = {
                "model": self.model_name,
                "prompt": clean_text
            }
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/embeddings",
                    json=request_data
                )
                response.raise_for_status()
                
                result = response.json()
                
                if "embedding" not in result:
                    raise EmbeddingError(f"No embedding in Ollama response: {result}")
                
                embedding = result["embedding"]
                
                if not isinstance(embedding, list) or not embedding:
                    raise EmbeddingError(f"Invalid embedding format from Ollama: {type(embedding)}")
                
                # Normalize the vector
                normalized_embedding = normalize_vector(embedding)
                
                logger.debug(
                    "Generated Ollama embedding",
                    text_length=len(clean_text),
                    embedding_dim=len(normalized_embedding),
                    model=self.model_name
                )
                
                return normalized_embedding
                
        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP error from Ollama embeddings: {e.response.status_code}"
            try:
                error_detail = e.response.json()
                error_msg += f" - {error_detail.get('error', 'Unknown error')}"
            except:
                error_msg += f" - {e.response.text}"
            logger.error(error_msg)
            raise EmbeddingError(error_msg) from e
            
        except httpx.RequestError as e:
            error_msg = f"Connection error to Ollama embeddings at {self.base_url}: {str(e)}"
            logger.error(error_msg)
            raise EmbeddingError(error_msg) from e
            
        except Exception as e:
            error_msg = f"Ollama embedding error: {str(e)}"
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
        
        logger.info(f"Embedding {len(texts)} documents with Ollama")
        
        try:
            embeddings = []
            
            # Process with limited concurrency to avoid overwhelming Ollama
            semaphore = asyncio.Semaphore(2)  # Max 2 concurrent requests
            
            async def embed_with_semaphore(text: str) -> List[float]:
                async with semaphore:
                    return await self._embed_single(text)
            
            # Process all texts
            results = await asyncio.gather(
                *[embed_with_semaphore(text) for text in texts],
                return_exceptions=True
            )
            
            # Handle results
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Failed to embed document {i}: {result}")
                    embeddings.append([0.0] * self.dimension)  # Fallback zero vector
                else:
                    embeddings.append(result)
            
            logger.info(f"Successfully embedded {len(embeddings)} documents with Ollama")
            return embeddings
            
        except Exception as e:
            error_msg = f"Failed to embed documents with Ollama: {str(e)}"
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
        logger.debug(f"Embedding query with Ollama: {text[:100]}...")
        
        try:
            embedding = await self._embed_single(text)
            logger.debug(f"Query embedded successfully, dimension: {len(embedding)}")
            return embedding
            
        except Exception as e:
            error_msg = f"Failed to embed query with Ollama: {str(e)}"
            logger.error(error_msg)
            raise EmbeddingError(error_msg) from e
    
    async def health_check(self) -> bool:
        """Check if the Ollama embedding service is healthy."""
        try:
            # Check if Ollama is running
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()
                
                # Check if embedding model is available
                models_data = response.json()
                available_models = [model["name"] for model in models_data.get("models", [])]
                
                if self.model_name not in available_models:
                    logger.warning(f"Embedding model {self.model_name} not found. Available: {available_models}")
                    # Try to pull the model
                    await self.pull_model()
            
            # Test embedding generation
            test_embedding = await self._embed_single("Health check test")
            
            if test_embedding and len(test_embedding) > 0:
                logger.info("Ollama embedding service health check passed")
                return True
            else:
                logger.warning("Ollama embedding service returned empty embedding")
                return False
                
        except Exception as e:
            logger.error(f"Ollama embedding service health check failed: {e}")
            return False
    
    async def pull_model(self) -> bool:
        """Pull/download the embedding model if it's not available."""
        try:
            logger.info(f"Pulling embedding model {self.model_name} from Ollama...")
            
            async with httpx.AsyncClient(timeout=300.0) as client:  # 5 minutes for model download
                response = await client.post(
                    f"{self.base_url}/api/pull",
                    json={"name": self.model_name}
                )
                response.raise_for_status()
                
                logger.info(f"Successfully pulled embedding model {self.model_name}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to pull embedding model {self.model_name}: {e}")
            return False
    
    async def get_model_info(self) -> dict:
        """Get information about the current embedding model."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()
                
                models_data = response.json()
                available_models = [model["name"] for model in models_data.get("models", [])]
                
                return {
                    "model_name": self.model_name,
                    "dimension": self.dimension,
                    "available": self.model_name in available_models,
                    "ollama_url": self.base_url,
                    "all_models": available_models
                }
                
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            return {
                "model_name": self.model_name,
                "dimension": self.dimension,
                "available": False,
                "error": str(e),
                "ollama_url": self.base_url
            }


class HybridEmbeddingService:
    """
    Hybrid embedding service that falls back from Ollama to Mock embeddings.
    """
    
    def __init__(self):
        self.ollama_embeddings = None
        self.mock_embeddings = None
        self.use_ollama = False
        
        # Try to initialize Ollama embeddings
        try:
            from .mock_embeddings import MockEmbeddings
            
            ollama_url = getattr(settings, 'ollama_url', 'http://localhost:11434')
            embedding_model = getattr(settings, 'ollama_embedding_model', 'nomic-embed-text:latest')
            
            self.ollama_embeddings = OllamaEmbeddings(
                base_url=ollama_url,
                model=embedding_model
            )
            self.mock_embeddings = MockEmbeddings()
            
            logger.info("Initialized hybrid embedding service (Ollama + Mock fallback)")
            
        except Exception as e:
            logger.warning(f"Failed to initialize Ollama embeddings: {e}")
            from .mock_embeddings import MockEmbeddings
            self.mock_embeddings = MockEmbeddings()
            logger.info("Using Mock embeddings only")
    
    async def _check_ollama_health(self) -> bool:
        """Check if Ollama embeddings are available."""
        if not self.ollama_embeddings:
            return False
        
        try:
            return await self.ollama_embeddings.health_check()
        except Exception:
            return False
    
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents with fallback."""
        if not texts:
            return []
        
        # Try Ollama first
        if self.ollama_embeddings and not self.use_ollama:
            ollama_healthy = await self._check_ollama_health()
            if ollama_healthy:
                self.use_ollama = True
                logger.info("Switching to Ollama embeddings")
        
        if self.use_ollama and self.ollama_embeddings:
            try:
                return await self.ollama_embeddings.embed_documents(texts)
            except Exception as e:
                logger.warning(f"Ollama embeddings failed, falling back to mock: {e}")
                self.use_ollama = False
        
        # Fallback to mock embeddings
        logger.info("Using mock embeddings")
        return await self.mock_embeddings.embed_documents(texts)
    
    async def embed_query(self, text: str) -> List[float]:
        """Embed query with fallback."""
        if self.use_ollama and self.ollama_embeddings:
            try:
                return await self.ollama_embeddings.embed_query(text)
            except Exception as e:
                logger.warning(f"Ollama query embedding failed, falling back to mock: {e}")
                self.use_ollama = False
        
        # Fallback to mock embeddings
        return await self.mock_embeddings.embed_query(text)
    
    async def health_check(self) -> bool:
        """Health check for hybrid service."""
        if self.ollama_embeddings:
            ollama_healthy = await self._check_ollama_health()
            if ollama_healthy:
                return True
        
        if self.mock_embeddings:
            return await self.mock_embeddings.health_check()
        
        return False
    
    def get_current_service(self) -> str:
        """Get name of currently used embedding service."""
        if self.use_ollama and self.ollama_embeddings:
            return f"ollama_{self.ollama_embeddings.model_name}"
        else:
            return "mock_embeddings"


# Factory function to get the appropriate embedding service
def get_embedding_service():
    """Get the best available embedding service."""
    return HybridEmbeddingService()

