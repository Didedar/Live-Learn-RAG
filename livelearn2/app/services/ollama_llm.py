"""Ollama LLM service for local Llama models."""

import asyncio
from typing import AsyncIterator, Optional, Dict, Any

import httpx
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from ..config import settings
from ..core.exceptions import LLMError
from .positive_prompts import build_positive_ollama_prompt, enhance_answer_positivity


class OllamaLLM:
    """Ollama LLM service for local Llama models."""
    
    def __init__(self, 
                 base_url: str = "http://localhost:11434",
                 model: str = "llama3.2:latest"):
        """
        Initialize Ollama LLM service.
        
        Args:
            base_url: Ollama server URL
            model: Model name to use (e.g., 'llama3.2:latest', 'llama3.2:3b')
        """
        self.base_url = base_url.rstrip('/')
        self.model_name = model
        self.timeout = getattr(settings, 'request_timeout', 60)
        
        # Generation parameters
        self.temperature = 0.1  # Default temperature for Ollama
        self.max_tokens = 8192  # Default max tokens for Ollama
        self.top_p = 0.8  # Default top_p for Ollama
        
        logger.info(f"Initialized Ollama LLM with model: {self.model_name} at {self.base_url}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((Exception,))
    )
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate text using Ollama.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature override
            max_tokens: Max tokens override
            
        Returns:
            Generated text
            
        Raises:
            LLMError: If generation fails
        """
        try:
            # Validate inputs
            if not prompt or not prompt.strip():
                raise LLMError("Empty prompt provided")
            
            # Prepare the full prompt
            full_prompt = prompt.strip()
            if system_prompt:
                full_prompt = f"System: {system_prompt.strip()}\n\nUser: {full_prompt}"
            
            # Prepare request parameters
            request_data = {
                "model": self.model_name,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": temperature or self.temperature,
                    "num_predict": max_tokens or self.max_tokens,
                    "top_p": self.top_p,
                }
            }
            
            logger.debug(
                "Generating with Ollama",
                model=self.model_name,
                prompt_length=len(full_prompt),
                temperature=request_data["options"]["temperature"]
            )
            
            # Make request to Ollama
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json=request_data
                )
                response.raise_for_status()
                
                result = response.json()
                
                if "response" not in result:
                    raise LLMError(f"Invalid response format from Ollama: {result}")
                
                generated_text = result["response"].strip()
                
                if not generated_text:
                    logger.warning("Ollama returned empty response")
                    return "I apologize, but I couldn't generate a proper response."
                
                logger.info(
                    "Ollama generation completed",
                    model=self.model_name,
                    response_length=len(generated_text),
                    done=result.get("done", False)
                )
                
                return generated_text
                
        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP error from Ollama: {e.response.status_code}"
            try:
                error_detail = e.response.json()
                error_msg += f" - {error_detail.get('error', 'Unknown error')}"
            except:
                error_msg += f" - {e.response.text}"
            logger.error(error_msg)
            raise LLMError(error_msg) from e
            
        except httpx.RequestError as e:
            error_msg = f"Connection error to Ollama at {self.base_url}: {str(e)}"
            logger.error(error_msg)
            raise LLMError(error_msg) from e
            
        except Exception as e:
            error_msg = f"Failed to generate with Ollama: {str(e)}"
            logger.error(error_msg)
            raise LLMError(error_msg) from e
    
    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> AsyncIterator[str]:
        """
        Generate text with streaming.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature override
            max_tokens: Max tokens override
            
        Yields:
            Text chunks as they are generated
        """
        try:
            # Prepare the full prompt
            full_prompt = prompt.strip()
            if system_prompt:
                full_prompt = f"System: {system_prompt.strip()}\n\nUser: {full_prompt}"
            
            # Prepare request parameters
            request_data = {
                "model": self.model_name,
                "prompt": full_prompt,
                "stream": True,
                "options": {
                    "temperature": temperature or self.temperature,
                    "num_predict": max_tokens or self.max_tokens,
                    "top_p": self.top_p,
                }
            }
            
            logger.debug(
                "Starting streaming generation with Ollama",
                model=self.model_name,
                prompt_length=len(full_prompt)
            )
            
            # Stream response from Ollama
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/api/generate",
                    json=request_data
                ) as response:
                    response.raise_for_status()
                    
                    async for line in response.aiter_lines():
                        if line.strip():
                            try:
                                chunk_data = httpx._content.json_loads(line)
                                if "response" in chunk_data and chunk_data["response"]:
                                    yield chunk_data["response"]
                                    
                                if chunk_data.get("done", False):
                                    break
                                    
                            except Exception as e:
                                logger.warning(f"Failed to parse streaming chunk: {e}")
                                continue
            
            logger.debug("Ollama streaming generation completed")
            
        except Exception as e:
            error_msg = f"Failed during streaming generation with Ollama: {str(e)}"
            logger.error(error_msg)
            raise LLMError(error_msg) from e
    
    async def health_check(self) -> bool:
        """
        Check if the Ollama service is healthy.
        
        Returns:
            True if service is healthy, False otherwise
        """
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                # Check if Ollama is running
                response = await client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()
                
                # Check if our model is available
                models_data = response.json()
                available_models = [model["name"] for model in models_data.get("models", [])]
                
                if self.model_name not in available_models:
                    logger.warning(f"Model {self.model_name} not found. Available: {available_models}")
                    # Try to pull the model
                    await self.pull_model()
                
                # Test generation
                test_response = await self.generate("Say 'OK' if you are working correctly.", temperature=0.0)
                
                if test_response and len(test_response) > 0:
                    logger.info("Ollama LLM service health check passed")
                    return True
                else:
                    logger.warning("Ollama LLM service returned empty response")
                    return False
                    
        except Exception as e:
            logger.error(f"Ollama LLM service health check failed: {e}")
            return False
    
    async def pull_model(self) -> bool:
        """
        Pull/download the model if it's not available.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Pulling model {self.model_name} from Ollama...")
            
            async with httpx.AsyncClient(timeout=300.0) as client:  # 5 minutes for model download
                response = await client.post(
                    f"{self.base_url}/api/pull",
                    json={"name": self.model_name}
                )
                response.raise_for_status()
                
                logger.info(f"Successfully pulled model {self.model_name}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to pull model {self.model_name}: {e}")
            return False
    
    async def list_models(self) -> Dict[str, Any]:
        """
        List available models in Ollama.
        
        Returns:
            Dictionary with model information
        """
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()
                
                models_data = response.json()
                available_models = [model["name"] for model in models_data.get("models", [])]
                
                return {
                    "current_model": self.model_name,
                    "available_models": available_models,
                    "model_available": self.model_name in available_models,
                    "ollama_url": self.base_url
                }
                
        except Exception as e:
            logger.error(f"Failed to list Ollama models: {e}")
            return {
                "current_model": self.model_name,
                "available_models": [],
                "model_available": False,
                "error": str(e),
                "ollama_url": self.base_url
            }


def build_rag_prompt_for_ollama(query: str, contexts: list, language: str = "auto") -> str:
    """
    Build a positive RAG prompt optimized for Llama models via Ollama.
    Uses the new positive prompting approach.
    """
    # Use the new positive prompt builder
    return build_positive_ollama_prompt(query, contexts, language="ru")


# Global LLM service instance - will be initialized based on config
def get_llm_service():
    """Get Ollama LLM service (local-only mode)."""
    # Always use Ollama now
    ollama_url = getattr(settings, 'ollama_url', 'http://localhost:11434')
    ollama_model = getattr(settings, 'ollama_model', 'llama3.2:latest')
    return OllamaLLM(base_url=ollama_url, model=ollama_model)


# For backward compatibility
llm_service = None  # Will be initialized when needed
