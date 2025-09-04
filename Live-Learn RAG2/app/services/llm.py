"""Gemini LLM service with streaming and retry support."""

from typing import AsyncIterator, Optional

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from ..config import settings
from ..core.exceptions import LLMError


class GeminiLLM:
    """Gemini LLM service with robust error handling."""
    
    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or settings.google_api_key
        self.model_name = model or settings.llm_model
        self.timeout = settings.request_timeout
        
        # Configure Google AI
        genai.configure(api_key=self.api_key)
        
        # Initialize the model with safety settings
        self.model = genai.GenerativeModel(
            self.model_name,
            safety_settings={
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            }
        )
        
        # Generation config
        self.generation_config = genai.types.GenerationConfig(
            temperature=settings.gemini_temperature,
            max_output_tokens=settings.gemini_max_tokens,
            top_p=settings.gemini_top_p,
            top_k=settings.gemini_top_k,
        )
        
        logger.info(f"Initialized Gemini LLM with model: {self.model_name}")
    
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
        Generate text using Gemini.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt (will be prepended)
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
                full_prompt = f"{system_prompt.strip()}\n\n{full_prompt}"
            
            # Override generation config if needed
            gen_config = self.generation_config
            if temperature is not None or max_tokens is not None:
                gen_config = genai.types.GenerationConfig(
                    temperature=temperature or settings.gemini_temperature,
                    max_output_tokens=max_tokens or settings.gemini_max_tokens,
                    top_p=settings.gemini_top_p,
                    top_k=settings.gemini_top_k,
                )
            
            logger.debug(
                "Generating with Gemini",
                model=self.model_name,
                prompt_length=len(full_prompt),
                temperature=gen_config.temperature,
                max_tokens=gen_config.max_output_tokens
            )
            
            # Generate response
            response = self.model.generate_content(
                full_prompt,
                generation_config=gen_config
            )
            
            # Handle blocked content
            if response.candidates[0].finish_reason.name in ["SAFETY", "RECITATION"]:
                logger.warning("Content was blocked by safety filters")
                return "I apologize, but I cannot provide a response due to safety guidelines."
            
            # Extract text
            generated_text = response.text.strip()
            
            if not generated_text:
                logger.warning("Gemini returned empty response")
                return "I apologize, but I couldn't generate a proper response."
            
            logger.info(
                "Gemini generation completed",
                model=self.model_name,
                response_length=len(generated_text),
                finish_reason=response.candidates[0].finish_reason.name
            )
            
            return generated_text
            
        except Exception as e:
            error_msg = f"Failed to generate with Gemini: {str(e)}"
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
                full_prompt = f"{system_prompt.strip()}\n\n{full_prompt}"
            
            # Override generation config if needed
            gen_config = self.generation_config
            if temperature is not None or max_tokens is not None:
                gen_config = genai.types.GenerationConfig(
                    temperature=temperature or settings.gemini_temperature,
                    max_output_tokens=max_tokens or settings.gemini_max_tokens,
                    top_p=settings.gemini_top_p,
                    top_k=settings.gemini_top_k,
                )
            
            logger.debug(
                "Starting streaming generation with Gemini",
                model=self.model_name,
                prompt_length=len(full_prompt)
            )
            
            # Generate streaming response
            response = self.model.generate_content(
                full_prompt,
                generation_config=gen_config,
                stream=True
            )
            
            for chunk in response:
                if chunk.text:
                    yield chunk.text
            
            logger.debug("Gemini streaming generation completed")
            
        except Exception as e:
            error_msg = f"Failed during streaming generation with Gemini: {str(e)}"
            logger.error(error_msg)
            raise LLMError(error_msg) from e
    
    async def health_check(self) -> bool:
        """
        Check if the Gemini service is healthy.
        
        Returns:
            True if service is healthy, False otherwise
        """
        try:
            # Try to generate a simple response
            test_prompt = "Say 'OK' if you are working correctly."
            response = await self.generate(test_prompt, temperature=0.0)
            
            if response and len(response) > 0:
                logger.info("Gemini LLM service health check passed")
                return True
            else:
                logger.warning("Gemini LLM service returned empty response")
                return False
                
        except Exception as e:
            logger.error(f"Gemini LLM service health check failed: {e}")
            return False
    
    async def get_model_info(self) -> dict:
        """
        Get information about the Gemini model.
        
        Returns:
            Model information dict
        """
        try:
            # Get model info from Google AI
            model_info = genai.get_model(f"models/{self.model_name}")
            
            logger.info(f"Found Gemini model: {self.model_name}")
            return {
                "model": self.model_name,
                "available": True,
                "details": {
                    "name": model_info.name,
                    "display_name": model_info.display_name,
                    "description": model_info.description,
                    "input_token_limit": model_info.input_token_limit,
                    "output_token_limit": model_info.output_token_limit,
                    "supported_generation_methods": model_info.supported_generation_methods,
                }
            }
                
        except Exception as e:
            logger.error(f"Failed to get Gemini model info: {e}")
            return {
                "model": self.model_name,
                "available": False,
                "error": str(e)
            }


def build_rag_prompt(query: str, contexts: list, language: str = "auto") -> str:
    """
    Build a RAG prompt optimized for Gemini.
    
    Args:
        query: User question
        contexts: List of context dictionaries
        language: Response language (auto-detect from query)
        
    Returns:
        Formatted prompt string
    """
    # Detect language from query for better response
    if any(ord(char) > 127 for char in query):
        # Contains non-ASCII characters, likely Russian/Kazakh
        lang_instruction = "ÐžÑ‚Ð²ÐµÑ‡Ð°Ð¹Ñ‚Ðµ Ð½Ð° Ñ‚Ð¾Ð¼ Ð¶Ðµ ÑÐ·Ñ‹ÐºÐµ, Ñ‡Ñ‚Ð¾ Ð¸ Ð²Ð¾Ð¿Ñ€Ð¾Ñ."
    else:
        lang_instruction = "Respond in the same language as the question."
    
    context_parts = []
    
    for i, ctx in enumerate(contexts, 1):
        # Format context with metadata
        metadata_info = ""
        if "metadata" in ctx and ctx["metadata"]:
            feedback_info = ""
            if ctx["metadata"].get("feedback_applied"):
                feedback_info = f" (ðŸ“Š feedback: {ctx['metadata'].get('feedback_count', 0)} events)"
            
            source_info = ctx["metadata"].get("source", "original")
            version_info = ctx["metadata"].get("version", 1)
            metadata_info = f" [source: {source_info}, v{version_info}{feedback_info}]"
        
        context_parts.append(
            f"[{i}] doc={ctx.get('doc_id', 'unknown')} "
            f"chunk={ctx.get('chunk_id', 'unknown')} "
            f"score={ctx.get('score', 0.0):.3f}{metadata_info}\n{ctx.get('content', ctx.get('text', ''))}"
        )
    
    context_text = "\n\n".join(context_parts)
    
    # Build the prompt optimized for Gemini
    prompt = f"""You are a helpful AI assistant that provides accurate answers based on provided context.

INSTRUCTIONS:
â€¢ Use ONLY the provided context to answer the question
â€¢ If the context doesn't contain sufficient information, clearly state that you don't know
â€¢ {lang_instruction}
â€¢ When referencing information, cite sources using [number] format
â€¢ Be concise but comprehensive
â€¢ If multiple sources provide conflicting information, mention this

CONTEXT:
{context_text}

QUESTION: {query}

ANSWER:"""
    
    return prompt


# Global LLM service instance
llm_service = GeminiLLM()