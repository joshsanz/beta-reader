"""Ollama client for beta-reader."""

from collections.abc import AsyncGenerator, Iterator

import ollama
from ollama import AsyncClient, Client

from ..core.config import Config, OllamaConfig
from .exceptions import ModelNotFoundError, OllamaConnectionError, ProcessingError


class OllamaClient:
    """Client for interacting with Ollama server."""

    def __init__(self, config: OllamaConfig) -> None:
        """Initialize Ollama client.
        
        Args:
            config: Ollama configuration.
        """
        self.config = config
        self._sync_client = Client(host=config.base_url)
        self._async_client = AsyncClient(host=config.base_url)

    def check_connection(self) -> bool:
        """Check if Ollama server is accessible.
        
        Returns:
            True if server is accessible, False otherwise.
        """
        try:
            self._sync_client.list()
            return True
        except Exception:
            return False

    def list_models(self) -> list[str]:
        """List available models.
        
        Returns:
            List of available model names.
            
        Raises:
            OllamaConnectionError: If connection to server fails.
        """
        try:
            response = self._sync_client.list()
            models = response.get("models", [])
            return [model.get("name", model.get("model", "unknown")) for model in models]
        except Exception as e:
            raise OllamaConnectionError(str(e), self.config.base_url) from e

    def model_exists(self, model_name: str) -> bool:
        """Check if a model exists.
        
        Args:
            model_name: Name of the model to check.
            
        Returns:
            True if model exists, False otherwise.
        """
        try:
            available_models = self.list_models()
            return model_name in available_models
        except OllamaConnectionError:
            return False

    def validate_model(self, model_name: str) -> None:
        """Validate that a model is available.
        
        Args:
            model_name: Name of the model to validate.
            
        Raises:
            OllamaConnectionError: If connection to server fails.
            ModelNotFoundError: If model is not available.
        """
        available_models = self.list_models()
        if model_name not in available_models:
            raise ModelNotFoundError(model_name, available_models)

    def generate(
        self,
        model: str,
        prompt: str,
        system_prompt: str | None = None,
    ) -> str:
        """Generate text using Ollama model (synchronous).
        
        Args:
            model: Model name to use.
            prompt: Input prompt.
            system_prompt: Optional system prompt.
            
        Returns:
            Generated text response.
            
        Raises:
            OllamaConnectionError: If connection fails.
            ModelNotFoundError: If model is not available.
            ProcessingError: If generation fails.
        """
        self.validate_model(model)

        try:
            response = self._sync_client.generate(
                model=model,
                prompt=prompt,
                system=system_prompt,
                stream=False,
                options={
                    "temperature": 0.1,  # Low temperature for consistency
                    "top_p": 0.9,
                    "repeat_penalty": 1.1,
                }
            )
            return response["response"]
        except ollama.RequestError as e:
            raise OllamaConnectionError(str(e), self.config.base_url) from e
        except Exception as e:
            raise ProcessingError(str(e), model) from e

    def generate_stream(
        self,
        model: str,
        prompt: str,
        system_prompt: str | None = None,
    ) -> Iterator[str]:
        """Generate text using Ollama model with streaming (synchronous).
        
        Args:
            model: Model name to use.
            prompt: Input prompt.
            system_prompt: Optional system prompt.
            
        Yields:
            Chunks of generated text.
            
        Raises:
            OllamaConnectionError: If connection fails.
            ModelNotFoundError: If model is not available.
            ProcessingError: If generation fails.
        """
        self.validate_model(model)

        try:
            response_stream = self._sync_client.generate(
                model=model,
                prompt=prompt,
                system=system_prompt,
                stream=True,
                options={
                    "temperature": 0.1,  # Low temperature for consistency
                    "top_p": 0.9,
                    "repeat_penalty": 1.1,
                }
            )

            for chunk in response_stream:
                if "response" in chunk:
                    yield chunk["response"]

        except ollama.RequestError as e:
            raise OllamaConnectionError(str(e), self.config.base_url) from e
        except Exception as e:
            raise ProcessingError(str(e), model) from e

    async def generate_async(
        self,
        model: str,
        prompt: str,
        system_prompt: str | None = None,
    ) -> str:
        """Generate text using Ollama model (asynchronous).
        
        Args:
            model: Model name to use.
            prompt: Input prompt.
            system_prompt: Optional system prompt.
            
        Returns:
            Generated text response.
            
        Raises:
            OllamaConnectionError: If connection fails.
            ModelNotFoundError: If model is not available.
            ProcessingError: If generation fails.
        """
        self.validate_model(model)

        try:
            response = await self._async_client.generate(
                model=model,
                prompt=prompt,
                system=system_prompt,
                stream=False,
                options={
                    "temperature": 0.1,  # Low temperature for consistency
                    "top_p": 0.9,
                    "repeat_penalty": 1.1,
                }
            )
            return response["response"]
        except ollama.RequestError as e:
            raise OllamaConnectionError(str(e), self.config.base_url) from e
        except Exception as e:
            raise ProcessingError(str(e), model) from e

    async def generate_stream_async(
        self,
        model: str,
        prompt: str,
        system_prompt: str | None = None,
    ) -> AsyncGenerator[str, None]:
        """Generate text using Ollama model with streaming (asynchronous).
        
        Args:
            model: Model name to use.
            prompt: Input prompt.
            system_prompt: Optional system prompt.
            
        Yields:
            Chunks of generated text.
            
        Raises:
            OllamaConnectionError: If connection fails.
            ModelNotFoundError: If model is not available.
            ProcessingError: If generation fails.
        """
        self.validate_model(model)

        try:
            response_stream = await self._async_client.generate(
                model=model,
                prompt=prompt,
                system=system_prompt,
                stream=True,
                options={
                    "temperature": 0.1,  # Low temperature for consistency
                    "top_p": 0.9,
                    "repeat_penalty": 1.1,
                }
            )

            async for chunk in response_stream:
                if "response" in chunk:
                    yield chunk["response"]

        except ollama.RequestError as e:
            raise OllamaConnectionError(str(e), self.config.base_url) from e
        except Exception as e:
            raise ProcessingError(str(e), model) from e


def create_client(config: Config | None = None) -> OllamaClient:
    """Create Ollama client with configuration.
    
    Args:
        config: Application configuration. If None, loads from default location.
        
    Returns:
        Configured Ollama client.
    """
    if config is None:
        config = Config.load_from_file()

    return OllamaClient(config.ollama)
