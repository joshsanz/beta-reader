"""Custom exceptions for LLM operations."""


class BetaReaderError(Exception):
    """Base exception for beta-reader application."""
    pass


class LLMError(BetaReaderError):
    """Base exception for LLM-related errors."""
    pass


class OllamaConnectionError(LLMError):
    """Raised when connection to Ollama server fails."""

    def __init__(self, message: str, base_url: str) -> None:
        self.base_url = base_url
        super().__init__(f"Failed to connect to Ollama at {base_url}: {message}")


class ModelNotFoundError(LLMError):
    """Raised when requested model is not available."""

    def __init__(self, model_name: str, available_models: list[str] | None = None) -> None:
        self.model_name = model_name
        self.available_models = available_models or []

        if available_models:
            message = (
                f"Model '{model_name}' not found. "
                f"Available models: {', '.join(available_models)}"
            )
        else:
            message = f"Model '{model_name}' not found."

        super().__init__(message)


class ProcessingError(LLMError):
    """Raised when LLM processing fails."""

    def __init__(self, message: str, model_name: str | None = None) -> None:
        self.model_name = model_name

        if model_name:
            message = f"Processing failed with model '{model_name}': {message}"

        super().__init__(message)


class ConfigurationError(BetaReaderError):
    """Raised when configuration is invalid."""
    pass


class FileProcessingError(BetaReaderError):
    """Raised when file processing fails."""

    def __init__(self, message: str, file_path: str | None = None) -> None:
        self.file_path = file_path

        if file_path:
            message = f"Failed to process file '{file_path}': {message}"

        super().__init__(message)
