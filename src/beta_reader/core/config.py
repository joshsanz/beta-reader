"""Configuration management for beta-reader."""

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator


# ============================================================================
# Configuration Data Models
# ============================================================================

class ModelSpecificConfig(BaseModel):
    """Configuration for a specific model."""

    timeout: int | None = Field(default=None, description="Model-specific timeout in seconds")
    max_tokens: int | None = Field(default=None, description="Maximum tokens to generate")
    temperature: float | None = Field(default=None, description="Temperature for randomness (0.0-1.0)")
    top_p: float | None = Field(default=None, description="Nucleus sampling parameter")
    top_k: int | None = Field(default=None, description="Top-k sampling parameter")
    repeat_penalty: float | None = Field(default=None, description="Penalty for repeated tokens")
    system_prompt_override: str | None = Field(default=None, description="Override system prompt for this model")


class OllamaConfig(BaseModel):
    """Ollama server configuration."""

    base_url: str = Field(default="http://localhost:11434", description="Ollama server URL")
    default_model: str = Field(default="llama3.1:8b", description="Default model to use")
    timeout: int = Field(default=60, description="Request timeout in seconds")
    model_configs: dict[str, ModelSpecificConfig] = Field(
        default_factory=dict,
        description="Model-specific configurations"
    )


class OutputConfig(BaseModel):
    """Output configuration."""

    default_format: str = Field(default="text", description="Default output format")
    streaming: bool = Field(default=True, description="Enable streaming output by default")


class DiffConfig(BaseModel):
    """Diff configuration."""

    default_format: str = Field(default="unified", description="Default diff format: unified or split")

    @field_validator("default_format")
    @classmethod
    def validate_diff_format(cls, v: str) -> str:
        """Validate diff format."""
        if v not in ("unified", "split"):
            raise ValueError("diff format must be 'unified' or 'split'")
        return v


class ChunkingConfig(BaseModel):
    """Text chunking configuration."""

    target_word_count: int = Field(default=500, description="Target number of words per chunk")
    max_word_count: int = Field(default=750, description="Maximum words per chunk before forced split")

    @field_validator("target_word_count")
    @classmethod
    def validate_target_word_count(cls, v: int) -> int:
        """Validate target word count."""
        if v < 50:
            raise ValueError("target_word_count must be at least 50")
        if v > 2000:
            raise ValueError("target_word_count must be at most 2000")
        return v

    @field_validator("max_word_count")
    @classmethod
    def validate_max_word_count(cls, v: int) -> int:
        """Validate max word count."""
        if v < 100:
            raise ValueError("max_word_count must be at least 100")
        if v > 3000:
            raise ValueError("max_word_count must be at most 3000")
        return v

    def model_post_init(self, __context) -> None:
        """Validate that max_word_count >= target_word_count."""
        if self.max_word_count < self.target_word_count:
            raise ValueError("max_word_count must be greater than or equal to target_word_count")


class Config(BaseModel):
    """Main application configuration."""

    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    diff: DiffConfig = Field(default_factory=DiffConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)

    # ========================================================================
    # Configuration Loading and Management
    # ========================================================================

    @classmethod
    def load_from_file(cls, config_path: Path | None = None) -> "Config":
        """Load configuration from YAML file.

        Args:
            config_path: Path to configuration file. If None, uses default locations.

        Returns:
            Config instance.
        """
        if config_path is None:
            config_path = cls._find_config_file()

        if config_path and config_path.exists():
            with open(config_path, encoding="utf-8") as f:
                config_data = yaml.safe_load(f) or {}
            return cls(**config_data)

        # Return default configuration if no file found
        return cls()

    @classmethod
    def _find_config_file(cls) -> Path | None:
        """Find configuration file in default locations.

        Returns:
            Path to configuration file, or None if not found.
        """
        possible_paths = [
            Path.cwd() / "config" / "config.yaml",
            Path.cwd() / "config.yaml",
            Path.cwd() / ".beta-reader.yaml",
            Path.home() / ".config" / "beta-reader" / "config.yaml",
            Path.home() / ".beta-reader.yaml",
        ]

        for path in possible_paths:
            if path.exists():
                return path

        return None

    def save_to_file(self, config_path: Path) -> None:
        """Save configuration to YAML file.

        Args:
            config_path: Path where to save the configuration.
        """
        config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(
                self.model_dump(),
                f,
                default_flow_style=False,
                indent=2,
                sort_keys=False
            )

    # ========================================================================
    # Model and System Configuration Helpers
    # ========================================================================

    def get_system_prompt_path(self) -> Path:
        """Get path to system prompt file.

        Returns:
            Path to system_prompt.txt file.
        """
        # Check config directory first (matches processor logic)
        config_prompt_path = Path.cwd() / "config" / "system_prompt.txt"
        if config_prompt_path.exists():
            return config_prompt_path

        # Look in current directory
        current_dir_prompt = Path.cwd() / "system_prompt.txt"
        if current_dir_prompt.exists():
            return current_dir_prompt

        # Fallback to package directory
        package_dir = Path(__file__).parent.parent.parent.parent
        return package_dir / "system_prompt.txt"

    def get_model_config(self, model_name: str) -> ModelSpecificConfig:
        """Get configuration for a specific model.

        Args:
            model_name: Name of the model to get config for.

        Returns:
            Model-specific configuration, or default config if not found.
        """
        return self.ollama.model_configs.get(model_name, ModelSpecificConfig())

    def get_effective_timeout(self, model_name: str) -> int:
        """Get effective timeout for a model (model-specific or default).

        Args:
            model_name: Name of the model.

        Returns:
            Timeout in seconds.
        """
        model_config = self.get_model_config(model_name)
        return model_config.timeout if model_config.timeout is not None else self.ollama.timeout

    def get_effective_system_prompt(self, model_name: str, default_prompt: str) -> str:
        """Get effective system prompt for a model.

        Args:
            model_name: Name of the model.
            default_prompt: Default system prompt.

        Returns:
            System prompt to use (model-specific override or default).
        """
        model_config = self.get_model_config(model_name)
        return model_config.system_prompt_override if model_config.system_prompt_override else default_prompt
