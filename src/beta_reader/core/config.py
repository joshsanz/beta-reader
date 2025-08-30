"""Configuration management for beta-reader."""

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, validator


class OllamaConfig(BaseModel):
    """Ollama server configuration."""

    base_url: str = Field(default="http://localhost:11434", description="Ollama server URL")
    default_model: str = Field(default="llama3.1:8b", description="Default model to use")
    timeout: int = Field(default=60, description="Request timeout in seconds")


class OutputConfig(BaseModel):
    """Output configuration."""

    default_format: str = Field(default="text", description="Default output format")
    streaming: bool = Field(default=True, description="Enable streaming output by default")


class DiffConfig(BaseModel):
    """Diff configuration."""

    default_format: str = Field(default="unified", description="Default diff format: unified or split")

    @validator("default_format")
    def validate_diff_format(cls, v: str) -> str:
        """Validate diff format."""
        if v not in ("unified", "split"):
            raise ValueError("diff format must be 'unified' or 'split'")
        return v


class Config(BaseModel):
    """Main application configuration."""

    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    diff: DiffConfig = Field(default_factory=DiffConfig)

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
                self.dict(),
                f,
                default_flow_style=False,
                indent=2,
                sort_keys=False
            )

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
