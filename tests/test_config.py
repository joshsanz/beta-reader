"""Tests for configuration management."""

import tempfile
from pathlib import Path
from unittest.mock import mock_open, patch

import pytest
import yaml
from pydantic import ValidationError

from beta_reader.core.config import (
    ChunkingConfig,
    Config,
    DiffConfig,
    ModelSpecificConfig,
    OllamaConfig,
    OutputConfig,
)


class TestModelSpecificConfig:
    """Test ModelSpecificConfig class."""

    def test_default_values(self):
        """Test default values for ModelSpecificConfig."""
        config = ModelSpecificConfig()
        
        assert config.timeout is None
        assert config.max_tokens is None
        assert config.temperature is None
        assert config.top_p is None
        assert config.top_k is None
        assert config.repeat_penalty is None
        assert config.system_prompt_override is None

    def test_custom_values(self):
        """Test ModelSpecificConfig with custom values."""
        config = ModelSpecificConfig(
            timeout=120,
            max_tokens=4000,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            repeat_penalty=1.1,
            system_prompt_override="Custom prompt"
        )
        
        assert config.timeout == 120
        assert config.max_tokens == 4000
        assert config.temperature == 0.7
        assert config.top_p == 0.9
        assert config.top_k == 50
        assert config.repeat_penalty == 1.1
        assert config.system_prompt_override == "Custom prompt"


class TestOllamaConfig:
    """Test OllamaConfig class."""

    def test_default_values(self):
        """Test default values for OllamaConfig."""
        config = OllamaConfig()
        
        assert config.base_url == "http://localhost:11434"
        assert config.default_model == "llama3.1:8b"
        assert config.timeout == 60
        assert config.model_configs == {}

    def test_custom_values(self):
        """Test OllamaConfig with custom values."""
        model_config = ModelSpecificConfig(timeout=120)
        config = OllamaConfig(
            base_url="http://custom.server:8080",
            default_model="custom:model",
            timeout=90,
            model_configs={"custom:model": model_config}
        )
        
        assert config.base_url == "http://custom.server:8080"
        assert config.default_model == "custom:model"
        assert config.timeout == 90
        assert "custom:model" in config.model_configs


class TestOutputConfig:
    """Test OutputConfig class."""

    def test_default_values(self):
        """Test default values for OutputConfig."""
        config = OutputConfig()
        
        assert config.default_format == "text"
        assert config.streaming is True

    def test_custom_values(self):
        """Test OutputConfig with custom values."""
        config = OutputConfig(
            default_format="json",
            streaming=False
        )
        
        assert config.default_format == "json"
        assert config.streaming is False


class TestDiffConfig:
    """Test DiffConfig class."""

    def test_default_values(self):
        """Test default values for DiffConfig."""
        config = DiffConfig()
        assert config.default_format == "unified"

    def test_valid_formats(self):
        """Test valid diff formats."""
        config1 = DiffConfig(default_format="unified")
        config2 = DiffConfig(default_format="split")
        
        assert config1.default_format == "unified"
        assert config2.default_format == "split"

    def test_invalid_format(self):
        """Test invalid diff format raises ValueError."""
        with pytest.raises(ValidationError, match="diff format must be"):
            DiffConfig(default_format="invalid")


class TestChunkingConfig:
    """Test ChunkingConfig class."""

    def test_default_values(self):
        """Test default values for ChunkingConfig."""
        config = ChunkingConfig()
        
        assert config.target_word_count == 500
        assert config.max_word_count == 750

    def test_valid_custom_values(self):
        """Test valid custom values."""
        config = ChunkingConfig(
            target_word_count=300,
            max_word_count=400
        )
        
        assert config.target_word_count == 300
        assert config.max_word_count == 400

    def test_target_word_count_too_small(self):
        """Test target_word_count below minimum."""
        with pytest.raises(ValidationError, match="target_word_count must be at least 50"):
            ChunkingConfig(target_word_count=25)

    def test_target_word_count_too_large(self):
        """Test target_word_count above maximum."""
        with pytest.raises(ValidationError, match="target_word_count must be at most 2000"):
            ChunkingConfig(target_word_count=2500)

    def test_max_word_count_too_small(self):
        """Test max_word_count below minimum."""
        with pytest.raises(ValidationError, match="max_word_count must be at least 100"):
            ChunkingConfig(max_word_count=50)

    def test_max_word_count_too_large(self):
        """Test max_word_count above maximum."""
        with pytest.raises(ValidationError, match="max_word_count must be at most 3000"):
            ChunkingConfig(max_word_count=3500)

    def test_max_less_than_target(self):
        """Test max_word_count less than target_word_count."""
        with pytest.raises(ValidationError, match="max_word_count must be greater than or equal to target_word_count"):
            ChunkingConfig(target_word_count=600, max_word_count=500)

    def test_max_equal_to_target(self):
        """Test max_word_count equal to target_word_count."""
        config = ChunkingConfig(target_word_count=500, max_word_count=500)
        assert config.target_word_count == 500
        assert config.max_word_count == 500


class TestConfig:
    """Test main Config class."""

    def test_default_values(self):
        """Test Config with all default values."""
        config = Config()
        
        assert isinstance(config.ollama, OllamaConfig)
        assert isinstance(config.output, OutputConfig)
        assert isinstance(config.diff, DiffConfig)
        assert isinstance(config.chunking, ChunkingConfig)
        
        # Check some defaults
        assert config.ollama.default_model == "llama3.1:8b"
        assert config.output.streaming is True
        assert config.diff.default_format == "unified"
        assert config.chunking.target_word_count == 500

    def test_custom_values(self):
        """Test Config with custom nested values."""
        config = Config(
            ollama=OllamaConfig(default_model="custom:model"),
            output=OutputConfig(streaming=False),
            diff=DiffConfig(default_format="split"),
            chunking=ChunkingConfig(target_word_count=300, max_word_count=400)
        )
        
        assert config.ollama.default_model == "custom:model"
        assert config.output.streaming is False
        assert config.diff.default_format == "split"
        assert config.chunking.target_word_count == 300

    def test_load_from_nonexistent_file(self):
        """Test loading from non-existent file returns defaults."""
        with patch.object(Config, '_find_config_file', return_value=None):
            config = Config.load_from_file()
            
            # Should have default values
            assert config.ollama.default_model == "llama3.1:8b"
            assert config.chunking.target_word_count == 500

    def test_load_from_existing_file(self):
        """Test loading from existing file."""
        config_data = {
            'ollama': {
                'default_model': 'custom:model',
                'timeout': 120
            },
            'chunking': {
                'target_word_count': 300,
                'max_word_count': 400
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.safe_dump(config_data, f)
            config_path = Path(f.name)
        
        try:
            config = Config.load_from_file(config_path)
            
            assert config.ollama.default_model == "custom:model"
            assert config.ollama.timeout == 120
            assert config.chunking.target_word_count == 300
            assert config.chunking.max_word_count == 400
            # Other values should be defaults
            assert config.output.streaming is True
        finally:
            config_path.unlink()

    def test_load_from_empty_file(self):
        """Test loading from empty YAML file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("")  # Empty file
            config_path = Path(f.name)
        
        try:
            config = Config.load_from_file(config_path)
            # Should have all default values
            assert config.ollama.default_model == "llama3.1:8b"
            assert config.chunking.target_word_count == 500
        finally:
            config_path.unlink()

    def test_find_config_file(self):
        """Test finding config file in various locations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Test config/config.yaml
            config_dir = temp_path / "config"
            config_dir.mkdir()
            config_file = config_dir / "config.yaml"
            config_file.write_text("test: value")
            
            with patch('pathlib.Path.cwd', return_value=temp_path):
                found_path = Config._find_config_file()
                assert found_path == config_file
            
            # Test direct config.yaml (should be found over config/config.yaml)
            direct_config = temp_path / "config.yaml"
            direct_config.write_text("test: other")
            
            with patch('pathlib.Path.cwd', return_value=temp_path):
                found_path = Config._find_config_file()
                assert found_path == config_file  # config/config.yaml takes precedence

    def test_find_config_file_none_exist(self):
        """Test finding config file when none exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('pathlib.Path.cwd', return_value=Path(temp_dir)):
                with patch('pathlib.Path.home', return_value=Path(temp_dir)):
                    found_path = Config._find_config_file()
                    assert found_path is None

    def test_save_to_file(self):
        """Test saving config to file."""
        config = Config(
            ollama=OllamaConfig(default_model="custom:model"),
            chunking=ChunkingConfig(target_word_count=300, max_word_count=400)
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.yaml"
            config.save_to_file(config_path)
            
            assert config_path.exists()
            
            # Load and verify
            with open(config_path) as f:
                saved_data = yaml.safe_load(f)
            
            assert saved_data['ollama']['default_model'] == "custom:model"
            assert saved_data['chunking']['target_word_count'] == 300
            assert saved_data['chunking']['max_word_count'] == 400

    def test_save_to_file_creates_directory(self):
        """Test saving config creates parent directory."""
        config = Config()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            nested_path = Path(temp_dir) / "nested" / "config.yaml"
            config.save_to_file(nested_path)
            
            assert nested_path.exists()
            assert nested_path.parent.exists()

    def test_get_system_prompt_path_config_dir(self):
        """Test getting system prompt path from config directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config_dir = temp_path / "config"
            config_dir.mkdir()
            prompt_file = config_dir / "system_prompt.txt"
            prompt_file.write_text("Config prompt")
            
            with patch('pathlib.Path.cwd', return_value=temp_path):
                config = Config()
                path = config.get_system_prompt_path()
                assert path == prompt_file

    def test_get_system_prompt_path_current_dir(self):
        """Test getting system prompt path from current directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            prompt_file = temp_path / "system_prompt.txt"
            prompt_file.write_text("Current dir prompt")
            
            with patch('pathlib.Path.cwd', return_value=temp_path):
                config = Config()
                path = config.get_system_prompt_path()
                assert path == prompt_file

    def test_get_system_prompt_path_fallback(self):
        """Test getting system prompt path fallback."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            with patch('pathlib.Path.cwd', return_value=temp_path):
                config = Config()
                path = config.get_system_prompt_path()
                # Should return package directory fallback
                assert path.name == "system_prompt.txt"

    def test_get_model_config_existing(self):
        """Test getting config for existing model."""
        model_config = ModelSpecificConfig(timeout=120)
        config = Config(
            ollama=OllamaConfig(
                model_configs={"test:model": model_config}
            )
        )
        
        retrieved = config.get_model_config("test:model")
        assert retrieved is model_config
        assert retrieved.timeout == 120

    def test_get_model_config_nonexistent(self):
        """Test getting config for non-existent model."""
        config = Config()
        
        retrieved = config.get_model_config("nonexistent:model")
        assert isinstance(retrieved, ModelSpecificConfig)
        assert retrieved.timeout is None  # Default values

    def test_get_effective_timeout_model_specific(self):
        """Test getting effective timeout with model-specific override."""
        model_config = ModelSpecificConfig(timeout=120)
        config = Config(
            ollama=OllamaConfig(
                timeout=60,
                model_configs={"test:model": model_config}
            )
        )
        
        timeout = config.get_effective_timeout("test:model")
        assert timeout == 120  # Model-specific override

    def test_get_effective_timeout_default(self):
        """Test getting effective timeout with default."""
        config = Config(
            ollama=OllamaConfig(timeout=90)
        )
        
        timeout = config.get_effective_timeout("nonexistent:model")
        assert timeout == 90  # Default from ollama config

    def test_get_effective_system_prompt_override(self):
        """Test getting effective system prompt with model override."""
        model_config = ModelSpecificConfig(
            system_prompt_override="Custom model prompt"
        )
        config = Config(
            ollama=OllamaConfig(
                model_configs={"test:model": model_config}
            )
        )
        
        prompt = config.get_effective_system_prompt("test:model", "Default prompt")
        assert prompt == "Custom model prompt"

    def test_get_effective_system_prompt_default(self):
        """Test getting effective system prompt with default."""
        config = Config()
        
        prompt = config.get_effective_system_prompt("nonexistent:model", "Default prompt")
        assert prompt == "Default prompt"

    def test_invalid_nested_config(self):
        """Test that invalid nested config raises validation error."""
        with pytest.raises(ValidationError):
            Config(
                chunking=ChunkingConfig(target_word_count=25)  # Invalid (too small)
            )

    def test_config_serialization_roundtrip(self):
        """Test config can be serialized and deserialized."""
        original_config = Config(
            ollama=OllamaConfig(
                default_model="test:model",
                timeout=120,
                model_configs={
                    "model1": ModelSpecificConfig(timeout=90),
                    "model2": ModelSpecificConfig(temperature=0.8)
                }
            ),
            chunking=ChunkingConfig(target_word_count=300, max_word_count=500)
        )
        
        # Serialize to dict
        config_dict = original_config.model_dump()
        
        # Create new config from dict
        restored_config = Config(**config_dict)
        
        assert restored_config.ollama.default_model == original_config.ollama.default_model
        assert restored_config.ollama.timeout == original_config.ollama.timeout
        assert len(restored_config.ollama.model_configs) == 2
        assert restored_config.chunking.target_word_count == original_config.chunking.target_word_count