# Beta Reader Fanfiction App - Project Plan

## Overview
A Python CLI application that acts as a beta reader for fanfictions, using local LLM via Ollama. Supports text files and epub files with individual chapter or batch processing. Designed for future web interface expansion.

## Architecture

```
beta-reader/
├── src/
│   ├── core/           # BetaReader engine, Config, BatchProcessor  
│   ├── processors/     # TextProcessor, EpubProcessor with extensible interface
│   ├── llm/           # OllamaClient, custom exceptions, StreamHandler
│   ├── diff/          # Unified/split diff formatting with --unified/--split flags
│   ├── cli/           # Typer CLI with comprehensive error handling
│   └── web/           # Future FastAPI upload/paste interface
├── tests/             # Pytest with sample fixtures
├── config/            # YAML configuration templates
├── examples/          # Sample files for testing
├── system_prompt.txt  # Existing beta reader prompt
└── project-plan.md    # This file
```

## Key Libraries & Technologies

- **CLI**: `typer` with rich error messages for LLM failures
- **Epub**: `ebooklib` for chapter extraction and reconstruction  
- **LLM**: `ollama-python` with configurable model selection
- **Diff**: `rich` + `difflib` supporting unified/split format flags
- **Config**: `pydantic` + YAML for model and output preferences
- **Async**: `asyncio` for streaming LLM responses
- **Testing**: `pytest` with sample file fixtures
- **Future Web**: `FastAPI` for simple upload/paste/display workflow

## Core Components

### 1. LLM Integration (`src/llm/`)
- `OllamaClient`: Connection management, configurable model selection
- `LLMException`: Custom exceptions for model failures, connection issues
- `PromptManager`: Load system prompt from existing system_prompt.txt
- `StreamHandler`: Real-time response streaming with error detection

### 2. File Processors (`src/processors/`)
- `BaseProcessor`: Abstract interface for extensible output formats
- `TextProcessor`: Handle plain text files
- `EpubProcessor`: Parse chapters, batch processing, epub reconstruction
- Support for individual chapter selection or full book processing

### 3. Diff Engine (`src/diff/`)
- `TextDiffer`: Generate diffs with `--unified` / `--split` flags
- `EpubDiffer`: Chapter-by-chapter comparison for epub files
- Rich terminal formatting for easy model comparison

### 4. Core Engine (`src/core/`)
- `BetaReader`: Main orchestration with robust error handling
- `Config`: Model selection, diff preferences, output options
- `BatchProcessor`: Handle multiple files/chapters with progress tracking

### 5. CLI Interface (`src/cli/`)
- Commands: `process`, `diff`, `models` (list available)
- Flags: `--model`, `--output`, `--stream`, `--unified`/`--split`
- Comprehensive error messages for LLM failures
- Progress bars for batch operations

## Error Handling Strategy

- **CLI**: Immediate failure with helpful messages (model unavailable, connection issues, parsing errors)
- **Web**: Structured error responses with troubleshooting hints  
- **Custom exceptions**: `ModelNotFoundError`, `OllamaConnectionError`, `ProcessingError`
- **Graceful degradation**: Save partial progress in batch operations

## Configuration Example

```yaml
ollama:
  base_url: "http://localhost:11434"
  default_model: "llama3.1:8b"
  timeout: 60

output:
  default_format: "text" 
  streaming: true

diff:
  default_format: "unified"  # or "split"
```

## Development Phases & TODO List

### Phase 1: Core Infrastructure ✅ COMPLETED

#### Project Setup
- [x] Initialize project with `pyproject.toml`
- [x] Set up virtual environment and dependencies (using uv with Python 3.12)
- [x] Create basic directory structure
- [x] Set up pytest configuration
- [x] Create sample text and epub files for testing

#### Configuration System  
- [x] Create `Config` class using pydantic
- [x] Implement YAML configuration loading
- [x] Add model selection and validation
- [x] Create default config template
- [x] Add configuration validation

#### Basic LLM Integration
- [x] Implement `OllamaClient` with connection management
- [x] Create custom exception hierarchy
- [x] Add model availability checking
- [x] Implement basic prompt sending (sync/async with streaming)
- [x] Add connection error handling

#### Basic CLI
- [x] Create CLI entry point with typer
- [x] Add `models` command to list available models
- [x] Add `config-show` command
- [x] Add stub commands for `process` and `diff`

### Phase 2: Text Processing ✅ COMPLETED

#### Text File Processing
- [x] Create `BaseProcessor` abstract class
- [x] Implement `TextProcessor` for plain text files
- [x] Integrate with existing system_prompt.txt
- [x] Add file reading/writing capabilities
- [x] Implement streaming output to terminal

#### CLI Foundation
- [x] Set up Typer CLI application
- [x] Create `process` command for text files
- [x] Add `--model` flag for model selection
- [x] Add `--output` flag for file output vs streaming
- [x] Implement comprehensive error handling and messages

#### Basic Diff Utility
- [x] Create `TextDiffer` class
- [x] Implement unified diff format
- [x] Implement split diff format  
- [x] Add `--unified`/`--split` flags
- [x] Add rich terminal formatting

### Phase 3: Epub Support ✅ COMPLETED

#### Epub Processing Infrastructure
- [x] Implement `EpubProcessor` class
- [x] Add epub file parsing with ebooklib
- [x] Implement chapter extraction
- [x] Add chapter identification and numbering
- [x] Create epub reconstruction functionality

#### Chapter Processing
- [x] Add individual chapter processing
- [x] Implement chapter selection by number/name
- [x] Add batch processing for entire epubs
- [x] Implement progress tracking for batch operations
- [x] Add partial save functionality for interrupted processing

#### Epub CLI Integration
- [x] Extend `process` command for epub files
- [x] Add `--chapter` flag for individual chapters
- [x] Add `--batch` flag for full epub processing
- [x] Implement epub-specific error handling
- [x] Add epub validation

### Phase 4: Enhanced Features

#### Advanced Diff Functionality
- [x] Implement `EpubDiffer` for chapter-by-chapter comparison
- [x] Add diff command to CLI for epub files
- [x] Implement side-by-side epub diff viewing
- [x] Add diff statistics and summary
- [ ] Create model comparison utilities

#### Model Management
- [x] Add `models` command to list available models
- [x] Implement model switching without restart
- [ ] Add model performance testing utilities
- [ ] Create model recommendation system
- [ ] Add model-specific configuration options

#### Batch Processing Enhancements
- [ ] Add resume functionality for interrupted batches
- [ ] Implement parallel chapter processing
- [ ] Add batch processing statistics
- [ ] Create processing queues
- [ ] Add batch operation logging

### Phase 5: Web Interface Foundation

#### API Layer
- [ ] Set up FastAPI application structure
- [ ] Create upload endpoints for files
- [ ] Implement paste/text input endpoints
- [ ] Add processing status endpoints
- [ ] Create download endpoints for results

#### Web Error Handling
- [ ] Adapt error handling for web context
- [ ] Create structured JSON error responses
- [ ] Add user-friendly error messages
- [ ] Implement error logging
- [ ] Add troubleshooting hints in responses

#### Web-CLI Integration
- [ ] Create shared processing core
- [ ] Implement async processing for web
- [ ] Add job queue system
- [ ] Create progress tracking for web
- [ ] Add result caching

### Phase 6: Testing & Documentation

#### Comprehensive Testing
- [ ] Create unit tests for all processors
- [ ] Add integration tests for CLI commands
- [ ] Implement LLM mocking for reliable tests
- [ ] Add epub parsing/reconstruction tests
- [ ] Create diff functionality tests

#### Documentation
- [ ] Create user documentation
- [ ] Add developer documentation
- [ ] Document configuration options
- [ ] Create troubleshooting guide
- [ ] Add example usage scenarios

## Key Features Summary

- CLI fails fast with helpful messages on LLM errors (no fallbacks)
- Configurable model selection via config file  
- Diff output via `--unified` or `--split` flags for model testing
- Individual chapter or full epub batch processing
- Extensible processor interface for future output formats
- Web-ready error handling for future upload/paste interface
- Preservation of HTML formatting using existing system prompt
- Streaming output support for real-time feedback
- Progress tracking for long operations
- Resume capability for interrupted batch jobs

## Future Enhancements (Post-MVP)

- [ ] Support for additional ebook formats (mobi, pdf)
- [ ] Integration with online fanfiction sites
- [ ] Custom prompt templates for different genres
- [ ] Multi-model ensemble processing
- [ ] Advanced diff visualization with syntax highlighting
- [ ] Plugin system for custom processors
- [ ] API rate limiting and user management
- [ ] Cloud deployment options