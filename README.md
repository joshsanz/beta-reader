# Beta Reader

A fanfiction beta reader using local LLM via Ollama.

## Features

- Process text files and epub files
- Individual chapter or batch processing for epubs
- Configurable model selection via Ollama
- Streaming output to terminal or file output
- Diff utility with unified/split formatting for model comparison

## Installation

```bash
# Create virtual environment with Python 3.12
uv venv --python 3.12
source .venv/bin/activate

# Install dependencies
uv pip install -e .
```

## Usage

```bash
# Process a text file
beta-reader process input.txt

# Process with specific model
beta-reader process --model llama3.1:8b input.txt

# Output to file instead of streaming
beta-reader process --output edited.txt input.txt

# Process epub file
beta-reader process book.epub

# Process specific chapter
beta-reader process --chapter 5 book.epub

# Generate diff
beta-reader diff original.txt edited.txt --unified
```

## Development

```bash
# Install development dependencies
uv pip install -e .[dev]

# Run program without activating venv
uv run beta-reader --help

# Run tests
pytest

# Run linting and formatting
ruff check src tests
ruff format src tests
mypy src
```