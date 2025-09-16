"""Model comparison utilities for beta-reader."""

import difflib
import time
from collections.abc import Iterator
from pathlib import Path

from rich.console import Console
from rich.progress import Progress
from rich.table import Table

from ..llm.client import OllamaClient
from ..llm.exceptions import FileProcessingError
from .performance import get_sample_warmup_text
from .text_chunker import TextChunk, TextChunker
from .utils import get_safe_filename


class ModelComparison:
    """Compare multiple models on the same input text."""

    def __init__(
        self,
        client: OllamaClient,
        chunk_size: int = 500,
        max_chunk_size: int = 750,
        enable_chunking: bool = True,
    ) -> None:
        """Initialize model comparison.

        Args:
            client: The Ollama client instance.
            chunk_size: Target words per chunk for long texts.
            max_chunk_size: Maximum words per chunk.
            enable_chunking: Whether to enable automatic text chunking.
        """
        self.client = client
        self.console = Console()
        self.enable_chunking = enable_chunking
        self.chunker = TextChunker(
            target_word_count=chunk_size,
            max_word_count=max_chunk_size
        )

    def compare_models(
        self,
        input_text: str,
        models: list[str],
        system_prompt: str,
        output_dir: Path | None = None,
        warmup: bool = True,
    ) -> dict[str, dict[str, str | float | list]]:
        """Compare multiple models on the same input text.

        Args:
            input_text: Text to process with each model.
            models: List of model names to compare.
            system_prompt: System prompt to use.
            output_dir: Optional directory to save individual outputs.
            warmup: Whether to warm up models before timing.

        Returns:
            Dictionary mapping model names to results and metrics.
            If chunking is used, includes 'chunks' key with per-chunk results.

        Raises:
            FileProcessingError: If comparison fails.
        """
        if not models:
            raise FileProcessingError("No models specified for comparison")

        # Validate all models exist
        available_models = self.client.list_models()
        invalid_models = [model for model in models if model not in available_models]
        if invalid_models:
            raise FileProcessingError(f"Models not found: {', '.join(invalid_models)}")

        # Determine if chunking is needed
        text_word_count = len(input_text.split())
        use_chunking = self.enable_chunking and text_word_count > self.chunker.target_word_count
        chunks = self.chunker.chunk_text(input_text) if use_chunking else None

        results = {}
        warmup_text = get_sample_warmup_text() if warmup else None

        self.console.print(f"\n[bold blue]Comparing {len(models)} models[/bold blue]")
        self.console.print(f"[dim]Input length: {len(input_text):,} characters ({text_word_count:,} words)[/dim]")

        if use_chunking and chunks:
            self.console.print(f"[dim]Text will be processed in {len(chunks)} chunks (target: {self.chunker.target_word_count} words/chunk)[/dim]")

        if warmup:
            self.console.print("[dim]Models will be warmed up to exclude loading time[/dim]")

        total_operations = len(models) * (len(chunks) if chunks else 1)

        with Progress() as progress:
            task = progress.add_task("[green]Processing models...", total=total_operations)

            for model in models:
                try:
                    # Warm up model if requested
                    if warmup and warmup_text:
                        self.client.generate(
                            model=model,
                            prompt=warmup_text,
                            system_prompt=system_prompt,
                        )

                    if use_chunking and chunks:
                        # Process with chunking
                        model_result = self._process_model_chunked(
                            model, chunks, system_prompt, progress, task
                        )
                    else:
                        # Process without chunking
                        progress.update(task, description=f"[green]Processing with {model}...")
                        model_result = self._process_model_single(
                            model, input_text, system_prompt
                        )
                        progress.update(task, advance=1)

                    results[model] = model_result

                    # Save individual output if requested
                    if output_dir and model_result["success"]:
                        self._save_model_output(output_dir, model, model_result, use_chunking)

                except Exception as e:
                    results[model] = {
                        "output": "",
                        "processing_time": 0,
                        "word_count": 0,
                        "char_count": 0,
                        "words_per_second": 0,
                        "success": False,
                        "error": str(e),
                        "chunked": use_chunking,
                        "chunks": [] if use_chunking else None,
                    }
                    if use_chunking and chunks:
                        progress.update(task, advance=len(chunks))
                    else:
                        progress.update(task, advance=1)

        # Generate unified diff comparison if output directory is specified
        if output_dir:
            diff_path = output_dir / "unified_diff_comparison.md"
            self.save_unified_diff_comparison(results, diff_path, input_text)

        return results

    def _process_model_single(
        self, model: str, input_text: str, system_prompt: str
    ) -> dict[str, str | float | bool | None]:
        """Process text with a single model without chunking."""
        start_time = time.time()
        output = self.client.generate(
            model=model,
            prompt=input_text,
            system_prompt=system_prompt,
        )
        processing_time = time.time() - start_time

        # Calculate metrics
        word_count = len(output.split())
        char_count = len(output)
        words_per_second = word_count / processing_time if processing_time > 0 else 0

        return {
            "output": output,
            "processing_time": processing_time,
            "word_count": word_count,
            "char_count": char_count,
            "words_per_second": words_per_second,
            "success": True,
            "error": None,
            "chunked": False,
            "chunks": None,
        }

    def _process_model_chunked(
        self,
        model: str,
        chunks: list[TextChunk],
        system_prompt: str,
        progress: Progress,
        task,
    ) -> dict[str, str | float | bool | None | list]:
        """Process text with a model using chunking."""
        chunk_results = []
        total_processing_time = 0.0
        all_outputs = []

        for i, chunk in enumerate(chunks):
            progress.update(
                task,
                description=f"[green]Processing {model} - chunk {i+1}/{len(chunks)}..."
            )

            start_time = time.time()
            try:
                # Add chunk context to the prompt
                chunk_prompt = f"[Processing chunk {chunk.chunk_number} of {chunk.total_chunks}]\n\n{chunk.content}"

                output = self.client.generate(
                    model=model,
                    prompt=chunk_prompt,
                    system_prompt=system_prompt,
                )
                processing_time = time.time() - start_time
                total_processing_time += processing_time

                # Calculate chunk metrics
                word_count = len(output.split())
                char_count = len(output)
                words_per_second = word_count / processing_time if processing_time > 0 else 0

                chunk_result = {
                    "chunk_number": chunk.chunk_number,
                    "input_word_count": chunk.word_count,
                    "output": output,
                    "processing_time": processing_time,
                    "word_count": word_count,
                    "char_count": char_count,
                    "words_per_second": words_per_second,
                    "success": True,
                    "error": None,
                }
                chunk_results.append(chunk_result)
                all_outputs.append(output)

            except Exception as e:
                processing_time = time.time() - start_time
                total_processing_time += processing_time

                chunk_result = {
                    "chunk_number": chunk.chunk_number,
                    "input_word_count": chunk.word_count,
                    "output": "",
                    "processing_time": processing_time,
                    "word_count": 0,
                    "char_count": 0,
                    "words_per_second": 0,
                    "success": False,
                    "error": str(e),
                }
                chunk_results.append(chunk_result)

            progress.update(task, advance=1)

        # Combine all outputs
        combined_output = self.chunker.reassemble_chunks([
            TextChunk(
                content=result["output"],
                chunk_number=result["chunk_number"],
                total_chunks=len(chunks),
                word_count=result["word_count"],
                start_position=0,  # Not needed for reassembly
                end_position=0,    # Not needed for reassembly
            )
            for result in chunk_results
            if result["success"] and result["output"]
        ])

        # Calculate aggregate metrics
        total_output_words = sum(r["word_count"] for r in chunk_results if r["success"])
        total_output_chars = sum(r["char_count"] for r in chunk_results if r["success"])
        overall_words_per_second = total_output_words / total_processing_time if total_processing_time > 0 else 0
        successful_chunks = sum(1 for r in chunk_results if r["success"])

        return {
            "output": combined_output,
            "processing_time": total_processing_time,
            "word_count": total_output_words,
            "char_count": total_output_chars,
            "words_per_second": overall_words_per_second,
            "success": successful_chunks > 0,
            "error": None if successful_chunks > 0 else "All chunks failed",
            "chunked": True,
            "chunks": chunk_results,
            "successful_chunks": successful_chunks,
            "total_chunks": len(chunks),
        }

    def _save_model_output(
        self, output_dir: Path, model: str, result: dict, use_chunking: bool
    ) -> None:
        """Save model output to file."""
        output_dir.mkdir(parents=True, exist_ok=True)
        safe_filename = get_safe_filename(model)
        output_file = output_dir / f"{safe_filename}_output.txt"

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"# Output from model: {model}\n")
            f.write(f"# Processing time: {result['processing_time']:.2f} seconds\n")
            f.write(f"# Words generated: {result['word_count']:,}\n")
            f.write(f"# Characters generated: {result['char_count']:,}\n")
            f.write(f"# Words per second: {result['words_per_second']:.1f}\n")

            if use_chunking and result.get("chunks"):
                f.write(f"# Processed in chunks: {result['successful_chunks']}/{result['total_chunks']} successful\n")

            f.write("#" + "="*50 + "\n\n")
            f.write(result["output"])

            if use_chunking and result.get("chunks"):
                f.write("\n\n" + "#" + "="*50 + "\n")
                f.write("# CHUNK DETAILS\n")
                f.write("#" + "="*50 + "\n\n")

                for chunk_result in result["chunks"]:
                    f.write(f"## Chunk {chunk_result['chunk_number']}\n")
                    f.write(f"- Input words: {chunk_result['input_word_count']:,}\n")
                    f.write(f"- Output words: {chunk_result['word_count']:,}\n")
                    f.write(f"- Processing time: {chunk_result['processing_time']:.2f}s\n")
                    f.write(f"- Success: {chunk_result['success']}\n")
                    if not chunk_result['success']:
                        f.write(f"- Error: {chunk_result['error']}\n")
                    f.write("\n")

    def display_comparison_results(self, results: dict[str, dict[str, str | float]]) -> None:
        """Display comparison results in a formatted table.

        Args:
            results: Results from compare_models method.
        """
        if not results:
            self.console.print("[yellow]No results to display[/yellow]")
            return

        # Create summary table
        table = Table(title="Model Comparison Results")
        table.add_column("Model", style="cyan")
        table.add_column("Status", style="white")
        table.add_column("Time (s)", style="yellow")
        table.add_column("Words", style="green")
        table.add_column("Chars", style="green")
        table.add_column("WPS", style="blue")  # Words per second
        table.add_column("Chunks", style="magenta")

        for model, result in results.items():
            if result["success"]:
                status = "✓ Success"
                processing_time_str = f"{result['processing_time']:.2f}"
                word_count_str = f"{result['word_count']:,}"
                char_count_str = f"{result['char_count']:,}"
                words_per_second_str = f"{result['words_per_second']:.1f}"

                # Handle chunking info
                if result.get("chunked"):
                    chunks_str = f"{result['successful_chunks']}/{result['total_chunks']}"
                else:
                    chunks_str = "Single"

                table.add_row(
                    model,
                    f"[green]{status}[/green]",
                    processing_time_str,
                    word_count_str,
                    char_count_str,
                    words_per_second_str,
                    chunks_str,
                )
            else:
                status = "✗ Failed"
                chunks_str = f"0/{result.get('total_chunks', 0)}" if result.get("chunked") else "Failed"
                table.add_row(
                    model,
                    f"[red]{status}[/red]",
                    "-",
                    "-",
                    "-",
                    "-",
                    chunks_str,
                )

        self.console.print(table)

        # Show chunking details if any models used chunking
        chunked_models = {model: result for model, result in results.items()
                         if result.get("chunked") and result["success"]}
        if chunked_models:
            self.console.print("\n[bold blue]Chunking Details:[/bold blue]")
            for model, result in chunked_models.items():
                chunks = result.get("chunks", [])
                if chunks:
                    avg_chunk_time = sum(c["processing_time"] for c in chunks) / len(chunks)
                    failed_chunks = [c for c in chunks if not c["success"]]
                    self.console.print(f"  [cyan]{model}:[/cyan] {len(chunks)} chunks, avg {avg_chunk_time:.2f}s/chunk")
                    if failed_chunks:
                        self.console.print(f"    [red]Failed chunks:[/red] {[c['chunk_number'] for c in failed_chunks]}")

        # Show errors if any
        failed_models = {model: result for model, result in results.items() if not result["success"]}
        if failed_models:
            self.console.print("\n[bold red]Errors:[/bold red]")
            for model, result in failed_models.items():
                self.console.print(f"  [red]{model}:[/red] {result['error']}")

    def compare_with_streaming(
        self,
        input_text: str,
        models: list[str],
        system_prompt: str,
    ) -> Iterator[tuple[str, str]]:
        """Compare models with streaming output.

        Args:
            input_text: Text to process with each model.
            models: List of model names to compare.
            system_prompt: System prompt to use.

        Yields:
            Tuples of (model_name, chunk) for streaming output.

        Raises:
            FileProcessingError: If comparison fails.
        """
        if not models:
            raise FileProcessingError("No models specified for comparison")

        # Validate all models exist
        available_models = self.client.list_models()
        invalid_models = [model for model in models if model not in available_models]
        if invalid_models:
            raise FileProcessingError(f"Models not found: {', '.join(invalid_models)}")

        # Determine if chunking is needed
        text_word_count = len(input_text.split())
        use_chunking = self.enable_chunking and text_word_count > self.chunker.target_word_count
        chunks = self.chunker.chunk_text(input_text) if use_chunking else None

        if use_chunking and chunks:
            self.console.print(f"\n[dim]Text will be streamed in {len(chunks)} chunks[/dim]")

        for model in models:
            self.console.print(f"\n[bold blue]--- {model} ---[/bold blue]")

            try:
                if use_chunking and chunks:
                    # Stream each chunk separately
                    for chunk in chunks:
                        self.console.print(f"\n[dim]Chunk {chunk.chunk_number}/{chunk.total_chunks}[/dim]")
                        chunk_prompt = f"[Processing chunk {chunk.chunk_number} of {chunk.total_chunks}]\n\n{chunk.content}"

                        for stream_chunk in self.client.generate_stream(
                            model=model,
                            prompt=chunk_prompt,
                            system_prompt=system_prompt,
                        ):
                            yield model, stream_chunk
                else:
                    # Stream without chunking
                    for stream_chunk in self.client.generate_stream(
                        model=model,
                        prompt=input_text,
                        system_prompt=system_prompt,
                    ):
                        yield model, stream_chunk
            except Exception as e:
                self.console.print(f"[red]Error with {model}: {e}[/red]")
                continue

    def save_comparison_report(
        self,
        results: dict[str, dict[str, str | float]],
        output_path: Path,
        input_text: str,
        system_prompt: str,
    ) -> None:
        """Save detailed comparison report to file.

        Args:
            results: Results from compare_models method.
            output_path: Path to save the report.
            input_text: Original input text.
            system_prompt: System prompt used.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("# Model Comparison Report\n\n")

            f.write(f"**Generated on:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Input length:** {len(input_text):,} characters\n")
            f.write(f"**Models compared:** {len(results)}\n\n")

            # Check if any model used chunking
            chunked_models = any(result.get("chunked", False) for result in results.values())

            # Summary table
            f.write("## Summary\n\n")
            if chunked_models:
                f.write("| Model | Status | Time (s) | Words | Characters | Words/sec | Chunks |\n")
                f.write("|-------|--------|----------|-------|------------|----------|--------|\n")
            else:
                f.write("| Model | Status | Time (s) | Words | Characters | Words/sec |\n")
                f.write("|-------|--------|----------|-------|------------|----------|\n")

            for model, result in results.items():
                if result["success"]:
                    base_row = f"| {model} | ✓ Success | {result['processing_time']:.2f} | " \
                              f"{result['word_count']:,} | {result['char_count']:,} | " \
                              f"{result['words_per_second']:.1f}"

                    if chunked_models:
                        if result.get("chunked"):
                            chunks_info = f" | {result['successful_chunks']}/{result['total_chunks']}"
                        else:
                            chunks_info = " | Single"
                        f.write(base_row + chunks_info + " |\n")
                    else:
                        f.write(base_row + " |\n")
                else:
                    if chunked_models:
                        chunks_info = f"0/{result.get('total_chunks', 0)}" if result.get("chunked") else "Failed"
                        f.write(f"| {model} | ✗ Failed | - | - | - | - | {chunks_info} |\n")
                    else:
                        f.write(f"| {model} | ✗ Failed | - | - | - | - |\n")

            f.write("\n")

            # Chunking details if applicable
            if chunked_models:
                f.write("## Chunking Details\n\n")
                for model, result in results.items():
                    if result.get("chunked") and result["success"]:
                        chunks = result.get("chunks", [])
                        if chunks:
                            f.write(f"### {model}\n\n")
                            f.write(f"- **Total chunks:** {result['total_chunks']}\n")
                            f.write(f"- **Successful chunks:** {result['successful_chunks']}\n")
                            avg_chunk_time = sum(c["processing_time"] for c in chunks) / len(chunks)
                            f.write(f"- **Average processing time per chunk:** {avg_chunk_time:.2f}s\n")

                            failed_chunks = [c for c in chunks if not c["success"]]
                            if failed_chunks:
                                f.write(f"- **Failed chunks:** {[c['chunk_number'] for c in failed_chunks]}\n")

                            f.write("\n| Chunk | Input Words | Output Words | Time (s) | WPS | Status |\n")
                            f.write("|-------|-------------|--------------|----------|-----|--------|\n")

                            for chunk in chunks:
                                status = "✓" if chunk["success"] else "✗"
                                f.write(f"| {chunk['chunk_number']} | {chunk['input_word_count']:,} | "
                                       f"{chunk['word_count']:,} | {chunk['processing_time']:.2f} | "
                                       f"{chunk['words_per_second']:.1f} | {status} |\n")
                            f.write("\n")

            # Individual outputs
            f.write("## Individual Outputs\n\n")
            for model, result in results.items():
                f.write(f"### {model}\n\n")
                if result["success"]:
                    f.write(f"**Processing time:** {result['processing_time']:.2f}s\n")
                    f.write(f"**Word count:** {result['word_count']:,}\n")
                    f.write(f"**Character count:** {result['char_count']:,}\n")

                    if result.get("chunked"):
                        f.write(f"**Processing method:** Chunked ({result['successful_chunks']}/{result['total_chunks']} chunks)\n")
                    else:
                        f.write("**Processing method:** Single request\n")

                    f.write("\n**Output:**\n\n")
                    f.write("```\n")
                    f.write(result["output"])
                    f.write("\n```\n\n")
                else:
                    f.write(f"**Error:** {result['error']}\n\n")

            # Appendices
            f.write("## Input Text\n\n")
            f.write("```\n")
            f.write(input_text)
            f.write("\n```\n\n")

            f.write("## System Prompt\n\n")
            f.write("```\n")
            f.write(system_prompt)
            f.write("\n```\n")

        self.console.print(f"[green]Comparison report saved to: {output_path}[/green]")

    def save_unified_diff_comparison(
        self,
        results: dict[str, dict[str, str | float]],
        output_path: Path,
        input_text: str,
        reference_model: str | None = None,
    ) -> None:
        """Save unified diff comparison between model outputs.

        Args:
            results: Results from compare_models method.
            output_path: Path to save the diff comparison.
            input_text: Original input text for context.
            reference_model: Model to use as reference (defaults to first successful model).
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Get successful results
        successful_results = {model: result for model, result in results.items() 
                            if result["success"]}
        
        if len(successful_results) < 2:
            self.console.print("[yellow]Need at least 2 successful models for diff comparison[/yellow]")
            return

        # Determine reference model
        if reference_model and reference_model in successful_results:
            ref_model = reference_model
        else:
            # Use first successful model as reference
            ref_model = next(iter(successful_results.keys()))

        ref_output = successful_results[ref_model]["output"]

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("# Model Output Unified Diff Comparison\n\n")
            f.write(f"**Generated on:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Reference model:** {ref_model}\n")
            f.write(f"**Input length:** {len(input_text):,} characters\n\n")

            # Compare each model against the reference
            for model, result in successful_results.items():
                if model == ref_model:
                    continue

                f.write(f"## {ref_model} vs {model}\n\n")
                
                # Add performance comparison
                ref_result = successful_results[ref_model]
                f.write("**Performance Comparison:**\n")
                f.write(f"- **{ref_model}:** {ref_result['processing_time']:.2f}s, "
                       f"{ref_result['words_per_second']:.1f} WPS, "
                       f"{ref_result['word_count']:,} words\n")
                f.write(f"- **{model}:** {result['processing_time']:.2f}s, "
                       f"{result['words_per_second']:.1f} WPS, "
                       f"{result['word_count']:,} words\n\n")

                # Generate unified diff
                ref_lines = ref_output.splitlines(keepends=True)
                model_lines = result["output"].splitlines(keepends=True)
                
                diff = difflib.unified_diff(
                    ref_lines,
                    model_lines,
                    fromfile=f"{ref_model} output",
                    tofile=f"{model} output",
                    lineterm="",
                )
                
                diff_content = "".join(diff)
                
                if diff_content:
                    f.write("**Unified Diff:**\n\n")
                    f.write("```diff\n")
                    f.write(diff_content)
                    f.write("\n```\n\n")
                else:
                    f.write("**No differences found between outputs.**\n\n")

                f.write("---\n\n")

            # Add context sections
            f.write("## Input Text\n\n")
            f.write("```\n")
            f.write(input_text)
            f.write("\n```\n\n")

            f.write("## Full Outputs\n\n")
            for model, result in successful_results.items():
                f.write(f"### {model}\n\n")
                f.write("```\n")
                f.write(result["output"])
                f.write("\n```\n\n")

        self.console.print(f"[green]Unified diff comparison saved to: {output_path}[/green]")
