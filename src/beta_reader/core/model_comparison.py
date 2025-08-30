"""Model comparison utilities for beta-reader."""

import time
from collections.abc import Iterator
from pathlib import Path

from rich.console import Console
from rich.progress import Progress
from rich.table import Table

from ..llm.client import OllamaClient
from ..llm.exceptions import FileProcessingError
from .performance import get_sample_warmup_text


class ModelComparison:
    """Compare multiple models on the same input text."""

    def __init__(self, client: OllamaClient) -> None:
        """Initialize model comparison.

        Args:
            client: The Ollama client instance.
        """
        self.client = client
        self.console = Console()

    def compare_models(
        self,
        input_text: str,
        models: list[str],
        system_prompt: str,
        output_dir: Path | None = None,
        warmup: bool = True,
    ) -> dict[str, dict[str, str | float]]:
        """Compare multiple models on the same input text.

        Args:
            input_text: Text to process with each model.
            models: List of model names to compare.
            system_prompt: System prompt to use.
            output_dir: Optional directory to save individual outputs.
            warmup: Whether to warm up models before timing.

        Returns:
            Dictionary mapping model names to results and metrics.

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

        results = {}
        warmup_text = get_sample_warmup_text() if warmup else None

        self.console.print(f"\n[bold blue]Comparing {len(models)} models[/bold blue]")
        self.console.print(f"[dim]Input length: {len(input_text):,} characters[/dim]")
        if warmup:
            self.console.print("[dim]Models will be warmed up to exclude loading time[/dim]")

        with Progress() as progress:
            task = progress.add_task("[green]Processing models...", total=len(models))

            for model in models:
                progress.update(task, description=f"[green]Processing with {model}...")

                try:
                    # Warm up model if requested
                    if warmup and warmup_text:
                        self.client.generate(
                            model=model,
                            prompt=warmup_text,
                            system_prompt=system_prompt,
                        )

                    # Process with model and measure time
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

                    results[model] = {
                        "output": output,
                        "processing_time": processing_time,
                        "word_count": word_count,
                        "char_count": char_count,
                        "words_per_second": words_per_second,
                        "success": True,
                        "error": None,
                    }

                    # Save individual output if requested
                    if output_dir:
                        output_dir.mkdir(parents=True, exist_ok=True)
                        safe_model_name = model.replace(":", "_").replace("/", "_").replace(" ", "_")
                        output_file = output_dir / f"{safe_model_name}_output.txt"
                        with open(output_file, "w", encoding="utf-8") as f:
                            f.write(f"# Output from model: {model}\n")
                            f.write(f"# Processing time: {processing_time:.2f} seconds\n")
                            f.write(f"# Words generated: {word_count:,}\n")
                            f.write(f"# Characters generated: {char_count:,}\n")
                            f.write(f"# Words per second: {words_per_second:.1f}\n")
                            f.write("#" + "="*50 + "\n\n")
                            f.write(output)

                except Exception as e:
                    results[model] = {
                        "output": "",
                        "processing_time": 0,
                        "word_count": 0,
                        "char_count": 0,
                        "words_per_second": 0,
                        "success": False,
                        "error": str(e),
                    }

                progress.update(task, advance=1)

        return results

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

        for model, result in results.items():
            if result["success"]:
                status = "✓ Success"
                time_str = f"{result['processing_time']:.2f}"
                words_str = f"{result['word_count']:,}"
                chars_str = f"{result['char_count']:,}"
                wps_str = f"{result['words_per_second']:.1f}"

                table.add_row(
                    model,
                    f"[green]{status}[/green]",
                    time_str,
                    words_str,
                    chars_str,
                    wps_str,
                )
            else:
                status = "✗ Failed"
                error = result["error"]
                table.add_row(
                    model,
                    f"[red]{status}[/red]",
                    "-",
                    "-",
                    "-",
                    "-",
                )

        self.console.print(table)

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

        for model in models:
            self.console.print(f"\n[bold blue]--- {model} ---[/bold blue]")

            try:
                for chunk in self.client.generate_stream(
                    model=model,
                    prompt=input_text,
                    system_prompt=system_prompt,
                ):
                    yield model, chunk
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

            # Summary table
            f.write("## Summary\n\n")
            f.write("| Model | Status | Time (s) | Words | Characters | Words/sec |\n")
            f.write("|-------|--------|----------|-------|------------|----------|\n")

            for model, result in results.items():
                if result["success"]:
                    f.write(f"| {model} | ✓ Success | {result['processing_time']:.2f} | "
                           f"{result['word_count']:,} | {result['char_count']:,} | "
                           f"{result['words_per_second']:.1f} |\n")
                else:
                    f.write(f"| {model} | ✗ Failed | - | - | - | - |\n")

            f.write("\n")

            # Individual outputs
            f.write("## Individual Outputs\n\n")
            for model, result in results.items():
                f.write(f"### {model}\n\n")
                if result["success"]:
                    f.write(f"**Processing time:** {result['processing_time']:.2f}s\n")
                    f.write(f"**Word count:** {result['word_count']:,}\n")
                    f.write(f"**Character count:** {result['char_count']:,}\n\n")
                    f.write("**Output:**\n\n")
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
