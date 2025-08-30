"""Model performance testing utilities."""

import statistics
import time
from pathlib import Path

from rich.console import Console
from rich.progress import Progress
from rich.table import Table

from ..llm.client import OllamaClient
from ..llm.exceptions import FileProcessingError


class ModelPerformanceTester:
    """Test and measure model performance metrics."""

    def __init__(self, client: OllamaClient) -> None:
        """Initialize performance tester.

        Args:
            client: The Ollama client instance.
        """
        self.client = client
        self.console = Console()

    def warm_up_model(self, model: str, warmup_text: str, system_prompt: str) -> None:
        """Warm up a model to avoid including loading time in measurements.

        Args:
            model: Model name to warm up.
            warmup_text: Text to use for warmup (should be short).
            system_prompt: System prompt to use.

        Raises:
            FileProcessingError: If warmup fails.
        """
        try:
            self.console.print(f"[dim]Warming up {model}...[/dim]")
            self.client.generate(
                model=model,
                prompt=warmup_text,
                system_prompt=system_prompt,
            )
        except Exception as e:
            raise FileProcessingError(f"Failed to warm up model {model}: {e}") from e

    def benchmark_model(
        self,
        model: str,
        test_text: str,
        system_prompt: str,
        warmup_text: str | None = None,
        runs: int = 3,
    ) -> dict[str, float]:
        """Benchmark a single model with multiple runs.

        Args:
            model: Model name to benchmark.
            test_text: Text to use for benchmarking.
            system_prompt: System prompt to use.
            warmup_text: Optional warmup text. If None, uses test_text[:100].
            runs: Number of runs to average.

        Returns:
            Dictionary with performance metrics.

        Raises:
            FileProcessingError: If benchmarking fails.
        """
        if warmup_text is None:
            warmup_text = test_text[:100]  # Use first 100 chars for warmup

        # Validate model exists
        if not self.client.model_exists(model):
            raise FileProcessingError(f"Model {model} not found")

        # Warm up the model
        self.warm_up_model(model, warmup_text, system_prompt)

        # Run benchmarks
        times = []
        word_counts = []
        char_counts = []

        self.console.print(f"[blue]Benchmarking {model} ({runs} runs)...[/blue]")

        with Progress() as progress:
            task = progress.add_task(f"Running {model}...", total=runs)

            for run in range(runs):
                progress.update(task, description=f"Run {run + 1}/{runs}")

                start_time = time.time()
                try:
                    output = self.client.generate(
                        model=model,
                        prompt=test_text,
                        system_prompt=system_prompt,
                    )
                    end_time = time.time()

                    processing_time = end_time - start_time
                    word_count = len(output.split())
                    char_count = len(output)

                    times.append(processing_time)
                    word_counts.append(word_count)
                    char_counts.append(char_count)

                except Exception as e:
                    raise FileProcessingError(f"Benchmark run {run + 1} failed for {model}: {e}") from e

                progress.update(task, advance=1)

        # Calculate statistics
        avg_time = statistics.mean(times)
        std_time = statistics.stdev(times) if len(times) > 1 else 0.0
        min_time = min(times)
        max_time = max(times)

        avg_words = statistics.mean(word_counts)
        avg_chars = statistics.mean(char_counts)

        words_per_second = avg_words / avg_time if avg_time > 0 else 0
        chars_per_second = avg_chars / avg_time if avg_time > 0 else 0

        return {
            "model": model,
            "runs": runs,
            "avg_time": avg_time,
            "std_time": std_time,
            "min_time": min_time,
            "max_time": max_time,
            "avg_words": avg_words,
            "avg_chars": avg_chars,
            "words_per_second": words_per_second,
            "chars_per_second": chars_per_second,
            "all_times": times,
            "all_word_counts": word_counts,
            "all_char_counts": char_counts,
        }

    def benchmark_multiple_models(
        self,
        models: list[str],
        test_text: str,
        system_prompt: str,
        warmup_text: str | None = None,
        runs: int = 3,
    ) -> list[dict[str, float]]:
        """Benchmark multiple models.

        Args:
            models: List of model names to benchmark.
            test_text: Text to use for benchmarking.
            system_prompt: System prompt to use.
            warmup_text: Optional warmup text.
            runs: Number of runs per model.

        Returns:
            List of benchmark results for each model.

        Raises:
            FileProcessingError: If benchmarking fails.
        """
        if not models:
            raise FileProcessingError("No models specified for benchmarking")

        # Validate all models exist
        available_models = self.client.list_models()
        invalid_models = [model for model in models if model not in available_models]
        if invalid_models:
            raise FileProcessingError(f"Models not found: {', '.join(invalid_models)}")

        results = []

        self.console.print(f"\n[bold blue]Benchmarking {len(models)} models[/bold blue]")
        self.console.print(f"[dim]Test text length: {len(test_text):,} characters[/dim]")
        self.console.print(f"[dim]Runs per model: {runs}[/dim]\n")

        for model in models:
            try:
                result = self.benchmark_model(model, test_text, system_prompt, warmup_text, runs)
                results.append(result)
            except Exception as e:
                self.console.print(f"[red]Failed to benchmark {model}: {e}[/red]")
                # Add failed result
                results.append({
                    "model": model,
                    "runs": 0,
                    "avg_time": 0.0,
                    "std_time": 0.0,
                    "min_time": 0.0,
                    "max_time": 0.0,
                    "avg_words": 0.0,
                    "avg_chars": 0.0,
                    "words_per_second": 0.0,
                    "chars_per_second": 0.0,
                    "error": str(e),
                })

        return results

    def display_benchmark_results(self, results: list[dict[str, float]]) -> None:
        """Display benchmark results in a formatted table.

        Args:
            results: List of benchmark results from benchmark_multiple_models.
        """
        if not results:
            self.console.print("[yellow]No benchmark results to display[/yellow]")
            return

        # Create performance table
        table = Table(title="Model Performance Benchmark")
        table.add_column("Model", style="cyan")
        table.add_column("Runs", style="white")
        table.add_column("Avg Time (s)", style="yellow")
        table.add_column("±Std (s)", style="dim")
        table.add_column("Words/sec", style="green")
        table.add_column("Chars/sec", style="blue")
        table.add_column("Min Time", style="dim")
        table.add_column("Max Time", style="dim")

        # Sort by average words per second (descending)
        sorted_results = sorted(results, key=lambda x: x.get("words_per_second", 0), reverse=True)

        for result in sorted_results:
            if result.get("error"):
                table.add_row(
                    result["model"],
                    "Failed",
                    "-",
                    "-",
                    "-",
                    "-",
                    "-",
                    "-",
                )
            else:
                table.add_row(
                    result["model"],
                    str(result["runs"]),
                    f"{result['avg_time']:.2f}",
                    f"±{result['std_time']:.2f}",
                    f"{result['words_per_second']:.1f}",
                    f"{result['chars_per_second']:.0f}",
                    f"{result['min_time']:.2f}",
                    f"{result['max_time']:.2f}",
                )

        self.console.print(table)

        # Show errors if any
        failed_models = [r for r in results if r.get("error")]
        if failed_models:
            self.console.print("\n[bold red]Benchmark Errors:[/bold red]")
            for result in failed_models:
                self.console.print(f"  [red]{result['model']}:[/red] {result['error']}")

    def save_benchmark_report(
        self,
        results: list[dict[str, float]],
        output_path: Path,
        test_text: str,
        system_prompt: str,
        runs: int,
    ) -> None:
        """Save detailed benchmark report to file.

        Args:
            results: Benchmark results to save.
            output_path: Path to save the report.
            test_text: Test text used.
            system_prompt: System prompt used.
            runs: Number of runs per model.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("# Model Performance Benchmark Report\n\n")

            f.write(f"**Generated on:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Test text length:** {len(test_text):,} characters\n")
            f.write(f"**Runs per model:** {runs}\n")
            f.write(f"**Models tested:** {len(results)}\n\n")

            # Summary table
            f.write("## Performance Summary\n\n")
            f.write("| Model | Runs | Avg Time (s) | ±Std (s) | Words/sec | Chars/sec | Min Time | Max Time |\n")
            f.write("|-------|------|--------------|----------|-----------|-----------|----------|----------|\n")

            # Sort by words per second
            sorted_results = sorted(results, key=lambda x: x.get("words_per_second", 0), reverse=True)

            for result in sorted_results:
                if result.get("error"):
                    f.write(f"| {result['model']} | Failed | - | - | - | - | - | - |\n")
                else:
                    f.write(f"| {result['model']} | {result['runs']} | "
                           f"{result['avg_time']:.2f} | ±{result['std_time']:.2f} | "
                           f"{result['words_per_second']:.1f} | {result['chars_per_second']:.0f} | "
                           f"{result['min_time']:.2f} | {result['max_time']:.2f} |\n")

            f.write("\n")

            # Detailed results
            f.write("## Detailed Results\n\n")
            for result in results:
                f.write(f"### {result['model']}\n\n")
                if result.get("error"):
                    f.write(f"**Error:** {result['error']}\n\n")
                else:
                    f.write(f"**Runs:** {result['runs']}\n")
                    f.write(f"**Average processing time:** {result['avg_time']:.2f}s\n")
                    f.write(f"**Standard deviation:** {result['std_time']:.2f}s\n")
                    f.write(f"**Min/Max time:** {result['min_time']:.2f}s / {result['max_time']:.2f}s\n")
                    f.write(f"**Average words generated:** {result['avg_words']:.0f}\n")
                    f.write(f"**Average characters generated:** {result['avg_chars']:.0f}\n")
                    f.write(f"**Words per second:** {result['words_per_second']:.1f}\n")
                    f.write(f"**Characters per second:** {result['chars_per_second']:.0f}\n\n")

                    if len(result.get("all_times", [])) > 1:
                        f.write("**Individual run times:** " +
                               ", ".join(f"{t:.2f}s" for t in result["all_times"]) + "\n\n")

            # Test configuration
            f.write("## Test Configuration\n\n")
            f.write("### Test Text\n\n")
            f.write("```\n")
            f.write(test_text)
            f.write("\n```\n\n")

            f.write("### System Prompt\n\n")
            f.write("```\n")
            f.write(system_prompt)
            f.write("\n```\n")

        self.console.print(f"[green]Benchmark report saved to: {output_path}[/green]")


def get_sample_warmup_text() -> str:
    """Get short text suitable for model warmup."""
    return "This is a short test sentence to warm up the model."


def load_sample_text(sample_path: Path | None = None) -> str:
    """Load sample text from examples directory or provided path.

    Args:
        sample_path: Optional path to sample text file.

    Returns:
        Sample text content.

    Raises:
        FileProcessingError: If sample file cannot be loaded.
    """
    if sample_path is None:
        sample_path = Path("examples") / "sample.txt"

    if not sample_path.exists():
        raise FileProcessingError(f"Sample text file not found: {sample_path}")

    try:
        with open(sample_path, encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        raise FileProcessingError(f"Failed to load sample text: {e}") from e
