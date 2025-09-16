"""Main CLI entry point for beta-reader."""

from pathlib import Path

import typer
from rich import print
from rich.console import Console
from rich.table import Table

from ..core.batch_state import BatchStateManager
from ..core.config import Config
from ..core.model_comparison import ModelComparison
from ..core.model_recommendations import ModelRecommendationEngine
from ..diff import EpubDiffer, TextDiffer
from ..llm.client import create_client
from ..llm.exceptions import BetaReaderError
from ..processors import EpubProcessor, TextProcessor

app = typer.Typer(
    name="beta-reader",
    help="A fanfiction beta reader using local LLM via Ollama",
    rich_markup_mode="rich",
)
console = Console()


# ============================================================================
# Model and Configuration Commands
# ============================================================================

@app.command()
def models() -> None:
    """List available Ollama models."""
    try:
        config = Config.load_from_file()
        client = create_client(config)

        available_models = client.list_models()

        if not available_models:
            print("[yellow]No models found. Make sure Ollama is running and has models installed.[/yellow]")
            raise typer.Exit(1)

        table = Table(title="Available Models")
        table.add_column("Model Name", style="cyan")
        table.add_column("Status", style="green")

        for model in available_models:
            status = "✓ Available"
            if model == config.ollama.default_model:
                status = "✓ Default"
                table.add_row(f"[bold]{model}[/bold]", f"[bold green]{status}[/bold green]")
            else:
                table.add_row(model, status)

        console.print(table)

    except BetaReaderError as e:
        print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def config_show() -> None:
    """Show current configuration."""
    try:
        config = Config.load_from_file()

        table = Table(title="Configuration")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="white")

        table.add_row("Ollama URL", config.ollama.base_url)
        table.add_row("Default Model", config.ollama.default_model)
        table.add_row("Timeout", f"{config.ollama.timeout}s")
        table.add_row("Default Output Format", config.output.default_format)
        table.add_row("Streaming", "Yes" if config.output.streaming else "No")
        table.add_row("Default Diff Format", config.diff.default_format)
        table.add_row("Chunk Target Size", f"{config.chunking.target_word_count} words")
        table.add_row("Chunk Max Size", f"{config.chunking.max_word_count} words")

        # Add system prompt path
        prompt_path = config.get_system_prompt_path()
        table.add_row("System Prompt Path", str(prompt_path))

        console.print(table)

        # Show model-specific configuration for default model if it exists
        default_model = config.ollama.default_model
        model_config = config.get_model_config(default_model)

        # Check if there are any non-None values in the model config
        model_settings = {
            "Timeout": f"{model_config.timeout}s" if model_config.timeout is not None else None,
            "Max Tokens": str(model_config.max_tokens) if model_config.max_tokens is not None else None,
            "Temperature": str(model_config.temperature) if model_config.temperature is not None else None,
            "Top P": str(model_config.top_p) if model_config.top_p is not None else None,
            "Top K": str(model_config.top_k) if model_config.top_k is not None else None,
            "Repeat Penalty": str(model_config.repeat_penalty) if model_config.repeat_penalty is not None else None,
            "System Prompt Override": "Yes" if model_config.system_prompt_override else None,
        }

        # Filter out None values
        active_settings = {k: v for k, v in model_settings.items() if v is not None}

        if active_settings:
            console.print(f"\n[bold blue]Model-Specific Settings for {default_model}:[/bold blue]")
            model_table = Table()
            model_table.add_column("Setting", style="cyan")
            model_table.add_column("Value", style="white")

            for setting, value in active_settings.items():
                model_table.add_row(setting, value)

            console.print(model_table)

    except BetaReaderError as e:
        print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command(name="system-prompt")
def system_prompt() -> None:
    """Show the current system prompt used for processing."""
    try:
        config = Config.load_from_file()
        prompt_path = config.get_system_prompt_path()

        if not prompt_path.exists():
            print(f"[red]Error: System prompt file not found: {prompt_path}[/red]")
            raise typer.Exit(1)

        console.print(f"\n[bold blue]System Prompt Location:[/bold blue] {prompt_path}")
        console.print(f"[bold blue]File size:[/bold blue] {prompt_path.stat().st_size:,} bytes")
        console.print("\n[bold blue]System Prompt Content:[/bold blue]")
        console.print("─" * 80)

        with open(prompt_path, encoding="utf-8") as f:
            content = f.read()

        # Display the prompt with syntax highlighting for readability
        console.print(f"[dim]{content}[/dim]")
        console.print("─" * 80)

    except BetaReaderError as e:
        print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        print(f"[red]Error reading system prompt: {e}[/red]")
        raise typer.Exit(1)


# ============================================================================
# File Processing Commands
# ============================================================================

@app.command()
def process(
    input_file: Path = typer.Argument(..., help="Input file to process"),
    model: str | None = typer.Option(None, "--model", "-m", help="Model to use"),
    output: Path | None = typer.Option(None, "--output", "-o", help="Output file"),
    stream: bool = typer.Option(True, "--stream/--no-stream", help="Stream output"),
    chapter: int | None = typer.Option(None, "--chapter", "-c", help="Process specific chapter (epub only)"),
    batch: bool = typer.Option(False, "--batch", "-b", help="Process all chapters (epub only)"),
    resume: str | None = typer.Option(None, "--resume", help="Resume interrupted batch by hash or batch ID"),
    debug_chunking: bool = typer.Option(False, "--debug-chunking", help="Show debug info about chunk boundaries"),
) -> None:
    """Process a text or epub file for beta reading."""
    try:
        # Load configuration
        config = Config.load_from_file()
        client = create_client(config)

        # Check if input file exists
        if not input_file.exists():
            print(f"[red]Error: Input file not found: {input_file}[/red]")
            raise typer.Exit(1)

        # Create appropriate processor
        file_type = input_file.suffix.lower()

        if file_type == ".txt":
            if chapter is not None or batch:
                print("[red]Error: --chapter and --batch options are only for epub files[/red]")
                raise typer.Exit(1)
            processor = TextProcessor(client, config)
        elif file_type == ".epub":
            processor = EpubProcessor(client, config)
        else:
            print(f"[red]Error: Unsupported file type: {input_file.suffix}[/red]")
            print("[dim]Currently supported: .txt, .epub[/dim]")
            raise typer.Exit(1)

        # Set debug chunking flag
        processor._debug_chunking = debug_chunking

        # Validate model if specified
        if model and not client.model_exists(model):
            available = client.list_models()
            print(f"[red]Error: Model '{model}' not found.[/red]")
            print(f"[dim]Available models: {', '.join(available)}[/dim]")
            raise typer.Exit(1)

        # Process the file
        if isinstance(processor, EpubProcessor):
            result = processor.process_file(
                input_file,
                output_path=output,
                stream=stream,
                model=model,
                chapter=chapter,
                batch=batch,
                resume_batch_id=resume,
            )
        else:
            result = processor.process_file(
                input_file,
                output_path=output,
                stream=stream,
                model=model,
            )

        # If not streaming or saving to file, print result
        if not stream and not output:
            console.print("\n[bold blue]Processed text:[/bold blue]\n")
            console.print(result)

    except BetaReaderError as e:
        print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
    except KeyboardInterrupt:
        print("\n[yellow]Process interrupted by user[/yellow]")
        raise typer.Exit(130)


# ============================================================================
# Diff and Comparison Commands
# ============================================================================

@app.command()
def diff(
    original: Path = typer.Argument(..., help="Original file"),
    edited: Path = typer.Argument(..., help="Edited file"),
    format: str = typer.Option("unified", "--format", "-f", help="Diff format: unified or split"),
    output: Path | None = typer.Option(None, "--output", "-o", help="Save diff to file"),
) -> None:
    """Generate diff between original and edited files."""
    try:
        # Validate format
        if format not in ("unified", "split"):
            print(f"[red]Error: Invalid format '{format}'. Must be 'unified' or 'split'.[/red]")
            raise typer.Exit(1)

        # Check if files exist
        for file_path, file_type in [(original, "Original"), (edited, "Edited")]:
            if not file_path.exists():
                print(f"[red]Error: {file_type} file not found: {file_path}[/red]")
                raise typer.Exit(1)

        # Validate file types match
        if original.suffix.lower() != edited.suffix.lower():
            print(f"[red]Error: File types must match. Got {original.suffix} and {edited.suffix}[/red]")
            raise typer.Exit(1)

        # Create appropriate differ
        file_type = original.suffix.lower()
        differ_map = {".epub": EpubDiffer, ".txt": TextDiffer}

        if file_type not in differ_map:
            print(f"[red]Error: Unsupported file type: {original.suffix}[/red]")
            print("[dim]Currently supported: .txt, .epub[/dim]")
            raise typer.Exit(1)

        differ = differ_map[file_type]()

        if output:
            # Save diff to file
            diff_content = differ.diff_files(original, edited, format)
            output.parent.mkdir(parents=True, exist_ok=True)
            with open(output, "w", encoding="utf-8") as f:
                f.write(diff_content)
            print(f"[green]Diff saved to: {output}[/green]")
        else:
            # Display diff in terminal
            differ.display_diff(original, edited, format)

    except BetaReaderError as e:
        print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
    except KeyboardInterrupt:
        print("\n[yellow]Diff interrupted by user[/yellow]")
        raise typer.Exit(130)


@app.command()
def compare(
    input_file: Path = typer.Argument(..., help="Input file to process with multiple models"),
    models: str | None = typer.Option(None, "--models", "-m", help="Comma-separated list of models to compare"),
    models_file: Path | None = typer.Option(None, "--models-file", "-f", help="File containing list of models (one per line)"),
    output_dir: Path | None = typer.Option(None, "--output-dir", "-d", help="Directory to save individual outputs"),
    report: Path | None = typer.Option(None, "--report", "-r", help="Save comparison report to file"),
    stream: bool = typer.Option(False, "--stream", help="Stream outputs for visual comparison"),
    chapter: int | None = typer.Option(None, "--chapter", "-c", help="Process specific chapter (epub only)"),
    no_warmup: bool = typer.Option(False, "--no-warmup", help="Skip model warmup (include loading time in measurements)"),
    recommend: bool = typer.Option(False, "--recommend", help="Generate model recommendations based on results"),
    recommend_report: Path | None = typer.Option(None, "--recommend-report", help="Save recommendations report to file"),
) -> None:
    """Compare multiple models on the same input file."""
    try:
        # Load configuration
        config = Config.load_from_file()
        client = create_client(config)

        # Check if input file exists
        if not input_file.exists():
            print(f"[red]Error: Input file not found: {input_file}[/red]")
            raise typer.Exit(1)

        # Determine which models to compare
        model_list = []

        if models_file:
            if not models_file.exists():
                print(f"[red]Error: Models file not found: {models_file}[/red]")
                raise typer.Exit(1)

            try:
                with open(models_file, encoding="utf-8") as f:
                    model_list = [line.strip() for line in f if line.strip() and not line.startswith("#")]
            except Exception as e:
                print(f"[red]Error reading models file: {e}[/red]")
                raise typer.Exit(1)

        elif models:
            model_list = [model.strip() for model in models.split(",") if model.strip()]
        else:
            print("[red]Error: Must specify models via --models or --models-file[/red]")
            raise typer.Exit(1)

        if not model_list:
            print("[red]Error: No models specified[/red]")
            raise typer.Exit(1)

        if len(model_list) < 2:
            print("[red]Error: At least 2 models required for comparison[/red]")
            raise typer.Exit(1)

        # Validate all models exist
        available_models = client.list_models()
        invalid_models = [model for model in model_list if model not in available_models]
        if invalid_models:
            print(f"[red]Error: Models not found: {', '.join(invalid_models)}[/red]")
            print(f"[dim]Available models: {', '.join(available_models)}[/dim]")
            raise typer.Exit(1)

        # Load system prompt
        prompt_path = config.get_system_prompt_path()
        if not prompt_path.exists():
            print(f"[red]Error: System prompt not found: {prompt_path}[/red]")
            raise typer.Exit(1)

        with open(prompt_path, encoding="utf-8") as f:
            system_prompt = f.read()

        # Extract text based on file type
        file_type = input_file.suffix.lower()

        if file_type == ".txt":
            if chapter is not None:
                print("[red]Error: --chapter option is only for epub files[/red]")
                raise typer.Exit(1)
            with open(input_file, encoding="utf-8") as f:
                input_text = f.read()
        elif file_type == ".epub":
            # Load epub and extract chapters
            from ebooklib import epub
            book = epub.read_epub(str(input_file))

            # Create temporary processor to use its chapter extraction method
            processor = EpubProcessor(client, config)
            chapters = processor._extract_chapters(book)

            if chapter is not None:
                if chapter < 1 or chapter > len(chapters):
                    print(f"[red]Error: Chapter {chapter} not found. Available: 1-{len(chapters)}[/red]")
                    raise typer.Exit(1)
                _, input_text = chapters[chapter - 1]
            else:
                # Use first chapter or prompt user to specify
                if len(chapters) > 1:
                    print(f"[yellow]Warning: Multiple chapters found ({len(chapters)}). Using first chapter.[/yellow]")
                    print("[dim]Use --chapter N to specify a different chapter[/dim]")
                _, input_text = chapters[0]
        else:
            print(f"[red]Error: Unsupported file type: {input_file.suffix}[/red]")
            print("[dim]Currently supported: .txt, .epub[/dim]")
            raise typer.Exit(1)

        if not input_text.strip():
            print("[red]Error: Input text is empty[/red]")
            raise typer.Exit(1)

        # Create model comparison instance
        comparison = ModelComparison(client)

        if stream:
            # Stream comparison
            console.print(f"\n[bold blue]Streaming comparison of {len(model_list)} models[/bold blue]")
            console.print(f"[dim]Input: {input_file} ({len(input_text):,} characters)[/dim]")

            for model, chunk in comparison.compare_with_streaming(input_text, model_list, system_prompt):
                console.print(f"[cyan]{model}:[/cyan] ", end="")
                console.print(chunk, end="")

            console.print("\n[green]Streaming comparison completed[/green]")
        else:
            # Batch comparison with warmup option
            results = comparison.compare_models(
                input_text,
                model_list,
                system_prompt,
                output_dir,
                warmup=not no_warmup
            )
            comparison.display_comparison_results(results)

            # Show saved files info if output directory was used
            if output_dir and output_dir.exists():
                output_files = list(output_dir.glob("*_output.txt"))
                if output_files:
                    console.print("\n[green]Individual model outputs saved:[/green]")
                    for file in sorted(output_files):
                        model_name = file.stem.replace("_output", "").replace("_", ":")
                        console.print(f"  [cyan]{model_name}:[/cyan] {file}")
                    console.print(f"\n[dim]Compare outputs manually using: diff {output_dir}/*_output.txt[/dim]")

            # Generate recommendations if requested
            if recommend or recommend_report:
                recommendation_engine = ModelRecommendationEngine()
                recommendations = recommendation_engine.analyze_results(results)

                if recommend:
                    recommendation_engine.display_recommendations(recommendations, show_metrics=True)

                if recommend_report:
                    recommendation_engine.save_recommendations_report(recommendations, recommend_report)

            # Save report if requested
            if report:
                comparison.save_comparison_report(results, report, input_text, system_prompt)

    except BetaReaderError as e:
        print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
    except KeyboardInterrupt:
        print("\n[yellow]Comparison interrupted by user[/yellow]")
        raise typer.Exit(130)


# ============================================================================
# Batch Processing Commands
# ============================================================================

@app.command(name="batch-list")
def batch_list() -> None:
    """List all batch processing states."""
    try:
        batch_manager = BatchStateManager()
        states = batch_manager.list_batch_states()

        if not states:
            print("[yellow]No batch states found[/yellow]")
            return

        table = Table(title="Batch Processing States")
        table.add_column("Hash", style="cyan")
        table.add_column("Input File", style="white")
        table.add_column("Model", style="green")
        table.add_column("Status", style="yellow")
        table.add_column("Progress", style="blue")
        table.add_column("Started", style="dim")

        import datetime

        for state in states:
            # Format timestamp
            start_time = datetime.datetime.fromtimestamp(state['start_time'])
            start_str = start_time.strftime('%Y-%m-%d %H:%M')

            # Style status
            status = state['status']
            if status == 'completed':
                status = f"[green]{status}[/green]"
            elif status == 'failed':
                status = f"[red]{status}[/red]"
            elif status == 'paused':
                status = f"[yellow]{status}[/yellow]"
            else:
                status = f"[blue]{status}[/blue]"

            # Truncate filename if too long
            filename = Path(state['input_file']).name
            if len(filename) > 30:
                filename = filename[:27] + "..."

            table.add_row(
                state['short_hash'],
                filename,
                state['model'],
                status,
                state['progress'],
                start_str
            )

        console.print(table)

        # Show resumable batches
        resumable = batch_manager.get_resumable_batches()
        if resumable:
            console.print(f"\n[green]{len(resumable)} batch(es) can be resumed with --resume <hash>[/green]")

    except Exception as e:
        print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command(name="batch-clean")
def batch_clean(
    days: int = typer.Option(7, "--days", "-d", help="Delete states older than N days"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
) -> None:
    """Clean up old batch processing states."""
    try:
        batch_manager = BatchStateManager()

        if not force:
            confirm = typer.confirm(f"Delete batch states older than {days} days?")
            if not confirm:
                print("Cancelled")
                return

        cleaned = batch_manager.cleanup_old_states(days)
        print(f"[green]Cleaned up {cleaned} old batch state(s)[/green]")

    except Exception as e:
        print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command(name="batch-status")
def batch_status(
    batch_id: str = typer.Argument(..., help="Batch ID or short hash to show status for"),
) -> None:
    """Show detailed status of a specific batch."""
    try:
        batch_manager = BatchStateManager()
        # Resolve short hash if necessary
        full_batch_id = batch_manager.resolve_short_hash(batch_id)
        state = batch_manager.load_batch_state(full_batch_id)

        import datetime

        # Show batch overview
        short_hash = batch_manager.get_short_hash(full_batch_id)
        console.print(f"\n[bold blue]Batch Status:[/bold blue] {short_hash}")
        console.print(f"[dim]Full ID:[/dim] {full_batch_id}")
        console.print(f"[dim]Input file:[/dim] {state.input_file}")
        console.print(f"[dim]Model:[/dim] {state.model}")
        console.print(f"[dim]Status:[/dim] {state.status}")
        console.print(f"[dim]Started:[/dim] {datetime.datetime.fromtimestamp(state.start_time)}")
        console.print(f"[dim]Progress:[/dim] {state.completed_chapters}/{state.total_chapters} chapters")

        if state.failed_chapters > 0:
            console.print(f"[dim]Failed chapters:[/dim] {state.failed_chapters}")

        # Show chapter details
        table = Table(title="Chapter Status")
        table.add_column("Chapter", style="cyan", width=8)
        table.add_column("Title", style="white", width=40)
        table.add_column("Status", style="yellow", width=12)
        table.add_column("Time", style="green", width=8)
        table.add_column("Words", style="blue", width=8)

        for i, chapter in enumerate(state.chapters, 1):
            # Format processing time
            time_str = ""
            if chapter.processing_time:
                time_str = f"{chapter.processing_time:.1f}s"

            # Format word count
            words_str = ""
            if chapter.word_count:
                words_str = f"{chapter.word_count:,}"

            # Format status with colors
            status = chapter.status
            if status == 'completed':
                status = f"[green]{status}[/green]"
            elif status == 'failed':
                status = f"[red]{status}[/red]"
            elif status == 'processing':
                status = f"[yellow]{status}[/yellow]"

            # Truncate title if needed
            title = chapter.chapter_title
            if len(title) > 37:
                title = title[:34] + "..."

            table.add_row(str(i), title, status, time_str, words_str)

        console.print(table)

        # Show failed chapters with errors
        failed_chapters = [c for c in state.chapters if c.status == 'failed']
        if failed_chapters:
            console.print("\n[bold red]Failed Chapters:[/bold red]")
            for chapter in failed_chapters:
                console.print(f"  [red]Chapter {chapter.chapter_index + 1}:[/red] {chapter.error_message}")

    except Exception as e:
        print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


# ============================================================================
# Main Entry Point
# ============================================================================

def main() -> None:
    """Entry point for the CLI application."""
    app()


if __name__ == "__main__":
    main()
