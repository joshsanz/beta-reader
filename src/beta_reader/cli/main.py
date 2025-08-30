"""Main CLI entry point for beta-reader."""

from pathlib import Path

import typer
from rich import print
from rich.console import Console
from rich.table import Table

from ..core.config import Config
from ..diff import TextDiffer
from ..llm.client import create_client
from ..llm.exceptions import BetaReaderError
from ..processors import TextProcessor

app = typer.Typer(
    name="beta-reader",
    help="A fanfiction beta reader using local LLM via Ollama",
    rich_markup_mode="rich",
)
console = Console()


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

        console.print(table)

    except BetaReaderError as e:
        print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def process(
    input_file: Path = typer.Argument(..., help="Input file to process"),
    model: str | None = typer.Option(None, "--model", "-m", help="Model to use"),
    output: Path | None = typer.Option(None, "--output", "-o", help="Output file"),
    stream: bool = typer.Option(True, "--stream/--no-stream", help="Stream output"),
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
        processor = None
        if input_file.suffix.lower() == ".txt":
            processor = TextProcessor(client, config)
        else:
            print(f"[red]Error: Unsupported file type: {input_file.suffix}[/red]")
            print("[dim]Currently supported: .txt[/dim]")
            raise typer.Exit(1)

        # Validate model if specified
        if model and not client.model_exists(model):
            available = client.list_models()
            print(f"[red]Error: Model '{model}' not found.[/red]")
            print(f"[dim]Available models: {', '.join(available)}[/dim]")
            raise typer.Exit(1)

        # Process the file
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
        if not original.exists():
            print(f"[red]Error: Original file not found: {original}[/red]")
            raise typer.Exit(1)

        if not edited.exists():
            print(f"[red]Error: Edited file not found: {edited}[/red]")
            raise typer.Exit(1)

        # Create differ and generate/display diff
        differ = TextDiffer()

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


def main() -> None:
    """Entry point for the CLI application."""
    app()


if __name__ == "__main__":
    main()
