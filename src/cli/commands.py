import typer
import os
from typing_extensions import Annotated
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich import print as rprint
from rich.live import Live

import tiktoken

from ..core.rag import RAG
from ..utils.lexical_utils import Lemmatizer
from ..utils.config import get_config
from ..utils.logger import Logger

logger = Logger(__name__)

console = Console()
app = typer.Typer(help="ChatPDF Command Line Interface")

rag_instance = None


def get_rag_instance() -> RAG:
    global rag_instance
    if rag_instance is None:
        config = get_config()

        # Initialize with configured models
        tokenizer = tiktoken.get_encoding("cl100k_base")
        rag_instance = RAG(
            reranker_model=config.get("models.reranker"),
            semantic_retriever_model=config.get(
                "models.semantic_retriever"
            ),
            generator_model=config.get(
                "models.generator"
            ),
            tokenize_func=lambda x: tokenizer.encode(x),
            lemmatizer=Lemmatizer(),
            cache_dir=config.get("cache.directory", ".chatpdf_cache")
        )

        # Try loading from cache
        if config.get("cache.use_embeddings_cache", True):
            if rag_instance.load_from_cache():
                return rag_instance

    return rag_instance


@app.command()
def ingest(
    path: Annotated[
        str,
        typer.Argument(
            help="Path to directory containing PDFs to index."
        )
    ]
):
    """Ingest PDF documents and build search indexes."""
    console.rule("[bold blue]Document Ingestion Pipeline")
    rprint(f"[italic]Scanning directory:[/italic] {path}")

    if not os.path.isdir(path):
        console.print(
            f"[bold red]Error:[/bold red] Directory '{path}' not found"
        )
        raise typer.Exit(code=1)

    try:
        rag = get_rag_instance()

        with console.status(
            "[bold green]Processing documents...[/bold green]",
            spinner="dots"
        ):
            chunk_count = rag.ingest_documents(
                path,
                chunk_size=512,
                chunk_overlap=64
            )

        console.print(
            Panel(
                f"[bold green]Success![/bold green] "
                f"Indexed {chunk_count} chunks",
                title="Ingestion Status"
            )
        )
        rprint("[green]✓ Documents ready for querying[/green]")

    except Exception as e:
        console.print(
            f"[bold red]Error during ingestion:[/bold red] {e}"
        )
        logger.log("Ingestion failed", "EXCEPTION")
        raise typer.Exit(code=1)


@app.command()
def chat():
    """Start interactive chat session with PDF documents."""
    console.clear()
    console.rule("[bold magenta]Interactive ChatPDF Terminal[/bold magenta]")
    rprint("[yellow]Type 'exit', 'quit', or Ctrl+C to stop.[/yellow]\n")

    try:
        rag = get_rag_instance()
        # Check if documents are indexed
        if not rag.is_indexed:
            msg = ("[bold red]No documents indexed.[/bold red]\n"
                   "Run 'chatpdf ingest /path/to/pdfs' first.")
            rprint(msg)
            raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[bold red]Error initializing RAG:[/bold red] {e}")
        raise typer.Exit(code=1)

    while True:
        try:
            user_query = typer.prompt(
                typer.style("You", fg=typer.colors.CYAN, bold=True)
            )
            user_query = user_query.strip()

            if user_query.lower() in ['exit', 'quit']:
                rprint("[yellow]Ending chat session. Goodbye![/yellow]")
                break
            if not user_query:
                continue

            stream_generator = rag.chat_stream(user_query)
            first_chunk = None

            status_msg = "[bold green]Retrieving and thinking...[/bold green]"
            with console.status(status_msg, spinner="dots"):
                try:
                    first_chunk = next(stream_generator)
                except StopIteration:
                    pass

            console.print("[bold magenta]AI[/bold magenta]:")

            full_response = ""
            sources = []

            # We use a separate Live context (NOT nested in status)
            with Live(
                Markdown(""),
                auto_refresh=False,
                vertical_overflow="visible"
            ) as live:

                # Helper to process chunks (DRY logic)
                def process_chunk(chunk):
                    nonlocal full_response, sources

                    # 1. Capture Sources (Metadata)
                    if isinstance(chunk, dict) and "__sources__" in chunk:
                        sources = chunk["__sources__"]
                        return

                    # 2. Render Text
                    if isinstance(chunk, str):
                        full_response += chunk
                        live.update(Markdown(full_response))
                        live.refresh()

                # Process the chunk we fetched during Phase 1
                if first_chunk:
                    process_chunk(first_chunk)

                # Process the rest of the stream
                for chunk in stream_generator:
                    process_chunk(chunk)

            if sources:
                rprint("\n[dim]--- Sources used for this answer ---[/dim]")
                source_table = Table(
                    show_header=True, header_style="bold blue"
                )
                source_table.add_column("Source File")
                for src in sources:
                    source_table.add_row(src)
                console.print(source_table)

            rprint("-" * 50 + "\n")

        except KeyboardInterrupt:
            rprint("\n[yellow]Session interrupted by user.[/yellow]")
            break
        except Exception as e:
            console.print(
                f"[bold red]Error processing query:[/bold red] {e}"
            )
            logger.log("Query processing failed", "ERROR")
            continue


@app.command()
def clear_cache():
    """Clear cached embeddings and indexes."""
    console.rule("[bold yellow]Cache Management[/bold yellow]")
    rprint("[yellow]Clearing ChatPDF cache...[/yellow]")

    try:
        rag = get_rag_instance()
        rag.serializer.clear_cache()
        rprint("[bold green]✓ Cache cleared successfully[/bold green]")
        logger.log("Cache cleared by user", "INFO")
    except Exception as e:
        console.print(
            f"[bold red]Error clearing cache:[/bold red] {e}"
        )
        logger.log("Cache clearing failed", "ERROR")
        raise typer.Exit(code=1)


@app.command()
def cache_info():
    """Display information about cached data."""
    console.rule("[bold blue]Cache Information[/bold blue]")

    try:
        rag = get_rag_instance()
        info = rag.serializer.get_cache_info()

        if not info.get("cached"):
            rprint("[yellow]No cache found[/yellow]")
            return

        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Property")
        table.add_column("Value")

        for key, value in info.items():
            if key == "size_mb":
                table.add_row("Cache Size", f"{value:.2f} MB")
            elif key == "n_docs":
                table.add_row("Cached Documents", str(value))
            elif key == "timestamp":
                table.add_row("Created", str(value))
            elif key != "cached":
                table.add_row(key.capitalize(), str(value))

        console.print(table)

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        logger.log("Cache info retrieval failed", "ERROR")
        raise typer.Exit(code=1)
