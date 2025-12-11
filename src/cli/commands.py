import typer
import time
import random
from typing_extensions import Annotated
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich import print as rprint

console = Console()
app = typer.Typer(help="ChatPDF Command Line Interface")


class MockRAGBackend:
    def ingest_documents(self, path: str):
        """Simulates loading and indexing documents."""
        files = [f"{path}/doc_{i}.txt" for i in range(1, 6)]
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            task = progress.add_task(description="Processing documents...", total=len(files))
            for file in files:
                time.sleep(0.5) # Simulate processing time
                progress.update(task, advance=1, description=f"Indexing {file}")
        return len(files)

    def stream_chat_response(self, query: str):
        """Simulates streaming an LLM response chunk by chunk."""
        dummy_response = (
            f"Based on the context regarding '{query}', here is the answer. "
            "RAG systems retrieve relevant documents before generating an answer. "
            "This ensures the LLM is grounded in factual data rather than hallucinating. "
            "Here is a bullet point list explaining why:\n"
            "- Improved accuracy\n"
            "- Source attribution\n"
            "- Ability to update knowledge base easily."
        )
        chunks = dummy_response.split(" ")
        for chunk in chunks:
            time.sleep(random.uniform(0.05, 0.2))  # Simulate token generation delay
            yield chunk + " "

    def get_sources(self, query: str):
        """Simulates retrieving the source documents used for the answer."""
        return [
            {"source": "wiki_rag.txt", "content": "RAG combines retrieval and generation..."},
            {"source": "internal_docs_v2.pdf", "content": "...grounding the LLM in facts..."},
        ]


backend = MockRAGBackend()


@app.command()
def ingest(
    path: Annotated[str, typer.Argument(help="Path to the directory containing documents to index.")]
):
    console.rule("[bold blue]Document Ingestion Pipeline")
    rprint(f"[italic]Scanning directory:[/italic] {path}")

    try:
        # TODO: Add actual document ingestion logic here
        count = backend.ingest_documents(path)
        console.print(Panel(f"[bold green]Success![/bold green] Indexed {count} documents. ", title="Ingestion Status"))
    except Exception as e:
        console.print(f"[bold red]Error during ingestion:[/bold red] {e}")


@app.command()
def chat():
    console.clear()
    console.rule("[bold magenta]Interactive ChatPDF Terminal[/bold magenta]")
    rprint("[yellow]Type 'exit', 'quit', or Ctrl+C to stop the session.[/yellow]\n")

    while True:
        try:
            user_query = typer.prompt(typer.style("You", fg=typer.colors.CYAN, bold=True))
            user_query = user_query.strip()

            if user_query.lower() in ['exit', 'quit']:
                rprint("[yellow]Ending chat session. Goodbye![/yellow]")
                break
            if not user_query:
                continue

            with console.status("[bold green]Thinking and Retrieving Documents...[/bold green]", spinner="dots"):
                # TODO: Add Reranking and query processing here
                time.sleep(1)

            console.print("[bold magenta]AI[/bold magenta]: ", end="\n")
            full_response = ""


            from rich.live import Live
            with Live(console=console, refresh_per_second=10, transient=False) as live:
                # TODO: Change PoC to actuall LLM token response streaming
                for chunk in backend.stream_chat_response(user_query):
                    full_response += chunk
                    
                    live.update(Markdown(full_response))

            console.print()

            sources = backend.get_sources(user_query)
            if sources:
                rprint("\n[dim]--- Sources used for this answer ---[/dim]")
                source_table = Table(show_header=True, header_style="bold blue")
                source_table.add_column("Filename")
                source_table.add_column("Snippet Snippet")
                for src in sources:
                    # Truncate snippet for app readability
                    snippet = (src['content'][:75] + '..') if len(src['content']) > 75 else src['content']
                    source_table.add_row(src["source"], snippet)
                console.print(source_table)

            rprint("-" * 50)

        except KeyboardInterrupt:
            rprint("\n[yellow]Session interrupted by user. Exiting.[/yellow]")
            break
