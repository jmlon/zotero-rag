"""`zotero-rag-ui` entry point: check the index folder, then draw the screen (§2)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.logging import RichHandler

from zotero_rag import PACKAGE_LOGGERS, __version__
from zotero_rag.config import load_env
from zotero_rag.tui import UI_DIRNAME
from zotero_rag.tui.history import HISTORY_FILENAME, PromptHistory
from zotero_rag.tui.session import open_session
from zotero_rag_core.errors import ZoteroRagError
from zotero_rag_core.retrieval import RetrievalMode

LOG_FILENAME = "ui.log"
# The CLI's list (`cli.NOISY_LOGGERS`) plus the HTTP clients, whose request lines
# would bury this package's own diagnostics in the log.
NOISY_LOGGERS = ("pypdf", "httpx", "httpcore", "openai")

err_console = Console(stderr=True)

app = typer.Typer(add_completion=False)


def _version_callback(value: bool) -> None:
    if value:
        print(f"zotero-rag-ui {__version__}")  # noqa: T201
        raise typer.Exit()


def _log_to_stderr() -> None:
    """Until the screen is up, warnings can go to the terminal as the CLI's do."""
    logging.basicConfig(
        level=logging.WARNING,
        format="%(message)s",
        handlers=[RichHandler(console=err_console, show_time=False, show_path=False)],
    )


def _log_while_drawing(index_dir: Path | None, verbose: bool) -> None:
    """Move logging off the terminal, where it would corrupt the screen (§8).

    With `-v` it goes to `<index-folder>/.zotero-rag-ui/ui.log`; without, nowhere.
    A `NullHandler` is still installed, because a root logger with no handlers
    makes Python fall back to writing warnings to stderr.
    """
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    if not verbose or index_dir is None:
        root.addHandler(logging.NullHandler())
        return
    log_dir = index_dir / UI_DIRNAME
    log_dir.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_dir / LOG_FILENAME, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root.addHandler(handler)
    for name in PACKAGE_LOGGERS:
        logging.getLogger(name).setLevel(logging.INFO)
    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


@app.command()
def main(
    top_k: Annotated[
        int | None,
        typer.Option("--top-k", help="Chunks to retrieve; overrides the configured top_k.", min=1),
    ] = None,
    window: Annotated[
        int | None,
        typer.Option(
            "--window", help="Also include N neighbouring chunks either side of each hit.", min=0
        ),
    ] = None,
    retrieval: Annotated[
        RetrievalMode | None,
        typer.Option(
            "--retrieval", help="vector (embeddings only) or hybrid (also BM25 keyword search)."
        ),
    ] = None,
    reranker: Annotated[
        str | None,
        typer.Option(
            "--reranker",
            help="Re-score the retrieved passages against the question: an OpenRouter "
            "reranker id (e.g. cohere/rerank-4-fast), cohere for Cohere's own API, or none.",
        ),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help=f"Write diagnostics to {UI_DIRNAME}/{LOG_FILENAME}."),
    ] = False,
    version: Annotated[
        bool,
        typer.Option("--version", callback=_version_callback, is_eager=True, help="Show version."),
    ] = False,
) -> None:
    """Ask questions of this index folder in a full-screen terminal UI.

    Must be run from an index folder that has been indexed at least once.
    """
    _log_to_stderr()
    load_env()
    try:
        session = open_session(top_k, window, retrieval, reranker)
    except ZoteroRagError as exc:
        # The same message and exit code as the CLI, and the screen is never drawn.
        err_console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(code=1) from None

    # Imported here so a failed check, or `--help`, does not pay for loading Textual.
    from zotero_rag.tui.app import ZoteroRagApp

    _log_while_drawing(session.index_dir, verbose)
    history_path = (session.index_dir or Path.cwd()) / UI_DIRNAME / HISTORY_FILENAME
    ZoteroRagApp(session, PromptHistory.load(history_path)).run()
