"""`zotero-rag` command line interface."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.markup import escape
from rich.tree import Tree

from zotero_rag import __version__
from zotero_rag.config import Config, load_config, load_env, packaged_defaults
from zotero_rag.errors import ZoteroRagError
from zotero_rag.scaffold import ENV_FILENAME, init_index_folder
from zotero_rag.zotero_api import (
    CollectionNode,
    build_client,
    build_collection_tree,
    count_collections,
    fetch_collections,
)

console = Console()
err_console = Console(stderr=True)

app = typer.Typer(
    help="Index and query a Zotero library using RAG.",
    no_args_is_help=True,
    add_completion=False,
)


def _setup_logging(verbose: bool) -> None:
    """Warnings and above by default; -v raises this package to INFO."""
    logging.basicConfig(
        level=logging.WARNING,
        format="%(message)s",
        handlers=[RichHandler(console=err_console, show_time=False, show_path=False)],
    )
    logging.getLogger("zotero_rag").setLevel(logging.INFO if verbose else logging.WARNING)


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"zotero-rag {__version__}")
        raise typer.Exit()


def _fail(error: ZoteroRagError) -> None:
    """Report an expected error and exit non-zero, without a traceback."""
    err_console.print(f"[bold red]Error:[/bold red] {error}")
    raise typer.Exit(code=1)


@app.callback()
def main(
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Show progress and diagnostic output.")
    ] = False,
    version: Annotated[
        bool,
        typer.Option("--version", callback=_version_callback, is_eager=True, help="Show version."),
    ] = False,
) -> None:
    """Index and query a Zotero library using RAG."""
    _setup_logging(verbose)
    load_env()


def _render_tree(nodes: list[CollectionNode], max_depth: int | None) -> Tree:
    """Build a rich Tree of collections, showing name, key, and item count."""
    root = Tree("[bold]Zotero collections[/bold]")
    for node in nodes:
        _add_branch(root, node, depth=1, max_depth=max_depth)
    return root


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def _add_branch(parent: Tree, node: CollectionNode, depth: int, max_depth: int | None) -> None:
    if max_depth is not None and depth > max_depth:
        return
    # Collection names are user data: escape them so square brackets in a name
    # are not parsed as rich markup tags.
    name = escape(node.name)
    label = (
        f"[bold]{name}[/bold]  [dim]{node.key}[/dim]  "
        f"[cyan]{_plural(node.num_items, 'item')}[/cyan]"
    )
    if node.children:
        hidden = max_depth is not None and depth >= max_depth
        if hidden:
            label += f"  [dim](+{_plural(len(node.children), 'subcollection')})[/dim]"
    branch = parent.add(label)
    for child in node.children:
        _add_branch(branch, child, depth + 1, max_depth)


@app.command("list-collections")
def list_collections(
    depth: Annotated[
        int | None,
        typer.Option("--depth", help="Limit the displayed hierarchy to N levels.", min=1),
    ] = None,
) -> None:
    """Print the Zotero collection hierarchy with keys and item counts.

    Run this before `init` to find the collection path or key to index. It does
    not require an index folder — only Zotero credentials.
    """
    try:
        config = load_config()
        zot = build_client(config)
        raw = fetch_collections(zot)
    except ZoteroRagError as exc:
        _fail(exc)
        return

    nodes = build_collection_tree(raw)
    if not nodes:
        console.print("[yellow]No collections found in this library.[/yellow]")
        return

    console.print(_render_tree(nodes, depth))
    total = count_collections(nodes)
    console.print(
        f"\n[dim]{_plural(total, 'collection')}. Use the path or key with "
        f"[/dim]zotero-rag index --collection-key KEY[dim].[/dim]"
    )


@app.command()
def init(
    path: Annotated[
        Path,
        typer.Argument(help="Directory to initialize; created if it does not exist."),
    ] = Path("."),
) -> None:
    """Create an index folder: config.yaml, a .env template, and a .gitignore.

    Offline — contacts nothing, pins no collection, and creates no vector store.
    The collection is pinned by the first `index` run.
    """
    try:
        # Packaged defaults only: an existing config.yaml means this is already
        # an index folder, which `init_index_folder` refuses.
        global_config = Config.model_validate(packaged_defaults()).global_
        result = init_index_folder(path, global_config)
    except ZoteroRagError as exc:
        _fail(exc)
        return

    console.print(f"Initialized index folder [bold]{result.directory}[/bold]")
    for name in result.created:
        console.print(f"  [green]created[/green]  {name}")
    for name in result.skipped:
        console.print(f"  [yellow]kept[/yellow]     {name} [dim](already existed)[/dim]")

    console.print(
        f"\nNext: add your Zotero credentials to "
        f"[bold]{result.directory / ENV_FILENAME}[/bold], then run "
        f"[bold]zotero-rag index --collection-key KEY[/bold] from that folder "
        f"to pin and index a collection."
    )


@app.command()
def index() -> None:
    """Index the pinned Zotero collection. Not yet implemented."""
    err_console.print(
        "[yellow]`index` is not implemented yet[/yellow] — planned for a later milestone."
    )
    raise typer.Exit(code=1)


@app.command()
def query() -> None:
    """Answer a question from the index. Not yet implemented."""
    err_console.print(
        "[yellow]`query` is not implemented yet[/yellow] — planned for a later milestone."
    )
    raise typer.Exit(code=1)


if __name__ == "__main__":  # pragma: no cover
    app()
