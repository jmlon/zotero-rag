"""`zotero-rag` command line interface."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated, Any, cast

import typer
from langchain_core.messages import AIMessage, BaseMessage
from rich.console import Console
from rich.logging import RichHandler
from rich.markup import escape
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table
from rich.tree import Tree

from zotero_rag import __version__
from zotero_rag.config import (
    CONFIG_FILENAME,
    ENV_FILENAME,
    Config,
    existing_env_files,
    load_config,
    load_env,
    packaged_defaults,
    pin_collection,
)
from zotero_rag.download import CACHE_DIRNAME, ensure_cached
from zotero_rag.embedding import build_embeddings
from zotero_rag.errors import (
    CollectionMismatchError,
    NotAnIndexFolderError,
    NotIndexedError,
    ZoteroRagError,
)
from zotero_rag.indexer import IndexStats, index_plan
from zotero_rag.llm import build_chat
from zotero_rag.planning import IndexPlan, PlannedAttachment, build_plan
from zotero_rag.querying import QueryResult, answer_question
from zotero_rag.scaffold import init_index_folder
from zotero_rag.vectorstore import CHROMA_DIRNAME, VectorStore
from zotero_rag.zotero_api import (
    CollectionNode,
    build_client,
    build_collection_tree,
    collection_scope,
    count_collections,
    fetch_collection_items,
    fetch_collections,
    fetch_username,
    find_collection_by_key,
    library_scope,
    resolve_collection_path,
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


# `-v` is accepted both before the subcommand (as a global flag) and on `index`
# and `query` themselves, because that is where a user reaches for it.
VerboseOption = Annotated[
    bool, typer.Option("--verbose", "-v", help="Show progress and diagnostic output.")
]


def _enable_verbose(verbose: bool) -> None:
    """Raise this package to INFO when a subcommand was given its own `-v`."""
    if verbose:
        logging.getLogger("zotero_rag").setLevel(logging.INFO)


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
        f"\n[dim]{_plural(total, 'collection')}. Index one by name — [/dim]"
        f'zotero-rag index --collection "NAME"[dim] — or, when a name is not '
        f"unique, by key — [/dim]zotero-rag index --collection-key KEY[dim].[/dim]"
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

    # `index` and `query` must run *inside* the folder, so spell out the `cd`.
    steps: list[str] = []
    if result.directory.resolve() != Path.cwd():
        steps.append(f"cd {result.directory}")
    steps.append(f"add your credentials to {ENV_FILENAME}")
    steps.append('zotero-rag index --collection "NAME"   (or --collection-key KEY)')

    console.print("\n[bold]Next:[/bold]")
    for number, step in enumerate(steps, start=1):
        console.print(f"  {number}. [bold]{escape(step)}[/bold]")

    console.print(
        "\n[dim]A collection NAME is matched case-insensitively and may be a path such as\n"
        '"Research/Data Trust/Legal". Use the key when a name is not unique.[/dim]'
    )

    # A .env further up (or at ~/.env) is used too, so say so rather than let the
    # empty template look like the only place credentials can go.
    # Resolved, so the hint cannot print a bare `.env` that reads as the local one.
    index_dir = result.directory.resolve()
    inherited = [path for path in existing_env_files(index_dir) if path.parent != index_dir]
    if inherited:
        console.print(
            f"\n[dim]Credentials already in {inherited[0]} will be used for any "
            f"variable you leave blank.[/dim]"
        )


def _is_verbose() -> bool:
    """Verbosity is carried by the package logger level, set in the callback."""
    return logging.getLogger("zotero_rag").isEnabledFor(logging.INFO)


def _resolve_target(
    config: Config,
    roots: list[CollectionNode],
    collection: str | None,
    collection_key: str | None,
    assume_yes: bool,
) -> tuple[list[CollectionNode], str, CollectionNode | None]:
    """Decide which collections are in scope, honouring any pinned collection.

    Returns the collections to walk, a human-readable target description, and the
    single collection to pin (`None` for the whole library). Raises when the
    request contradicts what the index is already pinned to.
    """
    pinned = config.index.collection_key
    if pinned:
        requested_key = collection_key
        if collection is not None:
            requested_key = resolve_collection_path(roots, collection).key
        if requested_key is not None and requested_key != pinned:
            raise CollectionMismatchError(
                f"This index is pinned to collection {config.index.collection_name!r} "
                f"({pinned}), but {requested_key} was requested. An index folder serves "
                "one collection; use a different folder."
            )
        node = find_collection_by_key(roots, pinned)
        return collection_scope(node, config.index.recursive), f"{node.name} ({node.key})", node

    if collection_key is not None:
        node = find_collection_by_key(roots, collection_key)
        return collection_scope(node, config.index.recursive), f"{node.name} ({node.key})", node

    if collection is not None:
        node = resolve_collection_path(roots, collection)
        return collection_scope(node, config.index.recursive), f"{node.name} ({node.key})", node

    # No target given: the whole library root.
    scope = library_scope(roots)
    err_console.print(
        f"[yellow]Warning:[/yellow] no collection given, so the whole library would be "
        f"indexed — {_plural(len(scope), 'collection')}. Items that belong to no "
        f"collection are not included."
    )
    if not assume_yes and not typer.confirm("Index the entire library?", default=False):
        raise typer.Abort()
    return scope, "the whole library", None


def _run_index(
    plan: IndexPlan,
    config: Config,
    store: VectorStore,
    zot: Any,
    embeddings: Any,
    cache_dir: Path,
) -> IndexStats:
    """Run the indexing pass with a progress bar over the planned attachments."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Indexing...", total=len(plan.attachments))

        def on_start(attachment: PlannedAttachment) -> None:
            progress.update(task, advance=1, description=escape(attachment.title[:50]))

        return index_plan(
            plan,
            store=store,
            download=lambda key: ensure_cached(zot, key, cache_dir),
            embed=embeddings.embed_documents,
            chunk_size=config.index.chunk_size,
            chunk_overlap=config.index.chunk_overlap,
            on_start=on_start,
        )


def _print_index_summary(
    plan: IndexPlan, stats: IndexStats, store: VectorStore, verbose: bool
) -> None:
    """Report what the run did, including everything it chose not to do."""
    console.print(
        f"\n[bold green]Indexed {_plural(stats.indexed, 'PDF')}[/bold green] "
        f"({_plural(stats.chunks, 'chunk')} from {_plural(stats.pages, 'page')}) "
        f"out of {_plural(plan.items_seen, 'item')} seen in "
        f"{_plural(len(plan.collections), 'collection')}."
    )

    if stats.already_indexed:
        console.print(f"  [dim]{stats.already_indexed:>4}[/dim]  already indexed, skipped")
    if stats.no_text:
        console.print(f"  [dim]{stats.no_text:>4}[/dim]  no extractable text (scanned PDF?)")
    for reason, count in plan.skip_counts.items():
        console.print(f"  [dim]{count:>4}[/dim]  {reason.value}")

    if stats.failures:
        console.print(f"\n[yellow]{_plural(stats.failed, 'failure')}:[/yellow]")
        for failure in stats.failures:
            console.print(f"  [yellow]•[/yellow] {escape(failure.title)}: {escape(failure.error)}")

    console.print(f"\nThe store now holds {_plural(store.count(), 'chunk')}.")
    if verbose:
        console.print(
            f"[dim]{stats.embedding_tokens} embedding tokens, {stats.elapsed_s:.1f}s elapsed.[/dim]"
        )


def _print_plan(plan: IndexPlan, target: str) -> None:
    """Render the dry-run plan: what would be indexed, and what would not."""
    console.print(
        f"\nTarget: [bold]{escape(target)}[/bold] "
        f"({_plural(len(plan.collections), 'collection')}, "
        f"{_plural(plan.items_seen, 'item')})\n"
    )

    if plan.attachments:
        table = Table(title="PDFs that would be indexed", title_justify="left")
        table.add_column("#", justify="right", style="dim")
        table.add_column("Title", overflow="ellipsis", max_width=54)
        table.add_column("Authors", overflow="ellipsis", max_width=28)
        table.add_column("Year", justify="right")
        table.add_column("Collection", overflow="ellipsis", max_width=20)
        table.add_column("Attachment", style="dim")
        for number, attachment in enumerate(plan.attachments, start=1):
            table.add_row(
                str(number),
                escape(attachment.title),
                escape(attachment.authors) or "[dim]—[/dim]",
                attachment.year or "[dim]—[/dim]",
                escape(attachment.collection_name),
                attachment.attachment_key,
            )
        console.print(table)
    else:
        console.print("[yellow]Nothing to index: no PDF attachments found.[/yellow]")

    console.print(
        f"\n[bold]{_plural(len(plan.attachments), 'PDF')}[/bold] from "
        f"{_plural(plan.item_count, 'item')} would be indexed."
    )
    if plan.skipped:
        console.print("Skipped:")
        for reason, count in plan.skip_counts.items():
            console.print(f"  [dim]{count:>4}[/dim]  {reason.value}")
    console.print("\n[dim]Dry run: nothing was downloaded, embedded, or written.[/dim]")


@app.command()
def index(
    collection: Annotated[
        str | None,
        typer.Option("--collection", help="Zotero collection path, e.g. 'Research/Legal'."),
    ] = None,
    collection_key: Annotated[
        str | None,
        typer.Option("--collection-key", help="Zotero collection key (unambiguous)."),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="List what would be indexed; make no changes."),
    ] = False,
    assume_yes: Annotated[
        bool, typer.Option("--yes", "-y", help="Skip the whole-library confirmation prompt.")
    ] = False,
    verbose: VerboseOption = False,
) -> None:
    """Index the PDF attachments of the pinned Zotero collection.

    The first run pins its collection into config.yaml; later runs reuse it.
    Must be run from an index folder.
    """
    _enable_verbose(verbose)

    if collection is not None and collection_key is not None:
        err_console.print(
            "[bold red]Error:[/bold red] give either --collection or --collection-key, not both."
        )
        raise typer.Exit(code=1)

    try:
        config = load_config()
        if config.source_dir is None:
            raise NotAnIndexFolderError(
                f"No {CONFIG_FILENAME} in {Path.cwd()}. Run `zotero-rag init` here first, "
                "or change to an index folder."
            )
        index_dir = config.source_dir

        # Fail on a missing OpenRouter key before downloading anything.
        if not dry_run:
            config.global_.require_openrouter()

        zot = build_client(config)
        roots = build_collection_tree(fetch_collections(zot))
        scope, target, node = _resolve_target(config, roots, collection, collection_key, assume_yes)

        # The citation URL needs the library owner's username; a dry run skips it.
        username = "" if dry_run else fetch_username(zot)

        items: list[tuple[str, list[dict[str, object]]]] = []
        with console.status("[dim]Fetching items from Zotero...[/dim]") as status:
            for collection_node in scope:
                status.update(f"[dim]Fetching items from {escape(collection_node.name)}...[/dim]")
                items.append(
                    (collection_node.name, fetch_collection_items(zot, collection_node.key))
                )

        plan = build_plan(items, username)

        if dry_run:
            _print_plan(plan, target)
            return

        # Pin before doing any work, so an interrupted run resumes on the same
        # collection rather than silently accepting a different one.
        if node is not None and not config.index.is_pinned:
            pin_collection(
                index_dir,
                collection_key=node.key,
                collection_name=node.name,
                collection_path=collection,
            )
            console.print(f"Pinned collection [bold]{escape(node.name)}[/bold] ({node.key})")

        store = VectorStore(index_dir / CHROMA_DIRNAME)
        embeddings = build_embeddings(config)
        cache_dir = index_dir / CACHE_DIRNAME
        stats = _run_index(plan, config, store, zot, embeddings, cache_dir)
    except ZoteroRagError as exc:
        _fail(exc)
        return

    _print_index_summary(plan, stats, store, verbose=_is_verbose())


def _open_index() -> tuple[Config, VectorStore]:
    """Load the index config and open its store, or explain what is missing."""
    config = load_config()
    if config.source_dir is None:
        raise NotAnIndexFolderError(
            f"No {CONFIG_FILENAME} in {Path.cwd()}. Run `zotero-rag init` here first, "
            "or change to an index folder."
        )
    chroma_dir = config.source_dir / CHROMA_DIRNAME
    if not VectorStore.exists(chroma_dir):
        raise NotIndexedError(
            f"This index folder has no vector store yet ({chroma_dir} is empty). "
            'Run `zotero-rag index --collection "NAME"` (or `--collection-key KEY`) first.'
        )
    store = VectorStore(chroma_dir)
    if store.count() == 0:
        raise NotIndexedError(
            "The vector store holds no chunks. Run `zotero-rag index` to populate it."
        )
    return config, store


def _print_answer(result: QueryResult, show_sources: bool, verbose: bool) -> None:
    """Print the answer, then the numbered source list it cites."""
    console.print()
    console.print(escape(result.answer))

    # Only what the answer actually cited, so an "I don't know" does not appear
    # to rest on the passages it declined to use. `--show-sources` shows them all.
    listed = [source for source in result.sources if source.cited or show_sources]
    if listed:
        console.print("\n[bold]Sources[/bold]")
        for source in listed:
            marker = "[cyan]" if source.cited else "[dim]"
            console.print(f"  {marker}\\[{source.number}][/] {escape(source.label)}")
            if source.zotero_url:
                console.print(f"       [dim]{source.zotero_url}[/dim]")
            if show_sources:
                for excerpt in source.excerpts:
                    console.print(f"       [dim]{escape(excerpt.strip())}[/dim]\n")
    elif result.sources:
        console.print(
            f"\n[dim]The answer cites no sources; "
            f"{_plural(result.chunks_retrieved, 'passage')} were retrieved "
            f"(see --show-sources).[/dim]"
        )

    if verbose:
        usage = result.usage
        console.print(
            f"\n[dim]{_plural(result.chunks_retrieved, 'chunk')} retrieved; "
            f"{usage.prompt_tokens} prompt + {usage.completion_tokens} completion tokens "
            f"({usage.embedding_tokens} to embed the question), "
            f"{usage.elapsed_s:.1f}s elapsed.[/dim]"
        )


def _answer_as_json(result: QueryResult, show_sources: bool) -> str:
    """The machine-readable form: answer, sources, usage (PRD §3.4)."""
    payload = {
        "answer": result.answer,
        "sources": [
            source.model_dump(exclude={"excerpts"} if not show_sources else set())
            for source in result.sources
        ],
        "usage": {
            **result.usage.model_dump(),
            "total_tokens": result.usage.total_tokens,
        },
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


@app.command()
def query(
    question: Annotated[str, typer.Argument(help="The question to answer, in any language.")],
    top_k: Annotated[
        int | None,
        typer.Option("--top-k", help="Chunks to retrieve; overrides the configured top_k.", min=1),
    ] = None,
    show_sources: Annotated[
        bool,
        typer.Option("--show-sources", help="Also print the full text of each retrieved chunk."),
    ] = False,
    as_json: Annotated[
        bool, typer.Option("--json", help="Emit a JSON object instead of formatted output.")
    ] = False,
    verbose: VerboseOption = False,
) -> None:
    """Answer a question from the index, with citations back to Zotero items.

    Must be run from an index folder that has been indexed at least once.
    """
    _enable_verbose(verbose)

    try:
        config, store = _open_index()
        embeddings = build_embeddings(config)
        chat = build_chat(config)

        def ask(messages: list[BaseMessage]) -> AIMessage:
            # `invoke` is typed as returning any BaseMessage; a chat model returns
            # an AIMessage, which is what carries `usage_metadata`.
            return cast(AIMessage, chat.invoke(messages))

        # Status goes to stderr so it never contaminates `--json` on stdout.
        with err_console.status("[dim]Retrieving and answering...[/dim]"):
            result = answer_question(
                question,
                store=store,
                embed_query=embeddings.embed_query,
                answer=ask,
                top_k=top_k if top_k is not None else config.index.top_k,
            )
    except ZoteroRagError as exc:
        _fail(exc)
        return

    if as_json:
        # Plain print: rich would wrap and highlight the JSON.
        print(_answer_as_json(result, show_sources))  # noqa: T201
        return
    _print_answer(result, show_sources, verbose=_is_verbose())


if __name__ == "__main__":  # pragma: no cover
    app()
