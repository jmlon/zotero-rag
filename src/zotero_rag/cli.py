"""`zotero-rag` command line interface."""

from __future__ import annotations

import csv
import json
import logging
import os
import shutil
import threading
import time
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from functools import partial
from pathlib import Path
from typing import Annotated, Any, cast

import typer
from langchain_core.messages import AIMessage, BaseMessage
from rich.console import Console
from rich.logging import RichHandler
from rich.markup import escape
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
)
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
from zotero_rag.conversion import (
    ConversionAction,
    ConversionEntry,
    ConversionPlan,
    ConversionState,
    build_conversion_plan,
)
from zotero_rag.converter_run import (
    REPORT_FIELDS,
    ConversionResult,
    ConversionRun,
    Outcome,
    report_rows,
    run_conversion,
)
from zotero_rag.documents import DocumentFormat
from zotero_rag.download import CACHE_DIRNAME, ensure_cached
from zotero_rag.embedding import build_embeddings
from zotero_rag.embedding_comparison import (
    ComparisonReport,
    ComparisonSettings,
    Variant,
    VariantResult,
    chunkings,
    context_warning,
    plan_variants,
    stored_tokens,
    variant_config,
)
from zotero_rag.errors import (
    CollectionMismatchError,
    NotAnIndexFolderError,
    NotIndexedError,
    ZoteroRagError,
)
from zotero_rag.evaluation import (
    MAX_SCORE,
    EvalQuestion,
    EvalReport,
    Grade,
    QuestionOutcome,
    QuestionSetError,
    TimeSpread,
    aggregate,
    build_context_assessor,
    build_verdict_fn,
    evaluate,
    load_questions,
    resolve_questions_path,
    run_settings,
)
from zotero_rag.fingerprint import StoreFingerprint
from zotero_rag.generation import (
    MATCH_DOCUMENT,
    DocumentText,
    GenerationStats,
    build_generator,
    collect_documents,
    file_header,
    generate_questions,
    write_questions,
)
from zotero_rag.index_folder import (
    REINDEX_REMEDY,
    Notice,
    build_keyword_index,
    check_fingerprint,
    open_index,
    resolve_reranker,
)
from zotero_rag.indexer import (
    DIMENSION_PROBE,
    IndexStats,
    drop_replaced_copies,
    find_orphans,
    index_plan,
    prune_orphans,
    refresh_collection_keys,
    repair_embeddings,
)
from zotero_rag.llm import build_chat
from zotero_rag.planning import (
    IndexPlan,
    PlannedAttachment,
    SkipReason,
    SkipReport,
    build_plan,
    item_membership,
)
from zotero_rag.quality import QualityVerdict
from zotero_rag.querying import QueryResult, answer_question
from zotero_rag.retrieval import KeywordIndex, RetrievalMode
from zotero_rag.scaffold import init_index_folder
from zotero_rag.shrink import (
    KeepReason,
    ShrinkEntry,
    Verdict,
    format_size,
    parse_size,
    shrink_candidates,
)
from zotero_rag.shrink_run import BACKUP_DIRNAME, LinkProber, check_candidates, trash_files
from zotero_rag.shrink_run import REPORT_FIELDS as SHRINK_REPORT_FIELDS
from zotero_rag.shrink_run import report_rows as shrink_report_rows
from zotero_rag.snapshot import write_snapshot
from zotero_rag.vectorstore import CHROMA_DIRNAME, VectorStore
from zotero_rag.zotero_api import (
    CollectionNode,
    FileDownloader,
    build_client,
    build_collection_tree,
    collection_scope,
    count_collections,
    fetch_annotated_attachments,
    fetch_collection_items,
    fetch_collections,
    fetch_username,
    find_collection_by_key,
    library_scope,
    require_write_access,
    resolve_collection_path,
    scope_tree,
)

console = Console()
err_console = Console(stderr=True)

app = typer.Typer(
    help="Index and query a Zotero library using RAG.",
    no_args_is_help=True,
    add_completion=False,
)


# pypdf logs a warning for every quirk it finds in a malformed file ("Ignoring
# wrong pointing object ...") and then recovers, so the page is still extracted.
# Those messages are not actionable, and they interleave with the progress bar; a
# page we genuinely cannot read is reported by `extraction.py` itself.
NOISY_LOGGERS = ("pypdf",)

#: Documents `index` prepares at once. The time goes on waiting for Zotero and the
#: embedding provider, so a few threads overlap it; both clients back off on a 429,
#: and a handful stays well under the free tiers' limits.
INDEX_JOBS = 4

#: Questions `eval` answers and grades at once. Each one is a chat request to the
#: answering model and another to the judge, both of which back off on a 429.
EVAL_JOBS = 4


def _setup_logging(verbose: bool) -> None:
    """Warnings and above by default; -v raises this package to INFO."""
    logging.basicConfig(
        level=logging.WARNING,
        format="%(message)s",
        handlers=[RichHandler(console=err_console, show_time=False, show_path=False)],
    )
    logging.getLogger("zotero_rag").setLevel(logging.INFO if verbose else logging.WARNING)
    _set_noisy_logger_level(verbose)


def _set_noisy_logger_level(verbose: bool) -> None:
    """Hide third-party recovery chatter unless the user asked for detail."""
    level = logging.WARNING if verbose else logging.ERROR
    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(level)


# `-v` is accepted both before the subcommand (as a global flag) and on `index`
# and `query` themselves, because that is where a user reaches for it.
VerboseOption = Annotated[
    bool, typer.Option("--verbose", "-v", help="Show progress and diagnostic output.")
]


def _enable_verbose(verbose: bool) -> None:
    """Raise this package to INFO when a subcommand was given its own `-v`."""
    if verbose:
        logging.getLogger("zotero_rag").setLevel(logging.INFO)
        _set_noisy_logger_level(True)


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
    verb: str = "index",
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
        f"{verb}ed — {_plural(len(scope), 'collection')}. Items that belong to no "
        f"collection are not included."
    )
    if not assume_yes and not typer.confirm(
        f"{verb.capitalize()} the entire library?", default=False
    ):
        raise typer.Abort()
    return scope, "the whole library", None


def _print_notice(notice: Notice) -> None:
    """Show a non-fatal notice from opening the index on stderr, beside the progress."""
    err_console.print(f"[yellow]{notice.level.capitalize()}:[/yellow] {escape(notice.text)}")


def _fetch_plan(
    config: Config,
    collection: str | None,
    collection_key: str | None,
    assume_yes: bool,
    with_username: bool,
) -> tuple[IndexPlan, list[CollectionNode], str, CollectionNode | None]:
    """Fetch the target collections from Zotero and plan what indexing them embeds.

    Returns the plan, the library's collection tree, a description of the target,
    and the single collection to pin (`None` for the whole library).
    """
    zot = build_client(config)
    roots = build_collection_tree(fetch_collections(zot))
    scope, target, node = _resolve_target(config, roots, collection, collection_key, assume_yes)

    # The citation URL needs the library owner's username; a dry run skips it.
    username = fetch_username(zot) if with_username else ""

    fetched: list[tuple[CollectionNode, list[dict[str, object]]]] = []
    with err_console.status("[dim]Fetching items from Zotero...[/dim]") as status:
        for collection_node in scope:
            status.update(f"[dim]Fetching items from {escape(collection_node.name)}...[/dim]")
            fetched.append((collection_node, fetch_collection_items(zot, collection_node.key)))

    plan = build_plan(
        [(collection_node.name, items) for collection_node, items in fetched],
        username,
        preference=config.index.attachment_preference,
        index_all_attachments=config.index.index_all_attachments,
        index_notes=config.index.index_notes,
        membership=item_membership(
            (collection_node.key, items) for collection_node, items in fetched
        ),
    )
    return plan, roots, target, node


def _run_index(
    plan: IndexPlan,
    config: Config,
    store: VectorStore,
    embeddings: Any,
    cache_dir: Path,
    jobs: int = 1,
) -> IndexStats:
    """Run the indexing pass with a progress bar over the planned attachments."""
    # The cache suffix depends on the format, so the downloader needs to know which
    # format each key is; the plan is the only place that knows.
    formats = {
        attachment.attachment_key: attachment.document_format for attachment in plan.attachments
    }
    # Not `zot.file()`: pyzotero cannot download content types it does not list (ODT).
    files = FileDownloader.from_config(config)

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
            download=lambda key: ensure_cached(files, key, cache_dir, formats[key]),
            embed=embeddings.embed_documents,
            chunk_size=config.index.chunk_size,
            chunk_overlap=config.index.chunk_overlap,
            embedding_model=config.embedding_model,
            on_start=on_start,
            workers=jobs,
        )


def _print_index_summary(
    plan: IndexPlan, stats: IndexStats, store: VectorStore, verbose: bool
) -> None:
    """Report what the run did, including everything it chose not to do."""
    console.print(
        f"\n[bold green]Indexed {_plural(stats.indexed, 'document')}[/bold green] "
        f"({_plural(stats.chunks, 'chunk')} from {stats.pages} {_segment_noun(plan)}) "
        f"out of {_plural(plan.items_seen, 'item')} seen in "
        f"{_plural(len(plan.collections), 'collection')}."
    )

    if stats.reindexed:
        console.print(f"  [dim]{stats.reindexed:>4}[/dim]  re-indexed (changed in Zotero)")
    if stats.already_indexed:
        console.print(f"  [dim]{stats.already_indexed:>4}[/dim]  already indexed, skipped")
    if stats.no_text:
        console.print(f"  [dim]{stats.no_text:>4}[/dim]  no extractable text (scanned PDF?)")
    _print_formats(plan)
    _print_skips(plan)

    if stats.failures:
        console.print(f"\n[yellow]{_plural(stats.failed, 'failure')}:[/yellow]")
        for failure in stats.failures:
            console.print(f"  [yellow]•[/yellow] {escape(failure.title)}: {escape(failure.error)}")

    console.print(f"\nThe store now holds {_plural(store.count(), 'chunk')}.")
    if verbose:
        console.print(
            f"[dim]{stats.embedding_tokens} embedding tokens, {stats.elapsed_s:.1f}s elapsed.[/dim]"
        )


def _segment_noun(plan: IndexPlan) -> str:
    """What this run's segments are, so the summary never calls a snapshot a page."""
    formats = set(plan.format_counts)
    # A transcript made by `convert` keeps its PDF's pages.
    paged = {DocumentFormat.PDF, DocumentFormat.MARKDOWN}
    if formats <= paged:
        return "pages"
    if formats & paged:
        return "pages and sections"
    return "sections"


def _print_formats(plan: IndexPlan) -> None:
    """Break the planned attachments down by format when it is not all PDFs."""
    counts = plan.format_counts
    if len(counts) < 2:
        return
    listed = ", ".join(f"{count} {name}" for name, count in counts.items())
    console.print(f"  [dim]formats:[/dim] {listed}")


def _print_skips(plan: SkipReport) -> None:
    """Report skips with the detail needed to act on them.

    A bare `10  unsupported content type` tells the user nothing about what those
    files were, which is the one thing that would let them decide whether the
    format is worth supporting (PRD §11.2).
    """
    if not plan.skipped:
        return
    for reason, count in plan.skip_counts.items():
        console.print(f"  [dim]{count:>4}[/dim]  {reason.value}")
        if reason in _DETAILED_SKIPS:
            for content_type, subcount in plan.content_types_for(reason).items():
                console.print(f"       [dim]{subcount:>3}  {escape(content_type)}[/dim]")
        if reason in _FIXABLE_SKIPS:
            # The user has to find these in Zotero, so name them.
            for entry in plan.skipped_for(reason):
                console.print(f"       [dim]{escape(entry.title)} ({entry.key})[/dim]")
            console.print(f"       [yellow]{_FIXABLE_SKIPS[reason]}[/yellow]")


_DETAILED_SKIPS = frozenset({SkipReason.UNSUPPORTED_TYPE, SkipReason.NOT_PREFERRED})

#: Skips the user can fix in Zotero, and how.
_FIXABLE_SKIPS: dict[SkipReason, str] = {
    SkipReason.NOT_IN_STORAGE: (
        "Zotero holds no file for these. In Zotero desktop, check Settings → Sync: "
        "file syncing on, using Zotero storage (not WebDAV), quota not full. Then sync "
        "and run this command again."
    ),
    SkipReason.LINKED_FILE: (
        "These files live only on your computer. Convert them to stored files in "
        "Zotero (Tools → Manage Attachments → Convert Linked Files to Stored Files), "
        "sync, and run this command again."
    ),
}


def _print_plan(plan: IndexPlan, target: str) -> None:
    """Render the dry-run plan: what would be indexed, and what would not."""
    console.print(
        f"\nTarget: [bold]{escape(target)}[/bold] "
        f"({_plural(len(plan.collections), 'collection')}, "
        f"{_plural(plan.items_seen, 'item')})\n"
    )

    if plan.attachments:
        table = Table(title="Documents that would be indexed", title_justify="left")
        table.add_column("#", justify="right", style="dim")
        table.add_column("Title", overflow="ellipsis", max_width=48)
        table.add_column("Authors", overflow="ellipsis", max_width=26)
        table.add_column("Year", justify="right")
        table.add_column("Format")
        table.add_column("Collection", overflow="ellipsis", max_width=18)
        table.add_column("Attachment", style="dim")
        for number, attachment in enumerate(plan.attachments, start=1):
            table.add_row(
                str(number),
                escape(attachment.title),
                escape(attachment.authors) or "[dim]—[/dim]",
                attachment.year or "[dim]—[/dim]",
                str(attachment.document_format),
                escape(attachment.collection_name),
                attachment.attachment_key,
            )
        console.print(table)
    else:
        console.print("[yellow]Nothing to index: no readable attachments found.[/yellow]")

    console.print(
        f"\n[bold]{_plural(len(plan.attachments), 'document')}[/bold] from "
        f"{_plural(plan.item_count, 'item')} would be indexed."
    )
    _print_formats(plan)
    if plan.skipped:
        console.print("Skipped:")
        _print_skips(plan)
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
    force: Annotated[
        bool,
        typer.Option("--force", help="Discard the store and re-embed everything from cache."),
    ] = False,
    refetch: Annotated[
        bool,
        typer.Option("--refetch", help="With --force, also delete cached PDFs and download again."),
    ] = False,
    repair: Annotated[
        bool,
        typer.Option(
            "--repair",
            help="Re-embed chunks written by a different model, reusing their stored text.",
        ),
    ] = False,
    prune: Annotated[
        bool,
        typer.Option("--prune", help="Delete chunks whose attachment left the collection."),
    ] = False,
    jobs: Annotated[
        int,
        typer.Option(
            "--jobs",
            help="Documents downloaded and embedded in parallel. Raise with care: "
            "more means more concurrent requests to Zotero and the embedding provider.",
            min=1,
        ),
    ] = INDEX_JOBS,
    verbose: VerboseOption = False,
) -> None:
    """Index the PDF attachments of the pinned Zotero collection.

    The first run pins its collection into config.yaml; later runs reuse it and
    re-embed only what changed in Zotero. Must be run from an index folder.
    """
    _enable_verbose(verbose)

    if refetch and not force:
        err_console.print("[bold red]Error:[/bold red] --refetch only applies with --force.")
        raise typer.Exit(code=1)

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

        plan, roots, target, node = _fetch_plan(
            config, collection, collection_key, assume_yes, with_username=not dry_run
        )

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

        if force:
            removed = store.count()
            store.reset()
            store.write_fingerprint(StoreFingerprint.intended(config))
            console.print(f"[yellow]--force:[/yellow] discarded {_plural(removed, 'chunk')}.")
            if refetch:
                shutil.rmtree(cache_dir, ignore_errors=True)
                console.print("[yellow]--refetch:[/yellow] cleared the PDF cache.")
        else:
            # --repair is the remedy for a model change, so it must not be blocked
            # by the very mismatch it exists to fix.
            if not repair:
                check_fingerprint(store, config, REINDEX_REMEDY, _print_notice)
            if repair:
                _run_repair(store, config, embeddings)

        stats = _run_index(plan, config, store, embeddings, cache_dir, jobs)
        store.write_fingerprint(
            StoreFingerprint.intended(config, embedding_dim=store.derived_dimension() or 0)
        )
        # Before counting orphans: a copy replaced by its transcript (or source) is
        # not an orphan to report, it is a duplicate to drop.
        replaced, _ = drop_replaced_copies(store, plan)
        orphans = prune_orphans(store, plan) if prune else (len(find_orphans(store, plan)), 0)
        # Folders can change without any attachment changing, so membership is
        # checked on every run, and the snapshot is written only once it is current.
        refreshed = refresh_collection_keys(store, plan)
        write_snapshot(
            index_dir,
            [scope_tree(node, config.index.recursive)] if node is not None else roots,
        )
    except ZoteroRagError as exc:
        _fail(exc)
        return

    _print_index_summary(plan, stats, store, verbose=_is_verbose())
    if replaced:
        console.print(
            f"Dropped the old copy of {_plural(replaced, 'document')} now indexed from "
            "its transcript or source, so its text is not retrieved twice."
        )
    _print_orphans(orphans, pruned=prune)
    if refreshed:
        console.print(
            f"Updated the folder membership of {_plural(refreshed, 'attachment')} "
            "(no re-embedding)."
        )


def _run_repair(store: VectorStore, config: Config, embeddings: Any) -> None:
    """Re-embed chunks written by a superseded model, from their stored text."""
    with err_console.status("[dim]Re-embedding chunks from a previous model...[/dim]"):
        stats = repair_embeddings(store, embeddings.embed_documents, config.embedding_model)

    if not stats.stale:
        console.print("[green]--repair:[/green] every chunk already matches the configured model.")
        return
    rebuilt = (
        " (store rebuilt: the new model has a different vector width)" if stats.rebuilt else ""
    )
    console.print(
        f"[green]--repair:[/green] re-embedded {_plural(stats.repaired, 'chunk')} with "
        f"[bold]{escape(config.embedding_model)}[/bold]{rebuilt}."
    )
    if _is_verbose():
        console.print(f"[dim]{stats.embedding_tokens} embedding tokens.[/dim]")


def _print_orphans(orphans: tuple[int, int], pruned: bool) -> None:
    """Report attachments that left the collection, and whether they were removed."""
    attachments, chunks = orphans
    if not attachments:
        return
    if pruned:
        console.print(
            f"[yellow]--prune:[/yellow] removed {_plural(chunks, 'chunk')} from "
            f"{_plural(attachments, 'attachment')} no longer in this collection."
        )
    else:
        console.print(
            f"\n[yellow]{_plural(attachments, 'attachment')}[/yellow] in the store "
            f"{'is' if attachments == 1 else 'are'} no longer in this collection "
            f"(deleted, moved out, or no longer a PDF). Run [bold]index --prune[/bold] "
            f"to remove {'it' if attachments == 1 else 'them'}."
        )


RetrievalOption = Annotated[
    RetrievalMode | None,
    typer.Option(
        "--retrieval",
        help="vector (embeddings only) or hybrid (also BM25 keyword search).",
    ),
]


RerankerOption = Annotated[
    str | None,
    typer.Option(
        "--reranker",
        help="Re-score the retrieved passages against the question: an OpenRouter "
        "reranker id (e.g. cohere/rerank-4-fast), cohere for Cohere's own API, or none.",
    ),
]


def _build_keyword_index(store: VectorStore, mode: RetrievalMode) -> KeywordIndex | None:
    """`build_keyword_index` behind a spinner, since it can take a moment."""
    if mode != "hybrid":
        return None
    with err_console.status("[dim]Building the keyword index...[/dim]"):
        return build_keyword_index(store, mode)


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
        # Windowing makes these two differ, and the gap is the whole story of what
        # it cost, so say both rather than one number that no longer means one thing.
        expanded = (
            f" expanded to {_plural(result.chunks_in_context, 'chunk')} with neighbours;"
            if result.chunks_in_context > result.chunks_retrieved
            else ";"
        )
        console.print(
            f"\n[dim]{_plural(result.chunks_retrieved, 'chunk')} retrieved{expanded} "
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
        "retrieval": {
            "chunks_retrieved": result.chunks_retrieved,
            "chunks_in_context": result.chunks_in_context,
            "candidates_considered": result.candidates_considered,
            "reranked": result.reranked,
        },
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


# --- convert (PRD-markdown) -------------------------------------------------------

#: What to tell the user about entries a run leaves alone, and how to change that.
_STATE_HINTS: dict[ConversionState, str] = {
    ConversionState.EDITED: "kept; --force overwrites them",
    ConversionState.OUTDATED: "kept; --reconvert regenerates them",
    ConversionState.USER_PROVIDED: "yours; never replaced",
    ConversionState.SOURCE_GONE: "kept; the original is in the trash",
    ConversionState.SOURCE_MISSING: "kept; delete them in Zotero if unwanted",
    ConversionState.SUPERSEDED: "kept; a preferred attachment was converted instead",
    ConversionState.DUPLICATE: "kept; delete the extra one in Zotero",
}


def _select_actions(
    actions: list[ConversionEntry], only_items: list[str] | None, limit: int | None
) -> list[ConversionEntry]:
    """Narrow a run to some items (`--only-item`) or a first batch (`--limit`)."""
    if only_items:
        wanted = {key.strip().upper() for key in only_items}
        actions = [
            entry
            for entry in actions
            if entry.source
            and (entry.source.item_key in wanted or entry.source.attachment_key in wanted)
        ]
    return actions[:limit] if limit else actions


def _print_conversion_states(plan: ConversionPlan) -> None:
    """The entries left alone, with what the user can do about each kind."""
    for state, count in plan.state_counts.items():
        if state in (ConversionState.MISSING, ConversionState.STALE):
            continue
        acted = sum(1 for e in plan.actions if e.state is state)
        if acted == count:
            continue
        hint = _STATE_HINTS.get(state, "")
        suffix = f"  [dim]({hint})[/dim]" if hint else ""
        console.print(f"  [dim]{count - acted:>4}[/dim]  {state.value}{suffix}")


def _print_conversion_plan(
    plan: ConversionPlan, actions: list[ConversionEntry], target: str, ocr: str
) -> None:
    console.print(
        f"\nTarget: [bold]{escape(target)}[/bold] "
        f"({_plural(len(plan.collections), 'collection')}, "
        f"{_plural(plan.items_seen, 'item')})\n"
    )
    if actions:
        table = Table(title="Transcripts that would be written", title_justify="left")
        table.add_column("#", justify="right", style="dim")
        table.add_column("Title", overflow="ellipsis", max_width=52)
        table.add_column("Format")
        table.add_column("Action")
        table.add_column("Why", overflow="ellipsis", max_width=30)
        table.add_column("Attachment", style="dim")
        for number, entry in enumerate(actions, start=1):
            assert entry.source is not None
            table.add_row(
                str(number),
                escape(entry.title),
                entry.source.source_format,
                str(entry.action),
                entry.state.value,
                entry.source.attachment_key,
            )
        console.print(table)
    else:
        console.print("[green]Nothing to convert: every transcript is up to date.[/green]")

    if actions:
        console.print(f"\n[bold]{_plural(len(actions), 'transcript')}[/bold] would be written.")
    if len(actions) < len(plan.actions):
        console.print(f"  [dim]{len(plan.actions) - len(actions)} more left for a later run.[/dim]")
    _print_conversion_states(plan)
    if plan.skipped:
        console.print("Skipped:")
        _print_skips(plan)
    if ocr == "hosted":
        console.print(
            "\n[dim]--ocr hosted: which documents have scanned pages is only known after "
            "converting; you will be asked before any page is sent.[/dim]"
        )
    console.print("\n[dim]Dry run: nothing was downloaded, converted, or written.[/dim]")


def _confirm_ocr(entries: list[ConversionEntry]) -> bool:
    verb = "has" if len(entries) == 1 else "have"
    err_console.print(
        f"\n[yellow]{_plural(len(entries), 'document')} {verb} scanned pages.[/yellow] "
        "With --ocr hosted, just those pages are uploaded to Firecrawl Parse "
        "(firecrawl.dev) to be read. Nothing else leaves this machine."
    )
    for entry in entries[:10]:
        err_console.print(f"  [dim]• {escape(entry.title)}[/dim]")
    if len(entries) > 10:
        err_console.print(f"  [dim]… and {len(entries) - 10} more[/dim]")
    return typer.confirm("Send those pages for OCR?", default=False)


def _print_conversion_summary(plan: ConversionPlan, run: ConversionRun, verbose: bool) -> None:
    counts = run.outcome_counts
    written = counts.get(Outcome.CREATED, 0) + counts.get(Outcome.REGENERATED, 0)
    console.print(
        f"\n[bold green]Wrote {_plural(written, 'transcript')}[/bold green] "
        f"({counts.get(Outcome.CREATED, 0)} new, {counts.get(Outcome.REGENERATED, 0)} "
        f"regenerated) for {_plural(plan.items_seen, 'item')} in "
        f"{_plural(len(plan.collections), 'collection')}."
    )
    if run.quality_counts:
        listed = ", ".join(f"{n} {verdict}" for verdict, n in run.quality_counts.items())
        console.print(f"  [dim]quality:[/dim] {listed}")
    _print_conversion_states(plan)
    if plan.skipped:
        _print_skips(plan)

    weak = run.with_quality(QualityVerdict.FAIL) + run.with_quality(QualityVerdict.UNVERIFIED)
    if weak:
        console.print(
            f"\n{_plural(len(weak), 'transcript')} did not pass the quality check "
            "(attached anyway; shrink-storage will keep their originals):"
        )
        shown = weak if verbose else weak[:15]
        for result in shown:
            assert result.quality is not None
            reasons = "; ".join(result.quality.reasons)
            console.print(
                f"  [yellow]•[/yellow] {escape(result.title)} "
                f"[dim]({result.quality.verdict}: {escape(reasons)})[/dim]"
            )
        if len(shown) < len(weak):
            console.print(f"  [dim]… and {len(weak) - len(shown)} more (-v lists all)[/dim]")

    if run.unread_scans and not run.ocr_declined:
        count = len(run.unread_scans)
        console.print(
            f"\n{_plural(count, 'document')} {'has' if count == 1 else 'have'} scanned pages "
            "with no text. --ocr hosted would send just those pages to Firecrawl Parse."
        )
    images = sum(result.images for result in run.results)
    if images:
        console.print(f"[dim]{images} images in the sources are not in the transcripts.[/dim]")

    problems = [r for r in run.results if r.outcome in (Outcome.FAILED, Outcome.CONFLICT)]
    if problems:
        console.print(f"\n[yellow]{_plural(len(problems), 'problem')}:[/yellow]")
        for result in problems:
            console.print(f"  [yellow]•[/yellow] {escape(result.title)}: {escape(result.error)}")


def _write_conversion_report(path: Path, plan: ConversionPlan, run: ConversionRun | None) -> None:
    _write_report(path, report_rows(plan.entries, run), REPORT_FIELDS)


def _write_report(path: Path, rows: list[dict[str, str | int]], fields: tuple[str, ...]) -> None:
    """Flat rows as JSON for a `.json` path, else as CSV."""
    if path.suffix.lower() == ".json":
        path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    else:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    console.print(f"Report written to {escape(str(path))}")


@app.command()
def convert(
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
        typer.Option("--dry-run", help="List what would be converted; download and write nothing."),
    ] = False,
    assume_yes: Annotated[
        bool, typer.Option("--yes", "-y", help="Answer yes to every confirmation.")
    ] = False,
    limit: Annotated[
        int | None,
        typer.Option("--limit", help="Convert at most N documents this run.", min=1),
    ] = None,
    only_item: Annotated[
        list[str] | None,
        typer.Option("--only-item", help="Convert just this item or attachment key; repeatable."),
    ] = None,
    reconvert: Annotated[
        bool,
        typer.Option("--reconvert", help="Regenerate transcripts made by an older converter."),
    ] = False,
    force: Annotated[
        bool,
        typer.Option("--force", help="Also regenerate transcripts edited in Zotero."),
    ] = False,
    ocr: Annotated[
        str,
        typer.Option(
            "--ocr",
            help="off, or hosted: send scanned PDF pages (only those) to Firecrawl Parse.",
        ),
    ] = "off",
    jobs: Annotated[
        int,
        typer.Option("--jobs", help="Documents converted in parallel.", min=1),
    ] = min(4, os.cpu_count() or 1),
    report: Annotated[
        Path | None,
        typer.Option("--report", help="Also write every entry to this .csv or .json file."),
    ] = None,
    verbose: VerboseOption = False,
) -> None:
    """Attach a Markdown transcript to each paper, for LLMs to read.

    Converts the attachment `index` would read (PDF, Word, OpenDocument, EPUB, a
    saved web page…) and uploads the transcript to Zotero beside it. Re-running
    only converts what is new or changed. Must be run from an index folder.
    """
    _enable_verbose(verbose)
    if collection is not None and collection_key is not None:
        err_console.print(
            "[bold red]Error:[/bold red] give either --collection or --collection-key, not both."
        )
        raise typer.Exit(code=1)
    if ocr not in ("off", "hosted"):
        err_console.print("[bold red]Error:[/bold red] --ocr must be off or hosted.")
        raise typer.Exit(code=1)

    run: ConversionRun | None = None
    try:
        config = load_config()
        if config.source_dir is None:
            raise NotAnIndexFolderError(
                f"No {CONFIG_FILENAME} in {Path.cwd()}. Run `zotero-rag init` here first, "
                "or change to an index folder."
            )
        zot = build_client(config)
        roots = build_collection_tree(fetch_collections(zot))
        scope, target, _node = _resolve_target(
            config, roots, collection, collection_key, assume_yes, verb="convert"
        )
        username = "" if dry_run else fetch_username(zot)
        fetched: list[tuple[str, list[dict[str, object]]]] = []
        with console.status("[dim]Fetching items from Zotero...[/dim]") as status:
            for collection_node in scope:
                status.update(f"[dim]Fetching items from {escape(collection_node.name)}...[/dim]")
                fetched.append(
                    (collection_node.name, fetch_collection_items(zot, collection_node.key))
                )

        plan = build_conversion_plan(
            fetched,
            username=username,
            preference=config.index.attachment_preference,
            every_attachment=config.index.index_all_attachments,
            force=force,
            reconvert=reconvert,
        )
        actions = _select_actions(plan.actions, only_item, limit)

        if dry_run:
            _print_conversion_plan(plan, actions, target, ocr)
            if report:
                _write_conversion_report(report, plan, None)
            return
        if not actions:
            console.print("[green]Nothing to convert: every transcript is up to date.[/green]")
            _print_conversion_states(plan)
            if report:
                _write_conversion_report(report, plan, None)
            return

        # A read-only key would otherwise fail after the downloads and conversions.
        require_write_access(zot, config)
        new = sum(1 for e in actions if e.action is ConversionAction.CREATE)
        if not assume_yes and not typer.confirm(
            f"Write {_plural(len(actions), 'transcript')} to Zotero "
            f"({new} new, {len(actions) - new} regenerated)?",
            default=True,
        ):
            raise typer.Abort()

        cache_dir = config.source_dir / CACHE_DIRNAME
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("Converting...", total=len(actions))

            def on_done(result: ConversionResult) -> None:
                progress.update(task, advance=1, description=escape(result.title[:50]))

            def confirm_ocr(entries: list[ConversionEntry]) -> bool:
                if assume_yes:
                    return True
                progress.stop()
                try:
                    return _confirm_ocr(entries)
                finally:
                    progress.start()

            run = run_conversion(
                actions,
                zot=zot,
                files=FileDownloader.from_config(config),
                cache_dir=cache_dir,
                thresholds=config.markdown.quality,
                today=date.today(),
                ocr="hosted" if ocr == "hosted" else "off",
                api_key=config.global_.firecrawl_api_key,
                jobs=jobs,
                confirm_ocr=confirm_ocr,
                on_done=on_done,
            )
    except ZoteroRagError as exc:
        _fail(exc)
        return

    _print_conversion_summary(plan, run, verbose=_is_verbose())
    if report:
        _write_conversion_report(report, plan, run)


# --- shrink-storage -------------------------------------------------------------

#: What the user can do about a kept file, when there is something.
_KEEP_HINTS: dict[KeepReason, str] = {
    KeepReason.NO_TRANSCRIPT: "run `zotero-rag convert` first",
    KeepReason.TRANSCRIPT_NOT_CURRENT: "run `zotero-rag convert` (--reconvert for older ones)",
    KeepReason.NO_LINK: "add a Link to URI attachment in Zotero to make one eligible",
}
#: Kept reasons listed file by file even without -v: each needs a look.
_LISTED_KEEPS = frozenset({KeepReason.UNREADABLE, KeepReason.WRITE_ERROR})


def _print_shrink_table(entries: list[ShrinkEntry], title: str) -> None:
    table = Table(title=title, title_justify="left")
    table.add_column("#", justify="right", style="dim")
    table.add_column("Title", overflow="ellipsis", max_width=44)
    table.add_column("File", overflow="ellipsis", max_width=28)
    table.add_column("Size", justify="right")
    table.add_column("Recoverable from", overflow="fold", max_width=48)
    for number, entry in enumerate(entries, start=1):
        table.add_row(
            str(number),
            escape(entry.title),
            escape(entry.candidate.source.filename or entry.candidate.source.attachment_key),
            format_size(entry.size or 0),
            escape(entry.link),
        )
    console.print(table)


def _print_check_manually(entries: list[ShrinkEntry]) -> None:
    if not entries:
        return
    console.print(
        f"\n[yellow]Check manually ({len(entries)}):[/yellow] the web link serves a file of "
        "the right kind, but not provably this one. Kept; compare them, and delete the file "
        "in Zotero yourself if they are the same."
    )
    for entry in entries:
        source = entry.candidate.source
        remote = format_size(entry.remote_size) if entry.remote_size is not None else "unknown"
        console.print(
            f"  [yellow]•[/yellow] {escape(entry.title)} — {escape(source.filename)} "
            f"[dim]({source.attachment_key})[/dim]\n"
            f"    {escape(entry.link)}\n"
            f"    [dim]here {format_size(entry.size or 0)} ({entry.size} bytes), "
            f"there {remote}"
            f"{f' ({entry.remote_size} bytes)' if entry.remote_size is not None else ''}[/dim]"
        )


def _print_kept(entries: list[ShrinkEntry], verbose: bool) -> None:
    kept = [entry for entry in entries if entry.verdict is Verdict.KEEP]
    if not kept:
        return
    console.print("\nKept:")
    by_reason: dict[KeepReason, list[ShrinkEntry]] = {}
    for entry in kept:
        assert entry.reason is not None  # every kept entry has its reason
        by_reason.setdefault(entry.reason, []).append(entry)
    for reason in KeepReason:  # in the order the checks run
        group = by_reason.get(reason)
        if not group:
            continue
        hint = _KEEP_HINTS.get(reason, "")
        console.print(
            f"  [dim]{len(group):>4}[/dim]  {reason.value}"
            + (f"  [dim]({hint})[/dim]" if hint else "")
        )
        if verbose or reason in _LISTED_KEEPS:
            for entry in group:
                detail = f": {escape(entry.detail)}" if entry.detail else ""
                console.print(
                    f"       [dim]{escape(entry.title)} "
                    f"({entry.candidate.source.attachment_key}){detail}[/dim]"
                )
    if not verbose and any(r not in _LISTED_KEEPS for r in by_reason):
        console.print("  [dim]-v lists every file with the reason in detail.[/dim]")


def _print_shrink_result(entries: list[ShrinkEntry], backup_dir: Path | None) -> None:
    trashed = [entry for entry in entries if entry.trashed]
    if not trashed:
        console.print("\nNothing was moved to the trash.")
        return
    total = sum(entry.size or 0 for entry in trashed)
    console.print(
        f"\n[bold green]Moved {_plural(len(trashed), 'file')} ({format_size(total)}) "
        "to the Zotero trash.[/bold green]"
    )
    if trashed:
        _print_shrink_table(trashed, "Moved to the Zotero trash")
        console.print(
            "[dim]Zotero frees the storage only when the trash is emptied, which is left to "
            "you: in the Zotero application, or automatically after 30 days by default. "
            "Until then each file can be restored from the trash.[/dim]"
        )
        if backup_dir is not None:
            console.print(f"[dim]Copies are kept in {escape(str(backup_dir))}.[/dim]")
    for entry in trashed:
        if entry.error:
            console.print(f"  [yellow]•[/yellow] {escape(entry.title)}: {escape(entry.error)}")


@app.command("shrink-storage")
def shrink_storage(
    max_size: Annotated[
        str,
        typer.Option(
            "--max-size",
            help="Only files larger than this, e.g. 500KB, 20MB, 1.5GB (1 KB = 1024 bytes).",
        ),
    ],
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
        typer.Option(
            "--dry-run",
            help="Check and report, but move nothing. Files are still downloaded to be measured.",
        ),
    ] = False,
    assume_yes: Annotated[
        bool, typer.Option("--yes", "-y", help="Answer yes to every confirmation.")
    ] = False,
    no_backup: Annotated[
        bool,
        typer.Option("--no-backup", help=f"Don't keep a copy of each file in {BACKUP_DIRNAME}/."),
    ] = False,
    jobs: Annotated[
        int,
        typer.Option("--jobs", help="Files checked in parallel.", min=1),
    ] = 4,
    report: Annotated[
        Path | None,
        typer.Option("--report", help="Also write every file checked to this .csv or .json file."),
    ] = None,
    verbose: VerboseOption = False,
) -> None:
    """Move large files to the Zotero trash when a transcript and a web link replace them.

    A file goes only if its Markdown transcript passed the quality check, it has
    no annotations, and a Link to URI attachment of the same item serves the
    very same file (same type and size). Nothing is deleted: files go to the
    Zotero trash, and a copy is kept locally unless --no-backup. Must be run
    from an index folder.
    """
    _enable_verbose(verbose)
    if collection is not None and collection_key is not None:
        err_console.print(
            "[bold red]Error:[/bold red] give either --collection or --collection-key, not both."
        )
        raise typer.Exit(code=1)
    try:
        limit = parse_size(max_size)
    except ValueError as exc:
        err_console.print(f"[bold red]Error:[/bold red] --max-size: {escape(str(exc))}")
        raise typer.Exit(code=1) from exc

    prober = LinkProber()
    try:
        config = load_config()
        if config.source_dir is None:
            raise NotAnIndexFolderError(
                f"No {CONFIG_FILENAME} in {Path.cwd()}. Run `zotero-rag init` here first, "
                "or change to an index folder."
            )
        zot = build_client(config)
        if not dry_run:
            # A read-only key would otherwise fail after every file was downloaded.
            require_write_access(zot, config)
        roots = build_collection_tree(fetch_collections(zot))
        scope, target, _node = _resolve_target(
            config, roots, collection, collection_key, assume_yes, verb="check"
        )
        fetched: list[tuple[str, list[dict[str, object]]]] = []
        with console.status("[dim]Fetching items from Zotero...[/dim]") as status:
            for collection_node in scope:
                status.update(f"[dim]Fetching items from {escape(collection_node.name)}...[/dim]")
                fetched.append(
                    (collection_node.name, fetch_collection_items(zot, collection_node.key))
                )
            status.update("[dim]Listing annotations...[/dim]")
            annotated = fetch_annotated_attachments(zot)

        plan = build_conversion_plan(
            fetched,
            preference=config.index.attachment_preference,
            every_attachment=config.index.index_all_attachments,
        )
        candidates = shrink_candidates(plan, fetched, annotated)
        console.print(
            f"\nTarget: [bold]{escape(target)}[/bold] "
            f"({_plural(len(plan.collections), 'collection')}, "
            f"{_plural(len(candidates), 'file')} with a transcript or eligible for one); "
            f"looking for files larger than {format_size(limit)}.\n"
        )

        cache_dir = config.source_dir / CACHE_DIRNAME
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("Checking...", total=len(candidates))

            def on_checked(entry: ShrinkEntry) -> None:
                progress.update(task, advance=1, description=escape(entry.title[:50]))

            entries = check_candidates(
                candidates,
                files=FileDownloader.from_config(config),
                cache_dir=cache_dir,
                max_size=limit,
                thresholds=config.markdown.quality,
                probe=prober,
                jobs=jobs,
                on_done=on_checked,
            )

        eligible = [entry for entry in entries if entry.verdict is Verdict.TRASH]
        total = sum(entry.size or 0 for entry in eligible)
        if eligible:
            _print_shrink_table(eligible, "Files that can move to the Zotero trash")
            console.print(
                f"\n[bold]{_plural(len(eligible), 'file')}[/bold], {format_size(total)} in all."
            )
        else:
            console.print("[green]No file can be moved to the trash.[/green]")
        _print_check_manually([entry for entry in entries if entry.verdict is Verdict.CHECK])

        backup_dir = None if no_backup else config.source_dir / BACKUP_DIRNAME
        if dry_run or not eligible:
            _print_kept(entries, verbose=_is_verbose())
            if dry_run:
                console.print("\n[dim]Dry run: nothing was moved or written to Zotero.[/dim]")
            if report:
                _write_report(report, shrink_report_rows(entries), SHRINK_REPORT_FIELDS)
            return

        if not assume_yes and not typer.confirm(
            f"\nMove {_plural(len(eligible), 'file')} ({format_size(total)}) to the Zotero trash?",
            default=False,
        ):
            for entry in eligible:
                entry.keep(KeepReason.DECLINED)
        else:
            with console.status("[dim]Moving files to the trash...[/dim]") as status:

                def on_trashed(entry: ShrinkEntry) -> None:
                    status.update(f"[dim]{escape(entry.title[:60])}[/dim]")

                trash_files(
                    eligible,
                    zot=zot,
                    backup_dir=backup_dir,
                    today=date.today(),
                    on_done=on_trashed,
                )
    except ZoteroRagError as exc:
        _fail(exc)
        return
    finally:
        prober.close()

    _print_shrink_result(entries, backup_dir)
    _print_kept(entries, verbose=_is_verbose())
    if report:
        _write_report(report, shrink_report_rows(entries), SHRINK_REPORT_FIELDS)


@app.command()
def query(
    question: Annotated[str, typer.Argument(help="The question to answer, in any language.")],
    top_k: Annotated[
        int | None,
        typer.Option("--top-k", help="Chunks to retrieve; overrides the configured top_k.", min=1),
    ] = None,
    window: Annotated[
        int | None,
        typer.Option(
            "--window",
            help="Also include N neighbouring chunks either side of each hit.",
            min=0,
        ),
    ] = None,
    retrieval: RetrievalOption = None,
    reranker: RerankerOption = None,
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
        config, store = open_index(_print_notice)
        mode = retrieval if retrieval is not None else config.index.retrieval
        rerank, _ = resolve_reranker(config, reranker)
        keyword_index = _build_keyword_index(store, mode)
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
                window=window if window is not None else config.index.retrieval_window,
                mode=mode,
                keyword_index=keyword_index,
                rerank=rerank,
                rerank_fanout=config.index.reranker_fanout,
            )
    except ZoteroRagError as exc:
        _fail(exc)
        return

    if as_json:
        # Plain print: rich would wrap and highlight the JSON.
        print(_answer_as_json(result, show_sources))  # noqa: T201
        return
    _print_answer(result, show_sources, verbose=_is_verbose())


_GRADE_STYLES: dict[Grade, str] = {
    Grade.PERFECT: "bold green",
    Grade.GOOD: "green",
    Grade.PARTIAL: "yellow",
    Grade.POOR: "red",
    Grade.WRONG: "bold red",
    Grade.NO_ANSWER: "dim",
    Grade.ERROR: "bold magenta",
}


def _flag(value: bool | None) -> str:
    """A tri-state cell: yes, no, or "the question did not say"."""
    if value is None:
        return "[dim]—[/dim]"
    return "[green]yes[/green]" if value else "[red]no[/red]"


def _time_spread(spread: TimeSpread | None) -> str:
    """`2.9s (2.4–3.4)`: the median question, and the middle half of them."""
    if spread is None:
        return "—"
    return f"{spread.median_s:.1f}s ({spread.p25_s:.1f}–{spread.p75_s:.1f})"


def _percent(value: float | None) -> str:
    return "[dim]n/a[/dim]" if value is None else f"{value * 100:.0f}%"


def _select_questions(questions: list[EvalQuestion], only: list[str]) -> list[EvalQuestion]:
    """Narrow the set to the requested ids, complaining about ids that do not exist."""
    if not only:
        return questions
    wanted = list(dict.fromkeys(only))
    by_id = {question.id: question for question in questions}
    if missing := [name for name in wanted if name not in by_id]:
        raise QuestionSetError(
            f"No question with id {', '.join(repr(name) for name in missing)}. "
            f"The set holds: {', '.join(by_id)}."
        )
    return [by_id[name] for name in wanted]


def _eval_table(outcomes: list[QuestionOutcome]) -> Table:
    """Per-question results, one row each."""
    table = Table(title="Evaluation", title_justify="left")
    table.add_column("Id", style="dim")
    table.add_column("Question", overflow="ellipsis", max_width=44)
    table.add_column("Grade")
    table.add_column("Points", justify="right")
    table.add_column("Hit", justify="center")
    # Only when the set has evidence quotes: an all-dash column says nothing.
    passages = any(outcome.passage_hit is not None for outcome in outcomes)
    if passages:
        table.add_column("Passage", justify="center")
    table.add_column("Cited", justify="center")
    table.add_column("Tokens", justify="right", style="dim")
    table.add_column("Time", justify="right", style="dim")

    for outcome in outcomes:
        coverage = outcome.coverage
        points = (
            f"{coverage.covered}+{coverage.partial}/{coverage.total}"
            if coverage.total
            else "[dim]—[/dim]"
        )
        style = _GRADE_STYLES[outcome.grade]
        tokens = (
            outcome.prompt_tokens
            + outcome.completion_tokens
            + outcome.embedding_tokens
            + outcome.judge_tokens
        )
        table.add_row(
            escape(outcome.id),
            escape(outcome.question),
            f"[{style}]{outcome.grade}[/]",
            points,
            _flag(outcome.hit),
            *([_flag(outcome.passage_hit)] if passages else []),
            _flag(outcome.cited_expected),
            str(tokens),
            f"{outcome.elapsed_s:.1f}s",
        )
    return table


def _print_eval_report(report: EvalReport, verbose: bool) -> None:
    """The table, the one-line summary, and the settings that produced them."""
    summary = report.aggregate
    console.print(_eval_table(report.results))

    # PRD §11.3 asks for "a table plus one summary line". This is that line.
    passage = (
        f"passage hit {_percent(summary.passage_hit_rate)}  ·  "
        if summary.passage_hit_rate is not None
        else ""
    )
    console.print(
        f"\n[bold]{summary.mean_score:.2f}/{MAX_SCORE}[/bold] mean score over "
        f"{_plural(summary.questions, 'question')}  ·  "
        f"retrieval hit {_percent(summary.hit_rate)}  ·  {passage}"
        f"expected item cited {_percent(summary.citation_rate)}  ·  "
        f"point coverage {_percent(summary.mean_coverage)}  ·  "
        f"{summary.total_tokens} tokens  ·  {summary.elapsed_s:.1f}s total, "
        f"{_time_spread(summary.question_time)} per question"
    )

    grades = "  ".join(
        f"[{_GRADE_STYLES[Grade(label)]}]{count} {label}[/]"
        for label, count in summary.grade_counts.items()
    )
    console.print(f"  {grades}")
    if summary.context_counts:
        listed = ", ".join(f"{count} {label}" for label, count in summary.context_counts.items())
        console.print(f"  [dim]retrieved context:[/dim] {listed}")
    if summary.errors:
        console.print(
            f"  [magenta]{_plural(summary.errors, 'question')} could not be graded[/magenta] "
            f"[dim](counted apart from a genuine 'wrong'; rerun to retry)[/dim]"
        )

    settings = report.settings
    windowing = f" ±{settings.retrieval_window}" if settings.retrieval_window else ""
    reranking = (
        f" · reranked by {settings.reranker_model} (×{settings.reranker_fanout})"
        if settings.reranker
        else ""
    )
    console.print(
        f"\n[dim]{settings.llm_model} answering, {settings.judge_model} grading, "
        f"{settings.embedding_model} embedding · top_k {settings.top_k} · "
        f"chunks {settings.chunk_size}/{settings.chunk_overlap} · "
        f"temperature {settings.temperature} · {settings.retrieval}{windowing} retrieval"
        f"{reranking} · {_plural(settings.store_chunks, 'chunk')} in the store[/dim]"
    )

    # A configured reranker that quietly did not run on every question would make two
    # reports incomparable while looking identical, so say so plainly.
    if settings.reranker and summary.reranked_questions < summary.questions:
        console.print(
            f"  [yellow]reranking applied to only {summary.reranked_questions} of "
            f"{summary.questions} questions[/yellow] [dim](the rest kept the retrieved "
            f"order after a provider failure; rerun to retry)[/dim]"
        )

    if verbose:
        for outcome in report.results:
            console.print(f"\n[bold]{escape(outcome.id)}[/bold] {escape(outcome.question)}")
            console.print(f"  [dim]{escape(outcome.rationale)}[/dim]")
            if outcome.error:
                console.print(f"  [red]{escape(outcome.error)}[/red]")


@app.command("eval")
def eval_command(
    questions_file: Annotated[
        Path | None,
        typer.Option("--questions", help="Question set to run; defaults to the configured one."),
    ] = None,
    top_k: Annotated[
        int | None,
        typer.Option("--top-k", help="Chunks to retrieve; overrides the configured top_k.", min=1),
    ] = None,
    window: Annotated[
        int | None,
        typer.Option(
            "--window",
            help="Also include N neighbouring chunks either side of each hit.",
            min=0,
        ),
    ] = None,
    retrieval: RetrievalOption = None,
    reranker: RerankerOption = None,
    only: Annotated[
        list[str] | None,
        typer.Option("--only", help="Run just this question id; repeatable."),
    ] = None,
    assess_context: Annotated[
        bool,
        typer.Option(
            "--assess-context",
            help="Also ask the judge whether the retrieved passages could support "
            "the expected answer. One extra call per question.",
        ),
    ] = False,
    jobs: Annotated[
        int,
        typer.Option(
            "--jobs",
            help="Questions answered and graded in parallel. More means more concurrent "
            "requests to the answering and grading models; 1 runs them in turn.",
            min=1,
        ),
    ] = EVAL_JOBS,
    as_json: Annotated[
        bool, typer.Option("--json", help="Emit the full report as JSON instead of a table.")
    ] = False,
    verbose: VerboseOption = False,
) -> None:
    """Score the index against a golden set of questions.

    Reports retrieval hit rate, point coverage judged by an LLM, tokens and
    latency, alongside the settings that produced them so two runs can be
    compared. Must be run from an index folder that has been indexed.
    """
    _enable_verbose(verbose)

    try:
        config, store = open_index(_print_notice)
        questions_path = resolve_questions_path(config, questions_file)
        questions = _select_questions(load_questions(questions_path), only or [])

        chosen_top_k = top_k if top_k is not None else config.index.top_k
        chosen_window = window if window is not None else config.index.retrieval_window
        mode = retrieval if retrieval is not None else config.index.retrieval
        rerank, rerank_name = resolve_reranker(config, reranker)
        keyword_index = _build_keyword_index(store, mode)
        embeddings = build_embeddings(config)
        chat = build_chat(config)
        verdict_of = build_verdict_fn(config)
        assessor = build_context_assessor(config) if assess_context else None

        def ask(messages: list[BaseMessage]) -> AIMessage:
            return cast(AIMessage, chat.invoke(messages))

        def answer_one(question: str) -> QueryResult:
            return answer_question(
                question,
                store=store,
                embed_query=embeddings.embed_query,
                answer=ask,
                top_k=chosen_top_k,
                window=chosen_window,
                mode=mode,
                keyword_index=keyword_index,
                rerank=rerank,
                rerank_fanout=config.index.reranker_fanout,
            )

        # Progress on stderr, so --json on stdout is never contaminated.
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=err_console,
            transient=True,
        ) as progress:
            task = progress.add_task("Evaluating...", total=len(questions))

            def on_start(question: EvalQuestion) -> None:
                progress.update(task, advance=1, description=escape(question.question[:48]))

            started = time.perf_counter()
            outcomes = evaluate(questions, answer_one, verdict_of, assessor, on_start, workers=jobs)
            elapsed = time.perf_counter() - started

        report = EvalReport(
            settings=run_settings(
                config,
                store,
                chosen_top_k,
                questions_path,
                window=chosen_window,
                mode=mode,
                reranker=rerank_name,
            ),
            aggregate=aggregate(outcomes, elapsed),
            results=outcomes,
        )
    except ZoteroRagError as exc:
        _fail(exc)
        return

    if as_json:
        # Plain print: rich would wrap and highlight the JSON.
        print(report.model_dump_json(indent=2))  # noqa: T201
        return
    _print_eval_report(report, verbose=_is_verbose())


def _money(value: float | None) -> str:
    return "[dim]n/a[/dim]" if value is None else f"${value:.4f}"


def _stored_chunks(directory: Path) -> int:
    """Chunks in the store at `directory`, without creating one where there is none."""
    return VectorStore(directory).count() if VectorStore.exists(directory) else 0


def _variant_remedy(variant: Variant) -> str:
    return (
        f"This comparison store does not match {variant.label}. Delete {variant.store_dir} "
        "and run `compare-embeddings` again to rebuild it."
    )


def _print_comparison_plan(
    out: Console,
    config: Config,
    variants: list[Variant],
    chunks: dict[Path, int],
    to_build: list[Variant],
    corpus_tokens: int,
    questions: int,
    retrieval: str,
) -> None:
    """What would be built and evaluated, and what building would cost."""
    index_dir = config.source_dir or Path.cwd()
    building = {variant.store_dir for variant in to_build}
    table = Table(title="Embedding comparison plan", title_justify="left")
    table.add_column("Embedding model")
    table.add_column("Chunk", justify="right")
    table.add_column("Overlap", justify="right", style="dim")
    table.add_column("Store", style="dim")
    table.add_column("Status")
    table.add_column("$/1M tok", justify="right")
    table.add_column("Est. index $", justify="right")

    estimated_total = 0.0
    for variant in variants:
        held = chunks[variant.store_dir]
        spec = config.embedding_model_spec(variant.embedding_model)
        estimate: float | None = None
        if variant.baseline:
            status = f"baseline · {_plural(held, 'chunk')}"
        elif held:
            status = f"built · {_plural(held, 'chunk')}"
        elif variant.store_dir in building:
            status = "[yellow]to build[/yellow]"
            # Every chunking of the same text at the same overlap ratio comes to about
            # the same token total, so the baseline's is a fair estimate.
            estimate = spec.cost(corpus_tokens) if spec else None
            estimated_total += estimate or 0.0
        else:
            status = "[red]not built[/red]"
        table.add_row(
            escape(variant.embedding_model),
            str(variant.chunk_size),
            str(variant.chunk_overlap),
            escape(str(variant.store_dir.relative_to(index_dir))),
            status,
            _money(spec.pricing.input if spec and spec.pricing else None),
            _money(estimate) if estimate is not None else "[dim]—[/dim]",
        )
    out.print(table)

    unbuilt = sum(1 for variant in to_build if not chunks[variant.store_dir])
    out.print(
        f"\n{_plural(unbuilt, 'store')} to build (~{corpus_tokens} embedding tokens each, "
        f"est. {_money(estimated_total)} in total); "
        f"{_plural(len(variants), 'variant')} × {_plural(questions, 'question')} = "
        f"{len(variants) * questions} answers by {escape(config.llm_model)}, each graded "
        f"by {escape(config.judge_model)}; {escape(retrieval)}."
    )
    for variant in variants:
        if warning := context_warning(config, variant):
            out.print(f"[yellow]Warning:[/yellow] {escape(warning)}")


#: Where `compare-embeddings` keeps a copy of every table it prints, in the index folder.
REPORTS_DIRNAME = "reports"

#: Variants scored at once by `compare-embeddings`. They all call the same answering
#: and grading models, so this bounds the requests in flight, not the variant count.
DEFAULT_EVAL_WORKERS = 4


class _Interrupted(Exception):
    """Raised inside a build or scoring worker once the user has pressed Ctrl-C."""


def _build_variant(
    plan: IndexPlan,
    config: Config,
    variant: Variant,
    store: VectorStore,
    download: Callable[[str], Path],
    on_start: Callable[[PlannedAttachment], None],
) -> IndexStats:
    """Index the planned attachments into a variant's own store.

    Incremental like `index`: a store built by an earlier run only embeds what
    changed in Zotero since. Downloads come from the folder's shared cache.
    """
    settings = variant_config(config, variant)
    embeddings = build_embeddings(settings)
    # One short call, so a model the provider does not serve fails here, once,
    # rather than once per attachment.
    embeddings.embed_documents([DIMENSION_PROBE])

    stats = index_plan(
        plan,
        store=store,
        download=download,
        embed=embeddings.embed_documents,
        chunk_size=settings.index.chunk_size,
        chunk_overlap=settings.index.chunk_overlap,
        embedding_model=settings.embedding_model,
        on_start=on_start,
    )
    store.write_fingerprint(
        StoreFingerprint.intended(settings, embedding_dim=store.derived_dimension() or 0)
    )
    # A comparison store mirrors the collection as it is now.
    drop_replaced_copies(store, plan)
    prune_orphans(store, plan)
    return stats


def _run_in_pool(
    jobs: list[Callable[[], None]], workers: int, prefix: str, stop: threading.Event
) -> None:
    """Run `jobs` on at most `workers` threads, re-raising the first failure.

    Ctrl-C reaches only the main thread, so it sets `stop` for the jobs to notice
    at their next check, waits for them, and is raised again.
    """
    pool = ThreadPoolExecutor(max_workers=min(workers, len(jobs)), thread_name_prefix=prefix)
    try:
        for future in [pool.submit(job) for job in jobs]:
            future.result()
    except KeyboardInterrupt:
        stop.set()
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


def _build_variants(
    plan: IndexPlan, config: Config, to_build: list[Variant], workers: int | None
) -> tuple[dict[Path, IndexStats], dict[Path, str]]:
    """Build the comparison stores, one worker thread per embedding model.

    The time goes on embedding calls, so threads overlap the waiting. A model's
    variants share one worker and run in turn, so no provider gets more than one
    request at a time from this run. Returns the stats of each store built and the
    error of each that failed.

    Two things stay off the workers: stores are opened here, because chromadb fills
    its client cache without a lock, and downloads take one lock, because two
    workers asking for the same uncached file would both write it.
    """
    index_dir = config.source_dir or Path.cwd()
    cache_dir = index_dir / CACHE_DIRNAME
    formats = {
        attachment.attachment_key: attachment.document_format for attachment in plan.attachments
    }
    files = FileDownloader.from_config(config)
    download_lock = threading.Lock()

    def download(key: str) -> Path:
        with download_lock:
            return ensure_cached(files, key, cache_dir, formats[key])

    built: dict[Path, IndexStats] = {}
    errors: dict[Path, str] = {}
    stores: dict[Path, VectorStore] = {}
    by_model: dict[str, list[Variant]] = {}
    for variant in to_build:
        try:
            store = VectorStore(variant.store_dir)
            check_fingerprint(
                store, variant_config(config, variant), _variant_remedy(variant), _print_notice
            )
        except ZoteroRagError as exc:
            errors[variant.store_dir] = str(exc)
            continue
        stores[variant.store_dir] = store
        by_model.setdefault(variant.embedding_model, []).append(variant)
    if not by_model:
        return built, errors

    stop = threading.Event()
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=err_console,
        transient=True,
    ) as progress:
        tasks = {
            variant.store_dir: progress.add_task(
                escape(variant.label), total=len(plan.attachments), start=False
            )
            for variants in by_model.values()
            for variant in variants
        }

        def run_model(variants: list[Variant]) -> None:
            for variant in variants:
                if stop.is_set():
                    return
                task = tasks[variant.store_dir]
                progress.start_task(task)

                def on_start(attachment: PlannedAttachment, task: TaskID = task) -> None:
                    if stop.is_set():
                        raise _Interrupted
                    progress.advance(task)

                try:
                    built[variant.store_dir] = _build_variant(
                        plan, config, variant, stores[variant.store_dir], download, on_start
                    )
                except _Interrupted:
                    return
                except Exception as exc:  # noqa: BLE001 - one model's failure must not stop the rest
                    errors[variant.store_dir] = str(exc) or type(exc).__name__
                    progress.update(task, description=f"[red]{escape(variant.label)}: failed[/red]")

        # On Ctrl-C workers finish the attachment in hand, then stop; the stores
        # keep what was written, and the next run picks up from there.
        _run_in_pool(
            [partial(run_model, variants) for variants in by_model.values()],
            workers or len(by_model),
            "embed",
            stop,
        )

    for variant in to_build:
        if variant.store_dir in errors:
            err_console.print(
                f"[yellow]Warning:[/yellow] could not build {escape(variant.label)}: "
                f"{escape(errors[variant.store_dir])}"
            )
        elif (stats := built.get(variant.store_dir)) is not None and stats.failed:
            err_console.print(
                f"[yellow]Built {escape(variant.label)} with gaps:[/yellow] "
                f"{_plural(stats.chunks, 'new chunk')}, but "
                f"{_plural(stats.failed, 'attachment')} failed "
                f"({escape(stats.failures[0].error[:80])}). Run again to retry them."
            )
        elif stats is not None and stats.chunks:
            err_console.print(
                f"Built [bold]{escape(variant.label)}[/bold]: "
                f"{_plural(stats.chunks, 'new chunk')} in {stats.elapsed_s:.0f}s."
            )
    return built, errors


def _evaluate_variant(
    variant: Variant,
    config: Config,
    store: VectorStore | None,
    stats: IndexStats | None,
    build_error: str,
    evaluate_with: Callable[[VectorStore, Config], list[QuestionOutcome]],
) -> VariantResult:
    """Score one variant. Any failure is recorded on its row, never raised.

    `store` is opened by the caller, on the main thread, and is `None` when the
    variant has nothing stored to score.
    """
    spec = config.embedding_model_spec(variant.embedding_model)
    result = VariantResult(
        embedding_model=variant.embedding_model,
        chunk_size=variant.chunk_size,
        chunk_overlap=variant.chunk_overlap,
        store=str(variant.store_dir),
        baseline=variant.baseline,
        provider=spec.provider if spec else "",
        price_per_million=spec.pricing.input if spec and spec.pricing else None,
        built_chunks=stats.chunks if stats else 0,
        build_failures=stats.failed if stats else 0,
    )
    if store is None:
        result.error = (
            f"could not build: {build_error}"
            if build_error
            else "no chunks in its store (not built, or every embedding call failed)"
        )
        return result

    result.store_chunks = store.count()
    result.embedding_dim = store.derived_dimension()
    result.index_tokens = stored_tokens(store)
    settings = variant_config(config, variant)
    try:
        if not variant.baseline:
            check_fingerprint(store, settings, _variant_remedy(variant), _print_notice)
        outcomes = evaluate_with(store, settings)
    except Exception as exc:  # noqa: BLE001 - one model's outage must not lose the others
        result.error = str(exc) or type(exc).__name__
        return result
    result.aggregate = aggregate(outcomes)
    result.results = outcomes
    return result


def _retrieval_summary(top_k: int, window: int, retrieval: str, reranker: str | None) -> str:
    """`top_k 10 · window 0 · hybrid retrieval · no reranker`: what every row shares."""
    reranking = f"reranked by {reranker}" if reranker else "no reranker"
    return f"top_k {top_k} · window {window} · {retrieval} retrieval · {reranking}"


def _settings_summary(settings: ComparisonSettings) -> str:
    return _retrieval_summary(
        settings.top_k, settings.retrieval_window, settings.retrieval, settings.reranker
    )


def _comparison_table(report: ComparisonReport) -> Table:
    """One row per variant: what it scored and what it cost."""
    table = Table(
        title=f"Embedding comparison · {escape(_settings_summary(report.settings))}",
        title_justify="left",
    )
    table.add_column("Embedding model", overflow="fold")
    table.add_column("Chunk", justify="right")
    table.add_column("Overlap", justify="right")
    table.add_column("Dim", justify="right", style="dim")
    table.add_column("Index $", justify="right")
    table.add_column("Score", justify="right")
    table.add_column("Hit", justify="right")
    table.add_column("Passage", justify="right")
    table.add_column("Cited", justify="right")
    table.add_column("Coverage", justify="right")
    table.add_column("Ans tok/q", justify="right", style="dim")
    table.add_column("s/q (IQR)", justify="right", style="dim")

    scores = [v.aggregate.mean_score for v in report.variants if v.aggregate is not None]
    best = max(scores, default=None)
    for variant in report.variants:
        name = escape(variant.embedding_model)
        if variant.baseline:
            name += " [dim](baseline)[/dim]"
        summary = variant.aggregate
        if summary is None:
            table.add_row(
                name,
                str(variant.chunk_size),
                str(variant.chunk_overlap),
                *["[dim]—[/dim]"] * 2,
                "[bold red]error[/bold red]",
                *["[dim]—[/dim]"] * 6,
            )
            continue
        score = f"{summary.mean_score:.2f}"
        if summary.mean_score == best and len(scores) > 1:
            score = f"[bold green]{score}[/bold green]"
        per_question = variant.answer_tokens_per_question
        table.add_row(
            name,
            str(variant.chunk_size),
            str(variant.chunk_overlap),
            str(variant.embedding_dim or "—"),
            _money(variant.index_cost),
            score,
            _percent(summary.hit_rate),
            _percent(summary.passage_hit_rate),
            _percent(summary.citation_rate),
            _percent(summary.mean_coverage),
            f"{per_question:.0f}" if per_question is not None else "—",
            _time_spread(summary.question_time),
        )
    return table


def _store_contents(report: ComparisonReport) -> list[str]:
    """What each chunking's stores hold, or which ones disagree.

    Stores of one chunking built from the same documents hold the same chunks, so
    the counts are one line rather than two columns. When they differ, a store is
    stale or missing documents, and its row is not comparable with the others.
    """
    by_chunking: dict[tuple[int, int], list[VariantResult]] = {}
    for variant in report.variants:
        if variant.store_chunks:
            by_chunking.setdefault((variant.chunk_size, variant.chunk_overlap), []).append(variant)
    lines: list[str] = []
    for (size, overlap), variants in by_chunking.items():
        held = Counter((v.store_chunks, v.index_tokens) for v in variants)
        # Measured against the baseline when it is in this group, which is the row
        # every other is read against; otherwise against the most common contents.
        reference = next((v for v in variants if v.baseline), None)
        if reference is not None:
            chunks, tokens = reference.store_chunks, reference.index_tokens
            holder = "the baseline holds"
        else:
            (chunks, tokens), _ = held.most_common(1)[0]
            holder = "most hold"
        if len(held) == 1:
            lines.append(
                f"[dim]Every {size}/{overlap} store holds {_plural(chunks, 'chunk')} "
                f"({tokens:,} tokens).[/dim]"
            )
            continue
        lines.append(
            f"[yellow]The {size}/{overlap} stores hold different documents[/yellow]: "
            f"{holder} {_plural(chunks, 'chunk')} ({tokens:,} tokens), but"
        )
        for v in variants:
            if (v.store_chunks, v.index_tokens) != (chunks, tokens):
                lines.append(
                    f"  [yellow]{escape(v.label)}[/yellow] holds "
                    f"{_plural(v.store_chunks, 'chunk')} ({v.index_tokens:,} tokens)"
                )
        lines.append(
            "  Those scores are not comparable. Re-run `index` if the baseline is stale, or "
            "delete a stale variant's store and compare again."
        )
    return lines


def _print_comparison(report: ComparisonReport, out: Console) -> None:
    out.print(_comparison_table(report))
    for line in _store_contents(report):
        out.print(line)
    for variant in report.variants:
        if variant.error:
            out.print(f"  [red]{escape(variant.label)}:[/red] {escape(variant.error)}")
        elif variant.build_failures:
            out.print(
                f"  [yellow]{escape(variant.label)}:[/yellow] "
                f"{_plural(variant.build_failures, 'attachment')} failed to embed, so this "
                "store is missing documents the others hold."
            )
        if variant.aggregate is not None and variant.aggregate.errors:
            out.print(
                f"  [magenta]{escape(variant.label)}: "
                f"{_plural(variant.aggregate.errors, 'question')} could not be graded[/magenta]"
            )

    settings = report.settings
    out.print(
        f"\n[dim]{_plural(settings.questions, 'question')} from {escape(settings.questions_file)}"
        f" · {escape(settings.llm_model)} answering, {escape(settings.judge_model)} grading · "
        f"{escape(_settings_summary(settings))}[/dim]"
    )
    out.print(
        "[dim]Score is out of 5. Tokens are counted with OpenAI's cl100k tokenizer, so costs "
        "for other providers are estimates. Point coverage moves a few points between "
        "identical runs; treat a small difference as noise.[/dim]"
    )


def _save_comparison(report: ComparisonReport, index_dir: Path) -> tuple[Path, Path]:
    """Write the table and the full JSON to `reports/`, for comparing runs later."""
    reports = index_dir / REPORTS_DIRNAME
    reports.mkdir(exist_ok=True)
    stem = f"compare-embeddings-{datetime.now():%Y%m%d-%H%M%S}"
    text_path, json_path = reports / f"{stem}.txt", reports / f"{stem}.json"
    with text_path.open("w", encoding="utf-8") as handle:
        # Wide enough that no model id wraps, and plain: a report is read in an
        # editor or a diff, where colour codes are noise.
        _print_comparison(report, Console(file=handle, width=200, color_system=None))
    json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return text_path, json_path


@app.command("compare-embeddings")
def compare_embeddings_command(
    models: Annotated[
        list[str] | None,
        typer.Option(
            "--model",
            help="Embedding model to compare; repeatable. Defaults to every model in "
            "the config's embedding_models list.",
        ),
    ] = None,
    chunk_sizes: Annotated[
        list[int] | None,
        typer.Option(
            "--chunk-size",
            help="Chunk size in characters; repeatable. Defaults to the configured one.",
            min=1,
        ),
    ] = None,
    chunk_overlaps: Annotated[
        list[int] | None,
        typer.Option(
            "--chunk-overlap",
            help="Chunk overlap in characters; repeatable, paired with every chunk size. "
            "Defaults to the configured overlap ratio applied to each size.",
            min=0,
        ),
    ] = None,
    questions_file: Annotated[
        Path | None,
        typer.Option("--questions", help="Question set to run; defaults to the configured one."),
    ] = None,
    top_k: Annotated[
        int | None,
        typer.Option("--top-k", help="Chunks to retrieve; overrides the configured top_k.", min=1),
    ] = None,
    retrieval: RetrievalOption = None,
    reranker: RerankerOption = None,
    only: Annotated[
        list[str] | None,
        typer.Option("--only", help="Run just this question id; repeatable."),
    ] = None,
    no_build: Annotated[
        bool,
        typer.Option("--no-build", help="Evaluate only the stores that already exist."),
    ] = False,
    workers: Annotated[
        int | None,
        typer.Option(
            "--workers",
            help="Embedding models built at once. Defaults to one worker per model; "
            "1 builds them one after another.",
            min=1,
        ),
    ] = None,
    eval_workers: Annotated[
        int,
        typer.Option(
            "--eval-workers",
            help="Variants scored at once. Every one calls the same answering and grading "
            "models, so this caps the requests they get at a time; 1 scores in turn.",
            min=1,
        ),
    ] = DEFAULT_EVAL_WORKERS,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Show the variants and the cost; build and run nothing."),
    ] = False,
    assume_yes: Annotated[
        bool, typer.Option("--yes", "-y", help="Skip the confirmation prompt.")
    ] = False,
    as_json: Annotated[
        bool, typer.Option("--json", help="Emit the full comparison as JSON instead of a table.")
    ] = False,
    verbose: VerboseOption = False,
) -> None:
    """Compare embedding models, chunk sizes and overlaps on the eval question set.

    The configured model, chunk size and overlap are the baseline, read from chroma/.
    Every other combination is indexed into
    embeddings/<model>/<chunk_size>-<chunk_overlap>/ from the download cache, then
    all are scored with the same questions, answering model and judge. Stores are
    kept, so a later comparison only builds what is new.
    """
    _enable_verbose(verbose)
    # The plan is a preamble to the JSON, not part of it.
    out = err_console if as_json else console

    try:
        config, baseline_store = open_index(_print_notice)
        questions_path = resolve_questions_path(config, questions_file)
        questions = _select_questions(load_questions(questions_path), only or [])

        chosen_models = models or [spec.id for spec in config.embedding_models]
        chosen_sizes = chunk_sizes or [config.index.chunk_size]
        pairs, impossible = chunkings(config, chosen_sizes, chunk_overlaps or None)
        for size, overlap in impossible:
            out.print(
                f"[yellow]Warning:[/yellow] skipping chunk size {size} with overlap {overlap}: "
                "the overlap must be smaller than the chunk."
            )
        variants = plan_variants(config, chosen_models, pairs)
        to_build = [] if no_build else [variant for variant in variants if not variant.baseline]
        chunks = {variant.store_dir: _stored_chunks(variant.store_dir) for variant in variants}
        corpus_tokens = stored_tokens(baseline_store)
        chosen_top_k = top_k if top_k is not None else config.index.top_k
        mode = retrieval if retrieval is not None else config.index.retrieval
        window = config.index.retrieval_window
        # Resolved before the prompt, so a bad reranker name fails before any work.
        rerank, rerank_name = resolve_reranker(config, reranker)

        _print_comparison_plan(
            out,
            config,
            variants,
            chunks,
            to_build,
            corpus_tokens,
            len(questions),
            _retrieval_summary(chosen_top_k, window, mode, rerank_name),
        )
        if dry_run:
            out.print("\n[dim]Dry run: nothing was embedded, answered, or graded.[/dim]")
            return
        if not assume_yes and not typer.confirm("Proceed?", default=False, err=True):
            raise typer.Abort()

        config.global_.require_openrouter()
        built: dict[Path, IndexStats] = {}
        build_errors: dict[Path, str] = {}
        if to_build:
            # One fetch serves every variant: they all index the same collection.
            plan, _, _, _ = _fetch_plan(config, None, None, assume_yes, with_username=True)
            built, build_errors = _build_variants(plan, config, to_build, workers)

        chat = build_chat(config)
        verdict_of = build_verdict_fn(config)

        def ask(messages: list[BaseMessage]) -> AIMessage:
            return cast(AIMessage, chat.invoke(messages))

        # Opened here rather than on the workers: chromadb fills its client cache
        # without a lock.
        stores = {
            variant.store_dir: baseline_store
            if variant.baseline
            else VectorStore(variant.store_dir)
            for variant in variants
            if variant.baseline or _stored_chunks(variant.store_dir)
        }
        results: list[VariantResult | None] = [None] * len(variants)
        stop = threading.Event()
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=err_console,
            transient=True,
        ) as progress:
            tasks = [
                progress.add_task(escape(variant.label), total=len(questions), start=False)
                for variant in variants
            ]

            def score(position: int) -> None:
                if stop.is_set():
                    return
                variant, task = variants[position], tasks[position]
                progress.start_task(task)

                def evaluate_with(store: VectorStore, settings: Config) -> list[QuestionOutcome]:
                    embeddings = build_embeddings(settings)
                    keyword_index = build_keyword_index(store, mode)

                    def answer_one(question: str) -> QueryResult:
                        return answer_question(
                            question,
                            store=store,
                            embed_query=embeddings.embed_query,
                            answer=ask,
                            top_k=chosen_top_k,
                            window=window,
                            mode=mode,
                            keyword_index=keyword_index,
                            rerank=rerank,
                            rerank_fanout=config.index.reranker_fanout,
                        )

                    def on_start(question: EvalQuestion) -> None:
                        if stop.is_set():
                            raise _Interrupted
                        progress.advance(task)

                    return evaluate(questions, answer_one, verdict_of, None, on_start)

                result = _evaluate_variant(
                    variant,
                    config,
                    stores.get(variant.store_dir),
                    built.get(variant.store_dir),
                    build_errors.get(variant.store_dir, ""),
                    evaluate_with,
                )
                if result.aggregate is None:
                    progress.update(
                        task,
                        completed=len(questions),
                        description=f"[red]{escape(variant.label)}: failed[/red]",
                    )
                results[position] = result

            # Each worker answers and grades one question at a time, so the LLM
            # sees at most `eval_workers` requests at once, whatever the variants.
            _run_in_pool(
                [partial(score, position) for position in range(len(variants))],
                eval_workers,
                "score",
                stop,
            )

        report = ComparisonReport(
            settings=ComparisonSettings(
                collection_name=config.index.collection_name,
                collection_key=config.index.collection_key,
                llm_model=config.llm_model,
                judge_model=config.judge_model,
                top_k=chosen_top_k,
                retrieval=mode,
                retrieval_window=window,
                reranker=rerank_name,
                questions_file=str(questions_path),
                questions=len(questions),
            ),
            variants=[result for result in results if result is not None],
        )
    except ZoteroRagError as exc:
        _fail(exc)
        return

    index_dir = config.source_dir or Path.cwd()
    text_path, json_path = _save_comparison(report, index_dir)
    if as_json:
        print(report.model_dump_json(indent=2))  # noqa: T201
    else:
        _print_comparison(report, console)
    err_console.print(
        f"[dim]Saved to {escape(str(text_path.relative_to(index_dir)))} "
        f"and {escape(json_path.name)}.[/dim]"
    )


#: Above this many documents, generation is worth confirming: it is one LLM call
#: per document over most of that document's text, which on a large collection is
#: a real bill rather than a rounding error.
_CONFIRM_DOCUMENTS = 30


def _select_documents(
    documents: list[DocumentText], only_item: list[str], limit: int | None
) -> list[DocumentText]:
    """Narrow to the requested items, complaining about keys that are not indexed."""
    if only_item:
        wanted = list(dict.fromkeys(only_item))
        by_key = {document.item_key: document for document in documents}
        if missing := [key for key in wanted if key not in by_key]:
            raise QuestionSetError(
                f"No indexed document for item {', '.join(repr(key) for key in missing)}. "
                "Item keys come from the store, so an item with no indexed attachment "
                "cannot have questions generated for it."
            )
        documents = [by_key[key] for key in wanted]
    return documents[:limit] if limit is not None else documents


def _print_generation_plan(documents: list[DocumentText], per_document: int) -> None:
    """The dry run: what would be read, and roughly what it would cost."""
    table = Table(title="Documents that would be used", title_justify="left")
    table.add_column("#", justify="right", style="dim")
    table.add_column("Title", overflow="ellipsis", max_width=46)
    table.add_column("Authors", overflow="ellipsis", max_width=22)
    table.add_column("Year", justify="right")
    table.add_column("Chunks", justify="right")
    table.add_column("Chars", justify="right")
    table.add_column("Item", style="dim")

    estimate = 0
    for number, document in enumerate(documents, start=1):
        estimate += document.prompt_tokens_estimate
        table.add_row(
            str(number),
            escape(document.title),
            escape(document.authors) or "[dim]—[/dim]",
            document.year or "[dim]—[/dim]",
            str(document.chunks),
            f"{len(document.text)}{'*' if document.truncated else ''}",
            document.item_key,
        )
    console.print(table)

    console.print(
        f"\n[bold]{_plural(per_document * len(documents), 'question')}[/bold] would be "
        f"written from {_plural(len(documents), 'document')} "
        f"({per_document} each), in {_plural(len(documents), 'call')} costing roughly "
        f"{estimate} prompt tokens."
    )
    if any(document.truncated for document in documents):
        console.print(
            "  [dim]* sampled as evenly spaced passages to fit "
            "eval.document_char_budget; the parts between them are not shown.[/dim]"
        )
    console.print("\n[dim]Dry run: nothing was generated and nothing was written.[/dim]")


def _print_generation_summary(
    path: Path, stats: GenerationStats, per_document: int, verbose: bool
) -> None:
    console.print(
        f"\n[bold green]Wrote {_plural(stats.questions, 'question')}[/bold green] from "
        f"{_plural(stats.documents, 'document')} to [bold]{path}[/bold]."
    )
    expected = per_document * stats.documents
    if stats.questions != expected:
        console.print(
            f"  [dim]{expected} were requested; the generator returned {stats.questions}.[/dim]"
        )
    if stats.rejected:
        console.print(
            f"  [dim]{stats.rejected} dropped for naming their document or pointing at "
            f'it ("the study"); -v lists them.[/dim]'
        )
    if stats.without_evidence:
        console.print(
            f"  [dim]{stats.without_evidence} have no evidence quote (the generator's was "
            f"not found verbatim), so eval cannot check their passage.[/dim]"
        )
    if stats.truncated:
        console.print(
            f"  [dim]{stats.truncated} document(s) were longer than "
            f"eval.document_char_budget and were sampled in passages.[/dim]"
        )
    if stats.failures:
        console.print(f"\n[yellow]{_plural(len(stats.failures), 'document')} produced nothing:")
        for failure in stats.failures:
            console.print(f"  [yellow]•[/yellow] {escape(failure)}")

    console.print(
        "\n[bold]Next:[/bold] read through the file and edit it — the questions are a "
        "model's reading of each paper, and a generated set cannot ask what the corpus "
        "fails to answer. Then run [bold]zotero-rag eval[/bold]."
    )
    if verbose:
        console.print(
            f"[dim]{stats.prompt_tokens} prompt + {stats.completion_tokens} completion "
            f"tokens, {stats.elapsed_s:.1f}s elapsed.[/dim]"
        )


@app.command("generate-questions")
def generate_questions_command(
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Where to write the set; defaults to the configured file."),
    ] = None,
    per_document: Annotated[
        int | None,
        typer.Option("--per-document", help="Questions per document.", min=1),
    ] = None,
    language: Annotated[
        str,
        typer.Option("--language", help="Language to write the questions in."),
    ] = MATCH_DOCUMENT,
    only_item: Annotated[
        list[str] | None,
        typer.Option("--only-item", help="Use just this Zotero item key; repeatable."),
    ] = None,
    limit: Annotated[
        int | None,
        typer.Option("--limit", help="Use only the first N documents.", min=1),
    ] = None,
    force: Annotated[
        bool, typer.Option("--force", help="Overwrite an existing question set.")
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="List the documents and the estimated cost; write nothing."),
    ] = False,
    assume_yes: Annotated[
        bool, typer.Option("--yes", "-y", help="Skip the confirmation on a large collection.")
    ] = False,
    verbose: VerboseOption = False,
) -> None:
    """Write a starting questions.yaml from the documents already indexed.

    Questions come from the vector store rather than from `cache/`, so every one of
    them is answerable from the index and each carries the exact item it was
    written from. The result is a draft to edit, not a finished golden set.
    """
    _enable_verbose(verbose)

    try:
        config, store = open_index(_print_notice)
        count = per_document if per_document is not None else config.eval.questions_per_document
        documents = _select_documents(
            collect_documents(store, config.eval.document_char_budget),
            only_item or [],
            limit,
        )
        if not documents:
            raise NotIndexedError(
                "No indexed document carries an item key, so no question can name the "
                "item it came from. Re-index this folder to record them."
            )

        if dry_run:
            _print_generation_plan(documents, count)
            return

        destination = out if out is not None else resolve_questions_path(config, None)
        # Refuse before spending anything, not after.
        if destination.exists() and not force:
            raise QuestionSetError(
                f"{destination} already exists. Generated questions would overwrite a set "
                "you may have edited; pass --force to replace it, or --out to write "
                "elsewhere."
            )

        if len(documents) > _CONFIRM_DOCUMENTS and not assume_yes:
            estimate = sum(document.prompt_tokens_estimate for document in documents)
            err_console.print(
                f"[yellow]Warning:[/yellow] this would make "
                f"{_plural(len(documents), 'call')} costing roughly {estimate} prompt "
                f"tokens. Try --dry-run, or --limit, first."
            )
            if not typer.confirm(
                f"Generate questions for {len(documents)} documents?", default=False
            ):
                raise typer.Abort()

        generator = build_generator(config)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=err_console,
            transient=True,
        ) as progress:
            task = progress.add_task("Writing questions...", total=len(documents))

            def on_start(document: DocumentText) -> None:
                progress.update(task, advance=1, description=escape(document.title[:48]))

            questions, stats = generate_questions(documents, generator, count, language, on_start)

        if not questions:
            raise QuestionSetError(
                "The generator returned no usable question for any document, so nothing "
                "was written. Run with -v to see what each document reported."
            )

        write_questions(
            destination, questions, header=file_header(config, stats, language), force=True
        )
    except ZoteroRagError as exc:
        _fail(exc)
        return

    _print_generation_summary(destination, stats, count, verbose=_is_verbose())


if __name__ == "__main__":  # pragma: no cover
    app()
