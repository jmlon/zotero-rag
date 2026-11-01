"""The `zotero-rag-ui` application (PRD-CursesUI.md)."""

from __future__ import annotations

import logging
from collections.abc import Coroutine
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar

from rich.text import Text
from textual import events, work
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Static, Tree

from zotero_rag.tui.clipboard import copy_with_system_tool
from zotero_rag.tui.export import (
    Exchange,
    answer_markdown,
    default_export_path,
    transcript_markdown,
    write_transcript,
)
from zotero_rag.tui.history import PromptHistory
from zotero_rag.tui.scopes import ScopeChoice, build_scope_tree
from zotero_rag.tui.session import Phase, QuerySession
from zotero_rag.tui.widgets import (
    Entry,
    ExportScreen,
    HelpScreen,
    PromptArea,
    ScopeTree,
    SessionNote,
    Transcript,
)
from zotero_rag_core.errors import ZoteroRagError
from zotero_rag_core.querying import QueryResult

logger = logging.getLogger(__name__)

MIN_WIDTH, MIN_HEIGHT = 80, 24
#: At most ten redraws a second while an answer streams in (§6.1).
REFRESH_INTERVAL_S = 0.1
SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

UNAVAILABLE_NOTICE = (
    "Folder filtering needs `zotero-rag index` to be run once more. "
    "Until then, questions search the whole index."
)
PHASE_TEXT: dict[Phase, str] = {
    "waiting": "waiting for the keyword index…",
    "retrieving": "retrieving…",
    "answering": "answering…",
}


class ZoteroRagApp(App[None]):
    CSS_PATH = "app.tcss"
    TITLE = "zotero-rag"
    # Every binding is listed in the help; the palette would be a second, partial list.
    ENABLE_COMMAND_PALETTE = False

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("ctrl+q", "quit", "Quit", priority=True),
        Binding("ctrl+s", "export", "Save transcript"),
        Binding("ctrl+l", "copy_answer", "Copy answer"),
        Binding("f1", "help", "Help", priority=True),
        # Not a priority binding, so `?` still types into the prompt.
        Binding("question_mark", "help", "Help", show=False),
    ]

    def __init__(self, session: QuerySession, history: PromptHistory) -> None:
        super().__init__()
        self.session = session
        self.history = history
        self.filtering = session.snapshot is not None
        self.scope: ScopeChoice = build_scope_tree(session.snapshot, session.root_name).choice
        #: The root's name, which titles an exported transcript.
        self.collection = self.scope.path
        self.phase: Phase | None = None
        self.active_entry: Entry | None = None
        self._spin = 0

    @property
    def busy(self) -> bool:
        return self.active_entry is not None

    def compose(self) -> ComposeResult:
        yield Static(
            f"The terminal is too small. zotero-rag-ui needs at least {MIN_WIDTH}×{MIN_HEIGHT}.",
            id="too-small",
        )
        with Horizontal(id="main"):
            tree_root = build_scope_tree(self.session.snapshot, self.session.root_name)
            yield ScopeTree(tree_root, self.filtering, id="tree")
            with Vertical(id="right"):
                yield Transcript(id="results")
                yield PromptArea(self.history, id="prompt")
        yield Static(id="status")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#tree").border_title = "Folders"
        self.query_one("#results").border_title = "Results"
        self.query_one("#prompt").border_title = "Prompt"
        transcript = self.query_one(Transcript)
        for notice in self.session.notices:
            transcript.add(SessionNote(f"{notice.level.capitalize()}: {notice.text}", notice.level))
        if not self.filtering:
            transcript.add(SessionNote(UNAVAILABLE_NOTICE, "warning"))
        self._check_size()
        self.query_one(PromptArea).focus()
        self.set_interval(REFRESH_INTERVAL_S, self._tick)
        if not self.session.keyword_index_ready:
            self._build_keyword_index()
        self._show_status()

    # --- layout -------------------------------------------------------------

    def on_resize(self, event: events.Resize) -> None:
        self._check_size()

    def _check_size(self) -> None:
        small = self.size.width < MIN_WIDTH or self.size.height < MIN_HEIGHT
        self.query_one("#too-small").display = small
        self.query_one("#main").display = not small

    def _show_status(self) -> None:
        status = Text.assemble(("Scope: ", "bold"), self.scope.path)
        status.append(f" · {self.session.settings.summary()}", style="dim")
        activity = None
        if self.phase is not None:
            activity = PHASE_TEXT[self.phase]
        elif not self.session.keyword_index_ready:
            activity = "building keyword index…"
        if activity is not None:
            status.append(f"   {SPINNER[self._spin % len(SPINNER)]} {activity}", style="yellow")
        self.query_one("#status", Static).update(status)

    async def _tick(self) -> None:
        """Draw queued answer text and advance the spinner, ten times a second."""
        entry = self.active_entry
        if entry is not None and await entry.flush():
            self.query_one(Transcript).content_changed()
        if self.phase is not None or not self.session.keyword_index_ready:
            self._spin += 1
            self._show_status()

    # --- scope --------------------------------------------------------------

    def on_tree_node_selected(self, event: Tree.NodeSelected[ScopeChoice]) -> None:
        tree = self.query_one(ScopeTree)
        choice = tree.activate(event.node)
        if choice is not None:
            self.scope = choice
            self._show_status()

    # --- asking -------------------------------------------------------------

    def on_prompt_area_submitted(self, message: PromptArea.Submitted) -> None:
        question = message.text
        if not question.strip():
            return
        if self.busy:
            self.notify("Wait for the current answer to finish.", severity="warning")
            return
        self.history.add(question)
        self.query_one(PromptArea).clear()
        entry = Entry(self.scope.path, question.strip(), scoped=self.scope.keys is not None)
        self.query_one(Transcript).add(entry)
        self.active_entry = entry
        self._ask(question.strip(), self.scope, entry)

    @work(thread=True, exclusive=True, group="query")
    def _ask(self, question: str, scope: ScopeChoice, entry: Entry) -> None:
        """Run one question on a worker thread, so the screen stays responsive."""
        try:
            result = self.session.ask(
                question,
                scope.keys,
                on_token=entry.feed,
                on_phase=lambda phase: self.call_from_thread(self._set_phase, phase),
            )
        except ZoteroRagError as exc:
            self._on_ui(self._failed(entry, str(exc)))
        except Exception as exc:  # noqa: BLE001 - one bad answer must not end the session
            logger.exception("query failed")
            self._on_ui(self._failed(entry, f"Unexpected error: {exc}"))
        else:
            self._on_ui(self._answered(entry, result))

    def _on_ui(self, step: Coroutine[Any, Any, None]) -> None:
        """Run `step` on the UI thread and wait for it, from a worker thread."""

        async def run() -> None:
            await step

        # mypy solves `T | Awaitable[T]` to Never when T is None; the call is fine.
        self.call_from_thread(run)  # type: ignore[arg-type]

    def _set_phase(self, phase: Phase) -> None:
        self.phase = phase
        self._show_status()

    async def _answered(self, entry: Entry, result: QueryResult) -> None:
        await entry.finish(result)
        self._done()

    async def _failed(self, entry: Entry, message: str) -> None:
        await entry.fail(message)
        self._done()

    def _done(self) -> None:
        self.active_entry = None
        self.phase = None
        self.query_one(Transcript).content_changed()
        self._show_status()

    @work(thread=True, group="keyword-index")
    def _build_keyword_index(self) -> None:
        self.session.build_keyword_index()
        self.call_from_thread(self._show_status)

    # --- export and copy ----------------------------------------------------

    def exchanges(self) -> list[Exchange]:
        return [entry.exchange for entry in self.query(Entry)]

    def action_export(self) -> None:
        """Ctrl+S: save the whole transcript as Markdown, to a path the user confirms."""
        if not isinstance(self.screen, (ExportScreen, HelpScreen)):
            if not self.exchanges():
                self.notify("Nothing to save yet: ask a question first.")
                return
            now = datetime.now()
            base = self.session.index_dir or Path.cwd()
            self.push_screen(
                ExportScreen(default_export_path(now), base, self._save_transcript),
                self._saved,
            )

    def _save_transcript(self, path: Path) -> None:
        """Render at the moment of saving, so an answer finished meanwhile is included."""
        text = transcript_markdown(
            self.exchanges(),
            collection=self.collection,
            settings=self.session.settings.summary(),
            exported_at=datetime.now(),
        )
        write_transcript(path, text)

    def _saved(self, path: Path | None) -> None:
        if path is not None:
            self.notify(f"Saved the transcript to {path}")

    def action_copy_answer(self) -> None:
        """Ctrl+L: copy the newest finished answer, with its sources."""
        answered = [exchange for exchange in self.exchanges() if exchange.result is not None]
        if not answered:
            self.notify("No answer to copy yet.")
            return
        latest = answered[-1]
        assert latest.result is not None
        text = answer_markdown(latest.result, latest.scoped)
        tool = copy_with_system_tool(text)
        # Sent as well when a tool worked: it is what reaches the clipboard over SSH.
        self.copy_to_clipboard(text)
        if tool is not None:
            self.notify("Copied the latest answer and its sources.")
        else:
            self.notify(
                "Sent the latest answer to the terminal's clipboard. If nothing pastes, "
                "this terminal does not support that: install xclip, xsel or wl-clipboard.",
                timeout=8,
            )

    # --- help ---------------------------------------------------------------

    def action_help(self) -> None:
        if not isinstance(self.screen, (HelpScreen, ExportScreen)):
            self.push_screen(HelpScreen())
