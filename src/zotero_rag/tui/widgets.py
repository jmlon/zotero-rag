"""The panes of `zotero-rag-ui`: folder tree, transcript, prompt (PRD-CursesUI §3–§8)."""

from __future__ import annotations

import asyncio
import threading
from typing import ClassVar

from rich.text import Text
from textual import events
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Vertical, VerticalScroll
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Markdown, Static, TextArea, Tree
from textual.widgets.tree import TreeNode

from zotero_rag.querying import QueryResult
from zotero_rag.tui.history import PromptHistory
from zotero_rag.tui.scopes import ScopeChoice, ScopeNode

# --- folder tree -------------------------------------------------------------


class ScopeTree(Tree[ScopeChoice]):
    """The pinned subtree. Moving the cursor browses; selecting sets the scope.

    The active scope is marked separately from the cursor, so the user can look
    around the tree without losing track of what their next question searches.
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("enter", "select_cursor", "Search here"),
        Binding("space", "select_cursor", "Search here", show=False),
        Binding("left", "collapse", "Collapse", show=False),
        Binding("right", "expand", "Expand", show=False),
    ]

    def __init__(self, root: ScopeNode, filtering: bool, *, id: str | None = None) -> None:
        super().__init__(_name(root.choice), root.choice, id=id)
        self.auto_expand = False
        self.show_root = True
        self.active: TreeNode[ScopeChoice] = self.root
        self._add_folders(self.root, root.children)
        # Without a snapshot only the root is shown, and it cannot be opened (§4.3.2).
        self.root.allow_expand = filtering and bool(root.children)
        self.root.expand()
        self._mark(self.root, active=True)

    def _add_folders(self, parent: TreeNode[ScopeChoice], children: list[ScopeNode]) -> None:
        for child in children:
            node = parent.add(_name(child.choice), child.choice, allow_expand=bool(child.children))
            self._add_folders(node, child.children)

    def activate(self, node: TreeNode[ScopeChoice]) -> ScopeChoice | None:
        """Make `node` the active scope; returns its choice."""
        self._mark(self.active, active=False)
        self.active = node
        self._mark(node, active=True)
        return node.data

    def _mark(self, node: TreeNode[ScopeChoice], active: bool) -> None:
        name = _name(node.data) if node.data is not None else str(node.label)
        node.set_label(Text(f"◆ {name}", style="bold underline") if active else Text(name))

    def action_collapse(self) -> None:
        """← closes the folder under the cursor, or moves up from an already closed one."""
        node = self.cursor_node
        if node is None:
            return
        if node.is_expanded and node.allow_expand:
            node.collapse()
        elif node.parent is not None:
            self.move_cursor(node.parent)

    def action_expand(self) -> None:
        node = self.cursor_node
        if node is not None and node.allow_expand:
            node.expand()


def _name(choice: ScopeChoice | None) -> str:
    return "" if choice is None else choice.path.rsplit("/", 1)[-1]


# --- transcript ------------------------------------------------------------


class Transcript(VerticalScroll):
    """The session's questions and answers, newest at the bottom (§6).

    It follows new output while the user is at the bottom. Once they scroll up it
    stays put and shows a "new output" hint instead, until they come back down.
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("home", "scroll_home", "Top", show=False),
        Binding("end", "scroll_end", "Bottom", show=False),
    ]

    def __init__(self, *, id: str | None = None) -> None:
        super().__init__(id=id)
        self.following = True

    def compose(self) -> ComposeResult:
        yield Static("new output ↓", id="new-output")

    def watch_scroll_y(self, old_value: float, new_value: float) -> None:
        super().watch_scroll_y(old_value, new_value)
        self.following = new_value >= self.max_scroll_y - 1
        if self.following:
            self.query_one("#new-output").display = False

    def content_changed(self) -> None:
        """Call after anything was added: follow it, or say it is there."""
        if self.following:
            self.call_after_refresh(self.scroll_end, animate=False)
        else:
            self.query_one("#new-output").display = True

    def add(self, widget: Static | Vertical) -> None:
        self.mount(widget)
        self.content_changed()

    def on_click(self, event: events.Click) -> None:
        if event.widget is not None and event.widget.id == "new-output":
            self.scroll_end(animate=False)


class SessionNote(Static):
    """A line from the session itself rather than an answer, e.g. a setup note."""

    def __init__(self, text: str, level: str = "note") -> None:
        super().__init__(Text(text), classes=f"notice {level}")


class Entry(Vertical):
    """One question and its answer: scope tag, question, answer, then sources.

    Answer text arrives from the query's worker thread through `feed`, and is drawn
    by `flush` on the UI thread, which the app calls on a timer. Rendering in
    batches like this keeps a fast stream from redrawing the Markdown per token.
    """

    def __init__(self, scope_path: str, question: str) -> None:
        super().__init__(classes="entry")
        self.scope_path = scope_path
        self.question = question
        self._pending: list[str] = []
        self._pending_lock = threading.Lock()
        self._render_lock = asyncio.Lock()
        self._streamed = False

    def compose(self) -> ComposeResult:
        yield Static(Text(f"[{self.scope_path}]"), classes="scope-tag")
        yield Static(Text(f"> {self.question}"), classes="question")
        yield Markdown("", classes="answer")
        yield Static("", classes="footnote")

    def feed(self, delta: str) -> None:
        """Queue answer text. Safe to call from any thread."""
        with self._pending_lock:
            self._pending.append(delta)

    async def flush(self) -> bool:
        """Draw the queued text; returns whether there was any."""
        async with self._render_lock:
            with self._pending_lock:
                text = "".join(self._pending)
                self._pending.clear()
            if not text:
                return False
            self._streamed = True
            await self.query_one(Markdown).append(text)
            return True

    async def finish(self, result: QueryResult, narrow_scope: bool) -> None:
        """Complete the entry once the stream has ended: final text, then sources."""
        await self.flush()
        if not self._streamed and result.answer:
            await self.query_one(Markdown).update(result.answer)
        self.query_one(".footnote", Static).update(_footnote(result, narrow_scope))

    async def fail(self, message: str) -> None:
        """Show the error under whatever part of the answer did arrive."""
        await self.flush()
        footnote = self.query_one(".footnote", Static)
        footnote.add_class("error")
        footnote.update(Text.assemble(("Error: ", "bold"), message))


def _footnote(result: QueryResult, narrow_scope: bool) -> Text:
    """The cited sources, as `zotero-rag query` lists them (PRD.md §8.2)."""
    cited = [source for source in result.sources if source.cited]
    text = Text()
    if cited:
        text.append("Sources", style="bold")
        for source in cited:
            text.append(f"\n  [{source.number}] ", style="cyan")
            text.append(source.label)
            if source.zotero_url:
                text.append(f"\n       {source.zotero_url}", style="dim")
    elif result.sources:
        passages = "passage was" if result.chunks_retrieved == 1 else "passages were"
        text.append(
            f"The answer cites no sources; {result.chunks_retrieved} {passages} retrieved.",
            style="dim",
        )
    elif narrow_scope:
        text.append(
            "Nothing in this folder matched. It may be too narrow; try a parent folder.",
            style="dim",
        )
    return text


# --- prompt -----------------------------------------------------------------


class PromptArea(TextArea):
    """The multi-line question editor (§7).

    Enter sends; Alt+Enter, or Shift+Enter where the terminal reports it, starts a
    new line. ↑ on the first line and ↓ on the last step through the history.
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        # Handled in `_on_key`, before TextArea would insert a newline; listed here so
        # the footer shows them.
        Binding("enter", "submit", "Send"),
        Binding("alt+enter", "newline", "Newline"),
    ]

    class Submitted(Message):
        def __init__(self, text: str) -> None:
            super().__init__()
            self.text = text

    def __init__(self, history: PromptHistory, *, id: str | None = None) -> None:
        super().__init__(
            id=id,
            soft_wrap=True,
            tab_behavior="focus",
            placeholder="Ask a question about this folder… (Enter to send)",
        )
        self.prompt_history = history

    async def _on_key(self, event: events.Key) -> None:
        if event.key == "enter":
            event.stop()
            event.prevent_default()
            self.action_submit()
            return
        if event.key in ("alt+enter", "shift+enter"):
            event.stop()
            event.prevent_default()
            self.action_newline()
            return
        await super()._on_key(event)

    def action_submit(self) -> None:
        self.post_message(self.Submitted(self.text))

    def action_newline(self) -> None:
        start, end = self.selection
        self.replace("\n", start, end)

    def action_cursor_up(self, select: bool = False) -> None:
        if not select and self.cursor_at_first_line:
            self._recall(self.prompt_history.previous(self.text))
            return
        super().action_cursor_up(select)

    def action_cursor_down(self, select: bool = False) -> None:
        if not select and self.cursor_at_last_line:
            self._recall(self.prompt_history.next())
            return
        super().action_cursor_down(select)

    def _recall(self, text: str | None) -> None:
        if text is None:
            return
        self.load_text(text)
        self.move_cursor(self.document.end)


# --- help -------------------------------------------------------------------

HELP_TEXT = """\
[b]Keys[/b]
  Tab / Shift+Tab        move between the prompt, the folders and the results
  F1 or ?                this help (? only outside the prompt)
  Ctrl+Q                 quit

[b]Prompt[/b]
  Enter                  send the question
  Alt+Enter, Shift+Enter new line
  ↑ / ↓                  earlier prompts (on the first / last line)

[b]Folders[/b]
  ↑ / ↓                  move
  ← / →                  collapse / expand
  Enter, Space, click    search this folder and everything below it

[b]Results[/b]
  PgUp PgDn Home End     scroll (the mouse wheel works everywhere)

Press Esc to close."""


class HelpScreen(ModalScreen[None]):
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "dismiss", "Close"),
        Binding("question_mark", "dismiss", "Close", show=False),
        Binding("f1", "dismiss", "Close", show=False),
    ]

    def compose(self) -> ComposeResult:
        yield Static(HELP_TEXT, id="help")

    def on_click(self) -> None:
        self.dismiss()
