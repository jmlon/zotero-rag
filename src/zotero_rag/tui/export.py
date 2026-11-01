"""The transcript as Markdown: exported to a file, or one answer copied (§1.2).

Plain functions over `Exchange` records, so what lands in the file or on the
clipboard is tested without a terminal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from zotero_rag_core.querying import QueryResult

TRANSCRIPTS_DIRNAME = "transcripts"


@dataclass
class Exchange:
    """One question and what came back for it, as the transcript recorded it."""

    scope_path: str
    question: str
    asked_at: datetime = field(default_factory=datetime.now)
    #: The text streamed so far; the whole answer once finished.
    answer: str = ""
    result: QueryResult | None = None
    error: str | None = None
    #: Set when the folder was scoped, for the "may be too narrow" note.
    scoped: bool = False

    @property
    def finished(self) -> bool:
        return self.result is not None or self.error is not None


def sources_markdown(result: QueryResult, scoped: bool = False) -> str:
    """The cited sources as a Markdown list, or the line that explains their absence."""
    cited = [source for source in result.sources if source.cited]
    if cited:
        lines = ["**Sources**", ""]
        for source in cited:
            line = f"- \\[{source.number}\\] {source.label}"
            if source.zotero_url:
                line += f"  \n  <{source.zotero_url}>"
            lines.append(line)
        return "\n".join(lines)
    if result.sources:
        passages = "passage was" if result.chunks_retrieved == 1 else "passages were"
        return f"*The answer cites no sources; {result.chunks_retrieved} {passages} retrieved.*"
    if scoped:
        return "*Nothing in this folder matched. It may be too narrow; try a parent folder.*"
    return ""


def answer_markdown(result: QueryResult, scoped: bool = False) -> str:
    """What Ctrl+L copies: the answer, then its sources, so `[n]` still resolves."""
    parts = [result.answer.strip(), sources_markdown(result, scoped)]
    return "\n\n".join(part for part in parts if part) + "\n"


def exchange_markdown(exchange: Exchange) -> str:
    """One entry: folder tag and time, the question as a quote, answer, sources."""
    quoted = "\n".join(f"> {line}" if line else ">" for line in exchange.question.splitlines())
    parts = [f"**\\[{exchange.scope_path}\\]** · {exchange.asked_at:%H:%M}", quoted]
    if exchange.result is not None:
        parts.append(answer_markdown(exchange.result, exchange.scoped).rstrip())
    else:
        if exchange.answer.strip():
            parts.append(exchange.answer.strip())
        if exchange.error is not None:
            parts.append(f"**Error:** {exchange.error}")
        else:
            parts.append("*(The answer was still arriving when this was exported.)*")
    return "\n\n".join(parts)


def transcript_markdown(
    exchanges: list[Exchange], collection: str, settings: str, exported_at: datetime
) -> str:
    """The whole session, oldest first, under a header saying where it came from."""
    header = "\n".join(
        [
            "# zotero-rag transcript",
            "",
            f"- Collection: {collection}",
            f"- Exported: {exported_at:%Y-%m-%d %H:%M}",
            f"- Settings: {settings}",
        ]
    )
    return "\n\n---\n\n".join([header, *map(exchange_markdown, exchanges)]) + "\n"


def default_export_path(exported_at: datetime) -> Path:
    """`transcripts/zotero-rag-YYYYMMDD-HHMMSS.md`, relative to the index folder."""
    return Path(TRANSCRIPTS_DIRNAME) / f"zotero-rag-{exported_at:%Y%m%d-%H%M%S}.md"


def write_transcript(path: Path, text: str) -> None:
    """Write the file, creating its folder. Raises `OSError` for the caller to show."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
