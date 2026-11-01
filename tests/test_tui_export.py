"""Saving the transcript and copying an answer (PRD-CursesUI §1.2, now built)."""

from __future__ import annotations

import os
import shutil
import stat
from datetime import datetime
from pathlib import Path

import pytest

from zotero_rag.tui import clipboard
from zotero_rag.tui.export import (
    Exchange,
    answer_markdown,
    default_export_path,
    exchange_markdown,
    transcript_markdown,
    write_transcript,
)
from zotero_rag_core.querying import QueryResult, Source

WHEN = datetime(2026, 10, 28, 14, 5, 9)
#: Resolved before any test narrows PATH to the fake tools.
CAT = shutil.which("cat") or "/bin/cat"


def result(cited: bool = True, sources: bool = True) -> QueryResult:
    listed = [
        Source(
            number=1,
            title="Fog Computing",
            authors="Bonomi, Flavio",
            year="2012",
            page=3,
            zotero_url="https://www.zotero.org/u/items/F1",
            cited=cited,
        ),
        Source(number=2, title="Unused", page=1, cited=False),
    ]
    return QueryResult(
        question="q",
        answer="Fog extends the cloud to the edge [1].\n",
        sources=listed if sources else [],
        chunks_retrieved=2 if sources else 0,
    )


# --- what is copied -----------------------------------------------------------


def test_a_copied_answer_carries_only_its_cited_sources() -> None:
    text = answer_markdown(result())
    assert text == (
        "Fog extends the cloud to the edge [1].\n\n"
        "**Sources**\n\n"
        "- \\[1\\] Fog Computing — Bonomi, Flavio (2012), p. 3  \n"
        "  <https://www.zotero.org/u/items/F1>\n"
    )
    assert "Unused" not in text


def test_an_answer_citing_nothing_says_so() -> None:
    text = answer_markdown(result(cited=False))
    assert "*The answer cites no sources; 2 passages were retrieved.*" in text


def test_an_empty_folder_says_it_may_be_too_narrow_only_when_scoped() -> None:
    assert "too narrow" in answer_markdown(result(sources=False), scoped=True)
    assert "too narrow" not in answer_markdown(result(sources=False), scoped=False)


# --- what is exported ---------------------------------------------------------


def test_an_exchange_is_tagged_quoted_and_answered() -> None:
    exchange = Exchange("Networking/Fog", "What is fog?\n\nBriefly.", WHEN, result=result())
    text = exchange_markdown(exchange)
    assert text.startswith("**\\[Networking/Fog\\]** · 14:05\n\n> What is fog?\n>\n> Briefly.")
    assert "Fog extends the cloud to the edge [1]." in text
    assert "<https://www.zotero.org/u/items/F1>" in text


def test_a_failed_exchange_keeps_its_partial_answer_and_the_error() -> None:
    exchange = Exchange("Fog", "why?", WHEN, answer="Partial", error="402 insufficient credits")
    text = exchange_markdown(exchange)
    assert "Partial\n\n**Error:** 402 insufficient credits" in text


def test_an_unfinished_exchange_says_it_was_still_arriving() -> None:
    exchange = Exchange("Fog", "why?", WHEN, answer="So far")
    assert "So far\n\n*(The answer was still arriving" in exchange_markdown(exchange)
    assert not exchange.finished


def test_the_transcript_has_a_header_then_each_exchange_in_order() -> None:
    exchanges = [
        Exchange("Fog", "first?", WHEN, result=result()),
        Exchange("Fog/Edge", "second?", WHEN, result=result(cited=False)),
    ]
    text = transcript_markdown(exchanges, "Fog", "hybrid · top_k 10 · m", WHEN)
    assert text.startswith(
        "# zotero-rag transcript\n\n- Collection: Fog\n- Exported: 2026-10-28 14:05\n"
        "- Settings: hybrid · top_k 10 · m\n\n---\n\n"
    )
    assert text.index("first?") < text.index("second?")
    assert text.count("\n---\n") == 2
    assert text.endswith("\n")


def test_the_default_path_is_timestamped_under_transcripts() -> None:
    assert default_export_path(WHEN) == Path("transcripts/zotero-rag-20261028-140509.md")


def test_writing_creates_the_folder(tmp_path: Path) -> None:
    path = tmp_path / "transcripts" / "a.md"
    write_transcript(path, "# hi\n")
    assert path.read_text(encoding="utf-8") == "# hi\n"


# --- the clipboard tool -----------------------------------------------------


@pytest.fixture
def tools(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An empty PATH directory to put fake clipboard tools in, on an X11 desktop."""
    directory = tmp_path / "bin"
    directory.mkdir()
    monkeypatch.setenv("PATH", str(directory))
    monkeypatch.setenv("DISPLAY", ":0")
    for name in ("WAYLAND_DISPLAY", "SSH_CONNECTION", "SSH_TTY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(clipboard.sys, "platform", "linux")
    return directory


def fake_tool(directory: Path, name: str, exit_code: int = 0) -> Path:
    """A shell script standing in for `name`; it saves what it was given."""
    received = directory / f"{name}.received"
    script = directory / name
    script.write_text(f'#!/bin/sh\n{CAT} > "{received}"\nexit {exit_code}\n', encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return received


@pytest.mark.skipif(os.name != "posix", reason="fake tools are shell scripts")
def test_xclip_receives_the_text(tools: Path) -> None:
    received = fake_tool(tools, "xclip")
    assert clipboard.copy_with_system_tool("¿Qué? [1]") == "xclip"
    assert received.read_text(encoding="utf-8") == "¿Qué? [1]"


@pytest.mark.skipif(os.name != "posix", reason="fake tools are shell scripts")
def test_a_failing_tool_falls_through_to_the_next(tools: Path) -> None:
    fake_tool(tools, "xclip", exit_code=1)
    received = fake_tool(tools, "xsel")
    assert clipboard.copy_with_system_tool("text") == "xsel"
    assert received.read_text(encoding="utf-8") == "text"


@pytest.mark.skipif(os.name != "posix", reason="fake tools are shell scripts")
def test_wayland_prefers_wl_copy(tools: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    fake_tool(tools, "xclip")
    fake_tool(tools, "wl-copy")
    assert clipboard.copy_with_system_tool("text") == "wl-copy"


def test_no_tool_means_none(tools: Path) -> None:
    assert clipboard.copy_with_system_tool("text") is None


@pytest.mark.skipif(os.name != "posix", reason="fake tools are shell scripts")
def test_over_ssh_no_local_tool_is_used(tools: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A local tool would fill the remote machine's clipboard, not the user's."""
    monkeypatch.setenv("SSH_CONNECTION", "10.0.0.1 5000 10.0.0.2 22")
    received = fake_tool(tools, "xclip")
    assert clipboard.copy_with_system_tool("text") is None
    assert not received.exists()
