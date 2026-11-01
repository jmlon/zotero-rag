"""The core package stands alone: no CLI, no terminal UI, and Chroma only on request.

`zotero-rag-web` installs `zotero-rag-core` without the `chroma` extra, so any core
module importing chromadb (or the CLI's own dependencies) at import time would
break it, while every test here would still pass with everything installed.
"""

from __future__ import annotations

import pkgutil
import subprocess
import sys
from pathlib import Path

import zotero_rag_core
from zotero_rag_core.store import ChunkStore
from zotero_rag_core.stores.chroma import ChromaStore

#: Modules that may only be needed by the CLI, or by an optional extra.
FORBIDDEN = ("chromadb", "zotero_rag", "typer", "textual", "dotenv", "firecrawl_anydoc")

#: Core modules that are allowed to need an extra.
NEEDS_EXTRA = {"zotero_rag_core.stores.chroma"}


def _core_modules() -> list[str]:
    found = pkgutil.walk_packages(zotero_rag_core.__path__, prefix="zotero_rag_core.")
    return sorted(module.name for module in found if module.name not in NEEDS_EXTRA)


def test_core_imports_without_extras_or_cli_dependencies() -> None:
    """Blocks the forbidden packages, then imports every core module in a fresh process."""
    script = "\n".join(
        [
            "import importlib, sys",
            # A `None` entry makes `import name` raise ImportError immediately.
            *(f"sys.modules[{name!r}] = None" for name in FORBIDDEN),
            f"for module in {_core_modules()!r}:",
            "    importlib.import_module(module)",
        ]
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", script],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent,
    )
    assert result.returncode == 0, result.stderr


def test_core_modules_are_discovered() -> None:
    """Guards the test above against silently importing nothing."""
    modules = _core_modules()
    assert "zotero_rag_core.querying" in modules
    assert "zotero_rag_core.indexer" in modules
    assert "zotero_rag_core.store" in modules


def test_chroma_store_is_a_chunk_store(tmp_path: Path) -> None:
    """Checked by mypy through this annotation, and opened for real here."""
    store: ChunkStore = ChromaStore(tmp_path / "chroma")
    assert store.count() == 0
