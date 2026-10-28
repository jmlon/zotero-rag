"""Prompt history: recalled with ↑/↓ and kept per index folder (PRD-CursesUI §7).

Each entry is stored as one JSON-encoded string per line, so a multi-line prompt
stays one entry. The file is a convenience, never a requirement: if it cannot be
read or written, history still works for the session and the failure is logged.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

HISTORY_FILENAME = "history"
HISTORY_LIMIT = 1000


class PromptHistory:
    """The sent prompts, oldest first, plus a cursor for stepping through them.

    The cursor is `None` while the user is editing their own text (the draft). The
    first step back saves that text, and stepping forward past the newest entry
    brings it back, so recalling an old prompt never costs what was being typed.
    """

    def __init__(
        self, entries: list[str] | None = None, path: Path | None = None, limit: int = HISTORY_LIMIT
    ) -> None:
        self.entries: list[str] = list(entries or [])[-limit:]
        self.path = path
        self.limit = limit
        self._position: int | None = None
        self._draft = ""

    @classmethod
    def load(cls, path: Path, limit: int = HISTORY_LIMIT) -> PromptHistory:
        """Read the history file; a missing file is an empty history.

        A line that is not a JSON string (a hand edit, a torn write) is skipped
        rather than discarding the rest of the file.
        """
        entries: list[str] = []
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            lines = []
        except OSError as exc:
            logger.warning("cannot read prompt history %s: %s", path, exc)
            lines = []
        for line in lines:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(entry, str) and entry:
                entries.append(entry)
        return cls(entries, path, limit)

    def add(self, prompt: str) -> None:
        """Record a sent prompt and return the cursor to the draft.

        A prompt identical to the newest entry is not added again, so resending the
        same question does not fill the history with copies of it.
        """
        self.reset()
        if not prompt or (self.entries and self.entries[-1] == prompt):
            return
        self.entries.append(prompt)
        trimmed = len(self.entries) > self.limit
        if trimmed:
            del self.entries[: -self.limit]
        self._save(prompt, rewrite=trimmed)

    def previous(self, current: str) -> str | None:
        """The entry before the cursor, or `None` when already at the oldest."""
        if not self.entries or self._position == 0:
            return None
        if self._position is None:
            self._draft = current
            self._position = len(self.entries)
        self._position -= 1
        return self.entries[self._position]

    def next(self) -> str | None:
        """The entry after the cursor, the draft past the newest, or `None` at the draft."""
        if self._position is None:
            return None
        self._position += 1
        if self._position >= len(self.entries):
            self._position = None
            return self._draft
        return self.entries[self._position]

    def reset(self) -> None:
        self._position = None
        self._draft = ""

    def _save(self, prompt: str, rewrite: bool) -> None:
        """Append the new entry, or rewrite the file once it has been trimmed."""
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if rewrite:
                partial = self.path.with_name(f".{self.path.name}.tmp")
                partial.write_text(
                    "".join(json.dumps(entry) + "\n" for entry in self.entries), encoding="utf-8"
                )
                partial.replace(self.path)
            else:
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(prompt) + "\n")
        except OSError as exc:
            logger.warning("cannot save prompt history to %s: %s", self.path, exc)
