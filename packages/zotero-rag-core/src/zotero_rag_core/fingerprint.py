"""What a store records about how it was built (PRD §11.1.1).

`config.yaml` states what the user *wants*; it is meant to be edited, and editing
`embedding_model` is exactly what makes a populated store invalid. So the store
carries what was actually *used*, and every run compares intent against fact.

Two settings are corrupting when they change and two are merely inconsistent:
re-embedding with a different model puts vectors from two incompatible spaces in
one collection, which silently returns meaningless neighbours, whereas a new
`chunk_size` only means old chunks are shaped differently from new ones.
"""

from __future__ import annotations

import logging
from enum import StrEnum
from typing import Any
from urllib.parse import urlsplit

from pydantic import BaseModel

from zotero_rag_core.config import Config

logger = logging.getLogger(__name__)

# Chroma reserves `hnsw:*` keys for its own configuration, so ours are namespaced.
METADATA_PREFIX = "zotero_rag."
SCHEMA_VERSION = 1


class Severity(StrEnum):
    """Whether a changed setting invalidates the store or merely mixes it."""

    REFUSE = "refuse"
    WARN = "warn"


class Mismatch(BaseModel):
    """One setting that differs between the store and the current config."""

    setting: str
    recorded: str
    intended: str
    severity: Severity

    def __str__(self) -> str:
        built = f"store was built with {self.recorded!r}"
        wanted = f"config says {self.intended!r}"
        return f"{self.setting}: {built}, {wanted}"


class StoreFingerprint(BaseModel):
    """How the vectors in a store were produced."""

    embedding_model: str = ""
    embedding_dim: int = 0
    chunk_size: int = 0
    chunk_overlap: int = 0
    endpoint_host: str = ""
    schema_version: int = SCHEMA_VERSION

    @classmethod
    def intended(cls, config: Config, embedding_dim: int = 0) -> StoreFingerprint:
        """The fingerprint a run with this config would produce."""
        return cls(
            embedding_model=config.embedding_model,
            embedding_dim=embedding_dim,
            chunk_size=config.index.chunk_size,
            chunk_overlap=config.index.chunk_overlap,
            endpoint_host=urlsplit(config.global_.endpoint).netloc,
        )

    def to_metadata(self) -> dict[str, str | int]:
        """Flat, prefixed, scalar-only mapping — Chroma rejects nested values."""
        return {f"{METADATA_PREFIX}{name}": value for name, value in self.model_dump().items()}

    @classmethod
    def from_metadata(cls, metadata: dict[str, Any] | None) -> StoreFingerprint | None:
        """Read a fingerprint back, or `None` for a store written before M4."""
        if not metadata:
            return None
        values = {
            name.removeprefix(METADATA_PREFIX): value
            for name, value in metadata.items()
            if name.startswith(METADATA_PREFIX)
        }
        if not values:
            return None
        try:
            return cls.model_validate(values)
        except ValueError:
            logger.warning("Ignoring unreadable store fingerprint: %r", values)
            return None

    def compare(self, intended: StoreFingerprint) -> list[Mismatch]:
        """Differences between this (recorded) fingerprint and what a run intends.

        A dimension of 0 means "not known", which is what a fingerprint adopted
        from an empty store carries, so it never counts as a difference.
        """
        mismatches: list[Mismatch] = []

        if self.embedding_model and self.embedding_model != intended.embedding_model:
            mismatches.append(
                Mismatch(
                    setting="embedding_model",
                    recorded=self.embedding_model,
                    intended=intended.embedding_model,
                    severity=Severity.REFUSE,
                )
            )
        dims_differ = self.embedding_dim != intended.embedding_dim
        if self.embedding_dim and intended.embedding_dim and dims_differ:
            mismatches.append(
                Mismatch(
                    setting="embedding dimension",
                    recorded=str(self.embedding_dim),
                    intended=str(intended.embedding_dim),
                    severity=Severity.REFUSE,
                )
            )
        for setting, recorded, wanted in (
            ("chunk_size", self.chunk_size, intended.chunk_size),
            ("chunk_overlap", self.chunk_overlap, intended.chunk_overlap),
        ):
            if recorded and recorded != wanted:
                mismatches.append(
                    Mismatch(
                        setting=setting,
                        recorded=str(recorded),
                        intended=str(wanted),
                        severity=Severity.WARN,
                    )
                )
        if self.endpoint_host and self.endpoint_host != intended.endpoint_host:
            mismatches.append(
                Mismatch(
                    setting="endpoint",
                    recorded=self.endpoint_host,
                    intended=intended.endpoint_host,
                    severity=Severity.WARN,
                )
            )
        return mismatches


def blocking(mismatches: list[Mismatch]) -> list[Mismatch]:
    return [mismatch for mismatch in mismatches if mismatch.severity is Severity.REFUSE]


def advisory(mismatches: list[Mismatch]) -> list[Mismatch]:
    return [mismatch for mismatch in mismatches if mismatch.severity is Severity.WARN]
