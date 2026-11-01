"""What the pipeline needs from a chunk store, independent of the database.

The indexer, retrieval, querying, evaluation and question generation are written
against `ChunkStore`, so the same code runs on the Chroma store an index folder
keeps (`zotero_rag_core.stores.chroma`) and on a shared database.

A chunk is identified by `<attachment_key>:<locator>:<chunk_index>` and carries its
citation metadata as a flat mapping (`planning.PlannedAttachment.chunk_metadata`).
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import Any, Protocol

from pydantic import BaseModel, Field

from zotero_rag_core.errors import ZoteroRagError
from zotero_rag_core.fingerprint import StoreFingerprint


class StoredChunk(BaseModel):
    """A chunk read back out of the store, with its citation metadata."""

    id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    distance: float | None = None


class VectorStoreError(ZoteroRagError):
    """The vector store could not be opened or written."""


def reading_order_key(chunk: StoredChunk) -> tuple[str, int, int]:
    """Where a chunk sits in its document: attachment, locator ordinal, chunk index.

    Chunk ids are `<attachment_key>:<locator>:<chunk_index>`, so sorting the *ids*
    as strings puts page 10 before page 2 and hands the reader a scrambled document.
    Reading order has to come from the numeric metadata instead.
    """

    def number(field: str) -> int:
        try:
            return int(str(chunk.metadata.get(field)))
        except (TypeError, ValueError):
            return 0

    return str(chunk.metadata.get("attachment_key") or ""), number("page"), number("chunk_index")


#: Collection keys a search is limited to; `None` searches the whole store.
Scope = Collection[str] | None


def in_scope(metadata: Mapping[str, Any], scope: Scope) -> bool:
    """Whether a chunk belongs to `scope`.

    A chunk with no `collection_keys` (written before they existed) is outside every
    scope, but inside an unscoped search. A store filtering in its own query
    language must agree with this rule.
    """
    if scope is None:
        return True
    keys = metadata.get("collection_keys")
    return isinstance(keys, list) and not set(scope).isdisjoint(keys)


class ChunkStore(Protocol):
    """A persistent store of embedded chunks.

    Implementations raise `VectorStoreError` when the database itself fails.
    """

    # --- writing -----------------------------------------------------------

    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """Store new chunks. Nothing happens for an empty `ids`."""
        ...

    def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """Replace chunks in place, keyed by id."""
        ...

    def delete_by_attachment(self, attachment_key: str) -> int:
        """Drop every chunk of one attachment. Returns how many were removed."""
        ...

    def delete_ids(self, ids: list[str]) -> None: ...

    def set_collection_keys(self, attachment_key: str, collection_keys: list[str]) -> int:
        """Rewrite `collection_keys` on every chunk of one attachment, re-embedding nothing.

        Returns how many chunks were updated. `collection_keys` must not be empty.
        """
        ...

    def reset(self) -> None:
        """Remove every chunk, so vectors of another width can be stored."""
        ...

    # --- reading -----------------------------------------------------------

    def query(self, embedding: list[float], top_k: int, scope: Scope = None) -> list[StoredChunk]:
        """The `top_k` nearest chunks to an embedded question, within `scope`, nearest first.

        Fewer come back when the scope holds fewer chunks; an empty scope matches
        nothing. `StoredChunk.distance` is the cosine distance.
        """
        ...

    def get_by_attachment(self, attachment_key: str) -> list[StoredChunk]:
        """Every chunk of one attachment, in reading order."""
        ...

    def get_by_ids(self, ids: list[str]) -> list[StoredChunk]: ...

    def all_chunks(self) -> list[StoredChunk]: ...

    def count(self) -> int: ...

    def has_attachment(self, attachment_key: str) -> bool: ...

    def attachment_keys(self) -> set[str]: ...

    def attachment_version(self, attachment_key: str) -> int | None:
        """The stored Zotero version of the attachment; `None` if never indexed or unknown."""
        ...

    def collection_keys_by_attachment(self) -> dict[str, list[str] | None]:
        """Each attachment's sorted `collection_keys`; `None` when missing or inconsistent."""
        ...

    def stale_ids(self, embedding_model: str) -> list[str]:
        """Ids of chunks not embedded by `embedding_model`, including those of unknown model."""
        ...

    # --- how the store was built (PRD §11.1.1) -----------------------------

    def read_fingerprint(self) -> StoreFingerprint | None: ...

    def write_fingerprint(self, fingerprint: StoreFingerprint) -> None: ...

    def derived_dimension(self) -> int | None:
        """The width of the stored vectors, read from the data; `None` when empty."""
        ...
