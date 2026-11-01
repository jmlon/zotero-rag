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

from pydantic import BaseModel, ConfigDict, Field, model_validator

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


#: `year_int` of a chunk whose item has no parseable year. Chroma rejects `None`.
YEAR_UNKNOWN = 0


class Filters(BaseModel):
    """What a search is limited to. Every condition given must hold (AND).

    `None` means "no condition". An *empty* set is a condition nothing satisfies,
    so a folder with no subfolders that nobody filed anything in returns nothing,
    rather than silently widening to the whole library.

    Chunks carry `collection_keys`, `year_int` and `tags` in their metadata
    (`planning.PlannedAttachment.chunk_metadata`). A chunk written before a field
    existed fails every condition on that field, and passes searches without one.
    A store filtering in its own query language must agree with `matches`.
    """

    model_config = ConfigDict(frozen=True)

    #: The chunk's item is filed in any of these collections.
    collection_keys: frozenset[str] | None = None
    #: Inclusive bounds on the item's publication year.
    year_min: int | None = None
    year_max: int | None = None
    #: With a year bound, also keep items that have no year.
    include_undated: bool = True
    #: The chunk's item carries any of these tags.
    tags: frozenset[str] | None = None

    @model_validator(mode="after")
    def _ordered_years(self) -> Filters:
        low, high = self.year_min, self.year_max
        if low is not None and high is not None and low > high:
            raise ValueError(f"year_min ({low}) is after year_max ({high})")
        return self

    @classmethod
    def in_collections(cls, keys: Collection[str] | None) -> Filters | None:
        """A folder filter, or `None` (no filter at all) for `keys=None`."""
        return None if keys is None else cls(collection_keys=frozenset(keys))

    @property
    def limits_year(self) -> bool:
        return self.year_min is not None or self.year_max is not None

    @property
    def selects_nothing(self) -> bool:
        """True when no chunk can match, so a store need not be asked at all."""
        return self.collection_keys == frozenset() or self.tags == frozenset()

    def matches(self, metadata: Mapping[str, Any]) -> bool:
        """Whether a chunk with this metadata passes every condition."""
        if self.collection_keys is not None:
            keys = metadata.get("collection_keys")
            if not isinstance(keys, list) or self.collection_keys.isdisjoint(keys):
                return False
        if self.tags is not None:
            tags = metadata.get("tags")
            if not isinstance(tags, list) or self.tags.isdisjoint(tags):
                return False
        if self.limits_year:
            year = metadata.get("year_int")
            if not isinstance(year, int):
                return False
            if year == YEAR_UNKNOWN:
                return self.include_undated
            if self.year_min is not None and year < self.year_min:
                return False
            if self.year_max is not None and year > self.year_max:
                return False
        return True


def passes(metadata: Mapping[str, Any], filters: Filters | None) -> bool:
    """`filters.matches`, with `None` meaning no filter."""
    return filters is None or filters.matches(metadata)


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

    def query(
        self, embedding: list[float], top_k: int, filters: Filters | None = None
    ) -> list[StoredChunk]:
        """The `top_k` nearest chunks to an embedded question that pass `filters`, nearest first.

        The filters apply before the cut, so a narrow filter still fills `top_k`
        when it holds that many chunks. `StoredChunk.distance` is the cosine distance.
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
