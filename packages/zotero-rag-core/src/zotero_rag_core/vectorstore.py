"""Persistent Chroma vector store.

chromadb is used directly rather than through a LangChain wrapper. Embeddings
are always supplied by us, so the collection is created with no embedding
function at all — that keeps Chroma's bundled ONNX model from ever loading.
"""

from __future__ import annotations

import logging
from collections.abc import Collection, Mapping
from pathlib import Path
from typing import Any, cast

import chromadb
from chromadb.config import Settings
from pydantic import BaseModel, Field

from zotero_rag_core.errors import ZoteroRagError
from zotero_rag_core.fingerprint import StoreFingerprint

logger = logging.getLogger(__name__)

CHROMA_DIRNAME = "chroma"
COLLECTION_NAME = "zotero_rag"


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


def scope_where(scope: Collection[str]) -> dict[str, Any]:
    """The Chroma filter for chunks held by any collection in `scope`.

    Verified against chromadb 1.5.9 (PRD-CursesUI §4.3.1): only `$contains` matches
    inside a list value. `$in` and plain equality silently match nothing, and `$or`
    refuses fewer than two clauses, so a single key needs the bare form. Keys are
    sorted so that one scope always yields one filter.
    """
    keys = sorted(set(scope))
    if not keys:
        raise ValueError("An empty scope selects nothing; check for it before filtering")
    clauses = [{"collection_keys": {"$contains": key}} for key in keys]
    return clauses[0] if len(clauses) == 1 else {"$or": clauses}


def in_scope(metadata: Mapping[str, Any], scope: Scope) -> bool:
    """Whether a chunk belongs to `scope`. The same rule as `scope_where`, in Python.

    A chunk with no `collection_keys` (written before they existed) is outside every
    scope, but inside an unscoped search.
    """
    if scope is None:
        return True
    keys = metadata.get("collection_keys")
    return isinstance(keys, list) and not set(scope).isdisjoint(keys)


class VectorStore:
    """Thin wrapper over one persistent Chroma collection."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        try:
            directory.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=str(directory),
                settings=Settings(anonymized_telemetry=False),
            )
            self._collection = self._client.get_or_create_collection(
                name=COLLECTION_NAME, embedding_function=None
            )
        except Exception as exc:  # noqa: BLE001 - chromadb raises assorted types
            raise VectorStoreError(f"Cannot open the vector store at {directory}: {exc}") from exc

    @classmethod
    def exists(cls, directory: Path) -> bool:
        """True if a persistent store has already been created here."""
        return (directory / "chroma.sqlite3").is_file()

    def has_attachment(self, attachment_key: str) -> bool:
        """True if any chunk of this attachment is already stored.

        This is the PoC's incremental criterion (PRD §7.3): presence of the
        attachment key, with no Zotero version comparison.
        """
        found = self._collection.get(where={"attachment_key": attachment_key}, limit=1)
        return bool(found["ids"])

    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        if not ids:
            return
        try:
            # chromadb's stubs describe narrower numpy-flavoured types than the
            # plain lists it accepts at runtime.
            self._collection.add(
                ids=ids,
                embeddings=cast(Any, embeddings),
                documents=documents,
                metadatas=cast(Any, metadatas),
            )
        except Exception as exc:  # noqa: BLE001 - chromadb raises assorted types
            raise VectorStoreError(f"Cannot write to the vector store: {exc}") from exc

    def get_by_attachment(self, attachment_key: str) -> list[StoredChunk]:
        """Every stored chunk of one attachment, in reading order."""
        found = self._collection.get(where={"attachment_key": attachment_key})
        return sorted(self._to_chunks(found), key=reading_order_key)

    def query(self, embedding: list[float], top_k: int, scope: Scope = None) -> list[StoredChunk]:
        """The `top_k` nearest chunks to an embedded question, within `scope`.

        Fewer come back when the scope holds fewer chunks. An empty scope matches
        nothing and makes no call at all.
        """
        if scope is not None and not scope:
            return []
        # Passed only when scoped, so an unscoped query is exactly the call it was.
        where = {"where": cast(Any, scope_where(scope))} if scope is not None else {}
        try:
            result = self._collection.query(
                query_embeddings=cast(Any, [embedding]),
                n_results=top_k,
                include=cast(Any, ["documents", "metadatas", "distances"]),
                **where,
            )
        except Exception as exc:  # noqa: BLE001 - chromadb raises assorted types
            raise VectorStoreError(f"Cannot query the vector store: {exc}") from exc

        # chromadb returns one list per query embedding; we always send one.
        ids = (result["ids"] or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        return [
            StoredChunk(
                id=identifier,
                text=documents[position] if position < len(documents) else "",
                metadata=dict(metadatas[position]) if position < len(metadatas) else {},
                distance=distances[position] if position < len(distances) else None,
            )
            for position, identifier in enumerate(ids)
        ]

    def count(self) -> int:
        return int(self._collection.count())

    # --- fingerprint (PRD §11.1.1) ---------------------------------------

    def read_fingerprint(self) -> StoreFingerprint | None:
        """What this store records about how it was built, if anything.

        `None` means a store written before M4; the caller adopts it rather than
        invalidating data that is probably fine.
        """
        return StoreFingerprint.from_metadata(self._collection.metadata)

    def write_fingerprint(self, fingerprint: StoreFingerprint) -> None:
        """Record how this store was built, preserving Chroma's own metadata."""
        existing = dict(self._collection.metadata or {})
        existing.update(fingerprint.to_metadata())
        try:
            self._collection.modify(metadata=existing)
        except Exception as exc:  # noqa: BLE001 - chromadb raises assorted types
            raise VectorStoreError(f"Cannot record the store fingerprint: {exc}") from exc

    def derived_dimension(self) -> int | None:
        """The true vector width, read from the data rather than from a record.

        This is the one fact that cannot drift from what is stored or be changed by
        editing a file, so it cross-checks the recorded fingerprint.
        """
        found = self._collection.get(limit=1, include=cast(Any, ["embeddings"]))
        embeddings = found.get("embeddings")
        if embeddings is None or len(embeddings) == 0:
            return None
        return len(embeddings[0])

    # --- repair and pruning ----------------------------------------------

    def stale_ids(self, embedding_model: str) -> list[str]:
        """Ids of chunks not embedded by `embedding_model`.

        Verified against chromadb 1.5.9: `$ne` also matches chunks that carry no
        `embedding_model` at all, so chunks written before M4 count as stale — which
        is what we want, since their true model is unknown.
        """
        where = cast(Any, {"embedding_model": {"$ne": embedding_model}})
        found = self._collection.get(where=where)
        return list(found["ids"])

    def get_by_ids(self, ids: list[str]) -> list[StoredChunk]:
        """Chunks with the given ids, text and metadata included."""
        if not ids:
            return []
        found = self._collection.get(ids=ids)
        return self._to_chunks(found)

    def all_chunks(self) -> list[StoredChunk]:
        """Every stored chunk. Used to rebuild a store from its own text."""
        return self._to_chunks(self._collection.get())

    @staticmethod
    def _to_chunks(found: Any) -> list[StoredChunk]:
        ids = found["ids"]
        documents = found.get("documents") or []
        metadatas = found.get("metadatas") or []
        return [
            StoredChunk(
                id=identifier,
                text=documents[position] if position < len(documents) else "",
                metadata=dict(metadatas[position]) if position < len(metadatas) else {},
            )
            for position, identifier in enumerate(ids)
        ]

    def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """Replace chunks in place, keyed by id."""
        if not ids:
            return
        try:
            self._collection.upsert(
                ids=ids,
                embeddings=cast(Any, embeddings),
                documents=documents,
                metadatas=cast(Any, metadatas),
            )
        except Exception as exc:  # noqa: BLE001 - chromadb raises assorted types
            raise VectorStoreError(f"Cannot update the vector store: {exc}") from exc

    def delete_by_attachment(self, attachment_key: str) -> int:
        """Drop every chunk of one attachment. Returns how many were removed."""
        ids = self._collection.get(where={"attachment_key": attachment_key})["ids"]
        if ids:
            self._collection.delete(ids=list(ids))
        return len(ids)

    def delete_ids(self, ids: list[str]) -> None:
        if ids:
            self._collection.delete(ids=ids)

    def attachment_keys(self) -> set[str]:
        """Every attachment key present in the store."""
        metadatas = self._collection.get(include=cast(Any, ["metadatas"]))["metadatas"] or []
        return {
            str(metadata["attachment_key"])
            for metadata in metadatas
            if metadata and metadata.get("attachment_key")
        }

    def collection_keys_by_attachment(self) -> dict[str, list[str] | None]:
        """Each stored attachment's `collection_keys`, in one read of the store.

        `None` when the attachment's chunks carry none, or disagree with one another,
        for example after an interrupted backfill. Either way it needs rewriting.
        """
        metadatas = self._collection.get(include=cast(Any, ["metadatas"]))["metadatas"] or []
        found: dict[str, list[str] | None] = {}
        for metadata in metadatas:
            if not metadata or not metadata.get("attachment_key"):
                continue
            key = str(metadata["attachment_key"])
            raw = metadata.get("collection_keys")
            keys = sorted(str(value) for value in raw) if isinstance(raw, list) else None
            if key in found and found[key] != keys:
                keys = None
            found[key] = keys
        return found

    def set_collection_keys(self, attachment_key: str, collection_keys: list[str]) -> int:
        """Rewrite `collection_keys` on every chunk of one attachment.

        Verified against chromadb 1.5.9: `update` merges the given metadata keys into
        what is stored and leaves the text and vector alone, so nothing is re-embedded.
        Returns how many chunks were updated.
        """
        if not collection_keys:
            raise ValueError("Chroma rejects an empty list; give at least one collection key")
        ids = list(self._collection.get(where={"attachment_key": attachment_key})["ids"])
        if not ids:
            return 0
        try:
            self._collection.update(
                ids=ids,
                metadatas=cast(Any, [{"collection_keys": list(collection_keys)}] * len(ids)),
            )
        except Exception as exc:  # noqa: BLE001 - chromadb raises assorted types
            raise VectorStoreError(f"Cannot update the vector store: {exc}") from exc
        return len(ids)

    def attachment_version(self, attachment_key: str) -> int | None:
        """The Zotero version of the attachment as stored, or `None` if unknown.

        `None` covers both "never indexed" and "indexed before versions were
        recorded"; the caller distinguishes them with `has_attachment()`.
        """
        found = self._collection.get(where={"attachment_key": attachment_key}, limit=1)
        metadatas = found.get("metadatas") or []
        if not metadatas or not metadatas[0]:
            return None
        raw = metadatas[0].get("attachment_version")
        try:
            return int(str(raw))
        except (TypeError, ValueError):
            return None

    def reset(self) -> None:
        """Drop the collection and recreate it empty.

        A collection's vector width is fixed once written, so switching to a model
        of a different dimension requires this rather than an in-place update.
        """
        try:
            self._client.delete_collection(name=COLLECTION_NAME)
        except Exception as exc:  # noqa: BLE001 - absent collection raises too
            logger.debug("nothing to delete: %s", exc)
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME, embedding_function=None
        )
