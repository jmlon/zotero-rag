"""Persistent Chroma vector store.

chromadb is used directly rather than through a LangChain wrapper. Embeddings
are always supplied by us, so the collection is created with no embedding
function at all — that keeps Chroma's bundled ONNX model from ever loading.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

import chromadb
from chromadb.config import Settings
from pydantic import BaseModel, Field

from zotero_rag.errors import ZoteroRagError

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
        """Every stored chunk belonging to one attachment, ordered by id."""
        found = self._collection.get(where={"attachment_key": attachment_key})
        ids = found["ids"]
        documents = found["documents"] or []
        metadatas = found["metadatas"] or []
        chunks = [
            StoredChunk(
                id=identifier,
                text=documents[position] if position < len(documents) else "",
                metadata=dict(metadatas[position]) if position < len(metadatas) else {},
            )
            for position, identifier in enumerate(ids)
        ]
        return sorted(chunks, key=lambda chunk: chunk.id)

    def query(self, embedding: list[float], top_k: int) -> list[StoredChunk]:
        """The `top_k` nearest chunks to an embedded question."""
        try:
            result = self._collection.query(
                query_embeddings=cast(Any, [embedding]),
                n_results=top_k,
                include=cast(Any, ["documents", "metadatas", "distances"]),
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
