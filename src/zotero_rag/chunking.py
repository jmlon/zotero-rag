"""Splitting segment text into embeddable chunks."""

from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel

from zotero_rag.documents import LocatorKind
from zotero_rag.extraction import Segment


class Chunk(BaseModel):
    """A piece of one segment, ready to embed.

    `locator` is the segment's ordinal — a page number for a PDF, a section number
    for a snapshot — and `kind`/`label` say how to render it in a citation.
    """

    text: str
    locator: int
    index: int
    kind: LocatorKind = LocatorKind.PAGE
    label: str = ""


def chunk_segments(segments: list[Segment], chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """Split each segment independently, so every chunk keeps an exact locator.

    `index` restarts at 0 in each segment; the pair (locator, index) is what makes
    a chunk's id unique within its attachment.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks: list[Chunk] = []
    for segment in segments:
        for index, piece in enumerate(splitter.split_text(segment.text)):
            if piece.strip():
                chunks.append(
                    Chunk(
                        text=piece,
                        locator=segment.number,
                        index=index,
                        kind=segment.kind,
                        label=segment.label,
                    )
                )
    return chunks
