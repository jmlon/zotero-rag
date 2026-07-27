"""Splitting page text into embeddable chunks."""

from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel

from zotero_rag.extraction import Page


class Chunk(BaseModel):
    """A piece of one page, ready to embed."""

    text: str
    page: int
    index: int


def chunk_pages(pages: list[Page], chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """Split each page independently, so every chunk keeps an exact page number.

    `index` restarts at 0 on each page; the pair (page, index) is what makes a
    chunk's id unique within its attachment.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks: list[Chunk] = []
    for page in pages:
        for index, piece in enumerate(splitter.split_text(page.text)):
            if piece.strip():
                chunks.append(Chunk(text=piece, page=page.number, index=index))
    return chunks
