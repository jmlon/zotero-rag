"""Reading Markdown transcripts.

Producing transcripts (`convert`) lives in the CLI and needs firecrawl-anydoc;
reading them, which indexing does, needs nothing beyond the standard library.
"""

from __future__ import annotations


def body(text: str) -> str:
    """A transcript without its YAML front matter; other Markdown is unchanged.

    Applied before judging a transcript, so the title and authors in its own
    header cannot count as text it recovered from the source.
    """
    if not text.startswith("---\n"):
        return text
    end = text.find("\n---\n", 4)
    return text if end < 0 else text[end + len("\n---\n") :].lstrip("\n")
