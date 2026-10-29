"""Expected, user-facing errors.

Anything raised as a `ZoteroRagError` is a condition the user can act on, so the
CLI prints its message and exits non-zero instead of showing a traceback.
"""

from __future__ import annotations


class ZoteroRagError(Exception):
    """Base class for expected, user-facing errors."""


class ConfigError(ZoteroRagError):
    """The configuration file is missing, malformed, or inconsistent."""


class MissingCredentialError(ConfigError):
    """A required credential is absent from the environment / `.env`."""

    def __init__(self, var_names: list[str], purpose: str) -> None:
        self.var_names = var_names
        self.purpose = purpose
        joined = ", ".join(var_names)
        super().__init__(
            f"Missing {purpose} credentials: {joined}. "
            f"Set {'them' if len(var_names) > 1 else 'it'} in your environment or a .env file."
        )


class NotAnIndexFolderError(ZoteroRagError):
    """The command must be run from an index folder (one containing config.yaml)."""


class NotIndexedError(ZoteroRagError):
    """The index folder has no vector store yet, so there is nothing to query."""


class FingerprintMismatchError(ZoteroRagError):
    """The store was built with settings that make the current config invalid.

    Raised rather than warned because the alternative is answering from a mixture
    of incompatible embedding spaces, which looks like a working answer.
    """

    def __init__(self, differences: list[str], remedy: str) -> None:
        self.differences = differences
        joined = "; ".join(differences)
        super().__init__(
            f"This index was built with different settings — {joined}. "
            f"Mixing embedding models in one store makes retrieval meaningless. {remedy}"
        )


class ZoteroApiError(ZoteroRagError):
    """The Zotero Web API call failed."""


class ZoteroWriteAccessError(ZoteroApiError):
    """The API key may read the library but not change it or upload files."""


class ZoteroConflictError(ZoteroApiError):
    """The item changed in Zotero after it was read, so the write was refused.

    Expected when someone edits the library while a run is going: the item is
    skipped and the next run sees the new state, rather than one side's edit
    silently overwriting the other's.
    """


class CollectionNotFoundError(ZoteroRagError):
    """No collection matches the given path or key."""


class AmbiguousCollectionError(ZoteroRagError):
    """A path segment matches more than one collection at the same level."""


class CollectionMismatchError(ZoteroRagError):
    """The requested collection differs from the one pinned in config.yaml."""
