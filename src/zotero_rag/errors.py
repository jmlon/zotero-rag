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


class ZoteroApiError(ZoteroRagError):
    """The Zotero Web API call failed."""


class CollectionNotFoundError(ZoteroRagError):
    """No collection matches the given path or key."""


class AmbiguousCollectionError(ZoteroRagError):
    """A path segment matches more than one collection at the same level."""


class CollectionMismatchError(ZoteroRagError):
    """The requested collection differs from the one pinned in config.yaml."""
