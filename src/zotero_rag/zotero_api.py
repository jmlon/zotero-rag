"""Zotero Web API access and collection-hierarchy handling."""

from __future__ import annotations

import hashlib
import io
import logging
import time
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, Field
from pyzotero import zotero
from pyzotero.zotero_errors import (
    PreConditionFailedError,
    PyZoteroError,
    RequestEntityTooLargeError,
)

from zotero_rag.config import Config
from zotero_rag.errors import (
    AmbiguousCollectionError,
    CollectionNotFoundError,
    ZoteroApiError,
    ZoteroConflictError,
    ZoteroWriteAccessError,
)

# pyzotero raises PyZoteroError for API-level problems, but it speaks HTTP through
# httpx, so a timeout or a DNS failure arrives as an httpx error instead. Both are
# expected conditions for a network tool (PRD §10), so both must reach the user as
# a one-line message rather than a traceback.
NETWORK_ERRORS = (PyZoteroError, httpx.HTTPError, OSError)

logger = logging.getLogger(__name__)


class CollectionNode(BaseModel):
    """A node in the Zotero collection hierarchy."""

    key: str
    name: str
    num_items: int = 0
    children: list[CollectionNode] = Field(default_factory=list)

    @property
    def total_items(self) -> int:
        """Item count for this collection plus all of its descendants."""
        return self.num_items + sum(child.total_items for child in self.children)

    def walk(self, depth: int = 0) -> list[tuple[CollectionNode, int]]:
        """Depth-first list of (node, depth) pairs, this node first."""
        pairs = [(self, depth)]
        for child in self.children:
            pairs.extend(child.walk(depth + 1))
        return pairs


def build_client(config: Config) -> zotero.Zotero:
    """Build an authenticated pyzotero client. Requires Zotero credentials."""
    config.global_.require_zotero()
    return zotero.Zotero(
        config.global_.zotero_library_id,
        config.global_.zotero_library_type,
        config.global_.zotero_api_key,
    )


def fetch_collections(zot: zotero.Zotero) -> list[dict[str, Any]]:
    """Fetch every collection in the library, following pagination."""
    try:
        return list(zot.everything(zot.collections()))
    except NETWORK_ERRORS as exc:
        raise ZoteroApiError(f"Could not fetch collections from Zotero: {exc}") from exc


def fetch_username(zot: zotero.Zotero) -> str:
    """The library owner's username, used to build zotero.org item URLs."""
    try:
        info = zot.key_info()
    except NETWORK_ERRORS as exc:
        raise ZoteroApiError(f"Could not verify the Zotero API key: {exc}") from exc
    username = info.get("username", "")
    return str(username)


def build_collection_tree(raw_collections: list[dict[str, Any]]) -> list[CollectionNode]:
    """Turn pyzotero's flat collection list into a sorted forest of roots.

    Each raw collection carries `data.parentCollection`, which is `False` for a
    top-level collection and a parent key otherwise. Collections whose parent is
    absent from the input (e.g. a partial fetch) are surfaced as roots rather
    than silently dropped.
    """
    nodes: dict[str, CollectionNode] = {}
    parents: dict[str, str | None] = {}

    for entry in raw_collections:
        data = entry.get("data", {})
        key = data.get("key") or entry.get("key")
        if not key:
            logger.warning("Skipping collection with no key: %r", entry)
            continue
        meta = entry.get("meta", {})
        nodes[key] = CollectionNode(
            key=key,
            name=data.get("name", "(unnamed)"),
            num_items=int(meta.get("numItems", 0) or 0),
        )
        parent = data.get("parentCollection")
        parents[key] = parent if isinstance(parent, str) and parent else None

    roots: list[CollectionNode] = []
    for key, node in nodes.items():
        parent_key = parents[key]
        if parent_key is None:
            roots.append(node)
        elif parent_key in nodes:
            nodes[parent_key].children.append(node)
        else:
            logger.warning(
                "Collection %r (%s) references unknown parent %s; treating as top level",
                node.name,
                key,
                parent_key,
            )
            roots.append(node)

    _sort_recursive(roots)
    return roots


def _sort_recursive(nodes: list[CollectionNode]) -> None:
    """Sort siblings by name, case-insensitively, at every level."""
    nodes.sort(key=lambda node: node.name.casefold())
    for node in nodes:
        _sort_recursive(node.children)


def count_collections(nodes: list[CollectionNode]) -> int:
    """Total number of collections in a forest."""
    return sum(len(node.walk()) for node in nodes)


def fetch_collection_items(zot: zotero.Zotero, collection_key: str) -> list[dict[str, Any]]:
    """Every item in one collection, including child attachments and notes."""
    try:
        return list(zot.everything(zot.collection_items(collection_key)))
    except NETWORK_ERRORS as exc:
        raise ZoteroApiError(
            f"Could not fetch items for collection {collection_key}: {exc}"
        ) from exc


def split_collection_path(path: str) -> list[str]:
    """Split a hierarchy path into segments, tolerating leading/trailing slashes."""
    return [segment.strip() for segment in path.split("/") if segment.strip()]


def resolve_collection_path(roots: list[CollectionNode], path: str) -> CollectionNode:
    """Walk a `Parent/Child/Grandchild` path through the hierarchy.

    Matching is case-insensitive. A segment matching more than one collection at
    its level is an error naming the candidate keys, so the user can retry with
    `--collection-key`.
    """
    segments = split_collection_path(path)
    if not segments:
        raise CollectionNotFoundError(f"{path!r} is not a valid collection path")

    level = roots
    trail: list[str] = []
    node: CollectionNode | None = None

    for segment in segments:
        matches = [child for child in level if child.name.casefold() == segment.casefold()]
        location = "/".join(trail) if trail else "the top level"
        if not matches:
            available = ", ".join(sorted(child.name for child in level)) or "(none)"
            raise CollectionNotFoundError(
                f"No collection named {segment!r} under {location}. Available: {available}"
            )
        if len(matches) > 1:
            candidates = ", ".join(f"{m.name} ({m.key})" for m in matches)
            raise AmbiguousCollectionError(
                f"{segment!r} is ambiguous under {location}: {candidates}. "
                "Disambiguate with --collection-key."
            )
        node = matches[0]
        trail.append(node.name)
        level = node.children

    assert node is not None  # guaranteed: segments is non-empty
    return node


def find_collection_by_key(roots: list[CollectionNode], key: str) -> CollectionNode:
    """Locate a collection anywhere in the hierarchy by its key."""
    for root in roots:
        for node, _depth in root.walk():
            if node.key == key:
                return node
    raise CollectionNotFoundError(f"No collection with key {key!r} in this library")


def collection_scope(node: CollectionNode, recursive: bool) -> list[CollectionNode]:
    """The collections to index: just `node`, or it and all its descendants."""
    if not recursive:
        return [node]
    return [descendant for descendant, _depth in node.walk()]


def scope_tree(node: CollectionNode, recursive: bool) -> CollectionNode:
    """`node` as `collection_scope` indexes it: with its subtree, or alone."""
    return node if recursive else node.model_copy(update={"children": []})


def library_scope(roots: list[CollectionNode]) -> list[CollectionNode]:
    """Every collection in the library, for the whole-library root target."""
    return [node for root in roots for node, _depth in root.walk()]


# --- Attachment files -------------------------------------------------------------

API_ENDPOINT = "https://api.zotero.org"
# Statuses worth waiting out: rate limiting, and the API saying it is busy.
_RETRY_STATUSES = frozenset({429, 503})
_MAX_ATTEMPTS = 4


class FileDownloader:
    """Fetches attachment files from the Web API, whatever their content type.

    `pyzotero`'s `file()` picks a parser from the response's Content-Type and
    falls back to JSON for a type it does not list — OpenDocument among them — so
    an ODT download failed with a decode error. This reads the bytes as bytes.

    The key travels as `Authorization: Bearer`, which httpx drops when the file
    endpoint redirects to storage on another host; a custom header would follow
    the redirect there.
    """

    def __init__(
        self,
        library_type: str,
        library_id: str,
        api_key: str,
        *,
        client: httpx.Client | None = None,
        endpoint: str = API_ENDPOINT,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._base = f"{endpoint.rstrip('/')}/{library_type}s/{library_id}"
        self._headers = {"Authorization": f"Bearer {api_key}", "Zotero-API-Version": "3"}
        self._client = client or httpx.Client(timeout=httpx.Timeout(30.0, read=120.0))
        self._sleep = sleep
        self._require: Callable[[], None] = lambda: None

    @classmethod
    def from_config(cls, config: Config) -> FileDownloader:
        """A downloader for the configured library.

        Credentials are checked at the first download, not here: a run whose files
        are all cached, or that indexes only notes, needs none.
        """
        downloader = cls(
            config.global_.zotero_library_type,
            config.global_.zotero_library_id,
            config.global_.zotero_api_key,
        )
        downloader._require = config.global_.require_zotero
        return downloader

    def file(self, itemkey: str, /) -> bytes:
        """The attachment's bytes. Raises `httpx.HTTPError` on failure."""
        self._require()
        url = f"{self._base}/items/{itemkey}/file"
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            response = self._client.get(url, headers=self._headers, follow_redirects=True)
            if response.status_code in _RETRY_STATUSES and attempt < _MAX_ATTEMPTS:
                wait = _retry_after(response, attempt)
                logger.info("Zotero asked to wait %.0fs before retrying %s", wait, itemkey)
                self._sleep(wait)
                continue
            response.raise_for_status()
            return _uncompressed(response)
        raise AssertionError("unreachable: the last attempt returns or raises")


def _retry_after(response: httpx.Response, attempt: int) -> float:
    for header in ("Retry-After", "Backoff"):
        value = response.headers.get(header)
        if value and value.isdigit():
            return float(value)
    return float(2**attempt)


def _uncompressed(response: httpx.Response) -> bytes:
    """A snapshot Zotero stored zipped arrives zipped; unwrap it as pyzotero does.

    The marker is on the API's redirect, not on the storage response.
    """
    compressed = any(
        hop.headers.get("Zotero-File-Compressed") == "Yes" for hop in (*response.history, response)
    )
    if not compressed:
        return response.content
    try:
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            return archive.read(archive.namelist()[0])
    except (zipfile.BadZipFile, IndexError) as exc:
        raise httpx.DecodingError(f"compressed attachment is not a usable zip: {exc}") from exc


# --- Writes (PRD-markdown, Phase 2) -------------------------------------------------


class UploadedAttachment(BaseModel):
    """An attachment as the server holds it after a write."""

    key: str
    version: int
    md5: str


def require_write_access(zot: zotero.Zotero, config: Config) -> None:
    """Fail before any work if the key cannot edit the library and upload files.

    Checked up front because a read-only key would otherwise fail halfway
    through a run, after the expensive downloads and conversions.
    """
    try:
        access = (zot.key_info() or {}).get("access", {})
    except NETWORK_ERRORS as exc:
        raise ZoteroApiError(f"Could not verify the Zotero API key: {exc}") from exc
    if config.global_.zotero_library_type == "group":
        groups = access.get("groups", {})
        rights = groups.get(config.global_.zotero_library_id) or groups.get("all") or {}
        allowed = bool(rights.get("write"))
    else:
        rights = access.get("user", {})
        allowed = bool(rights.get("write") and rights.get("files"))
    if not allowed:
        raise ZoteroWriteAccessError(
            "The Zotero API key cannot edit this library or upload files. Create a key "
            "with write access at https://www.zotero.org/settings/keys."
        )


def file_md5(path: Path) -> str:
    """The checksum Zotero records for an uploaded file."""
    return hashlib.md5(path.read_bytes()).hexdigest()  # noqa: S324 - Zotero's own checksum


def create_markdown_attachment(
    zot: zotero.Zotero,
    parent_key: str,
    path: Path,
    *,
    title: str,
    note: str,
    tags: list[str],
) -> UploadedAttachment:
    """Upload `path` as a new child attachment, with its metadata, and verify it.

    One call carries the title, type, tags and provenance note, so there is no
    moment when an untagged, unexplained file sits in the library.
    """
    template = zot.item_template("attachment", "imported_file")
    template.update(
        title=title,
        filename=path.name,
        contentType="text/markdown",
        tags=[{"tag": tag} for tag in tags],
        note=note,
    )
    try:
        result = zot.upload_attachments([template], parentid=parent_key, basedir=str(path.parent))
    except NETWORK_ERRORS as exc:
        raise _write_error(f"Could not upload {path.name}", exc) from exc
    return _verified(zot, _uploaded_key(result, path), path)


def replace_attachment_file(
    zot: zotero.Zotero, key: str, path: Path, *, expected_md5: str
) -> UploadedAttachment:
    """Put new contents in an existing attachment, keeping its key and metadata.

    `expected_md5` is the file as last seen. If anyone has changed it since,
    Zotero refuses (412) and this raises `ZoteroConflictError`: an edited
    transcript is never overwritten by accident.
    """
    payload = [{"key": key, "filename": path.name, "md5": expected_md5}]
    try:
        zot.upload_attachments(payload, basedir=str(path.parent))
    except NETWORK_ERRORS as exc:
        raise _write_error(f"Could not replace the file of {key}", exc) from exc
    return _verified(zot, key, path)


def update_attachment_note(zot: zotero.Zotero, key: str, version: int, note: str) -> int:
    """Rewrite an attachment's note; returns the item's new version."""
    return _patch(zot, {"key": key, "version": version, "note": note}, f"update {key}")


def trash_item(zot: zotero.Zotero, key: str, version: int) -> int:
    """Move an item to the Zotero trash; returns its new version.

    Never a permanent delete: the user empties the trash in Zotero, or restores
    from it. A trashed item disappears from `/items` and from its parent's
    children, though its file stays downloadable until the trash is emptied.
    """
    return _patch(zot, {"key": key, "version": version, "deleted": 1}, f"trash {key}")


def fetch_annotated_attachments(zot: zotero.Zotero) -> set[str]:
    """Keys of every attachment that carries annotations (highlights, comments).

    Annotations are children of the PDF, not of the paper, so the item listings
    used for planning do not include them. One paginated listing covers the
    library.
    """
    try:
        annotations = zot.everything(zot.items(itemType="annotation"))
    except NETWORK_ERRORS as exc:
        raise ZoteroApiError(f"Could not list annotations: {exc}") from exc
    return {
        str(entry["data"]["parentItem"])
        for entry in annotations
        if entry.get("data", {}).get("parentItem")
    }


def _uploaded_key(result: dict[str, list[dict[str, Any]]], path: Path) -> str:
    # "unchanged" means the server already held a file with this md5 and linked
    # it instead of storing another copy: the attachment has its file either way.
    done = result.get("success", []) + result.get("unchanged", [])
    if not done:
        reason = (result.get("failure") or [{}])[0].get("error", "no reason given")
        raise ZoteroApiError(f"Zotero refused the upload of {path.name}: {reason}")
    return str(done[0]["key"])


def _verified(zot: zotero.Zotero, key: str, path: Path) -> UploadedAttachment:
    """Re-read the attachment and check Zotero holds exactly the local file."""
    try:
        data = zot.item(key)["data"]
    except NETWORK_ERRORS as exc:
        raise ZoteroApiError(f"Could not re-read attachment {key}: {exc}") from exc
    local = file_md5(path)
    if data.get("md5") != local:
        raise ZoteroApiError(
            f"Attachment {key} does not hold the uploaded file "
            f"(Zotero has md5 {data.get('md5')}, the file is {local})"
        )
    return UploadedAttachment(key=key, version=int(data["version"]), md5=local)


def _patch(zot: zotero.Zotero, payload: dict[str, Any], action: str) -> int:
    try:
        zot.update_item(payload)
    except NETWORK_ERRORS as exc:
        raise _write_error(f"Could not {action}", exc) from exc
    # pyzotero's write decorator returns True and keeps the response on the client.
    response = getattr(zot, "request", None)
    version = getattr(response, "headers", {}).get("Last-Modified-Version")
    if version:
        return int(version)
    # Library versions jump by however many writes happened, so re-read, never guess.
    try:
        return int(zot.item(payload["key"])["version"])
    except NETWORK_ERRORS as exc:
        raise ZoteroApiError(f"Could not re-read {payload['key']}: {exc}") from exc


def _write_error(context: str, exc: Exception) -> ZoteroApiError:
    if isinstance(exc, PreConditionFailedError):
        return ZoteroConflictError(f"{context}: it changed in Zotero after it was read")
    if isinstance(exc, RequestEntityTooLargeError):
        return ZoteroApiError(f"{context}: the Zotero storage quota is full")
    return ZoteroApiError(f"{context}: {exc}")
