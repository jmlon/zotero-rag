"""The store fingerprint: what it records, and which changes it refuses.

The distinction under test is corruption versus inconsistency. Re-embedding with a
different model mixes two incompatible vector spaces, which retrieval cannot
detect — so it must refuse. A new `chunk_size` only means old chunks are shaped
differently from new ones, which is worth a warning and nothing more.
"""

from __future__ import annotations

from zotero_rag.config import Config
from zotero_rag_core.fingerprint import (
    METADATA_PREFIX,
    Severity,
    StoreFingerprint,
    advisory,
    blocking,
)


def config_with(**index_values: object) -> Config:
    config = Config()
    for name, value in index_values.items():
        setattr(config.index, name, value)
    return config


def test_intended_reads_the_effective_settings() -> None:
    config = config_with(chunk_size=800, chunk_overlap=100, embedding_model="acme/embed")
    fingerprint = StoreFingerprint.intended(config, embedding_dim=1536)
    assert fingerprint.embedding_model == "acme/embed"
    assert fingerprint.embedding_dim == 1536
    assert fingerprint.chunk_size == 800
    assert fingerprint.chunk_overlap == 100
    # The host, not the whole URL: a path or scheme change is not a model change.
    assert fingerprint.endpoint_host == "openrouter.ai"


def test_inherits_the_global_default_model() -> None:
    """`index.embedding_model: null` means "inherit", and the fact recorded is the
    model actually used, not the null."""
    fingerprint = StoreFingerprint.intended(Config())
    assert fingerprint.embedding_model == "openai/text-embedding-3-small"


def test_metadata_round_trip_is_lossless() -> None:
    original = StoreFingerprint.intended(config_with(chunk_size=750), embedding_dim=3)
    restored = StoreFingerprint.from_metadata(original.to_metadata())
    assert restored == original


def test_metadata_keys_are_namespaced() -> None:
    """Chroma reserves `hnsw:*`; ours must not collide with its own settings."""
    keys = StoreFingerprint.intended(Config()).to_metadata()
    assert all(key.startswith(METADATA_PREFIX) for key in keys)


def test_absent_or_foreign_metadata_reads_as_no_fingerprint() -> None:
    assert StoreFingerprint.from_metadata(None) is None
    assert StoreFingerprint.from_metadata({}) is None
    # A store Chroma configured but we never fingerprinted.
    assert StoreFingerprint.from_metadata({"hnsw:space": "l2"}) is None


def test_unreadable_metadata_degrades_rather_than_crashing() -> None:
    broken = {f"{METADATA_PREFIX}embedding_dim": "not-a-number"}
    assert StoreFingerprint.from_metadata(broken) is None


def test_identical_settings_produce_no_mismatch() -> None:
    recorded = StoreFingerprint.intended(Config(), embedding_dim=1536)
    assert recorded.compare(StoreFingerprint.intended(Config(), embedding_dim=1536)) == []


def test_a_changed_model_refuses() -> None:
    recorded = StoreFingerprint.intended(config_with(embedding_model="old/model"))
    mismatches = recorded.compare(
        StoreFingerprint.intended(config_with(embedding_model="new/model"))
    )
    assert [m.severity for m in mismatches] == [Severity.REFUSE]
    assert blocking(mismatches)
    assert "old/model" in str(mismatches[0]) and "new/model" in str(mismatches[0])


def test_a_changed_dimension_refuses() -> None:
    recorded = StoreFingerprint.intended(Config(), embedding_dim=1536)
    mismatches = recorded.compare(StoreFingerprint.intended(Config(), embedding_dim=3072))
    assert [m.setting for m in blocking(mismatches)] == ["embedding dimension"]


def test_an_unknown_dimension_is_not_a_mismatch() -> None:
    """A fingerprint adopted from an empty store records dimension 0."""
    recorded = StoreFingerprint.intended(Config(), embedding_dim=0)
    assert recorded.compare(StoreFingerprint.intended(Config(), embedding_dim=1536)) == []


def test_changed_chunking_only_warns() -> None:
    """Old chunks stay as they were: inconsistent, but every vector is still valid."""
    recorded = StoreFingerprint.intended(config_with(chunk_size=1000, chunk_overlap=200))
    mismatches = recorded.compare(
        StoreFingerprint.intended(config_with(chunk_size=500, chunk_overlap=50))
    )
    assert not blocking(mismatches)
    assert {m.setting for m in advisory(mismatches)} == {"chunk_size", "chunk_overlap"}


def test_a_changed_endpoint_warns() -> None:
    recorded = StoreFingerprint.intended(Config())
    other = Config()
    other.global_.endpoint = "https://api.openai.com/v1"
    mismatches = recorded.compare(StoreFingerprint.intended(other))
    assert not blocking(mismatches)
    assert [m.setting for m in advisory(mismatches)] == ["endpoint"]


def test_a_model_change_reports_alongside_a_chunking_change() -> None:
    """The refusal must not hide the other differences the user also made."""
    recorded = StoreFingerprint.intended(config_with(embedding_model="a/b", chunk_size=1000))
    mismatches = recorded.compare(
        StoreFingerprint.intended(config_with(embedding_model="c/d", chunk_size=400))
    )
    assert len(blocking(mismatches)) == 1
    assert len(advisory(mismatches)) == 1
