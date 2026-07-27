"""Pinning the collection into config.yaml.

Pinning rewrites a file the user is expected to edit by hand, so the comments
`init` wrote must survive it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from zotero_rag.config import Config, load_config, packaged_defaults, pin_collection
from zotero_rag.errors import ConfigError
from zotero_rag.scaffold import init_index_folder


@pytest.fixture
def index_dir(tmp_path: Path) -> Path:
    init_index_folder(tmp_path, Config.model_validate(packaged_defaults()).global_)
    return tmp_path


def test_pinning_is_visible_to_the_loader(index_dir: Path) -> None:
    pin_collection(
        index_dir,
        collection_key="TGXVGXGP",
        collection_name="Writing",
        collection_path="Writing",
    )
    config = load_config(index_dir)
    assert config.index.is_pinned
    assert config.index.collection_key == "TGXVGXGP"
    assert config.index.collection_name == "Writing"
    assert config.index.collection_path == "Writing"


def test_comments_survive_pinning(index_dir: Path) -> None:
    before = (index_dir / "config.yaml").read_text(encoding="utf-8")
    assert "#" in before  # the template is commented

    pin_collection(index_dir, collection_key="K", collection_name="N", collection_path=None)
    after = (index_dir / "config.yaml").read_text(encoding="utf-8")

    for comment in [line for line in before.splitlines() if line.strip().startswith("#")]:
        assert comment in after


def test_other_settings_are_untouched(index_dir: Path) -> None:
    (index_dir / "config.yaml").write_text(
        "index:\n  chunk_size: 321\n  top_k: 9\n", encoding="utf-8"
    )
    pin_collection(index_dir, collection_key="K", collection_name="N", collection_path=None)
    config = load_config(index_dir)
    assert config.index.chunk_size == 321
    assert config.index.top_k == 9
    assert config.index.collection_key == "K"


def test_a_null_path_is_written_as_yaml_null(index_dir: Path) -> None:
    """Pinning by key leaves no path; it must not become the string 'None'."""
    pin_collection(index_dir, collection_key="K", collection_name="N", collection_path=None)
    assert load_config(index_dir).index.collection_path is None


def test_pinning_twice_overwrites_rather_than_duplicates(index_dir: Path) -> None:
    pin_collection(index_dir, collection_key="FIRST", collection_name="One", collection_path=None)
    pin_collection(index_dir, collection_key="SECOND", collection_name="Two", collection_path=None)

    text = (index_dir / "config.yaml").read_text(encoding="utf-8")
    assert text.count("collection_key:") == 1
    assert load_config(index_dir).index.collection_key == "SECOND"


def test_missing_keys_are_inserted(tmp_path: Path) -> None:
    """The user may have deleted the null placeholders from their config."""
    (tmp_path / "config.yaml").write_text("index:\n  chunk_size: 500\n", encoding="utf-8")
    pin_collection(tmp_path, collection_key="K", collection_name="N", collection_path="A/B")
    config = load_config(tmp_path)
    assert config.index.collection_key == "K"
    assert config.index.collection_path == "A/B"
    assert config.index.chunk_size == 500


def test_config_without_an_index_block_gains_one(tmp_path: Path) -> None:
    (tmp_path / "config.yaml").write_text(
        "global:\n  zotero_library_type: user\n", encoding="utf-8"
    )
    pin_collection(tmp_path, collection_key="K", collection_name="N", collection_path=None)
    config = load_config(tmp_path)
    assert config.index.collection_key == "K"
    assert config.global_.zotero_library_type == "user"


def test_names_needing_quotes_round_trip(index_dir: Path) -> None:
    """A name with a colon would break naive YAML emission."""
    tricky = "Writing: tips & tricks #1"
    pin_collection(index_dir, collection_key="K", collection_name=tricky, collection_path=None)
    assert load_config(index_dir).index.collection_name == tricky


def test_pinning_without_a_config_file_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        pin_collection(tmp_path, collection_key="K", collection_name="N", collection_path=None)


def test_no_temporary_file_is_left_behind(index_dir: Path) -> None:
    pin_collection(index_dir, collection_key="K", collection_name="N", collection_path=None)
    assert not (index_dir / "config.yaml.tmp").exists()
