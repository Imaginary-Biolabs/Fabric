"""Phase 2 toll station — dataset release."""

from __future__ import annotations

import json
from pathlib import Path

import grumpy as gr
import pytest

import fabric
from fabric import Settings
from fabric.core.externals.local import LocalAdapter
from fabric.core.factory import Factory
from fabric.core.parsers.pdb import PdbParser
from fabric.utils import constants
from fabric.utils.config import load_config
from fabric.utils.errors import ConfigError, DatasetError, ExternalError, ParseError
from fabric.utils.hashing import config_hash


def test_pdb_parser_mini_pdb() -> None:
    parser = PdbParser()
    data = parser.parse(Path("tests/fixtures/structures/mini.pdb"))
    assert len(data.data) == 1
    assert len(data.data.molecule_id) == 1


def test_pdb_parser_bad_file() -> None:
    parser = PdbParser()
    with pytest.raises(ParseError, match="No ATOM records"):
        parser.parse(Path("tests/fixtures/structures/bad.pdb"))


def test_config_external_and_parent_mutually_exclusive(tmp_path: Path) -> None:
    cfg_path = tmp_path / "both.yaml"
    cfg_path.write_text(
        "external:\n  LocalAdapter:\n    paths: ['*.pdb']\nparent: D_parent\n"
    )
    with pytest.raises(ConfigError, match="both"):
        Factory.dataset(cfg_path)


def test_local_adapter_empty_glob(tmp_path: Path) -> None:
    adapter = LocalAdapter(["no_such_files/*.pdb"], config_dir=tmp_path)
    with pytest.raises(ExternalError, match="matched no files"):
        adapter.load()


def test_local_adapter_batches_by_chunk_size(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(constants, "GRUMPY_CHUNK_SIZE", 3)
    adapter = LocalAdapter(
        [
            "tests/fixtures/structures/mini.pdb",
            "tests/fixtures/structures/mol_*.pdb",
        ],
        config_dir=Path("tests/fixtures/datasets"),
    )
    batches, assets, splits = adapter.load()
    batch_list = list(batches)
    assert len(batch_list) == 4
    assert sum(len(batch.data.data) for batch in batch_list) == 12
    assert len(assets.data["files"]) == 12
    assert splits.partitions == {}


def test_local_adapter_parallel_workers(tmp_path: Path) -> None:
    with Settings(home=tmp_path, workers=2):
        adapter = LocalAdapter(
            [
                "tests/fixtures/structures/mini.pdb",
                "tests/fixtures/structures/mol_*.pdb",
            ],
            config_dir=Path("tests/fixtures/datasets"),
        )
        batches, assets, _ = adapter.load()
        batch_list = list(batches)
        assert len(batch_list) == 1
        assert len(batch_list[0].data.data) == 12
        assert len(assets.data["files"]) == 12


def test_dataset_release_and_cache_hit(tmp_path: Path) -> None:
    config_path = Path("tests/fixtures/datasets/D_local.yaml")
    with Settings(home=tmp_path):
        ds = Factory.dataset(config_path)
        path = ds.release()
        assert path == ds.cache_path
        assert path.parents[2] == Settings.transformed_path()
        assert (path / "grumpy.json").exists()
        assert (ds.processed_cache_path / "grumpy.json").exists()
        meta = json.loads((path / "release.json").read_text())
        assert meta["stage"] == "transformed"
        assert meta["config_hash"] == ds.config_hash
        assert meta["ingest_hash"] == ds.ingest_hash
        assert meta["transform_hash"] == ds.hash
        assert meta["grumpy_version"] == gr.__version__
        assert meta["fabric_version"] == fabric.__version__
        assert meta["molecules"] == 12

        processed_meta = json.loads((ds.processed_cache_path / "release.json").read_text())
        assert processed_meta["stage"] == "processed"
        assert processed_meta["ingest_hash"] == ds.ingest_hash
        assert processed_meta["molecules"] == 12

        ds2 = Factory.dataset(config_path)
        path2 = ds2.release()
        assert path2 == path
        assert len(ds2.data.data) > 0
        assert "random_8_2_2" in ds2.splits.partitions
        assert ds2.release_metadata()["transform_hash"] == ds2.hash


def test_dataset_stage_to_scratch(tmp_path: Path) -> None:
    scratch = tmp_path / "scratch"
    with Settings(home=tmp_path / "home", scratch=scratch):
        ds = Factory.dataset("tests/fixtures/datasets/D_local.yaml")
        ds.release()
        staged = ds.stage_to_scratch()
        assert staged == ds.scratch_path
        assert (staged / "grumpy.json").exists()
        assert ds.stage_to_scratch() == staged


def test_identical_config_has_identical_hash() -> None:
    cfg_a = load_config("tests/fixtures/datasets/D_local.yaml")
    cfg_b = load_config("tests/fixtures/datasets/D_local.yaml")
    ds_a = Factory.dataset("tests/fixtures/datasets/D_local.yaml")
    ds_b = Factory.dataset("tests/fixtures/datasets/D_local.yaml")
    assert config_hash(cfg_a) == config_hash(cfg_b)
    assert ds_a.hash == ds_b.hash


def test_parent_dataset_requires_release(tmp_path: Path) -> None:
    with Settings(home=tmp_path):
        child = Factory.dataset("tests/fixtures/datasets/D_child.yaml")
        with pytest.raises(DatasetError, match="Parent dataset 'D_parent' is not released"):
            child.release()


def test_parent_dataset_release_chain(tmp_path: Path) -> None:
    with Settings(home=tmp_path):
        parent = Factory.dataset("tests/fixtures/datasets/D_parent.yaml")
        parent.release()
        child = Factory.dataset("tests/fixtures/datasets/D_child.yaml")
        child_path = child.release()
        assert (child_path / "release.json").exists()
        meta = child.release_metadata()
        assert meta["parent"] == "D_parent"
        assert meta["molecules"] == 2
