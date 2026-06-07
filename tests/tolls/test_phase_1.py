"""Phase 1 toll station — repository setup."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import fabric
from fabric import Settings
from fabric.core.data import Batch, Data
from fabric.core.transform import Compose
from fabric.core.transforms import Identity
from fabric.utils.config import load_config
from fabric.utils.errors import (
    ConfigurationError,
    PlatformExtraRequired,
    SchemaError,
    TransformError,
)
from fabric.utils.hashing import config_hash


def test_import_and_version() -> None:
    assert re.match(r"^\d+\.\d+\.\d+", fabric.__version__)


def test_settings_default_home() -> None:
    assert Settings.dataset_path == Path("~/.imaginary")
    assert Settings.home == Path("~/.imaginary")


def test_settings_context_override(tmp_path: Path) -> None:
    with Settings(home=tmp_path):
        assert Settings.dataset_path == tmp_path
        assert Settings.home == tmp_path
    assert Settings.dataset_path == Path("~/.imaginary")


def test_settings_nested_context_raises(tmp_path: Path) -> None:
    with Settings(home=tmp_path):
        with pytest.raises(ConfigurationError, match="already active"):
            with Settings(home=tmp_path / "nested"):
                pass


def test_config_load_and_hash() -> None:
    fixture = Path("tests/fixtures/configs/D_mini.yaml")
    cfg = load_config(fixture)
    assert cfg.external.LocalAdapter.paths[0].endswith("*.pdb")

    h1 = config_hash(cfg)
    h2 = config_hash(cfg)
    assert h1 == h2
    assert h1.startswith("sha256:")

    cfg_override = load_config(fixture, overrides={"transforms": [{"Identity": {}}]})
    assert config_hash(cfg_override) != h1


def test_config_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="not found"):
        load_config(tmp_path / "missing.yaml")


def test_compose_requires_transforms() -> None:
    with pytest.raises(TransformError, match="at least one"):
        Compose([])


def test_compose_hash_order() -> None:
    chain_a = Compose([Identity()])
    chain_b = Compose([Identity(), Identity()])
    assert chain_a.transform_hash == Identity().transform_hash
    assert chain_b.transform_hash != chain_a.transform_hash
    assert "Identity" in repr(chain_a)


def test_platform_extra_required(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real_import = builtins.__import__

    def mock_import(name: str, *args, **kwargs):
        if name == "httpx":
            raise ImportError("httpx not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    import fabric.platform as platform_mod

    with pytest.raises(PlatformExtraRequired, match="platform"):
        _ = platform_mod.registry


def test_data_grumpy_integration() -> None:
    grumpy = pytest.importorskip("grumpy")
    df = grumpy.dataframe({"value": [1, 2, 3]}, schema=["molecule"])
    data = Data(df)
    assert data.molecule is not None
    batch = Batch(data=data)
    assert batch.data is data

    with pytest.raises(TypeError, match="GrumpyDataFrame"):
        Data({"not": "grumpy"})

    with pytest.raises(SchemaError, match="atom"):
        _ = Data(df).atom
