from __future__ import annotations

import errno
import importlib.util
from pathlib import Path

import pytest


_SCRIPT = Path(__file__).parents[1] / "scripts" / "build_autodl_release.py"
_SPEC = importlib.util.spec_from_file_location("build_autodl_release", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_stage_file_uses_hardlink_on_same_filesystem(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    destination = tmp_path / "destination.bin"
    source.write_bytes(b"legume" * 100)
    assert _MODULE.stage_file(source, destination) == "hardlink"
    assert source.stat().st_ino == destination.stat().st_ino
    assert destination.read_bytes() == source.read_bytes()


def test_stage_file_falls_back_to_independent_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.bin"
    destination = tmp_path / "destination.bin"
    source.write_bytes(b"soybean" * 100)

    def cross_device(*args: object, **kwargs: object) -> None:
        raise OSError(errno.EXDEV, "cross-device link")

    monkeypatch.setattr(_MODULE.os, "link", cross_device)
    monkeypatch.setattr(_MODULE, "_reflink", lambda *_: False)
    assert _MODULE.stage_file(source, destination) == "copy"
    assert source.stat().st_ino != destination.stat().st_ino
    assert destination.read_bytes() == source.read_bytes()


def test_stage_file_rejects_symlink_source(tmp_path: Path) -> None:
    target = tmp_path / "target.bin"
    source = tmp_path / "source.bin"
    target.write_bytes(b"x")
    source.symlink_to(target)
    with pytest.raises(ValueError, match="regular file"):
        _MODULE.stage_file(source, tmp_path / "destination.bin")
