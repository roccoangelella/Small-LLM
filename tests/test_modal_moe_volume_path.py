"""Resolve Modal volume mount aliases before checking containment."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_dataset_volume_accepts_mount_symlink_and_rejects_escape(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('modal_moe_path_test', Path(__file__).parents[1] / 'modal/moe_production_launch.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    actual = tmp_path / 'mounted-data'
    actual.mkdir()
    mount = tmp_path / 'data'
    mount.symlink_to(actual, target_is_directory=True)
    marker = object()
    monkeypatch.setattr(module, 'DATA_ROOT', mount)
    monkeypatch.setattr(module, 'DATA_VOLUME', marker)
    assert module._dataset_volume(SimpleNamespace(dataset_dir=str(mount / 'corpus'))) is marker
    escaped = actual / 'escape'
    escaped.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match='mounted'):
        module._dataset_volume(SimpleNamespace(dataset_dir=str(escaped / 'outside')))
