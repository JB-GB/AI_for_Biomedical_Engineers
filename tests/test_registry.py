import json

import pytest

from rausnet.registry import ModelEntry, Registry


def entry(version, parent=None):
    return ModelEntry(version, "2026-11-13T00:00:00", f"{version}/model.pt", "datasets/x/manifest.csv",
                      "abc", "S2", parent, test_metrics={"qwk": 0.9})


def test_promotion_lifecycle(tmp_path):
    reg = Registry(tmp_path)
    assert reg.current() is None and reg.next_version() == "v0001"
    reg.add(entry("v0001"), promote=True)
    reg.add(entry("v0002", "v0001"), promote=False)
    reg.add(entry("v0003", "v0001"), promote=True)
    reloaded = Registry(tmp_path)
    assert [e.status for e in reloaded.entries] == ["archived", "rejected", "current"]
    assert reloaded.current().version == "v0003"
    assert reloaded.next_version() == "v0004"


def test_duplicate_version_rejected(tmp_path):
    reg = Registry(tmp_path)
    reg.add(entry("v0001"), promote=True)
    with pytest.raises(ValueError):
        reg.add(entry("v0001"), promote=False)


def test_corrupt_registry_with_two_current(tmp_path):
    data = {"models": [dict(vars(entry("v0001")), status="current"),
                       dict(vars(entry("v0002")), status="current")]}
    (tmp_path / "registry.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="more than one current"):
        Registry(tmp_path)
