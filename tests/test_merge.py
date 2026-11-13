from dataclasses import replace

import pytest

from rausnet.data.manifest import Record, read_manifest, write_manifest
from rausnet.data.merge import load_source, merge_records, write_dataset


def make_source(root, name, ids):
    src = root / name
    (src / "gs").mkdir(parents=True)
    recs = []
    for i in ids:
        (src / "gs" / f"{i}.png").write_bytes(b"x")
        recs.append(Record(i, f"P-{i}", "MCP2", "L", f"gs/{i}.png", "", 1, 0, "hospital"))
    write_manifest(recs, src / "manifest.csv")
    return src / "manifest.csv"


def test_load_source_rebases_paths(tmp_path):
    m = make_source(tmp_path, "hosp/batch1", ["a"])
    recs = load_source(tmp_path, m)
    assert recs[0].gs_path == "hosp/batch1/gs/a.png"
    assert load_source(tmp_path, "hosp/batch1/manifest.csv") == recs


def test_missing_image_rejected(tmp_path):
    m = make_source(tmp_path, "s", ["a"])
    (tmp_path / "s/gs/a.png").unlink()
    with pytest.raises(FileNotFoundError):
        load_source(tmp_path, m)


def test_source_outside_root_rejected(tmp_path):
    m = make_source(tmp_path, "outside", ["a"])
    with pytest.raises(ValueError, match="outside the data root"):
        load_source(tmp_path / "root", m)


def test_merge_dedupes_and_detects_conflicts(tmp_path):
    a = load_source(tmp_path, make_source(tmp_path, "s1", ["a", "b"]))
    assert len(merge_records(a, a)) == 2
    with pytest.raises(ValueError, match="different content"):
        merge_records(a, [replace(a[0], gs_grade=3)])


def test_write_dataset_round_trip(tmp_path):
    recs = load_source(tmp_path, make_source(tmp_path, "s1", ["a"]))
    rel = write_dataset(tmp_path, "v0001", recs)
    assert read_manifest(tmp_path / rel) == recs
    with pytest.raises(FileExistsError):
        write_dataset(tmp_path, "v0001", recs)
