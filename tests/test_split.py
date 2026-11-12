import pytest

from rausnet.data.manifest import Record
from rausnet.data.split import assign_split, split_records


def rec(image_id, patient_id):
    return Record(image_id, patient_id, "WRIST", "L", f"gs/{image_id}.png", "", 1, 0, "synthetic")


def test_deterministic():
    assert assign_split("patient-42") == assign_split("patient-42")


def test_no_patient_in_two_splits():
    records = [rec(f"{p}-{i}", f"p{p}") for p in range(200) for i in range(5)]
    groups = split_records(records)
    patients = [{r.patient_id for r in g} for g in groups.values()]
    assert not (patients[0] & patients[1] or patients[0] & patients[2] or patients[1] & patients[2])
    assert sum(len(g) for g in groups.values()) == len(records)


def test_ratios_approximate():
    counts = {"train": 0, "val": 0, "test": 0}
    for p in range(20_000):
        counts[assign_split(f"p{p}")] += 1
    assert abs(counts["train"] / 20_000 - 0.70) < 0.02
    assert abs(counts["test"] / 20_000 - 0.15) < 0.02


def test_test_set_frozen_when_data_grows():
    old = {f"p{p}": assign_split(f"p{p}") for p in range(500)}
    for p in range(500, 1500):
        assign_split(f"p{p}")
    assert all(assign_split(pid) == split for pid, split in old.items())


def test_salt_changes_assignment():
    ids = [f"p{p}" for p in range(100)]
    assert [assign_split(i) for i in ids] != [assign_split(i, salt="other") for i in ids]


@pytest.mark.parametrize("ratios", [(0.5, 0.5, 0.5), (1.0, 0.0), (-0.1, 0.6, 0.5)])
def test_invalid_ratios(ratios):
    with pytest.raises(ValueError):
        assign_split("p1", ratios)
