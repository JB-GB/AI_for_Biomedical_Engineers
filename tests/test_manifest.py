import pytest

from rausnet.data.manifest import Record, read_manifest, write_manifest


def make(image_id="img1", patient_id="p1", **kw):
    base = {
        "image_id": image_id, "patient_id": patient_id, "joint": "MCP2", "side": "R",
        "gs_path": "gs/img1.png", "pd_path": "pd/img1.png", "gs_grade": 2, "pd_grade": 1,
        "source": "synthetic",
    }
    base.update(kw)
    return Record(**base)


def test_round_trip(tmp_path):
    records = [make(), make("img2", "p2", gs_grade=0, pd_grade=0, pd_path="")]
    path = tmp_path / "manifest.csv"
    write_manifest(records, path)
    assert read_manifest(path) == records
    assert records[0].combined_grade == 2


@pytest.mark.parametrize(
    "kw",
    [{"joint": "KNEE"}, {"side": "X"}, {"source": "web"}, {"gs_grade": 5},
     {"gs_grade": 0, "pd_grade": 2}, {"gs_path": ""}, {"patient_id": ""}],
)
def test_invalid_record(kw):
    with pytest.raises(ValueError):
        make(**kw)


def test_absolute_path_rejected(tmp_path):
    with pytest.raises(ValueError):
        make(gs_path=str(tmp_path / "a.png"))


def test_duplicate_image_id(tmp_path):
    path = tmp_path / "manifest.csv"
    write_manifest([make(), make()], path)
    with pytest.raises(ValueError, match="duplicate"):
        read_manifest(path)


def test_bad_row_reports_line(tmp_path):
    path = tmp_path / "manifest.csv"
    write_manifest([make()], path)
    path.write_text(path.read_text().replace("MCP2", "KNEE"))
    with pytest.raises(ValueError, match=":2:"):
        read_manifest(path)
