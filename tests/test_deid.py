import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, PngImagePlugin

from rausnet.data import deid

REPO = Path(__file__).resolve().parents[1]
PATIENT = "DOE_JOHN_123456"


@pytest.fixture
def salt(tmp_path):
    path = tmp_path / "secret" / "salt.txt"
    deid.make_salt(path)
    return deid.load_salt(path)


@pytest.fixture
def raw(tmp_path):
    d = tmp_path / "raw" / PATIENT
    d.mkdir(parents=True)
    rgb = np.random.default_rng(0).integers(0, 255, (100, 80, 3), np.uint8)
    info = PngImagePlugin.PngInfo()
    info.add_text("Comment", f"patient {PATIENT}")
    Image.fromarray(rgb).save(d / f"{PATIENT}_wrist.png", pnginfo=info)
    exif = Image.Exif()
    exif[0x010E] = f"patient {PATIENT}"  # ImageDescription
    exif[0x0132] = "2026:01:01 10:00:00"  # DateTime
    Image.fromarray(rgb).save(d / f"{PATIENT}_mcp2.jpg", exif=exif.tobytes())
    (d / "notes.txt").write_text(PATIENT)
    return tmp_path / "raw"


def _outputs(out):
    return sorted(p for p in out.rglob("*.png"))


def test_metadata_removed_and_names_pseudonymous(raw, salt, tmp_path):
    src_png = Image.open(next(raw.rglob("*.png")))
    src_jpg = Image.open(next(raw.rglob("*.jpg")))
    assert PATIENT in src_png.text["Comment"] and src_jpg.getexif()  # fixtures really carry metadata
    out = tmp_path / "out"
    report = deid.deidentify(raw, out, salt)
    files = _outputs(out)
    assert len(files) == 2 and report["counts"]["images_written"] == 2
    assert report["counts"]["skipped_unsupported"] == 1 and report["n_patients"] == 1
    for f in files:
        img = Image.open(f)
        assert not getattr(img, "text", {}) and not img.getexif() and "exif" not in img.info
        assert PATIENT not in f.read_bytes().decode("latin-1")
        assert PATIENT not in str(f.relative_to(out)) and "wrist" not in f.name and "mcp2" not in f.name
    audit = (out / deid.AUDIT_NAME).read_text()
    assert PATIENT not in audit and "wrist" not in audit and salt.decode() not in audit


def test_crop_and_blackout(raw, salt, tmp_path):
    preset = deid.Preset("t", crop=(10, 5, 0.25, 0), blackout=((0, 0, 1000, 20),))
    deid.deidentify(raw, tmp_path / "out", salt, preset)
    for f in _outputs(tmp_path / "out"):
        a = np.asarray(Image.open(f))
        assert a.shape[:2] == (100 - 10 - 5, 80 - 20)
        assert a[:10].max() == 0 and a[10:].max() > 0  # original rows 10..19 blacked out, now rows 0..9
    with pytest.raises(ValueError):
        deid.apply_preset(Image.new("L", (20, 20)), deid.Preset(crop=(0.5, 0.5, 0, 0)))


def test_preset_yaml_and_cli(raw, tmp_path, capsys):
    pytest.importorskip("yaml")
    preset = deid.load_preset(REPO / "configs" / "deid" / "devices.yaml")
    assert preset.name == "default" and preset.crop[0] > 0
    with pytest.raises(ValueError, match="not found"):
        deid.load_preset(REPO / "configs" / "deid" / "devices.yaml", "no_such_device")
    salt_file = tmp_path / "s.txt"
    deid.main(["--make-salt", str(salt_file)])
    deid.main(["--in", str(raw), "--out", str(tmp_path / "out"), "--salt-file", str(salt_file),
               "--crop-config", str(REPO / "configs" / "deid" / "devices.yaml"), "--crop", "0,0,0,0"])
    printed = capsys.readouterr().out
    assert salt_file.read_text().strip() not in printed and PATIENT not in printed
    report = json.loads((tmp_path / "out" / deid.AUDIT_NAME).read_text())
    assert report["preset"]["name"] == "default+cli"
    assert Image.open(_outputs(tmp_path / "out")[0]).size == (80, 100)


def test_pseudonym_deterministic_and_salted(salt, tmp_path):
    p = deid.pseudonymise(PATIENT, salt)
    assert p == deid.pseudonymise(f" {PATIENT.lower()} ", salt)
    assert p != deid.pseudonymise("OTHER", salt) and PATIENT not in p
    deid.make_salt(tmp_path / "other.txt")
    assert p != deid.pseudonymise(PATIENT, deid.load_salt(tmp_path / "other.txt"))


def test_rerun_is_stable(raw, salt, tmp_path):
    names = []
    for sub in ("a", "b"):
        deid.deidentify(raw, tmp_path / sub, salt)
        names.append([p.relative_to(tmp_path / sub) for p in _outputs(tmp_path / sub)])
    assert names[0] == names[1]


def test_refuses_repo_paths(raw, salt, tmp_path):
    with pytest.raises(ValueError, match="git"):
        deid.deidentify(raw, REPO / "data" / "deid_out", salt)
    assert not (REPO / "data" / "deid_out").exists()
    with pytest.raises(ValueError, match="git"):
        deid.make_salt(REPO / "salt.txt")
    with pytest.raises(ValueError, match="at least"):
        (tmp_path / "short.txt").write_text("abc")
        deid.load_salt(tmp_path / "short.txt")
    with pytest.raises(SystemExit):
        deid.main(["--in", str(raw), "--out", str(tmp_path / "o")])


def test_dicom_without_pydicom_is_clear(tmp_path, salt):
    if importlib.util.find_spec("pydicom") is not None:
        pytest.skip("pydicom installed")
    d = tmp_path / "raw" / PATIENT
    d.mkdir(parents=True)
    (d / "IM0001").write_bytes(b"\0" * 128 + b"DICM" + b"\0" * 64)
    with pytest.raises(deid.MissingDependencyError, match="pydicom"):
        deid.deidentify(tmp_path / "raw", tmp_path / "out", salt)
