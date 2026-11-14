import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "eda.py"


def _eda():
    pytest.importorskip("matplotlib")
    spec = importlib.util.spec_from_file_location("eda", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_colour_fraction():
    eda = _eda()
    rgb = np.full((10, 10, 3), 120, np.uint8)
    rgb[:2] = (255, 120, 10)
    assert eda.colour_fraction(rgb) == pytest.approx(0.2)


def test_eda_on_synthetic(synth_small, tmp_path):
    eda = _eda()
    out = tmp_path / "eda"
    eda.main(["--manifest", str(synth_small / "manifest.csv"), "--root", str(synth_small), "--out", str(out)])
    summary = json.loads((out / "eda_summary.json").read_text())
    assert summary["n_patients"] == 2 and summary["images"]["missing_files"] == 0
    assert sum(summary["class_balance"]["all"]["gs"].values()) == summary["n_records"]
    assert summary["images"]["sizes"] == {"96x96": summary["n_records"]}
    frac = summary["pd_colour_fraction_by_pd_grade"]
    if frac["0"]["n"]:
        assert frac["0"]["max"] < 0.01
    assert all((out / n).is_file() for n in ("eda_class_balance.png", "eda_pd_colour.png",
                                              "eda_images_per_patient.png"))
