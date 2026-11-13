"""Notebook hygiene: committed notebooks carry no outputs (patient images could leak) and no secrets."""

import json
import re
from pathlib import Path

import pytest

NOTEBOOKS = sorted(Path(__file__).resolve().parents[1].glob("notebooks/*.ipynb"))
SECRET = re.compile(r"(ghp_|github_pat_|hf_)[A-Za-z0-9_]{10,}|api[_-]?key\s*=\s*['\"]\w", re.I)


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_is_clean(path):
    nb = json.loads(path.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        src = "".join(cell["source"])
        assert not SECRET.search(src), f"possible secret in {path.name}"
        assert "share=True" not in src
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None, f"strip outputs of {path.name}"


def test_colab_notebook_present():
    assert any(p.name == "colab_train.ipynb" for p in NOTEBOOKS)
