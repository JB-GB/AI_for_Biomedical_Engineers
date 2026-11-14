"""Static checks of the deployment files (no Docker or PyInstaller needed)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _read(rel: str) -> str:
    path = ROOT / rel
    if not path.exists():
        pytest.fail(f"{rel} is missing")
    return path.read_text(encoding="utf-8")


def _code_lines(text: str) -> list[str]:
    return [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#")]


def test_dockerfile_serves_without_torch():
    lines = _code_lines(_read("Dockerfile"))
    text = "\n".join(lines)
    assert re.search(r"^FROM python:3\.1[12]-slim", text, re.M)
    assert '".[serve]"' in text or "'.[serve]'" in text or ".[serve]" in text
    for forbidden in ("torch", "torchvision", "timm", "[train]", "nvidia", "cuda"):
        assert forbidden not in text.lower(), f"Dockerfile must not install {forbidden}"


def test_dockerfile_runs_unprivileged_and_does_not_bake_models():
    lines = _code_lines(_read("Dockerfile"))
    users = [ln for ln in lines if ln.startswith("USER ")]
    assert users and users[-1] not in ("USER root", "USER 0")
    copies = [ln for ln in lines if ln.startswith(("COPY", "ADD"))]
    assert not any("models" in ln or ".onnx" in ln for ln in copies)
    assert any(ln.startswith("HEALTHCHECK") for ln in lines)
    assert "/health" in "\n".join(lines)


def test_compose_binds_localhost_and_mounts_models_read_only():
    text = _read("docker-compose.yml")
    ports = re.findall(r'^\s*-\s*"?([^"\n]*:8000)"?\s*$', text, re.M)
    assert ports, "no port mapping for 8000"
    assert all(p.startswith("127.0.0.1:") for p in ports), ports
    assert "${RAUSNET_MODELS_DIR:-./models}" in text
    assert re.search(r"read_only:\s*true", text) or ":/models:ro" in text
    assert "restart: unless-stopped" in text


def test_dockerignore_excludes_private_and_heavy_paths():
    entries = {ln.strip().strip("/") for ln in _code_lines(_read(".dockerignore"))}
    for required in ("models", "data", ".venv", ".git", "outputs"):
        assert required in entries, f".dockerignore must exclude {required}"


def test_pyinstaller_spec_does_not_bundle_models():
    spec = _read("packaging/rausnet_desktop.spec")
    code = "\n".join(_code_lines(spec))
    assert "onnxruntime" in code and "PySide6" in code
    assert "COLLECT(" in code, "expected a one-folder build"
    assert "exclude_binaries=True" in code
    assert "models" not in code, "the model registry must not be bundled"
    assert ".onnx" not in code and "registry.json" not in code
    assert '"torch"' in code  # listed in excludes
    assert (ROOT / "packaging" / "rausnet_desktop_entry.py").exists()
    assert (ROOT / "scripts" / "build_installer.py").exists()


def test_build_outputs_are_git_ignored():
    entries = {ln.strip().strip("/") for ln in _code_lines(_read(".gitignore"))}
    assert {"build", "dist", "models"} <= entries
