"""De-identification of raw hospital ultrasound exports before they enter the data root.

Usage:
    python -m rausnet.data.deid --make-salt PATH          # once, PATH outside any git repo
    python -m rausnet.data.deid --in RAW_DIR --out OUT_DIR --salt-file PATH
        [--device NAME --crop-config configs/deid/devices.yaml] [--crop top,bottom,left,right]
        [--blackout x0,y0,x1,y1 ...]

Input layout: ``RAW_DIR/<patient id>/**/<image>``. For DICOM the PatientID tag is used
when present, else the first-level folder name. Supported: DICOM (needs the optional
``pydicom`` package), PNG and JPEG.

What it does:
- re-encodes pixels only (no EXIF, PNG text chunks, ICC profile or DICOM header survive);
- crops device margins with burned-in text and blacks out extra rectangles (blackout
  coordinates refer to the original image, before cropping);
- replaces patient IDs with ``H`` + HMAC-SHA256(salt, id)[:12]; output names are
  ``<pseudonym>/<pseudonym>_<HMAC(salt, relative path)[:10]>.png`` so neither IDs nor
  original file names leak, and re-runs are stable;
- writes ``deid_audit.json`` with counts and settings only (no identifiers, no mapping).
The salt is never logged or written. OUT_DIR and the salt file must be outside any git repo.
Crop/blackout values below 1 are fractions of the image side, otherwise pixels.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import secrets
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
DICOM_EXTS = {".dcm", ".dicom"}
MIN_SALT_BYTES = 16
AUDIT_NAME = "deid_audit.json"


class MissingDependencyError(RuntimeError):
    """An optional package needed for this input is not installed."""


@dataclass(frozen=True)
class Preset:
    name: str = "default"
    crop: tuple[float, float, float, float] = (0, 0, 0, 0)  # top, bottom, left, right
    blackout: tuple[tuple[float, float, float, float], ...] = ()  # (x0, y0, x1, y1)


@dataclass
class Audit:
    preset: Preset
    counts: Counter = field(default_factory=Counter)
    patients: set = field(default_factory=set)  # pseudonyms only, never written (only counted)
    outputs: list = field(default_factory=list)  # pseudonymous output paths


# --- secrets and pseudonyms -------------------------------------------------------------------

def find_git_root(path: str | Path) -> Path | None:
    """Return the enclosing git work tree of ``path`` (existing or not), or None."""
    p = Path(path).expanduser().resolve()
    for candidate in (p, *p.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def make_salt(path: str | Path) -> None:
    """Create a new random salt file; refuses to overwrite or to write inside a git repo."""
    path = Path(path)
    if find_git_root(path) is not None:
        raise ValueError("the salt file must be kept outside any git repository")
    if path.exists():
        raise FileExistsError("salt file already exists; refusing to overwrite it")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(secrets.token_hex(32) + "\n")


def load_salt(path: str | Path) -> bytes:
    path = Path(path)
    if find_git_root(path) is not None:
        raise ValueError("the salt file must be kept outside any git repository")
    salt = path.read_bytes().strip()
    if len(salt) < MIN_SALT_BYTES:
        raise ValueError(f"salt file must contain at least {MIN_SALT_BYTES} bytes")
    return salt


def _hmac_hex(salt: bytes, text: str) -> str:
    return hmac.new(salt, text.encode("utf-8"), hashlib.sha256).hexdigest()


def pseudonymise(patient_id: str, salt: bytes) -> str:
    """Deterministic, salted pseudonym; IDs are stripped and upper-cased first."""
    pid = patient_id.strip().upper()
    if not pid:
        raise ValueError("empty patient id")
    return "H" + _hmac_hex(salt, f"patient:{pid}")[:12].upper()


# --- presets ------------------------------------------------------------------------------------

def _rect(values) -> tuple[float, float, float, float]:
    vals = tuple(float(v) for v in values)
    if len(vals) != 4 or any(v < 0 for v in vals):
        raise ValueError(f"expected four non-negative numbers, got {values!r}")
    return vals


def parse_crop(text: str) -> tuple[float, float, float, float]:
    """``"top,bottom,left,right"`` -> tuple."""
    return _rect(text.split(","))


def load_preset(config: str | Path | None, device: str = "default") -> Preset:
    """Read a device preset from a YAML file (``{name: {crop: {...}, blackout: [[x0,y0,x1,y1]]}}``)."""
    if config is None:
        return Preset()
    try:
        import yaml
    except ImportError as exc:  # pyyaml ships with the 'train' extra
        raise MissingDependencyError("--crop-config needs pyyaml: pip install pyyaml") from exc
    spec = yaml.safe_load(Path(config).read_text()) or {}
    if device not in spec:
        raise ValueError(f"device preset {device!r} not found in {config} (have {sorted(spec)})")
    entry = spec[device] or {}
    crop = entry.get("crop") or {}
    unknown = set(crop) - {"top", "bottom", "left", "right"}
    if unknown:
        raise ValueError(f"unknown crop keys in preset {device!r}: {sorted(unknown)}")
    return Preset(
        name=device,
        crop=_rect([crop.get(k, 0) for k in ("top", "bottom", "left", "right")]),
        blackout=tuple(_rect(r) for r in entry.get("blackout") or []),
    )


def _px(value: float, side: int) -> int:
    return round(value * side) if value < 1 else int(value)


def apply_preset(img: Image.Image, preset: Preset) -> Image.Image:
    """Black out rectangles (original coordinates), then crop margins."""
    w, h = img.size
    if preset.blackout:
        a = np.array(img)
        for x0, y0, x1, y1 in preset.blackout:
            a[_px(y0, h):_px(y1, h), _px(x0, w):_px(x1, w)] = 0
        img = Image.fromarray(a, img.mode)
    top, bottom, left, right = preset.crop
    box = (_px(left, w), _px(top, h), w - _px(right, w), h - _px(bottom, h))
    if box[2] - box[0] < 8 or box[3] - box[1] < 8:
        raise ValueError("crop leaves less than 8 px")
    return img.crop(box) if box != (0, 0, w, h) else img


# --- readers ------------------------------------------------------------------------------------

def _pixels_only(img: Image.Image) -> Image.Image:
    """Fresh image built from pixel bytes only, so no metadata can be carried over."""
    img = ImageOps.exif_transpose(img)
    if img.mode not in ("L", "RGB"):
        img = img.convert("RGB")
    return Image.frombytes(img.mode, img.size, img.tobytes())


def is_dicom(path: Path) -> bool:
    if path.suffix.lower() in DICOM_EXTS:
        return True
    if path.suffix:
        return False
    with open(path, "rb") as fh:
        head = fh.read(132)
    return head[128:132] == b"DICM"


def _require_pydicom():
    try:
        import pydicom
    except ImportError as exc:
        raise MissingDependencyError(
            "DICOM input needs the optional 'pydicom' package (pip install pydicom); "
            "or export the studies as PNG/JPEG"
        ) from exc
    return pydicom


def _to_uint8(a: np.ndarray) -> np.ndarray:
    if a.dtype == np.uint8:
        return a
    a = a.astype(np.float64)
    lo, hi = float(a.min()), float(a.max())
    return ((a - lo) / (hi - lo or 1.0) * 255).round().astype(np.uint8)


def read_dicom(path: Path) -> tuple[list[Image.Image], str, bool]:
    """Return (frames, PatientID or '', burned-in-annotation flag). Metadata is discarded."""
    pydicom = _require_pydicom()
    ds = pydicom.dcmread(str(path))
    a = ds.pixel_array
    photometric = str(getattr(ds, "PhotometricInterpretation", ""))
    if photometric.startswith("YBR") and int(pydicom.__version__.split(".")[0]) < 3:
        from pydicom.pixel_data_handlers.util import convert_color_space

        a = convert_color_space(a, photometric, "RGB")
    samples = int(getattr(ds, "SamplesPerPixel", 1))
    frames = int(getattr(ds, "NumberOfFrames", 1) or 1)
    if frames == 1:
        a = a[None]
    a = _to_uint8(a)
    if photometric == "MONOCHROME1":
        a = 255 - a
    images = [Image.fromarray(f, "RGB" if samples == 3 else "L") for f in a]
    burned = str(getattr(ds, "BurnedInAnnotation", "")).upper() == "YES"
    return images, str(getattr(ds, "PatientID", "") or ""), burned


def read_image(path: Path) -> Image.Image:
    with Image.open(path) as img:
        img.load()
        return _pixels_only(img)


# --- pipeline -----------------------------------------------------------------------------------

def _check_out_dir(out_dir: Path) -> None:
    if find_git_root(out_dir) is not None:
        raise ValueError("OUT_DIR is inside a git repository; write de-identified data to the data root")


def deidentify(raw_dir: str | Path, out_dir: str | Path, salt: bytes, preset: Preset | None = None) -> dict:
    """De-identify every supported file under ``raw_dir`` into ``out_dir``. Returns the audit dict."""
    raw_dir, out_dir = Path(raw_dir).resolve(), Path(out_dir).resolve()
    _check_out_dir(out_dir)
    if not raw_dir.is_dir():
        raise NotADirectoryError("input directory not found")
    if out_dir == raw_dir or raw_dir in out_dir.parents:
        raise ValueError("OUT_DIR must not be inside RAW_DIR")
    audit = Audit(preset or Preset())
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(p for p in raw_dir.rglob("*") if p.is_file()):
        rel = path.relative_to(raw_dir)
        audit.counts["files_seen"] += 1
        try:
            if is_dicom(path):
                kind = "dicom"
                frames, tag_id, burned = read_dicom(path)
                audit.counts["dicom_burned_in_annotation"] += int(burned)
            elif path.suffix.lower() in IMAGE_EXTS:
                kind = "jpeg" if path.suffix.lower() != ".png" else "png"
                frames, tag_id = [read_image(path)], ""
            else:
                audit.counts["skipped_unsupported"] += 1
                continue
            patient_id = tag_id or (rel.parts[0] if len(rel.parts) > 1 else "")
            if not patient_id:
                audit.counts["skipped_no_patient_id"] += 1
                continue
            pseudo = pseudonymise(patient_id, salt)
            stem = f"{pseudo}_{_hmac_hex(salt, f'file:{rel.as_posix()}')[:10]}"
            (out_dir / pseudo).mkdir(exist_ok=True)
            for i, frame in enumerate(frames):
                name = f"{stem}.png" if len(frames) == 1 else f"{stem}_f{i:03d}.png"
                apply_preset(_pixels_only(frame), audit.preset).save(out_dir / pseudo / name, format="PNG")
                audit.outputs.append(f"{pseudo}/{name}")
            audit.counts[f"read_{kind}"] += 1
            audit.counts["images_written"] += len(frames)
            audit.patients.add(pseudo)
        except MissingDependencyError:  # missing optional dependency: stop, the whole batch would fail
            raise
        except Exception as exc:  # never echo the path or message: both may hold identifiers
            audit.counts[f"failed_{type(exc).__name__}"] += 1
    report = {
        "tool": "rausnet.data.deid",
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "pseudonym": "H + HMAC-SHA256(secret salt, upper(patient id))[:12]",
        "metadata": "stripped (pixels re-encoded to PNG; DICOM header, EXIF, text chunks dropped)",
        "preset": {"name": audit.preset.name, "crop_top_bottom_left_right": list(audit.preset.crop),
                   "blackout_rects": [list(r) for r in audit.preset.blackout]},
        "counts": dict(sorted(audit.counts.items())),
        "n_patients": len(audit.patients),
        "outputs": sorted(audit.outputs),
    }
    (out_dir / AUDIT_NAME).write_text(json.dumps(report, indent=2))
    return report


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="raw_dir")
    ap.add_argument("--out", dest="out_dir")
    ap.add_argument("--salt-file")
    ap.add_argument("--make-salt", metavar="PATH", help="create a new random salt file and exit")
    ap.add_argument("--crop-config", help="YAML with device presets (e.g. configs/deid/devices.yaml)")
    ap.add_argument("--device", default="default", help="preset name in --crop-config")
    ap.add_argument("--crop", type=parse_crop, help="top,bottom,left,right (overrides the preset crop)")
    ap.add_argument("--blackout", type=parse_crop, action="append", default=[], metavar="x0,y0,x1,y1",
                    help="extra rectangle to black out (repeatable)")
    args = ap.parse_args(argv)
    if args.make_salt:
        make_salt(args.make_salt)
        print("salt file created; keep it secret and outside the repository")
        return
    if not (args.raw_dir and args.out_dir and args.salt_file):
        ap.error("--in, --out and --salt-file are required")
    _check_out_dir(Path(args.out_dir))
    preset = load_preset(args.crop_config, args.device)
    preset = Preset(preset.name if args.crop is None else f"{preset.name}+cli",
                    args.crop or preset.crop, preset.blackout + tuple(args.blackout))
    report = deidentify(args.raw_dir, args.out_dir, load_salt(args.salt_file), preset)
    print(json.dumps({"counts": report["counts"], "n_patients": report["n_patients"]}, indent=2))


if __name__ == "__main__":
    main()
