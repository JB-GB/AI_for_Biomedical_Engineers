"""Thin HTTP client for the RAUSNet API (used by the desktop app)."""

from __future__ import annotations

from pathlib import Path

import httpx


class ApiError(RuntimeError):
    pass


class ApiClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8000", http: httpx.Client | None = None,
                 timeout: float = 30.0):
        self.http = http or httpx.Client(base_url=base_url, timeout=timeout)

    def _check(self, res: httpx.Response) -> dict:
        if res.status_code != 200:
            try:
                detail = res.json().get("detail", res.text)
            except ValueError:
                detail = res.text
            raise ApiError(f"{res.status_code}: {detail}")
        return res.json()

    @staticmethod
    def _files(gs: str | Path, pd: str | Path | None) -> dict:
        files = {"gs": (Path(gs).name, Path(gs).read_bytes())}
        if pd:
            files["pd"] = (Path(pd).name, Path(pd).read_bytes())
        return files

    def health(self) -> dict:
        return self._check(self.http.get("/health"))

    def model(self) -> dict:
        return self._check(self.http.get("/model"))

    def predict(self, gs: str | Path, pd: str | Path | None = None) -> dict:
        return self._check(self.http.post("/predict", files=self._files(gs, pd)))

    def correct(self, gs: str | Path, pd: str | Path | None, *, patient_id: str, joint: str, side: str,
                gs_grade: int, pd_grade: int) -> dict:
        data = {"patient_id": patient_id, "joint": joint, "side": side,
                "gs_grade": str(gs_grade), "pd_grade": str(pd_grade)}
        return self._check(self.http.post("/corrections", files=self._files(gs, pd), data=data))
