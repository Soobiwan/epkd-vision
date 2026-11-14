from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


class JsonRunLogger:
    def __init__(self, *, log_dir: str | Path, run_name: str, metadata: dict[str, Any] | None = None):
        safe_run_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in run_name).strip("_")
        safe_run_name = safe_run_name or "run"
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.path = Path(log_dir) / f"{safe_run_name}_{timestamp}.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.payload = {
            "run_name": safe_run_name,
            "created_at": _now_iso(),
            "metadata": _jsonable(metadata or {}),
            "events": [],
        }
        self._flush()

    def log(self, event: str, **data: Any) -> None:
        entry = {"event": event, "timestamp": _now_iso(), **_jsonable(data)}
        self.payload["events"].append(entry)
        self._flush()

    def finalize(self, *, status: str, **summary: Any) -> None:
        self.payload["status"] = status
        self.payload["finished_at"] = _now_iso()
        if summary:
            self.payload["summary"] = _jsonable(summary)
        self._flush()

    def _flush(self) -> None:
        self.path.write_text(json.dumps(self.payload, indent=2) + "\n")
