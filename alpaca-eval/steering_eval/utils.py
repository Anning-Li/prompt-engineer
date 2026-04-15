from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable


def sha256_text(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_json_files(directory: Path) -> Iterable[Path]:
    if not directory.exists():
        return []
    return sorted(directory.glob("*.json"))


def timestamp_run_id(prefix: str = "alpaca") -> str:
    return f"{prefix}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"


def normalize_column_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = re.sub(r"^json\s*", "", text, flags=re.IGNORECASE)
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    candidate = match.group(0) if match else text
    return json.loads(candidate)
