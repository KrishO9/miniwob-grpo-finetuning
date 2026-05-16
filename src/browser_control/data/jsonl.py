from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schemas import json_dumps


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as f:
        f.write(json_dumps(record) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    input_path = Path(path)
    if not input_path.exists():
        return []
    with input_path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
