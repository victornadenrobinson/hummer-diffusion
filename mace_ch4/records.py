"""Generic append-only JSON-list persistence, shared by benchmark/EOS records."""
from __future__ import annotations

import json
from pathlib import Path


def append_record(path: str | Path, record: dict) -> None:
    """Append one record to a growing JSON list, written atomically."""
    path = Path(path)
    records = load_records(path) if path.is_file() else []
    records.append(record)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(records, indent=2))
    tmp_path.replace(path)


def load_records(path: str | Path) -> list[dict]:
    return json.loads(Path(path).read_text())
