from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable


def log_metrics(step: int, metrics: Dict[str, float], path: Path | None = None) -> None:
    """Écrit un dictionnaire de métriques en JSONL (ou affiche si path None)."""
    record = {"step": step, **metrics}
    line = json.dumps(record, ensure_ascii=True)
    if path is None:
        print(line)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
