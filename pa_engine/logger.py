"""Event log: one JSON line per event (signal, rejection, order, fill, stop change, exit) as the spec's section 29 requires."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class EventLog:
    def __init__(self, path: Optional[str | Path] = None, keep: bool = True):
        self.path = Path(path) if path else None
        self.rows: List[Dict[str, Any]] = [] if keep else None  # type: ignore
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(self.path, "w", encoding="utf-8")
        else:
            self._fh = None

    def log(self, ts, event: str, **fields: Any) -> None:
        row = {"ts": str(ts), "event": event}
        for k, v in fields.items():
            if isinstance(v, float):
                v = round(v, 4)
            row[k] = v
        if self.rows is not None:
            self.rows.append(row)
        if self._fh:
            self._fh.write(json.dumps(row, default=str) + "\n"); self._fh.flush()

    def close(self) -> None:
        if self._fh:
            self._fh.close(); self._fh = None
