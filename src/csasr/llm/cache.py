"""On-disk response cache. This is what makes every LLM stage resumable.

Kaggle sessions die at 12 hours. A generation run that cannot be killed and
resumed is a run that cannot finish. Every request is keyed by a hash of
(model, messages, sampling params, sample index) and appended to a JSONL as soon
as it returns, so a re-run skips everything already done.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

__all__ = ["ResponseCache"]


def _stable_hash(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


class ResponseCache:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._store: dict[str, str] = {}
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # tolerate a truncated final line from a killed run
                    self._store[rec["key"]] = rec["response"]
        self._fh = self.path.open("a", encoding="utf-8")

    @staticmethod
    def key(model: str, messages: list[dict], params: dict, sample: int = 0) -> str:
        return _stable_hash(
            {"model": model, "messages": messages, "params": params, "sample": sample}
        )

    def get(self, key: str) -> str | None:
        return self._store.get(key)

    def put(self, key: str, response: str) -> None:
        self._store[key] = response
        self._fh.write(json.dumps({"key": key, "response": response}, ensure_ascii=False) + "\n")
        self._fh.flush()

    def __len__(self) -> int:
        return len(self._store)

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "ResponseCache":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
