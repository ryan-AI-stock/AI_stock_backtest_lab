from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

ALLOWED_DECISIONS = {
    "approve_add": "列入候選",
    "keep_watch": "保留觀察",
    "reject": "排除",
    "keep_current": "維持現有",
}


class CandidateReviewDecisionStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def state(self) -> dict[str, Any]:
        with self.lock:
            data = self._load()
            decisions = list(data.get("decisions", []))
            return {
                "schema_version": 1,
                "decisions": decisions,
                "latest_by_key": _latest_by_key(decisions),
            }

    def record(self, payload: dict[str, Any]) -> dict[str, Any]:
        decision = str(payload.get("decision") or "").strip()
        if decision not in ALLOWED_DECISIONS:
            raise ValueError("不支援的候選審核決策。")
        pool_id = str(payload.get("pool_id") or "").strip()
        ticker = str(payload.get("ticker") or "").strip()
        signal_date = str(payload.get("signal_date") or "").strip()
        if not pool_id or not ticker or not signal_date:
            raise ValueError("pool_id、ticker、signal_date 不可空白。")
        entry = {
            "key": _decision_key(pool_id, ticker),
            "pool_id": pool_id,
            "pool_name": str(payload.get("pool_name") or "").strip(),
            "ticker": ticker,
            "display": str(payload.get("display") or ticker).strip(),
            "decision": decision,
            "decision_label": ALLOWED_DECISIONS[decision],
            "signal_date": signal_date,
            "note": str(payload.get("note") or "").strip(),
            "source_status": str(payload.get("source_status") or "").strip(),
            "recorded_at": datetime.now().isoformat(timespec="seconds"),
        }
        with self.lock:
            data = self._load()
            decisions = [item for item in data.get("decisions", []) if item.get("key") != entry["key"]]
            decisions.append(entry)
            data["decisions"] = decisions
            self._save(data)
            return json.loads(json.dumps(entry, ensure_ascii=False))

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"schema_version": 1, "decisions": []}
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if data.get("schema_version") != 1 or "decisions" not in data:
            raise ValueError("候選審核決策檔格式不支援。")
        return data

    def _save(self, data: dict[str, Any]) -> None:
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.path)


def _decision_key(pool_id: str, ticker: str) -> str:
    return f"{pool_id}|{ticker}"


def _latest_by_key(decisions: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for item in decisions:
        key = str(item.get("key") or _decision_key(str(item.get("pool_id") or ""), str(item.get("ticker") or "")))
        if key:
            latest[key] = item
    return latest
