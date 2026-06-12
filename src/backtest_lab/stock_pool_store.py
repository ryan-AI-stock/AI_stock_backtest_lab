from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from typing import Any


KNOWN_SYMBOLS: dict[str, dict[str, str]] = {
    "0050.TW": {"symbol": "0050", "name": "0050", "asset_type": "etf"},
    "00631L.TW": {"symbol": "00631L", "name": "0050正二", "asset_type": "etf"},
    "2330.TW": {"symbol": "2330", "name": "台積電", "asset_type": "stock"},
    "2454.TW": {"symbol": "2454", "name": "聯發科", "asset_type": "stock"},
    "2308.TW": {"symbol": "2308", "name": "台達電", "asset_type": "stock"},
    "2317.TW": {"symbol": "2317", "name": "鴻海", "asset_type": "stock"},
    "2382.TW": {"symbol": "2382", "name": "廣達", "asset_type": "stock"},
    "3231.TW": {"symbol": "3231", "name": "緯創", "asset_type": "stock"},
    "6669.TW": {"symbol": "6669", "name": "緯穎", "asset_type": "stock"},
}


class StockPoolStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def list_pools(self, *, latest_signal: dict | None = None) -> list[dict[str, Any]]:
        with self.lock:
            data = self._load()
            pools = []
            for pool in data["pools"]:
                resolved = json.loads(json.dumps(pool, ensure_ascii=False))
                resolved["resolved_symbols"] = self._resolve_symbols(pool, latest_signal=latest_signal)
                pools.append(resolved)
            return pools

    def upsert_pool(self, payload: dict[str, Any]) -> dict[str, Any]:
        pool_id = _clean_id(str(payload.get("pool_id") or payload.get("name") or "custom_pool"))
        name = str(payload.get("name") or pool_id).strip()
        if not name:
            raise ValueError("股票池名稱不可空白。")
        symbols = parse_symbol_lines(str(payload.get("symbols_text") or ""))
        dynamic_binding = payload.get("dynamic_binding") or None
        with self.lock:
            data = self._load()
            existing = next((pool for pool in data["pools"] if pool["pool_id"] == pool_id), None)
            if existing and existing.get("locked"):
                raise ValueError("內建股票池不可覆蓋，請另建自訂池。")
            pool = {
                "pool_id": pool_id,
                "name": name,
                "kind": str(payload.get("kind") or "custom"),
                "locked": False,
                "strategy_preset": str(payload.get("strategy_preset") or "universal_pool_custom"),
                "description": str(payload.get("description") or ""),
                "symbols": symbols,
                "dynamic_binding": dynamic_binding,
            }
            if existing:
                data["pools"] = [pool if item["pool_id"] == pool_id else item for item in data["pools"]]
            else:
                data["pools"].append(pool)
            self._save(data)
            return pool

    def delete_pool(self, pool_id: str) -> None:
        with self.lock:
            data = self._load()
            pool = next((item for item in data["pools"] if item["pool_id"] == pool_id), None)
            if pool is None:
                raise ValueError("找不到股票池。")
            if pool.get("locked"):
                raise ValueError("內建股票池不可刪除。")
            data["pools"] = [item for item in data["pools"] if item["pool_id"] != pool_id]
            self._save(data)

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return default_stock_pool_data()
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if data.get("schema_version") != 1 or "pools" not in data:
            raise ValueError("股票池設定檔格式不支援。")
        return data

    def _save(self, data: dict[str, Any]) -> None:
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.path)

    def _resolve_symbols(self, pool: dict[str, Any], *, latest_signal: dict | None) -> list[dict[str, Any]]:
        symbols = [dict(item) for item in pool.get("symbols", [])]
        binding = pool.get("dynamic_binding")
        if not binding:
            return symbols
        target = None
        if binding.get("source") == "latest_model_top1" and latest_signal:
            target = latest_signal.get("target_ticker")
        target = target or binding.get("fallback_ticker")
        if not target:
            return symbols
        dynamic_symbol = symbol_entry(target, source="dynamic")
        replace_index = int(binding.get("replace_index", len(symbols)))
        if 0 <= replace_index < len(symbols):
            symbols[replace_index] = dynamic_symbol
        else:
            symbols.append(dynamic_symbol)
        return symbols


def default_stock_pool_data() -> dict[str, Any]:
    large_symbols = [
        "0050.TW",
        "00631L.TW",
        "2330.TW",
        "2454.TW",
        "2308.TW",
        "2317.TW",
        "2382.TW",
        "3231.TW",
        "6669.TW",
    ]
    return {
        "schema_version": 1,
        "pools": [
            {
                "pool_id": "large_cap_best_v20260605",
                "name": "AI中大型權值股池最佳版 v20260605",
                "kind": "built_in",
                "locked": True,
                "strategy_preset": "best_v20260605",
                "description": "正式最佳版使用的 0050、0050正二與七檔 AI 中大型權值股。",
                "symbols": [symbol_entry(ticker, source="fixed") for ticker in large_symbols],
            },
            {
                "pool_id": "radar_mid_small_calibrated_v1",
                "name": "雷達中小型校準版",
                "kind": "built_in",
                "locked": True,
                "strategy_preset": "radar_core_mid_small_calibrated_v1",
                "description": "由 AI_stock_rotation_radar snapshot 與雷達核心成員池 v1 動態決定候選股，不在介面寫死成固定清單。",
                "symbols": [],
            },
            {
                "pool_id": "model_scorecard_ep10",
                "name": "模型延遲公開成績單池",
                "kind": "task",
                "locked": False,
                "strategy_preset": "delayed_public_scorecard_v1",
                "description": "0050、0050正二，加上跟隨池1每日模型第一名的第三檔股票。",
                "symbols": [
                    symbol_entry("0050.TW", source="fixed"),
                    symbol_entry("00631L.TW", source="fixed"),
                    symbol_entry("2454.TW", source="dynamic"),
                ],
                "dynamic_binding": {
                    "source": "latest_model_top1",
                    "source_pool_id": "large_cap_best_v20260605",
                    "replace_index": 2,
                    "fallback_ticker": "2454.TW",
                },
            },
        ],
    }


def parse_symbol_lines(text: str) -> list[dict[str, Any]]:
    entries = []
    seen = set()
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        ticker = normalize_ticker(line)
        if ticker in seen:
            continue
        entries.append(symbol_entry(ticker, source="manual"))
        seen.add(ticker)
    return entries


def normalize_ticker(value: str) -> str:
    text = value.strip().upper()
    match = re.search(r"([0-9A-Z]{4,6})(?:\.(TW|TWO))?", text)
    if not match:
        raise ValueError(f"無法解析股票代號：{value}")
    symbol = match.group(1)
    suffix = match.group(2) or "TW"
    return f"{symbol}.{suffix}"


def symbol_entry(ticker: str, *, source: str) -> dict[str, Any]:
    normalized = normalize_ticker(ticker)
    known = KNOWN_SYMBOLS.get(normalized, {})
    symbol = known.get("symbol") or normalized.split(".")[0]
    name = known.get("name") or symbol
    return {
        "ticker": normalized,
        "symbol": symbol,
        "name": name,
        "display": f"{name}({symbol})",
        "asset_type": known.get("asset_type") or "stock",
        "source": source,
    }


def _clean_id(value: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z_-]+", "_", value.strip()).strip("_")
    return cleaned or "custom_pool"
