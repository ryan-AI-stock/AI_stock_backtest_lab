from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from typing import Any

from backtest_lab.strategy_preset_dispatcher import dispatch_pool, resolve_strategy_preset

KNOWN_SYMBOLS: dict[str, dict[str, str]] = {
    "0050.TW": {"symbol": "0050", "name": "0050", "asset_type": "etf"},
    "00631L.TW": {"symbol": "00631L", "name": "0050正二", "asset_type": "etf"},
    "1216.TW": {"symbol": "1216", "name": "統一", "asset_type": "stock"},
    "1301.TW": {"symbol": "1301", "name": "台塑", "asset_type": "stock"},
    "1303.TW": {"symbol": "1303", "name": "南亞", "asset_type": "stock"},
    "2002.TW": {"symbol": "2002", "name": "中鋼", "asset_type": "stock"},
    "2207.TW": {"symbol": "2207", "name": "和泰車", "asset_type": "stock"},
    "2412.TW": {"symbol": "2412", "name": "中華電", "asset_type": "stock"},
    "2603.TW": {"symbol": "2603", "name": "長榮", "asset_type": "stock"},
    "2609.TW": {"symbol": "2609", "name": "陽明", "asset_type": "stock"},
    "2881.TW": {"symbol": "2881", "name": "富邦金", "asset_type": "stock"},
    "2882.TW": {"symbol": "2882", "name": "國泰金", "asset_type": "stock"},
    "2891.TW": {"symbol": "2891", "name": "中信金", "asset_type": "stock"},
    "2892.TW": {"symbol": "2892", "name": "第一金", "asset_type": "stock"},
    "2912.TW": {"symbol": "2912", "name": "統一超", "asset_type": "stock"},
    "3045.TW": {"symbol": "3045", "name": "台灣大", "asset_type": "stock"},
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
                resolved["dispatch"] = dispatch_pool(resolved)
                pools.append(resolved)
            return pools

    def upsert_pool(self, payload: dict[str, Any]) -> dict[str, Any]:
        pool_id = _clean_id(str(payload.get("pool_id") or payload.get("name") or "custom_pool"))
        name = str(payload.get("name") or pool_id).strip()
        if not name:
            raise ValueError("股票池名稱不可空白。")
        preset = resolve_strategy_preset(str(payload.get("strategy_preset") or "universal_pool_custom"))
        symbols = parse_symbol_lines(str(payload.get("symbols_text") or ""))
        dynamic_binding = payload.get("dynamic_binding") or None
        with self.lock:
            data = self._load()
            existing = next((pool for pool in data["pools"] if pool["pool_id"] == pool_id), None)
            if existing and existing.get("locked") and existing.get("ui_section") != "official_core":
                raise ValueError("內建股票池不可覆蓋，請另建自訂池。")
            if existing and existing.get("locked") and existing.get("ui_section") == "official_core":
                pool = {
                    **existing,
                    "name": name,
                    "strategy_preset": preset.preset,
                    "operational_observation": preset.operational_observation,
                    "description": str(payload.get("description") or ""),
                    "symbols": symbols,
                }
                if dynamic_binding:
                    pool["dynamic_binding"] = dynamic_binding
                data["pools"] = [pool if item["pool_id"] == pool_id else item for item in data["pools"]]
                self._save(data)
                return pool
            pool = {
                "pool_id": pool_id,
                "name": name,
                "kind": str(payload.get("kind") or "custom"),
                "locked": False,
                "ui_section": str(payload.get("ui_section") or "experiment"),
                "strategy_preset": preset.preset,
                "operational_observation": preset.operational_observation,
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
        return merge_default_pools(data)

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
    core_symbols = [
        "0050.TW",
        "00631L.TW",
        "2330.TW",
        "2412.TW",
        "3045.TW",
        "1216.TW",
        "2912.TW",
        "2881.TW",
        "2882.TW",
        "2891.TW",
        "2892.TW",
        "1301.TW",
        "1303.TW",
        "2002.TW",
        "2207.TW",
        "2603.TW",
        "2609.TW",
    ]
    return {
        "schema_version": 1,
        "pools": [
            {
                "pool_id": "ai_theme_large_cap_v20260613",
                "name": "AI主線攻擊池 v20260613",
                "kind": "built_in",
                "locked": True,
                "ui_section": "official_core",
                "strategy_preset": "ai_theme_large_cap_v20260613",
                "operational_observation": True,
                "vote_group": "three_perspective_v1",
                "role_name": "主線攻擊專家",
                "role_description": "檢查 AI/半導體/伺服器供應鏈主線是否仍值得攻擊，重視已驗證的實戰策略與市場環境切換。",
                "candidate_update_policy": "v1 先固定候選名單；未來改為週頻或月頻候選維護，不能用每日短線漲幅自動亂換成員。",
                "description": "AI 主線攻擊池：0050、0050正二與七檔 AI 中大型權值股，使用 v20260613 規則。此池不是完整 AI 產業地圖，候選名單來源需持續驗證。",
                "symbols": [symbol_entry(ticker, source="fixed") for ticker in large_symbols],
            },
            {
                "pool_id": "tw50_dynamic_constituents_v0",
                "name": "大型市場廣度池 v0",
                "kind": "built_in",
                "locked": True,
                "ui_section": "official_core",
                "strategy_preset": "universal_pool_custom",
                "operational_observation": True,
                "vote_group": "three_perspective_v1",
                "role_name": "市場廣度專家",
                "role_description": "檢查整體大型權值市場目前由誰領先，避免正式結論只被 AI 主線或少數指定股綁住。",
                "candidate_update_policy": "依帶有效日期的台灣50/0050成分股表更新，不用現代成分股硬回推歷史。",
                "description": "大型市場廣度池：讀取帶有效日期的台灣50/0050歷史成分股表；若資料不存在，批次報告會明確跳過。",
                "symbols": [],
                "dynamic_constituents": {
                    "source": "tw50_history_csv",
                    "path": "data/tw50_constituents.csv",
                },
            },
            {
                "pool_id": "large_core_bluechip_v0",
                "name": "核心防守風格池 v1",
                "kind": "built_in",
                "locked": True,
                "ui_section": "official_core",
                "strategy_preset": "core_defensive_style_v1",
                "operational_observation": True,
                "vote_group": "three_perspective_v1",
                "role_name": "核心防守與風格轉移專家",
                "role_description": "檢查資金是否從主線攻擊轉向金融、電信、消費、傳產、航運或半導體核心股，偏好較不過熱、回撤較受控的核心權值。",
                "candidate_update_policy": "v1 先固定跨產業核心代理池；未來可改成依市值、產業代表性、低波動與基本面品質定期維護。",
                "description": "核心防守風格池：跨金融、電信、消費、傳產、航運與半導體核心的中大型權值代理池；台積電在此視為市場核心股，不是純 AI 題材股。",
                "symbols": [symbol_entry(ticker, source="fixed") for ticker in core_symbols],
            },
            {
                "pool_id": "large_cap_best_v20260605",
                "name": "AI中大型權值股池最佳版 v20260605",
                "kind": "built_in",
                "locked": True,
                "ui_section": "legacy",
                "strategy_preset": "best_v20260605",
                "operational_observation": False,
                "description": "正式最佳版使用的 0050、0050正二與七檔 AI 中大型權值股。",
                "symbols": [symbol_entry(ticker, source="fixed") for ticker in large_symbols],
            },
            {
                "pool_id": "radar_mid_small_calibrated_v1",
                "name": "雷達中小型校準版",
                "kind": "built_in",
                "locked": True,
                "ui_section": "legacy",
                "strategy_preset": "radar_core_mid_small_calibrated_v1",
                "operational_observation": False,
                "description": "由 AI_stock_rotation_radar snapshot 與雷達核心成員池 v1 動態決定候選股，不在介面寫死成固定清單。",
                "symbols": [],
            },
            {
                "pool_id": "model_scorecard_ep10",
                "name": "模型延遲公開成績單池",
                "kind": "task",
                "locked": False,
                "ui_section": "experiment",
                "strategy_preset": "delayed_public_scorecard_v1",
                "operational_observation": False,
                "description": "0050、0050正二，加上跟隨池1每日模型第一名的第三檔股票。",
                "symbols": [
                    symbol_entry("0050.TW", source="fixed"),
                    symbol_entry("00631L.TW", source="fixed"),
                    symbol_entry("2454.TW", source="dynamic"),
                ],
                "dynamic_binding": {
                    "source": "latest_model_top1",
                    "source_pool_id": "ai_theme_large_cap_v20260613",
                    "replace_index": 2,
                    "fallback_ticker": "2454.TW",
                },
            },
        ],
    }


def merge_default_pools(data: dict[str, Any]) -> dict[str, Any]:
    defaults = default_stock_pool_data()
    current = {pool["pool_id"]: pool for pool in data.get("pools", [])}
    merged = []
    for default_pool in defaults["pools"]:
        pool_id = default_pool["pool_id"]
        if pool_id in current:
            pool = {**default_pool, **current[pool_id]}
            if pool.get("ui_section") is None:
                pool["ui_section"] = default_pool.get("ui_section", "experiment")
            if pool.get("ui_section") == "legacy":
                pool["operational_observation"] = False
            merged.append(pool)
        else:
            merged.append(default_pool)
    default_ids = {pool["pool_id"] for pool in defaults["pools"]}
    for pool in data.get("pools", []):
        if pool["pool_id"] not in default_ids:
            merged.append(pool)
    return {
        **data,
        "schema_version": 1,
        "pools": merged,
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
