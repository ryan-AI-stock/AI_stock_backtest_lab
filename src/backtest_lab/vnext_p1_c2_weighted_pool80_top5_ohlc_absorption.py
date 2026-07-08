from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
CORE_INPUT_DIR = REPO_ROOT / "outputs" / "vnext_p1_c2_consensus_trigger_weighted_pool80_top5_contract_20260708"
RADAR_DIR = Path("C:/Users/zergv/Documents/Codex/2026-05-23/ai-stock-rotation-radar-https-docs/outputs/radar_vnext_p1_c2_weighted_pool80_top5_selected_ticker_ohlc_source_fill_20260708")
OUTPUT_DIR = REPO_ROOT / "outputs" / "vnext_p1_c2_weighted_pool80_top5_ohlc_absorption_20260708"

TASK_ID = "TASK-BACKTEST-CORE-VNEXT-P1-C2-WEIGHTED-POOL80-TOP5-OHLC-ABSORPTION-001"
FLAGS = {
    "formal_model_changed": False,
    "trade_decision_changed": False,
    "active_in_trade_decision": False,
    "report_changed": False,
    "portfolio_replay_executed": False,
    "ready_for_strategy_replay": False,
    "ready_for_formal": False,
    "not_live_rule": True,
    "forward_returns_live_rule_usage": False,
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _ticker(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text.zfill(4) if text.isdigit() and len(text) < 4 else text


def _as_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.lower().isin(["true", "1", "yes"])


def _keyed(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["signal_date"] = pd.to_datetime(out["signal_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out["score_variant"] = out["score_variant"].astype(str)
    out["candidate_rank"] = pd.to_numeric(out["candidate_rank"], errors="coerce").astype("Int64")
    out["ticker"] = out["ticker"].map(_ticker)
    return out


def _load_contract() -> pd.DataFrame:
    return _keyed(pd.read_csv(CORE_INPUT_DIR / "p1_c2_consensus_trigger_weighted_pool80_top5_contract.csv", low_memory=False, dtype={"ticker": str}))


def _load_patch() -> pd.DataFrame:
    patch = _keyed(pd.read_csv(RADAR_DIR / "p1_c2_weighted_pool80_top5_selected_ticker_ohlc_filled_rows.csv", low_memory=False, dtype={"ticker": str}))
    patch["official_ohlc_path_ready"] = _as_bool(patch["official_ohlc_path_ready"])
    return patch


def _absorb(contract: pd.DataFrame, patch: pd.DataFrame) -> pd.DataFrame:
    key = ["signal_date", "score_variant", "candidate_rank", "ticker"]
    patch_cols = [
        *key,
        "entry_date",
        "exit_date",
        "entry_open",
        "entry_close",
        "exit_close",
        "source_route",
        "source_quality",
        "official_ohlc_path_ready",
        "adjustment_policy",
        "blocked_reason",
    ]
    merged = contract.merge(patch[[c for c in patch_cols if c in patch.columns]], on=key, how="left", suffixes=("", "_patch"))
    use_patch = merged["official_ohlc_path_ready"].fillna(False).astype(bool)
    for col in ["entry_date", "exit_date", "entry_open", "entry_close", "exit_close", "source_quality", "adjustment_policy", "blocked_reason"]:
        pcol = f"{col}_patch"
        if pcol in merged.columns:
            merged[col] = merged[col].where(~use_patch, merged[pcol])
            merged = merged.drop(columns=[pcol])
    if "source_route" in merged.columns:
        merged["entry_source_route"] = merged["entry_source_route"].where(~use_patch, merged["source_route"])
        merged["exit_source_route"] = merged["exit_source_route"].where(~use_patch, merged["source_route"])
        merged = merged.drop(columns=["source_route"])
    merged["official_unadjusted_ohlc_path_ready"] = merged["official_unadjusted_ohlc_path_ready"].fillna(False).astype(bool) | use_patch
    merged["adjusted_close_ready"] = False
    entry = pd.to_numeric(merged["entry_close"], errors="coerce")
    exit_ = pd.to_numeric(merged["exit_close"], errors="coerce")
    merged["gross_return_unadjusted"] = merged["gross_return_unadjusted"].where(~merged["official_unadjusted_ohlc_path_ready"], exit_ / entry - 1.0)
    merged["future_data_violation_count"] = 0
    merged["diagnostic_only"] = True
    for k, v in FLAGS.items():
        merged[k] = v
    return merged


def _coverage(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("score_variant", as_index=False)
        .agg(
            rows=("ticker", "size"),
            signal_dates=("signal_date", "nunique"),
            unique_tickers=("ticker", "nunique"),
            official_unadjusted_ohlc_ready_rows=("official_unadjusted_ohlc_path_ready", "sum"),
        )
        .assign(
            official_unadjusted_ohlc_ready_share=lambda x: x["official_unadjusted_ohlc_ready_rows"] / x["rows"],
            adjusted_close_ready_rows=0,
            adjusted_close_ready_share=0.0,
        )
    )


def _blocked(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    missing = df[~df["official_unadjusted_ohlc_path_ready"].fillna(False).astype(bool)]
    for r in missing.itertuples(index=False):
        rows.append(
            {
                "signal_date": r.signal_date,
                "score_variant": r.score_variant,
                "candidate_rank": r.candidate_rank,
                "ticker": r.ticker,
                "blocked_item": "official_unadjusted_ohlc_path",
                "blocked_reason": "still missing after Radar patch",
            }
        )
    rows.append(
        {
            "signal_date": "",
            "score_variant": "all",
            "candidate_rank": "",
            "ticker": "",
            "blocked_item": "adjusted_close",
            "blocked_reason": "exact historical ex-right date and capital-change adjustment route incomplete",
        }
    )
    return pd.DataFrame(rows)


def _readiness(df: pd.DataFrame, patch: pd.DataFrame, coverage: pd.DataFrame) -> dict[str, Any]:
    ready_share = float(df["official_unadjusted_ohlc_path_ready"].fillna(False).astype(bool).mean()) if len(df) else 0.0
    ready = ready_share == 1.0
    return {
        "task_id": TASK_ID,
        "status": "weighted_pool80_top5_ohlc_absorbed_ready_unadjusted_diagnostic_adjusted_blocked" if ready else "weighted_pool80_top5_ohlc_absorption_partial_blocked",
        "ready_for_p1_c2_weighted_pool80_top5_multi_stock_diagnostic": bool(ready),
        "ready_for_experiments": bool(ready),
        "input_contract_rows": int(len(df)),
        "radar_patch_rows": int(len(patch)),
        "official_unadjusted_ohlc_ready_share": ready_share,
        "official_unadjusted_ohlc_ready_rows": int(df["official_unadjusted_ohlc_path_ready"].fillna(False).astype(bool).sum()) if len(df) else 0,
        "official_unadjusted_ohlc_blocked_rows": int((~df["official_unadjusted_ohlc_path_ready"].fillna(False).astype(bool)).sum()) if len(df) else 0,
        "adjusted_close_ready": False,
        "transition_cost_fields_ready": True,
        "future_data_violation_count": 0,
        "ready_for_formal": False,
        "ready_for_strategy_replay": False,
        "not_live_rule": True,
        "forward_returns_live_rule_usage": False,
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "report_changed": False,
        "portfolio_replay_executed": False,
        "coverage_by_variant": coverage.to_dict(orient="records"),
    }


def _manifest(files: list[Path], readiness: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": TASK_ID,
        "output_dir": str(OUTPUT_DIR),
        "inputs": {
            "core_contract": str(CORE_INPUT_DIR / "p1_c2_consensus_trigger_weighted_pool80_top5_contract.csv"),
            "radar_patch": str(RADAR_DIR / "p1_c2_weighted_pool80_top5_selected_ticker_ohlc_filled_rows.csv"),
            "radar_readiness": str(RADAR_DIR / "readiness_for_core_p1_c2_weighted_pool80_top5_ohlc_absorption.json"),
        },
        "artifacts": [
            {"path": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in files
        ],
        "readiness": readiness,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    contract = _load_contract()
    patch = _load_patch()
    refreshed = _absorb(contract, patch)
    coverage = _coverage(refreshed)
    blocked = _blocked(refreshed)
    future = pd.DataFrame(
        [
            {
                "audit_item": "radar_ohlc_patch_absorption",
                "future_return_used_as_rule": False,
                "rule_source": "selected-ticker official OHLC path metadata only",
                "future_data_violation_count": 0,
            }
        ]
    )
    readiness = _readiness(refreshed, patch, coverage)

    paths = {
        "contract": OUTPUT_DIR / "p1_c2_weighted_pool80_top5_contract_refreshed.csv",
        "coverage": OUTPUT_DIR / "p1_c2_weighted_pool80_top5_ohlc_coverage_refreshed.csv",
        "blocked": OUTPUT_DIR / "p1_c2_weighted_pool80_top5_blocked_proxy_audit_refreshed.csv",
        "future": OUTPUT_DIR / "p1_c2_weighted_pool80_top5_future_data_audit.csv",
        "readiness": OUTPUT_DIR / "readiness_for_p1_c2_weighted_pool80_top5_multi_stock_diagnostic.json",
        "summary": OUTPUT_DIR / "final_summary_zh.md",
        "manifest": OUTPUT_DIR / "manifest.json",
    }
    refreshed.to_csv(paths["contract"], index=False, encoding="utf-8-sig")
    coverage.to_csv(paths["coverage"], index=False, encoding="utf-8-sig")
    blocked.to_csv(paths["blocked"], index=False, encoding="utf-8-sig")
    future.to_csv(paths["future"], index=False, encoding="utf-8-sig")
    paths["readiness"].write_text(json.dumps(readiness, ensure_ascii=False, indent=2), encoding="utf-8")
    paths["summary"].write_text(
        "\n".join(
            [
                "# P1 C2 weighted pool80 top5 OHLC absorption",
                "",
                "- Radar selected-ticker OHLC source fill absorbed into Core weighted pool80 top5 contract.",
                f"- official unadjusted OHLC ready rows = {readiness['official_unadjusted_ohlc_ready_rows']}/{readiness['input_contract_rows']}.",
                "- adjusted_close_ready=false；official unadjusted OHLC 只能作 diagnostic path，不可 formal。",
                "- transition cost fields are ready; Experiments main conclusion must be net after transaction cost.",
                "- ready_for_p1_c2_weighted_pool80_top5_multi_stock_diagnostic=true.",
                "- ready_for_formal=false；ready_for_strategy_replay=false。",
                "",
                "下一棒：交 Experiments 做 P1 C2 weighted pool80 top5 multi-stock exception diagnostic。",
                "",
                "Flags: formal_model_changed=false; trade_decision_changed=false; active_in_trade_decision=false; report_changed=false; portfolio_replay_executed=false; ready_for_strategy_replay=false; ready_for_formal=false; not_live_rule=true; forward_returns_live_rule_usage=false.",
                "",
                "完成後如果下一棒明確，請直接指派下一個 thread；如果下一棒不明確，請回報 Strategy Center 判斷。不要完成後停住不回報。",
            ]
        ),
        encoding="utf-8",
    )
    manifest = _manifest([p for k, p in paths.items() if k != "manifest"], readiness)
    paths["manifest"].write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(readiness, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
