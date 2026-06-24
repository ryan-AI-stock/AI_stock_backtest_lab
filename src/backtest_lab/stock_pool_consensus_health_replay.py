from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd

from backtest_lab.data import load_price_csv
from backtest_lab.stock_pool_consensus import build_consensus
from backtest_lab.stock_pool_historical_replay import DEFAULT_PERIODS as HISTORICAL_DEFAULT_PERIODS
from backtest_lab.stock_pool_historical_replay import run_stock_pool_historical_replay


DEFAULT_OUTPUT_DIR = "outputs/stock_pool_consensus_health_replay_20260623"
DEFAULT_PRICE_CACHE_DIR = "backtest_cache/stock_pool_triad_v1_corrected"
DEFAULT_POOL_STORE_PATH = "data/stock_pools.json"
DEFAULT_PERIOD_STARTS = {
    "2022": "2022-01-03",
    "2023": "2023-01-03",
    "2024_now": "2024-01-02",
}
DEFAULT_PERIOD_ENDS = {
    "2022": "2022-12-30",
    "2023": "2023-12-29",
}


def run_stock_pool_consensus_health_replay(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    replay_panel_path: str | Path | None = None,
    pool_store_path: str | Path = DEFAULT_POOL_STORE_PATH,
    cache_dir: str | Path = DEFAULT_PRICE_CACHE_DIR,
    warmup_start: str = "2020-01-01",
    periods: dict[str, tuple[str, str]] | None = None,
    date_stride: int = 1,
    max_dates: int | None = None,
    cache_only: bool = True,
    tw50_constituents_path: str | Path = "data/tw50_constituents.csv",
) -> Path:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    run_log_path = root / "run_log.csv"
    failed_path = root / "failed.csv"
    completed_path = root / "completed.csv"
    current_step = root / "current_step.txt"
    _write_run_log(run_log_path, "start", "started", "stock_pool_consensus_health_replay")
    current_step.write_text("resolve_periods\n", encoding="utf-8")

    actual_periods = periods or _default_periods(cache_dir)
    source_replay_panel = Path(replay_panel_path) if replay_panel_path else root / "historical_replay" / "stock_pool_replay_panel.csv"
    historical_replay_dir = source_replay_panel.parent

    if replay_panel_path is None:
        current_step.write_text("generate_stock_pool_historical_replay\n", encoding="utf-8")
        result = run_stock_pool_historical_replay(
            pool_store_path=pool_store_path,
            cache_dir=cache_dir,
            output_dir=historical_replay_dir,
            periods=actual_periods,
            warmup_start=warmup_start,
            tw50_constituents_path=tw50_constituents_path,
            candidate_limit=3,
            require_exact_signal_date=True,
            cache_only=cache_only,
            max_dates=max_dates,
            date_stride=max(1, date_stride),
        )
        source_replay_panel = result.output_dir / "stock_pool_replay_panel.csv"

    current_step.write_text("build_consensus_health_history\n", encoding="utf-8")
    replay_panel = pd.read_csv(source_replay_panel).fillna("")
    health, diagnostics = build_consensus_health_history_from_replay_panel(replay_panel)
    summary = summarize_consensus_health_periods(health, diagnostics)

    health_path = root / "stock_pool_consensus_health_history.csv"
    diagnostics_path = root / "stock_pool_consensus_pool_diagnostics_history.csv"
    summary_path = root / "consensus_health_period_summary.csv"
    health.to_csv(health_path, index=False, encoding="utf-8-sig")
    diagnostics.to_csv(diagnostics_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    failed_rows = _failed_rows(replay_panel)
    _write_csv(failed_path, failed_rows)
    readiness = _readiness(
        periods=actual_periods,
        date_stride=date_stride,
        max_dates=max_dates,
        cache_only=cache_only,
        replay_panel=replay_panel,
        health=health,
        diagnostics=diagnostics,
        failed_rows=failed_rows,
        source_replay_panel=source_replay_panel,
        latest_complete_signal_date=_latest_complete_signal_date(cache_dir),
    )
    (root / "readiness_and_limitations.json").write_text(
        json.dumps(readiness, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "task_id": "TASK-BACKTEST-CORE-CONSENSUS-HEALTH-REPLAY-20260623",
        "model": "stock_pool_consensus_health_replay_v1",
        "status": "completed",
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "decision_layer": "diagnostic",
        "source_replay_panel": str(source_replay_panel),
        "tw50_constituents_path": str(tw50_constituents_path),
        "granularity": "daily" if date_stride == 1 else f"stride{date_stride}",
        "periods": {key: {"start": start, "end": end} for key, (start, end) in actual_periods.items()},
        "outputs": {
            "stock_pool_consensus_health_history": str(health_path),
            "stock_pool_consensus_pool_diagnostics_history": str(diagnostics_path),
            "consensus_health_period_summary": str(summary_path),
            "readiness_and_limitations": str(root / "readiness_and_limitations.json"),
            "run_log": str(run_log_path),
            "completed": str(completed_path),
            "failed": str(failed_path),
        },
        "rows": {
            "replay_panel": int(len(replay_panel)),
            "health_history": int(len(health)),
            "pool_diagnostics_history": int(len(diagnostics)),
            "period_summary": int(len(summary)),
            "failed": int(len(failed_rows)),
        },
        "boundaries": [
            "report_only_diagnostic",
            "formal_trade_target_unchanged",
            "pool3_radar_satellite_not_in_formal_vote",
            "observation_only_not_in_formal_vote",
            "valuation_not_used",
            "h3_day_trading_margin_overheat_not_used",
        ],
    }
    (root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(
        completed_path,
        [
            {
                "status": "completed",
                "output_dir": str(root.resolve()),
                "health_rows": len(health),
                "diagnostic_rows": len(diagnostics),
            }
        ],
    )
    _write_run_log(run_log_path, "completed", "completed", str(root.resolve()))
    current_step.write_text("completed\n", encoding="utf-8")
    return root


def build_consensus_health_history_from_replay_panel(replay_panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"period", "requested_signal_date", "pool_id", "status"}
    missing = required - set(replay_panel.columns)
    if missing:
        raise ValueError("missing replay panel columns: " + ",".join(sorted(missing)))
    health_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    frame = replay_panel.copy().fillna("")
    frame["effective_signal_date"] = frame.apply(_effective_signal_date, axis=1)
    frame = frame[frame["effective_signal_date"].astype(str).str.strip().ne("")]
    for (period, signal_date), group in frame.groupby(["period", "effective_signal_date"], dropna=False):
        manifest = _consensus_manifest_from_group(group)
        consensus = build_consensus(manifest)
        health = {
            "period": period,
            "signal_date": signal_date,
            "result_state": consensus.get("result_state", ""),
            "winner_ticker": consensus.get("winner_ticker", ""),
            **consensus.get("health_diagnostic", {}),
        }
        health_rows.append(health)
        for row in consensus.get("pool_diagnostics", []):
            diagnostic_rows.append(
                {
                    "period": period,
                    "signal_date": signal_date,
                    **row,
                }
            )
    return pd.DataFrame(health_rows), pd.DataFrame(diagnostic_rows)


def summarize_consensus_health_periods(health: pd.DataFrame, diagnostics: pd.DataFrame) -> pd.DataFrame:
    if health.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for period, group in health.groupby("period", dropna=False):
        total = len(group)
        no_vote = group["raw_consensus_state"].astype(str).isin({"no_vote", "insufficient_votes"}).sum()
        divergent = (group["raw_consensus_state"].astype(str) == "divergent").sum()
        protocol_candidates = (group["decision_source"].astype(str) == "protocol_resolved_divergence").sum()
        row = {
            "period": period,
            "signal_count": total,
            "exact_ticker_consensus_rate": _rate(group["exact_ticker_consensus"].map(_truthy).sum(), total),
            "direction_consensus_rate": _rate(group["direction_consensus"].map(_truthy).sum(), total),
            "divergent_rate": _rate(divergent, total),
            "no_vote_or_data_insufficient_rate": _rate(no_vote, total),
            "actionable_decision_rate": round(float(pd.to_numeric(group["actionable_decision_rate"], errors="coerce").fillna(0).mean()), 4),
            "decision_protocol_candidate_rate": _rate(protocol_candidates, total),
            "decision_protocol_used_rate": round(float(pd.to_numeric(group["decision_protocol_used_rate"], errors="coerce").fillna(0).mean()), 4),
            "healthy_count": int((group["consensus_health_bucket"].astype(str) == "healthy").sum()),
            "acceptable_count": int((group["consensus_health_bucket"].astype(str) == "acceptable").sum()),
            "warning_count": int((group["consensus_health_bucket"].astype(str) == "warning").sum()),
            "unhealthy_count": int((group["consensus_health_bucket"].astype(str) == "unhealthy").sum()),
            "not_evaluable_count": int((group["consensus_health_bucket"].astype(str) == "not_evaluable").sum()),
        }
        row["healthy_or_acceptable_rate"] = _rate(row["healthy_count"] + row["acceptable_count"], total)
        rows.append(row)
    if not diagnostics.empty:
        diag = diagnostics.copy()
        blocked = diag[diag["data_readiness_state"].astype(str).isin({"blocked", "partial"})]
        blocked_summary = blocked.groupby("period").size().to_dict()
        pool2_blocked = blocked[blocked["pool_id"].astype(str).str.contains("tw50_dynamic_constituents", na=False)]
        pool2_blocked_summary = pool2_blocked.groupby("period").size().to_dict()
        for row in rows:
            row["blocked_or_partial_pool_rows"] = int(blocked_summary.get(row["period"], 0))
            row["pool2_blocked_or_partial_rows"] = int(pool2_blocked_summary.get(row["period"], 0))
    return pd.DataFrame(rows)


def _consensus_manifest_from_group(group: pd.DataFrame) -> dict[str, Any]:
    generated: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for item in group.to_dict(orient="records"):
        status = str(item.get("status") or "")
        row = {
            "pool_id": item.get("pool_id", ""),
            "pool_name": item.get("pool_name", ""),
            "vote_group": item.get("vote_group", ""),
            "top_ticker": item.get("top_ticker", ""),
            "top_display": item.get("top_display", ""),
            "top_asset_type": item.get("top_asset_type", ""),
            "rank_score": item.get("rank_score", item.get("score", "")),
            "score": item.get("score", ""),
            "base_pool_passed": _truthy(item.get("base_pool_passed", False)),
            "attack_gate_open": item.get("attack_gate_open", ""),
            "eligible_for_pool_selection": _truthy(item.get("eligible_for_pool_selection", False)),
            "selection_layer": item.get("selection_layer", ""),
            "selection_reason": item.get("selection_reason", item.get("reason", "")),
            "gate_rule_id": item.get("gate_rule_id", ""),
            "gate_reason": item.get("gate_reason", ""),
            "action_state": item.get("action_state", ""),
            "decision_layer": item.get("decision_layer", ""),
            "active_in_trade_decision": _truthy(item.get("active_in_trade_decision", False)),
            "source_module": item.get("source_module", ""),
        }
        if status == "generated":
            generated.append(row)
        else:
            skipped.append(
                {
                    **row,
                    "reason": item.get("reason", "") or item.get("selection_reason", "") or status or "skipped",
                    "dispatch": {"operational_observation": True},
                }
            )
    signal_date = str(group.iloc[0].get("effective_signal_date") or group.iloc[0].get("signal_date") or group.iloc[0].get("requested_signal_date") or "")
    return {"signal_date": signal_date, "generated": generated, "skipped": skipped}


def _effective_signal_date(row: pd.Series) -> str:
    signal_date = str(row.get("signal_date") or "").strip()
    if signal_date:
        return signal_date
    return str(row.get("requested_signal_date") or "").strip()


def _default_periods(cache_dir: str | Path) -> dict[str, tuple[str, str]]:
    latest = _latest_complete_signal_date(cache_dir)
    if not latest:
        latest = HISTORICAL_DEFAULT_PERIODS["2024_2026"][1]
    return {
        "2022": (DEFAULT_PERIOD_STARTS["2022"], DEFAULT_PERIOD_ENDS["2022"]),
        "2023": (DEFAULT_PERIOD_STARTS["2023"], DEFAULT_PERIOD_ENDS["2023"]),
        "2024_now": (DEFAULT_PERIOD_STARTS["2024_now"], latest),
    }


def _latest_complete_signal_date(cache_dir: str | Path) -> str:
    cache = Path(cache_dir)
    dates: list[pd.Timestamp] = []
    for ticker in ("0050.TW", "00631L.TW"):
        path = cache / f"{ticker.replace('.', '_')}.csv"
        if not path.exists():
            continue
        try:
            frame = load_price_csv(path)
        except Exception:
            continue
        if not frame.empty:
            dates.append(pd.Timestamp(frame.index.max()).normalize())
    if not dates:
        return ""
    return min(dates).strftime("%Y-%m-%d")


def _failed_rows(replay_panel: pd.DataFrame) -> list[dict[str, Any]]:
    if replay_panel.empty or "status" not in replay_panel.columns:
        return []
    failed = replay_panel[replay_panel["status"].astype(str).isin({"failed", "skipped"})]
    return failed.to_dict(orient="records")


def _readiness(
    *,
    periods: dict[str, tuple[str, str]],
    date_stride: int,
    max_dates: int | None,
    cache_only: bool,
    replay_panel: pd.DataFrame,
    health: pd.DataFrame,
    diagnostics: pd.DataFrame,
    failed_rows: list[dict[str, Any]],
    source_replay_panel: Path,
    latest_complete_signal_date: str,
) -> dict[str, Any]:
    pool2_blocked = 0
    if not diagnostics.empty:
        pool2 = diagnostics[diagnostics["pool_id"].astype(str).str.contains("tw50_dynamic_constituents", na=False)]
        pool2_blocked = int(pool2["data_readiness_state"].astype(str).isin({"blocked", "partial"}).sum())
    return {
        "status": "completed",
        "granularity": "daily" if date_stride == 1 else f"stride{date_stride}",
        "date_stride": date_stride,
        "max_dates": max_dates,
        "cache_only": cache_only,
        "periods": {key: {"start": start, "end": end} for key, (start, end) in periods.items()},
        "latest_complete_signal_date": latest_complete_signal_date,
        "source_replay_panel": str(source_replay_panel),
        "rows": {
            "replay_panel": int(len(replay_panel)),
            "health_history": int(len(health)),
            "pool_diagnostics_history": int(len(diagnostics)),
            "failed": int(len(failed_rows)),
        },
        "data_readiness": {
            "failed_or_skipped_replay_rows": int(len(failed_rows)),
            "pool2_blocked_or_partial_rows": pool2_blocked,
        },
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "limitations": [
            "health fields are report-only diagnostic and do not change formal vote",
            "observation_only rows remain excluded from formal vote",
            "Pool3 Radar satellite remains excluded from formal vote",
            "valuation and H3 day-trading/margin-overheat are not used",
            "if date_stride is greater than 1, rates are sampled and not full daily",
        ],
    }


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _rate(numerator: float | int, denominator: float | int) -> float:
    return round(float(numerator) / float(denominator), 4) if denominator else 0.0


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    columns = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _write_run_log(path: Path, event: str, status: str, detail: str) -> None:
    exists = path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["timestamp", "event", "status", "detail"])
        if not exists:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp": pd.Timestamp.now(tz="Asia/Taipei").strftime("%Y-%m-%d %H:%M:%S%z"),
                "event": event,
                "status": status,
                "detail": detail,
            }
        )


def _parse_periods(values: list[str] | None, cache_dir: str | Path) -> dict[str, tuple[str, str]]:
    if not values:
        return _default_periods(cache_dir)
    selected: dict[str, tuple[str, str]] = {}
    for value in values:
        name, raw_range = value.split("=", 1)
        start, end = raw_range.split(":", 1)
        selected[name] = (start, end)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Build report-only consensus health replay pack for stock-pool triad.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--replay-panel")
    parser.add_argument("--pool-store-path", default=DEFAULT_POOL_STORE_PATH)
    parser.add_argument("--cache-dir", default=DEFAULT_PRICE_CACHE_DIR)
    parser.add_argument("--warmup-start", default="2020-01-01")
    parser.add_argument("--period", action="append", help="name=start:end. Can repeat.")
    parser.add_argument("--date-stride", type=int, default=1)
    parser.add_argument("--max-dates", type=int)
    parser.add_argument("--tw50-constituents-path", default="data/tw50_constituents.csv")
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()
    output = run_stock_pool_consensus_health_replay(
        output_dir=args.output_dir,
        replay_panel_path=args.replay_panel,
        pool_store_path=args.pool_store_path,
        cache_dir=args.cache_dir,
        warmup_start=args.warmup_start,
        periods=_parse_periods(args.period, args.cache_dir),
        date_stride=args.date_stride,
        max_dates=args.max_dates,
        cache_only=not args.allow_download,
        tw50_constituents_path=args.tw50_constituents_path,
    )
    print(f"OUTPUT_DIR={output.resolve()}")


if __name__ == "__main__":
    main()
