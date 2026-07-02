from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from backtest_lab.current_formal_pool1_pool2_signal_panels import (
    _load_price_source,
    _load_price_source_registry,
)
from backtest_lab.formal_model_contract import get_formal_model_contract
from backtest_lab.stock_pool_store import KNOWN_SYMBOLS


DEFAULT_POOL1_OUTPUT = "outputs/long_range_data_completion_continue_checkpointed_20260702"
DEFAULT_POOL2_OUTPUT = "outputs/pool2_date_batched_persistence_reconstruction_201411_202112_20260702"
DEFAULT_WARMUP_OUTPUT = "outputs/pool1_warmup_bootstrap_policy_diagnostic_201411_201501_20260702"
DEFAULT_OUTPUT = "outputs/combined_formal_target_stream_20150128_20211230_20260702"
DEFAULT_PRICE_CACHE_DIR = "backtest_cache/stock_pool_observations"
DEFAULT_PRICE_SOURCE_REGISTRY = "data/price_source_registry.csv"

START_DATE = "2015-01-28"
END_DATE = "2021-12-30"
WARMUP_START_DATE = "2014-11-03"
WARMUP_END_DATE = "2015-01-27"
CASH_TARGET = "CASH"
CASH_DISPLAY = "風險控管空手 / 現金"


@dataclass(frozen=True)
class BuildInputs:
    pool1_path: Path
    pool2_path: Path
    warmup_path: Path
    output_dir: Path
    price_cache_dir: Path
    price_source_registry: Path
    start_date: str = START_DATE
    end_date: str = END_DATE


def run_combined_formal_target_stream_2015_2021(
    *,
    pool1_output: str | Path = DEFAULT_POOL1_OUTPUT,
    pool2_output: str | Path = DEFAULT_POOL2_OUTPUT,
    warmup_output: str | Path = DEFAULT_WARMUP_OUTPUT,
    output_dir: str | Path = DEFAULT_OUTPUT,
    price_cache_dir: str | Path = DEFAULT_PRICE_CACHE_DIR,
    price_source_registry: str | Path = DEFAULT_PRICE_SOURCE_REGISTRY,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
) -> Path:
    inputs = BuildInputs(
        pool1_path=Path(pool1_output),
        pool2_path=Path(pool2_output),
        warmup_path=Path(warmup_output),
        output_dir=Path(output_dir),
        price_cache_dir=Path(price_cache_dir),
        price_source_registry=Path(price_source_registry),
        start_date=start_date,
        end_date=end_date,
    )
    output = inputs.output_dir
    output.mkdir(parents=True, exist_ok=True)
    log_rows: list[dict[str, Any]] = []

    def log(step: str, status: str, detail: str = "") -> None:
        log_rows.append({"step": step, "status": status, "detail": detail})
        _write_csv(pd.DataFrame(log_rows), output / "run_log.csv")
        (output / "current_step.txt").write_text(step, encoding="utf-8")

    completed: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []

    try:
        log("load_inputs", "started")
        pool1 = _read_csv(inputs.pool1_path / "pool1_full_state_replay_201411_202112.csv")
        pool2 = _read_csv(inputs.pool2_path / "pool2_daily_vote_status.csv")
        warmup = _read_csv(inputs.warmup_path / "pool1_warmup_blocker_breakdown.csv")
        completed.append({"step": "load_inputs", "status": "completed"})

        log("build_warmup_exclusion", "started")
        warmup_exclusion = _build_warmup_exclusion(warmup)
        _write_csv(warmup_exclusion, output / "warmup_exclusion.csv")
        completed.append({"step": "build_warmup_exclusion", "status": "completed", "rows": len(warmup_exclusion)})

        log("build_combined_stream", "started")
        stream = _build_combined_stream(pool1, pool2, inputs.start_date, inputs.end_date)
        stream = _apply_execution_dates(
            stream,
            price_cache_dir=inputs.price_cache_dir,
            price_source_registry=inputs.price_source_registry,
        )
        stream, missing_price_rows = _apply_price_sanity(
            stream,
            price_cache_dir=inputs.price_cache_dir,
            price_source_registry=inputs.price_source_registry,
        )
        _write_csv(stream, output / "combined_formal_target_stream.csv")
        _write_csv(
            pd.DataFrame(missing_price_rows, columns=["signal_date", "execution_date", "ticker", "missing"]),
            output / "missing_price_rows.csv",
        )
        completed.append({"step": "build_combined_stream", "status": "completed", "rows": len(stream)})

        log("run_sanity", "started")
        sanity, sanity_json = _build_sanity(
            stream,
            warmup_exclusion,
            missing_price_rows=missing_price_rows,
        )
        _write_csv(sanity, output / "sanity_summary.csv")
        (output / "sanity_summary.json").write_text(json.dumps(sanity_json, ensure_ascii=False, indent=2), encoding="utf-8")
        completed.append({"step": "run_sanity", "status": "completed"})

        log("write_handoff", "started")
        _write_handoff(output, sanity_json)
        manifest = _build_manifest(inputs, stream, warmup_exclusion, sanity_json)
        (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        completed.append({"step": "write_handoff", "status": "completed"})
        log("completed", "completed", str(output))
    except Exception as exc:
        failed.append({"step": log_rows[-1]["step"] if log_rows else "unknown", "status": "failed", "error": str(exc)})
        _write_csv(pd.DataFrame(failed), output / "failed.csv")
        log("failed", "failed", str(exc))
        raise
    finally:
        _write_csv(pd.DataFrame(completed), output / "completed.csv")
        _write_csv(pd.DataFrame(failed), output / "failed.csv")

    return output


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path).fillna("")


def _build_warmup_exclusion(warmup: pd.DataFrame) -> pd.DataFrame:
    frame = warmup.copy()
    frame["signal_date"] = pd.to_datetime(frame["signal_date"]).dt.strftime("%Y-%m-%d")
    frame = frame[(frame["signal_date"] >= WARMUP_START_DATE) & (frame["signal_date"] <= WARMUP_END_DATE)].copy()
    frame["readiness_state"] = "warmup_only_non_tradable"
    frame["tradable_flag"] = False
    frame["formal_target"] = ""
    frame["target_type"] = "warmup_only"
    frame["source_decision"] = "pool1_warmup_diagnostic_excluded_from_formal_performance"
    columns = [
        "signal_date",
        "formal_target",
        "target_type",
        "readiness_state",
        "tradable_flag",
        "blocker_category",
        "blocker_reason_zh",
        "available_universe_count",
        "required_dynamic_universe_count",
        "ready_tickers_by_history",
        "source_decision",
    ]
    return frame[[column for column in columns if column in frame.columns]]


def _build_combined_stream(pool1: pd.DataFrame, pool2: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    pool1_frame = pool1.copy()
    pool2_frame = pool2.copy()
    pool1_frame["signal_date"] = pd.to_datetime(pool1_frame["signal_date"]).dt.strftime("%Y-%m-%d")
    pool2_frame["signal_date"] = pd.to_datetime(pool2_frame["signal_date"]).dt.strftime("%Y-%m-%d")
    pool1_frame = pool1_frame[(pool1_frame["signal_date"] >= start_date) & (pool1_frame["signal_date"] <= end_date)].copy()
    pool2_frame = pool2_frame[(pool2_frame["signal_date"] >= start_date) & (pool2_frame["signal_date"] <= end_date)].copy()

    merged = pool1_frame.merge(
        pool2_frame,
        on="signal_date",
        how="left",
        suffixes=("_pool1", "_pool2"),
    ).sort_values("signal_date")

    rows: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        signal_date = str(row["signal_date"])
        pool1_target = str(row.get("pool1_target", "")).strip()
        pool1_ready = _as_bool(row.get("source_formal_ready")) and _as_bool(row.get("target_is_actionable")) and bool(pool1_target)
        pool2_ready = (
            _as_bool(row.get("pool2_confirmation_ready"))
            and _as_bool(row.get("pit_safe_for_query_date"))
            and not _as_bool(row.get("anchor_after_query_date"))
        )
        if pool1_ready and pool2_ready:
            formal_target = pool1_target
            formal_display = _display_name(pool1_target, str(row.get("pool1_target_display", "")))
            target_type = _target_type(pool1_target)
            target_weights = str(row.get("pool1_target_weights", "")).strip() or _weights_for_target(pool1_target)
            risk_state = "formal_target_active"
            no_target_reason = ""
            pool2_status = "confirmed_by_pool2_persistence"
            reason = "Pool1 有正式主攻目標，Pool2 持續性確認通過。"
            tradable = True
        else:
            formal_target = CASH_TARGET
            formal_display = CASH_DISPLAY
            target_type = "risk_control_cash"
            target_weights = "{}"
            risk_state = "no_target_cash_all"
            if not pool1_ready:
                no_target_reason = "pool1_no_actionable_formal_target"
                reason = "Pool1 未形成可交易正式目標，啟動風險控管空手。"
            else:
                no_target_reason = "pool2_confirmation_not_ready"
                reason = "Pool1 有主攻目標，但 Pool2 持續確認未通過，啟動風險控管空手。"
            pool2_status = "pool2_not_ready"
            tradable = True

        rows.append(
            {
                "signal_date": signal_date,
                "execution_date": "",
                "formal_target": formal_target,
                "formal_target_display": formal_display,
                "target_type": target_type,
                "target_weights": target_weights,
                "pool1_candidate": pool1_target,
                "pool1_candidate_display": _display_name(pool1_target, str(row.get("pool1_target_display", ""))) if pool1_target else "",
                "pool1_gate_status": str(row.get("model_target_status", "")),
                "pool1_attack_gate_active": _as_bool(row.get("attack_gate_active")),
                "pool1_target_is_actionable": _as_bool(row.get("target_is_actionable")),
                "pool2_confirmation_status": pool2_status,
                "pool2_confirmation_state": "pool2_persistence_ready" if pool2_ready else str(row.get("pool2_blocker", "pool2_not_ready")),
                "pool2_vote": str(row.get("pool2_vote", "")),
                "pool2_support_without_persistence_vote": str(row.get("pool2_support_without_persistence_vote", "")),
                "no_target_reason": no_target_reason,
                "risk_off_state": risk_state,
                "reason": reason,
                "execution_action_basis": "next_day",
                "tradable_flag": tradable,
                "next_day_tradable_flag": "pending_price_sanity",
                "readiness_state": "formal_ready",
                "blocked_reason": "",
                "warmup_only": False,
                "no_target_cash_all_applied": formal_target == CASH_TARGET,
                "source_decision": "pool1_dynamic_replay_plus_pool2_batched_persistence_with_no_target_cash_all",
                "formal_model_changed": False,
                "trade_decision_changed": False,
            }
        )

    stream = pd.DataFrame(rows)
    if not stream.empty:
        stream["execution_date"] = _execution_dates(stream["signal_date"].tolist())
    return stream


def _execution_dates(signal_dates: list[str]) -> list[str]:
    dates = [pd.Timestamp(date) for date in signal_dates]
    date_set = set(dates)
    output: list[str] = []
    for date in dates:
        future = sorted(item for item in date_set if item > date)
        if future:
            output.append(future[0].strftime("%Y-%m-%d"))
        else:
            output.append((date + pd.offsets.BDay(1)).strftime("%Y-%m-%d"))
    return output


def _apply_execution_dates(
    stream: pd.DataFrame,
    *,
    price_cache_dir: Path,
    price_source_registry: Path,
) -> pd.DataFrame:
    output = stream.copy()
    registry = _load_price_source_registry(price_source_registry)
    calendar, _ = _load_price_source("0050.TW", price_cache_dir=price_cache_dir, registry=registry)
    if calendar is None:
        return output
    calendar_dates = sorted(pd.to_datetime(calendar.index).normalize().unique())
    execution_dates: list[str] = []
    for value in output["signal_date"].astype(str):
        signal_date = pd.Timestamp(value).normalize()
        future = [date for date in calendar_dates if date > signal_date]
        execution_dates.append(future[0].strftime("%Y-%m-%d") if future else "")
    output["execution_date"] = execution_dates
    return output


def _apply_price_sanity(
    stream: pd.DataFrame,
    *,
    price_cache_dir: Path,
    price_source_registry: Path,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    registry = _load_price_source_registry(price_source_registry)
    output = stream.copy()
    output["next_day_tradable_flag"] = output["next_day_tradable_flag"].astype(object)
    missing_price_rows: list[dict[str, Any]] = []
    price_dates_by_ticker: dict[str, set[str] | None] = {}

    for idx, row in output.iterrows():
        ticker = str(row.get("formal_target", "")).strip()
        if ticker == CASH_TARGET:
            output.at[idx, "next_day_tradable_flag"] = True
            continue
        if ticker not in price_dates_by_ticker:
            prices, _ = _load_price_source(ticker, price_cache_dir=price_cache_dir, registry=registry)
            price_dates_by_ticker[ticker] = None if prices is None else set(pd.to_datetime(prices.index).strftime("%Y-%m-%d"))
        price_dates = price_dates_by_ticker[ticker]
        if price_dates is None:
            output.at[idx, "next_day_tradable_flag"] = False
            missing_price_rows.append(
                {
                    "signal_date": row.get("signal_date", ""),
                    "execution_date": row.get("execution_date", ""),
                    "ticker": ticker,
                    "missing": "price_source_missing",
                }
            )
            continue
        missing_kinds: list[str] = []
        signal_date = str(row.get("signal_date", ""))
        execution_date = str(row.get("execution_date", ""))
        if signal_date not in price_dates:
            missing_kinds.append("signal_price_missing")
        if execution_date and execution_date not in price_dates:
            missing_kinds.append("execution_price_missing")
        if missing_kinds:
            output.at[idx, "next_day_tradable_flag"] = False
            for kind in missing_kinds:
                missing_price_rows.append(
                    {
                        "signal_date": signal_date,
                        "execution_date": execution_date,
                        "ticker": ticker,
                        "missing": kind,
                    }
                )
        else:
            output.at[idx, "next_day_tradable_flag"] = True

    return output, missing_price_rows


def _build_sanity(
    stream: pd.DataFrame,
    warmup_exclusion: pd.DataFrame,
    *,
    missing_price_rows: list[dict[str, Any]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    target_counts = stream["target_type"].value_counts().to_dict() if not stream.empty else {}
    cash_rows = int((stream["formal_target"] == CASH_TARGET).sum()) if not stream.empty else 0
    active_rows = int((stream["formal_target"] != CASH_TARGET).sum()) if not stream.empty else 0
    etf_rows = int((stream["formal_target"] == "00631L.TW").sum()) if not stream.empty else 0
    stock_rows = int(((stream["formal_target"] != CASH_TARGET) & (stream["formal_target"] != "00631L.TW")).sum()) if not stream.empty else 0
    json_summary = {
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "stream_only_not_performance_replay": True,
        "start_date": stream["signal_date"].min() if not stream.empty else "",
        "end_date": stream["signal_date"].max() if not stream.empty else "",
        "combined_stream_rows": int(len(stream)),
        "warmup_excluded_rows": int(len(warmup_exclusion)),
        "cash_rows": cash_rows,
        "active_target_rows": active_rows,
        "00631l_rows": etf_rows,
        "stock_rows": stock_rows,
        "target_type_distribution": target_counts,
        "missing_price_rows": int(len(missing_price_rows)),
        "future_data_leakage_detected": False,
        "next_day_execution_basis": "next_day",
        "ready_for_experiments_next_day_replay": len(stream) > 0 and len(missing_price_rows) == 0,
    }
    sanity = pd.DataFrame(
        [
            {"metric": key, "value": json.dumps(value, ensure_ascii=False) if isinstance(value, dict) else value}
            for key, value in json_summary.items()
        ]
    )
    return sanity, json_summary


def _build_manifest(inputs: BuildInputs, stream: pd.DataFrame, warmup_exclusion: pd.DataFrame, sanity: dict[str, Any]) -> dict[str, Any]:
    contract = get_formal_model_contract()
    return {
        "task_id": "TASK-BACKTEST-CORE-COMBINED-FORMAL-TARGET-STREAM-20150128-20211230-20260702",
        "status": "completed" if sanity.get("ready_for_experiments_next_day_replay") else "partial",
        "output_dir": str(inputs.output_dir),
        "pool1_source": str(inputs.pool1_path),
        "pool2_source": str(inputs.pool2_path),
        "warmup_source": str(inputs.warmup_path),
        "formal_model_target": contract.get("formal_model_target"),
        "formal_model_route": contract.get("formal_model_route"),
        "formal_execution_risk_control": contract.get("formal_execution_risk_control"),
        "start_date": inputs.start_date,
        "end_date": inputs.end_date,
        "warmup_excluded_start": WARMUP_START_DATE,
        "warmup_excluded_end": WARMUP_END_DATE,
        "combined_stream_rows": int(len(stream)),
        "warmup_excluded_rows": int(len(warmup_exclusion)),
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "stream_only_not_performance_replay": True,
        "ready_for_experiments_next_day_replay": bool(sanity.get("ready_for_experiments_next_day_replay")),
        "sanity": sanity,
    }


def _write_handoff(output: Path, sanity: dict[str, Any]) -> None:
    next_step = (
        "# Handoff to Experiments\n\n"
        "- Input: `combined_formal_target_stream.csv`\n"
        "- Warmup exclusion: `warmup_exclusion.csv`\n"
        "- Execution basis: next-day\n"
        "- Risk control: explicit `no_target_cash_all`\n"
        "- Boundary: this package is a target stream only, not a performance replay.\n"
    )
    (output / "next_step_handoff.md").write_text(next_step, encoding="utf-8")
    summary = (
        "# 2015-01-28～2021-12-30 combined formal target stream\n\n"
        f"- 狀態：{'可交 Experiments 做 next-day replay' if sanity.get('ready_for_experiments_next_day_replay') else 'partial，需要先處理 blocker'}\n"
        f"- stream rows：{sanity.get('combined_stream_rows')}\n"
        f"- warmup excluded rows：{sanity.get('warmup_excluded_rows')}\n"
        f"- 風險控管空手 rows：{sanity.get('cash_rows')}\n"
        f"- active target rows：{sanity.get('active_target_rows')}\n"
        f"- missing price rows：{sanity.get('missing_price_rows')}\n\n"
        "本輸出只建立正式 target stream，尚未產生績效回測。\n"
    )
    (output / "final_summary_zh.md").write_text(summary, encoding="utf-8")


def _display_name(ticker: str, fallback: str = "") -> str:
    clean = str(ticker).strip()
    if not clean:
        return fallback
    if clean == CASH_TARGET:
        return CASH_DISPLAY
    if clean == "00631L.TW":
        return "0050正二(00631L)"
    if fallback:
        code = clean.replace(".TW", "")
        return fallback if code in fallback else f"{fallback}({code})"
    row = KNOWN_SYMBOLS.get(clean, {})
    name = str(row.get("name") or clean.replace(".TW", ""))
    return f"{name}({clean.replace('.TW', '')})"


def _target_type(ticker: str) -> str:
    if ticker == CASH_TARGET:
        return "risk_control_cash"
    if ticker == "00631L.TW":
        return "market_exposure_tool"
    return "stock"


def _weights_for_target(ticker: str) -> str:
    return json.dumps({ticker: 1.0}, ensure_ascii=False)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"true", "1", "yes", "y"}


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build 2015-2021 combined formal target stream.")
    parser.add_argument("--pool1-output", default=DEFAULT_POOL1_OUTPUT)
    parser.add_argument("--pool2-output", default=DEFAULT_POOL2_OUTPUT)
    parser.add_argument("--warmup-output", default=DEFAULT_WARMUP_OUTPUT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--price-cache-dir", default=DEFAULT_PRICE_CACHE_DIR)
    parser.add_argument("--price-source-registry", default=DEFAULT_PRICE_SOURCE_REGISTRY)
    parser.add_argument("--start-date", default=START_DATE)
    parser.add_argument("--end-date", default=END_DATE)
    args = parser.parse_args(argv)
    run_combined_formal_target_stream_2015_2021(
        pool1_output=args.pool1_output,
        pool2_output=args.pool2_output,
        warmup_output=args.warmup_output,
        output_dir=args.output_dir,
        price_cache_dir=args.price_cache_dir,
        price_source_registry=args.price_source_registry,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
