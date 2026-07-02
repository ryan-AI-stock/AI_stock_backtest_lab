from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from backtest_lab.current_formal_pool1_pool2_signal_panels import (
    DEFAULT_PRICE_CACHE_DIR,
    DEFAULT_PRICE_SOURCE_REGISTRY,
    POOL1_TICKERS,
    TW50_BENCHMARK,
    _load_price_source_registry,
)
from backtest_lab.long_range_data_completion_continue import _extend_benchmark_with_best_local_source
from backtest_lab.pool1_dynamic_adapter_2022_equivalence import _load_required_prices


TASK_ID = "TASK-BACKTEST-CORE-LONG-RANGE-CHECKPOINT-SANITY-POOL2-PERSISTENCE-20260702"
DEFAULT_SOURCE_OUTPUT = "outputs/long_range_data_completion_continue_checkpointed_20260702"
DEFAULT_OUTPUT = "outputs/long_range_checkpoint_sanity_pool2_persistence_20260702"


def run_checkpoint_sanity_package(
    *,
    source_output: str | Path = DEFAULT_SOURCE_OUTPUT,
    output_dir: str | Path = DEFAULT_OUTPUT,
    price_cache_dir: str | Path = DEFAULT_PRICE_CACHE_DIR,
    price_source_registry: str | Path = DEFAULT_PRICE_SOURCE_REGISTRY,
) -> Path:
    source = Path(source_output)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    run_log: list[dict[str, str]] = []

    def log(step: str, status: str, detail: str = "") -> None:
        run_log.append({"step": step, "status": status, "detail": detail})
        pd.DataFrame(run_log).to_csv(output / "run_log.csv", index=False, encoding="utf-8-sig")
        (output / "current_step.txt").write_text(step, encoding="utf-8")

    try:
        log("load_source_output", "started", str(source))
        manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
        checkpoints = _read_optional_csv(source / "pool1_dynamic_state_checkpoints.csv")
        pool2_gate = _read_optional_csv(source / "pool2_gate_breakdown.csv")

        log("write_price_continuity_evidence", "started", "")
        price_evidence = _price_continuity_evidence(price_cache_dir, price_source_registry)
        price_evidence.to_csv(output / "00631l_price_continuity_evidence.csv", index=False, encoding="utf-8-sig")

        log("write_pool1_checkpoint_sanity", "started", "")
        sanity = _pool1_checkpoint_sanity(checkpoints, manifest)
        sanity.to_csv(output / "pool1_checkpoint_sanity.csv", index=False, encoding="utf-8-sig")

        log("write_pool2_diagnosis", "started", "")
        diagnosis = _pool2_persistence_gate_diagnosis(pool2_gate)
        diagnosis.to_csv(output / "pool2_persistence_gate_diagnosis.csv", index=False, encoding="utf-8-sig")

        log("write_summary", "started", "")
        package_manifest = {
            "schema_version": 1,
            "task_id": TASK_ID,
            "status": "completed_partial_precise_blocker",
            "source_long_range_output": str(source),
            "pool1_rows": int(manifest.get("pool1_rows", 0)),
            "pool1_blocked_rows": int(manifest.get("pool1_blocked_rows", 0)),
            "pool1_remaining_blocker": "2014-11-03..2015-01-27 warmup/insufficient dynamic universe",
            "pool2_blocker": (
                "persistence gate was hardcoded false and is fixed in code; "
                "full panel rerun needs a date-batched/checkpointed Pool2 runner"
            ),
            "formal_model_changed": False,
            "trade_decision_changed": False,
            "active_in_trade_decision": False,
            "outputs": {
                "pool1_checkpoint_sanity": "pool1_checkpoint_sanity.csv",
                "price_continuity_evidence": "00631l_price_continuity_evidence.csv",
                "pool2_persistence_gate_diagnosis": "pool2_persistence_gate_diagnosis.csv",
                "handoff": "next_step_handoff.md",
                "summary": "final_summary_zh.md",
            },
        }
        (output / "manifest.json").write_text(json.dumps(package_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        (output / "next_step_handoff.md").write_text(_handoff_text(), encoding="utf-8")
        (output / "final_summary_zh.md").write_text(_summary_text(package_manifest), encoding="utf-8")
        pd.DataFrame([{"step": TASK_ID, "status": "completed_partial_precise_blocker", "output_dir": str(output.resolve())}]).to_csv(
            output / "completed.csv",
            index=False,
            encoding="utf-8-sig",
        )
        pd.DataFrame(columns=["step", "status", "reason"]).to_csv(output / "failed.csv", index=False, encoding="utf-8-sig")
        log("completed", "completed", str(output.resolve()))
        (output / "current_step.txt").write_text("completed\n", encoding="utf-8")
        return output
    except Exception as exc:
        pd.DataFrame([{"step": TASK_ID, "status": "failed", "reason": str(exc)}]).to_csv(
            output / "failed.csv",
            index=False,
            encoding="utf-8-sig",
        )
        log("failed", "failed", str(exc))
        raise


def _read_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path).fillna("")


def _price_continuity_evidence(price_cache_dir: str | Path, price_source_registry: str | Path) -> pd.DataFrame:
    registry = _load_price_source_registry(price_source_registry)
    prices, meta = _load_required_prices(
        price_cache_dir=price_cache_dir,
        registry=registry,
        required_tickers=sorted(set(POOL1_TICKERS) | {TW50_BENCHMARK}),
    )
    prices, _meta = _extend_benchmark_with_best_local_source(
        prices=prices,
        price_meta=meta,
        price_cache_dir=price_cache_dir,
        benchmark=TW50_BENCHMARK,
    )
    frame = prices["00631L.TW"].loc[
        (prices["00631L.TW"].index >= pd.Timestamp("2015-03-18"))
        & (prices["00631L.TW"].index <= pd.Timestamp("2015-03-24")),
        ["open", "close", "adj_close", "price_source_type"],
    ]
    return frame.reset_index(names="date")


def _pool1_checkpoint_sanity(checkpoints: pd.DataFrame, manifest: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if checkpoints.empty:
        return pd.DataFrame(
            [
                {
                    "check": "state_exported_all_chunks",
                    "status": "blocked",
                    "detail": "no checkpoint rows",
                }
            ]
        )
    state_exported = checkpoints["state_exported"].astype(str).str.lower().isin({"true", "1"}).all()
    max_cash = pd.to_numeric(checkpoints["account_cash"], errors="coerce").abs().max()
    rows.append({"check": "state_exported_all_chunks", "status": "pass" if state_exported else "fail", "detail": f"chunks={len(checkpoints)}"})
    rows.append(
        {
            "check": "state_value_no_trillion_scale_after_price_fix",
            "status": "pass" if max_cash < 1e10 else "fail",
            "detail": f"max_abs_cash={max_cash}",
        }
    )
    rows.append(
        {
            "check": "pool1_checkpoint_progress",
            "status": "partial",
            "detail": f"pool1_rows={manifest.get('pool1_rows')}; blocked_rows={manifest.get('pool1_blocked_rows')}",
        }
    )
    return pd.DataFrame(rows)


def _pool2_persistence_gate_diagnosis(pool2_gate: pd.DataFrame) -> pd.DataFrame:
    rows = pool2_gate.to_dict(orient="records") if not pool2_gate.empty else []
    rows.append(
        {
            "gate": "persistence_gate_code_path",
            "rows_true": "",
            "rows_false": "",
            "rows_missing": "",
            "status": "fixed_in_code_not_full_panel_rerun",
            "blocker_class": "runner_checkpoint_missing",
        }
    )
    rows.append(
        {
            "gate": "pool2_full_panel_rerun",
            "rows_true": "",
            "rows_false": "",
            "rows_missing": "",
            "status": "blocked_needs_batched_checkpoint_runner",
            "blocker_class": "runner_observability",
        }
    )
    return pd.DataFrame(rows)


def _handoff_text() -> str:
    return """# Next step handoff

- Pool1: 00631L price-source priority bug fixed. Checkpointed dynamic replay reaches 2015-01-28..2021-12-30 with 1689 rows; remaining 60 rows are 2014-11-03..2015-01-27 warmup/insufficient universe.
- Pool2: persistence gate was hardcoded false in current_formal_pool1_pool2_signal_panels.py and is now wired to _tw50_persistence_days.
- Pool2 full panel rerun is still blocked by runner observability/performance. Build a date-batched/checkpointed Pool2 panel runner next.
- Combined formal stream is not ready until Pool2 eligible rows are regenerated and validated.
"""


def _summary_text(manifest: dict[str, Any]) -> str:
    return f"""# Long-range formal replay checkpoint sanity + Pool2 persistence

- status: {manifest['status']}
- Pool1 rows: {manifest['pool1_rows']}
- Pool1 blocked rows: {manifest['pool1_blocked_rows']}
- Pool1 blocker: {manifest['pool1_remaining_blocker']}
- Pool2 blocker: {manifest['pool2_blocker']}
- formal_model_changed=false
- trade_decision_changed=false
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Package long-range checkpoint sanity and Pool2 persistence diagnosis.")
    parser.add_argument("--source-output", default=DEFAULT_SOURCE_OUTPUT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--price-cache-dir", default=DEFAULT_PRICE_CACHE_DIR)
    parser.add_argument("--price-source-registry", default=DEFAULT_PRICE_SOURCE_REGISTRY)
    args = parser.parse_args(argv)
    output = run_checkpoint_sanity_package(
        source_output=args.source_output,
        output_dir=args.output_dir,
        price_cache_dir=args.price_cache_dir,
        price_source_registry=args.price_source_registry,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
