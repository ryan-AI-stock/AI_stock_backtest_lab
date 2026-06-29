from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from backtest_lab.data import load_price_csv
from backtest_lab.supplemental_price_sources import DEFAULT_PRICE_SOURCE_REGISTRY, load_price_source_registry


TASK_ID = "TASK-BACKTEST-CORE-00631L-BACKFILL-CACHE-INTEGRATION-PHASE5-20260629"
DEFAULT_PHASE4_SOURCE = (
    "outputs/core_00631l_price_backfill_201411_201512_phase4_20260629/"
    "00631l_201411_201512_twse_stock_day_normalized.csv"
)
DEFAULT_OUTPUT_DIR = "outputs/core_00631l_backfill_cache_integration_phase5_20260629"
DEFAULT_NORMALIZED_PRICE_DIR = "data/normalized_prices"
DEFAULT_INTEGRATED_FILENAME = "00631L_twse_stock_day_201411_201512.csv"


def run_00631l_backfill_cache_integration(
    *,
    source_csv: str | Path = DEFAULT_PHASE4_SOURCE,
    normalized_price_dir: str | Path = DEFAULT_NORMALIZED_PRICE_DIR,
    registry_path: str | Path = DEFAULT_PRICE_SOURCE_REGISTRY,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    run_log: list[dict[str, str]] = []

    def log(step: str, status: str, detail: str = "") -> None:
        run_log.append(
            {
                "timestamp": pd.Timestamp.now(tz="Asia/Taipei").strftime("%Y-%m-%d %H:%M:%S%z"),
                "step": step,
                "status": status,
                "detail": detail,
            }
        )
        pd.DataFrame(run_log).to_csv(output / "run_log.csv", index=False, encoding="utf-8-sig")
        (output / "current_step.txt").write_text(step, encoding="utf-8")

    try:
        log("load_phase4_source", "started", str(source_csv))
        source = Path(source_csv)
        if not source.exists():
            raise FileNotFoundError(f"Phase 4 normalized source missing: {source}")
        frame = load_price_csv(source)
        _validate_phase4_frame(frame)

        log("write_normalized_price_source", "started", "")
        normalized_dir = Path(normalized_price_dir)
        normalized_dir.mkdir(parents=True, exist_ok=True)
        integrated_path = normalized_dir / DEFAULT_INTEGRATED_FILENAME
        frame_out = frame.copy()
        frame_out.index.name = "date"
        frame_out.reset_index().to_csv(integrated_path, index=False, encoding="utf-8-sig")

        log("update_registry", "started", str(registry_path))
        registry = _upsert_registry_row(load_price_source_registry(registry_path), integrated_path)
        Path(registry_path).parent.mkdir(parents=True, exist_ok=True)
        registry.to_csv(registry_path, index=False, encoding="utf-8-sig")

        log("write_outputs", "started", "")
        source_manifest = registry[registry["source_id"].eq("00631l_twse_stock_day_201411_201512")].copy()
        coverage = _coverage(frame, integrated_path)
        readiness = _readiness_ledger(coverage)
        manifest = _manifest(output, integrated_path, registry_path, coverage)
        (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        source_manifest.to_csv(output / "integrated_price_source_manifest.csv", index=False, encoding="utf-8-sig")
        coverage.to_csv(output / "00631l_price_coverage_after_integration.csv", index=False, encoding="utf-8-sig")
        readiness.to_csv(output / "readiness_ledger.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame([{"status": "completed", "output_dir": str(output.resolve())}]).to_csv(
            output / "completed.csv", index=False, encoding="utf-8-sig"
        )
        pd.DataFrame(columns=["step", "error"]).to_csv(output / "failed.csv", index=False, encoding="utf-8-sig")
        (output / "final_summary_zh.md").write_text(_summary_zh(manifest, coverage), encoding="utf-8")
        log("completed", "completed", str(output.resolve()))
        (output / "current_step.txt").write_text("completed\n", encoding="utf-8")
        return output
    except Exception as exc:
        pd.DataFrame([{"step": "run_00631l_backfill_cache_integration", "error": str(exc)}]).to_csv(
            output / "failed.csv", index=False, encoding="utf-8-sig"
        )
        log("failed", "failed", str(exc))
        raise


def _validate_phase4_frame(frame: pd.DataFrame) -> None:
    required = {"open", "high", "low", "close", "adj_close", "volume", "source", "source_type"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Phase 4 source missing columns: {', '.join(sorted(missing))}")
    source_types = set(frame["source_type"].astype(str))
    if source_types != {"official_real_price"}:
        raise ValueError(f"Unexpected source_type for 00631L backfill: {source_types}")
    if frame.index.min() > pd.Timestamp("2014-11-03"):
        raise ValueError("00631L backfill source does not cover 2014-11-03")
    if frame.index.max() < pd.Timestamp("2015-12-31"):
        raise ValueError("00631L backfill source does not cover 2015-12-31")


def _upsert_registry_row(registry: pd.DataFrame, integrated_path: Path) -> pd.DataFrame:
    row = {
        "ticker": "00631L.TW",
        "source_id": "00631l_twse_stock_day_201411_201512",
        "source_path": str(integrated_path.as_posix()),
        "source_type": "twse_stock_day_backfill",
        "first_date": "2014-11-03",
        "last_date": "2015-12-31",
        "price_source_ready": True,
        "strategy_ready": False,
        "synthetic_used": False,
        "provenance": "TASK-BACKTEST-CORE-00631L-PRICE-BACKFILL-201411-201512-PHASE4-20260629",
        "notes": "Official TWSE STOCK_DAY real price rows. Price-only source; PIT/target stream/execution ledger still required.",
    }
    registry = registry[registry.get("source_id", pd.Series(dtype=str)).astype(str).ne(row["source_id"])].copy()
    return pd.concat([registry, pd.DataFrame([row])], ignore_index=True)


def _coverage(frame: pd.DataFrame, integrated_path: Path) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ticker": "00631L.TW",
                "source_path": str(integrated_path.as_posix()),
                "source_type": "twse_stock_day_backfill",
                "first_date": frame.index.min().strftime("%Y-%m-%d"),
                "last_date": frame.index.max().strftime("%Y-%m-%d"),
                "row_count": int(len(frame)),
                "price_source_ready": True,
                "strategy_ready": False,
                "synthetic_used": False,
                "0050x2_proxy_used": False,
                "strategy_ready_blocker": "TW50 PIT constituents, formal target stream, execution ledger, and final adjusted-close/distribution policy are not complete.",
            }
        ]
    )


def _readiness_ledger(coverage: pd.DataFrame) -> pd.DataFrame:
    row = coverage.iloc[0].to_dict()
    return pd.DataFrame(
        [
            {
                "layer": "00631l_supplemental_price_source",
                "status": "ready",
                "price_source_ready": True,
                "strategy_ready": False,
                "detail": f"00631L supplemental source covers {row['first_date']} to {row['last_date']} with {row['row_count']} rows.",
            },
            {
                "layer": "strategy_replay_readiness",
                "status": "blocked_not_strategy_ready",
                "price_source_ready": True,
                "strategy_ready": False,
                "detail": row["strategy_ready_blocker"],
            },
        ]
    )


def _manifest(output: Path, integrated_path: Path, registry_path: str | Path, coverage: pd.DataFrame) -> dict[str, Any]:
    row = coverage.iloc[0].to_dict()
    return {
        "schema_version": 1,
        "task_id": TASK_ID,
        "status": "completed",
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "cache_overwritten": False,
        "source_registry_updated": True,
        "integrated_price_source_path": str(integrated_path.as_posix()),
        "registry_path": str(Path(registry_path).as_posix()),
        "ticker": "00631L.TW",
        "source_type": "twse_stock_day_backfill",
        "first_date": row["first_date"],
        "last_date": row["last_date"],
        "row_count": int(row["row_count"]),
        "price_source_ready": True,
        "strategy_ready": False,
        "synthetic_used": False,
        "0050x2_proxy_used": False,
        "output_dir": str(output.resolve()),
    }


def _summary_zh(manifest: dict[str, Any], coverage: pd.DataFrame) -> str:
    row = coverage.iloc[0].to_dict()
    return (
        "# 00631L 回補價格 source integration\n\n"
        "## 結論\n\n"
        f"- 已把 Phase 4 TWSE STOCK_DAY 真實價格資料接入可重用 supplemental price source：`{manifest['integrated_price_source_path']}`。\n"
        f"- coverage：{row['first_date']} 到 {row['last_date']}，共 {row['row_count']} 筆。\n"
        "- `price_source_ready=true`，但 `strategy_ready=false`。\n"
        "- 未覆蓋既有 `backtest_cache`，未使用 synthetic，也未使用 0050x2 proxy。\n\n"
        "## 為什麼還不是完整回測 ready\n\n"
        f"- {row['strategy_ready_blocker']}\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Integrate Phase 4 00631L TWSE backfill into reusable supplemental price source registry.")
    parser.add_argument("--source-csv", default=DEFAULT_PHASE4_SOURCE)
    parser.add_argument("--normalized-price-dir", default=DEFAULT_NORMALIZED_PRICE_DIR)
    parser.add_argument("--registry-path", default=DEFAULT_PRICE_SOURCE_REGISTRY)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    output = run_00631l_backfill_cache_integration(
        source_csv=args.source_csv,
        normalized_price_dir=args.normalized_price_dir,
        registry_path=args.registry_path,
        output_dir=args.output_dir,
    )
    print(f"00631L_BACKFILL_CACHE_INTEGRATION_OUTPUT={output.resolve()}")


if __name__ == "__main__":
    main()
