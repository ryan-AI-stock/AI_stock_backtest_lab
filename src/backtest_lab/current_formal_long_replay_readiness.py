from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from backtest_lab.formal_model_contract import FORMAL_MODEL_ROUTE, FORMAL_MODEL_TARGET


DEFAULT_OUTPUT_DIR = "outputs/current_formal_long_replay_readiness_201411_20260630"
PIT_READINESS_MANIFEST = Path("outputs/core_0050_pit_candidate_backtest_data_readiness_201411_202312_20260629/manifest.json")
PRICE_ABSORPTION_MANIFEST = Path("outputs/core_0050_pit_price_coverage_absorption_201411_202312_20260629/manifest.json")
PHASE7_DEPENDENCY_MATRIX = Path("outputs/core_formal_target_stream_reconstruction_phase7_20260629/formal_target_stream_dependency_matrix.csv")


def run_current_formal_long_replay_readiness(
    *,
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
        log("load_existing_readiness", "started", "")
        pit = _read_json(PIT_READINESS_MANIFEST)
        price = _read_json(PRICE_ABSORPTION_MANIFEST)
        phase7 = _read_csv(PHASE7_DEPENDENCY_MATRIX)

        log("build_readiness_tables", "started", "")
        available = _available_inputs(pit, price)
        missing = _missing_inputs()
        blockers = _data_blockers()
        next_steps = _next_steps()
        target_stream_stub = _target_stream_stub()
        source_dependency = _source_dependency_snapshot(phase7)

        log("write_outputs", "started", "")
        available.to_csv(output / "available_inputs.csv", index=False, encoding="utf-8-sig")
        missing.to_csv(output / "missing_inputs.csv", index=False, encoding="utf-8-sig")
        blockers.to_csv(output / "data_blockers.csv", index=False, encoding="utf-8-sig")
        next_steps.to_csv(output / "next_steps.csv", index=False, encoding="utf-8-sig")
        target_stream_stub.to_csv(output / "candidate_formal_target_stream_blocked.csv", index=False, encoding="utf-8-sig")
        source_dependency.to_csv(output / "source_dependency_snapshot.csv", index=False, encoding="utf-8-sig")
        (output / "final_summary_zh.md").write_text(_summary_markdown(pit, price), encoding="utf-8")

        manifest = {
            "schema_version": 1,
            "task_id": "TASK-BACKTEST-CORE-CURRENT-FORMAL-LONG-REPLAY-READINESS-201411-20260630",
            "status": "blocked_by_missing_formal_target_signal_stream",
            "formal_model_target": FORMAL_MODEL_TARGET,
            "formal_model_route": FORMAL_MODEL_ROUTE,
            "requested_replay_start": "2014-11-03",
            "requested_first_period": "2014-11_to_2023-12",
            "requested_second_period": "2024-01_to_latest",
            "can_run_2014_2021_current_formal_replay_now": False,
            "can_emit_candidate_target_stream_now": False,
            "price_only_coverage_ready": bool(price.get("price_blocker_cleared_for_price_only", False)),
            "pit_candidate_monthly_anchor_ready": bool(pit.get("monthly_anchor_readable", False)),
            "pit_candidate_formal_exact": bool(pit.get("formal_exact", False)),
            "adjusted_close_total_return_ready": bool(price.get("adjusted_close_blocker_cleared", False)),
            "blocking_layer_count": int(len(blockers[blockers["blocks_long_replay"].eq(True)])),
            "formal_model_changed": False,
            "trade_decision_changed": False,
            "active_in_trade_decision": False,
            "execution_basis_for_future_replay": "next_day_full_rotation",
            "same_day_result_not_used_as_next_day_proof": True,
            "outputs": {
                "available_inputs": "available_inputs.csv",
                "missing_inputs": "missing_inputs.csv",
                "data_blockers": "data_blockers.csv",
                "next_steps": "next_steps.csv",
                "target_stream_blocked_stub": "candidate_formal_target_stream_blocked.csv",
                "summary": "final_summary_zh.md",
            },
        }
        (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        pd.DataFrame([{"status": "completed_readiness_blocked", "output_dir": str(output.resolve())}]).to_csv(
            output / "completed.csv", index=False, encoding="utf-8-sig"
        )
        pd.DataFrame(columns=["step", "error"]).to_csv(output / "failed.csv", index=False, encoding="utf-8-sig")
        log("completed", "completed", str(output.resolve()))
        (output / "current_step.txt").write_text("completed\n", encoding="utf-8")
        return output
    except Exception as exc:
        pd.DataFrame([{"step": "run_current_formal_long_replay_readiness", "error": str(exc)}]).to_csv(
            output / "failed.csv", index=False, encoding="utf-8-sig"
        )
        log("failed", "failed", str(exc))
        raise


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path).fillna("")


def _available_inputs(pit: dict[str, Any], price: dict[str, Any]) -> pd.DataFrame:
    rows = [
        {
            "input_name": "current_formal_contract",
            "status": "available",
            "coverage": "effective_from_2026-06-29_as_contract",
            "evidence": "src/backtest_lab/formal_model_contract.py",
            "notes": f"{FORMAL_MODEL_TARGET} / {FORMAL_MODEL_ROUTE}; contract exists but historical inputs are still missing.",
        },
        {
            "input_name": "0050_pcf_daily_monthly_anchor_candidate",
            "status": "available_candidate_not_exact",
            "coverage": f"{pit.get('covered_months', '')}/110 months; rows={pit.get('anchor_rows', '')}; tickers={pit.get('unique_anchor_tickers', '')}",
            "evidence": str(PIT_READINESS_MANIFEST.parent),
            "notes": "source_backed_manual_candidate; formal_exact=false; usable as PIT candidate layer, not exact TW50 official constituent truth.",
        },
        {
            "input_name": "pit_universe_price_only_coverage",
            "status": "available_price_only",
            "coverage": f"{price.get('price_only_ready_tickers', '')}/{price.get('pit_universe_tickers', '')} tickers",
            "evidence": str(PRICE_ABSORPTION_MANIFEST.parent),
            "notes": "Price-only coverage cleared for 76/76 PIT candidate tickers.",
        },
        {
            "input_name": "00631L_real_price_source_from_2014_11",
            "status": "available_price_source",
            "coverage": "2014-11-03 onward through supplemental TWSE source plus base cache",
            "evidence": "data/normalized_prices/00631L_twse_stock_day_201411_201512.csv; data/price_source_registry.csv",
            "notes": "synthetic_used=false; adjusted/distribution policy still needs final validation for total-return replay.",
        },
        {
            "input_name": "pool1_price_cache",
            "status": "available_with_listing_lifecycle_caveat",
            "coverage": "0050/00631L/2330/2454/2308/2317/2382/3231 from 2014-11; 6669 from 2017-11",
            "evidence": "backtest_cache/stock_pool_observations",
            "notes": "2014-2017 Pool1 must exclude not-yet-listed 6669 via formal listing eligibility contract; cannot forward-fill pre-listing history.",
        },
    ]
    return pd.DataFrame(rows)


def _missing_inputs() -> pd.DataFrame:
    rows = [
        {
            "input_name": "pool1_daily_candidate_ranking_panel_201411_202112",
            "required_for": "Pool1 vote / target / score margin / report top candidates",
            "missing_detail": "No formal daily Pool1 ranking panel exists for 2014-2021 using current report contract.",
            "blocker_type": "signal_feature_and_target_contract",
            "owner": "Core",
            "minimum_fix": "Build daily Pool1 ranking from price cache with listing eligibility, score, rank, rank2/rank3 margin, and raw winner.",
        },
        {
            "input_name": "pool1_listing_eligibility_contract",
            "required_for": "2014-2017 Pool1 universe correctness",
            "missing_detail": "6669 has no pre-2017 listing price; fixed Pool1 universe must be date-aware instead of treating missing ticker as fatal or forward-filled.",
            "blocker_type": "universe_lifecycle_policy",
            "owner": "Core",
            "minimum_fix": "Define not-yet-listed handling: exclude until first valid price/date with enough warmup; record exclusion reason.",
        },
        {
            "input_name": "pool2_daily_confirmation_panel_201411_202112",
            "required_for": "Pool2 vote / confirmation / disagreement state",
            "missing_detail": "0050 PIT candidate universe and prices exist price-only, but no daily Pool2 scoring/confirmation output has been rebuilt for 2014-2021.",
            "blocker_type": "signal_feature",
            "owner": "Core",
            "minimum_fix": "Use monthly anchor PIT candidate set by signal date, load price-only series, score candidates, emit pool2_vote and readiness flags.",
        },
        {
            "input_name": "formal_pool1_pool2_target_stream_201411_202112",
            "required_for": "current formal target stream / next-day replay",
            "missing_detail": "The current formal policy requires Pool1 vote plus Pool2 confirmation1 state. Without both vote streams, target_weights cannot be truthfully emitted.",
            "blocker_type": "target_stream",
            "owner": "Core",
            "minimum_fix": "Apply pool1_primary_pool2_confirmation policy to completed Pool1/Pool2 vote panel; emit target_weights/action/cash/no-target state.",
        },
        {
            "input_name": "next_day_execution_ledger_201411_202112",
            "required_for": "performance replay",
            "missing_detail": "Execution ledger depends on completed formal target stream; execution policy itself is fixed and available.",
            "blocker_type": "execution_dependency",
            "owner": "Core",
            "minimum_fix": "After target stream exists, run existing next-day full-rotation ledger with Taiwan cost model and 1,000,000 TWD base.",
        },
        {
            "input_name": "adjusted_close_total_return_policy_for_four_pit_tickers",
            "required_for": "total-return-quality historical replay",
            "missing_detail": "Four PIT universe tickers are unadjusted TWSE OHLCV only. Price-only replay can proceed, but total-return adjusted replay remains partial.",
            "blocker_type": "price_policy_not_target_stream_blocker",
            "owner": "Core/Data",
            "minimum_fix": "Either accept price-only for first pass and label it, or source adjusted/total-return series for the four tickers.",
        },
    ]
    return pd.DataFrame(rows)


def _data_blockers() -> pd.DataFrame:
    rows = [
        {
            "blocker_id": "missing_pool1_daily_candidate_ranking_panel",
            "blocks_long_replay": True,
            "layer": "Pool1 signal",
            "severity": "blocking",
            "detail": "Need daily Pool1 rank/score/winner/margin panel for 2014-2021 under current formal report contract.",
        },
        {
            "blocker_id": "missing_pool2_daily_confirmation_panel",
            "blocks_long_replay": True,
            "layer": "Pool2 signal",
            "severity": "blocking",
            "detail": "Need daily Pool2 vote/confirmation state using 0050 PIT monthly anchor candidate universe.",
        },
        {
            "blocker_id": "missing_formal_target_stream",
            "blocks_long_replay": True,
            "layer": "formal target",
            "severity": "blocking",
            "detail": "Cannot emit current formal target_weights until Pool1 and Pool2 vote streams exist.",
        },
        {
            "blocker_id": "execution_ledger_waits_for_target_stream",
            "blocks_long_replay": True,
            "layer": "execution",
            "severity": "blocking",
            "detail": "Next-day full switch ledger is available but cannot run without target stream.",
        },
        {
            "blocker_id": "four_tickers_unadjusted_only",
            "blocks_long_replay": False,
            "layer": "price policy",
            "severity": "caveat",
            "detail": "Price-only first pass can proceed if labeled; total-return adjusted replay remains partial until adjusted source is supplied.",
        },
    ]
    return pd.DataFrame(rows)


def _next_steps() -> pd.DataFrame:
    rows = [
        {
            "step_order": 1,
            "task_id": "TASK-BACKTEST-CORE-CURRENT-FORMAL-POOL1-POOL2-SIGNAL-PANELS-201411-202112-001",
            "owner": "Core",
            "action": "Build Pool1 daily ranking panel and Pool2 daily confirmation panel for 2014-11 to 2021-12.",
            "acceptance": "Outputs include pool1_vote, pool2_vote, score/rank/margin, PIT universe source date, listing eligibility exclusions, readiness flags.",
        },
        {
            "step_order": 2,
            "task_id": "TASK-BACKTEST-CORE-CURRENT-FORMAL-TARGET-STREAM-201411-202112-001",
            "owner": "Core",
            "action": "Apply current formal pool1_primary_pool2_confirmation policy to completed vote panels.",
            "acceptance": "Emit daily target stream with signal_date, target_weights, raw decision, action, cash/no-target state, and blockers=0.",
        },
        {
            "step_order": 3,
            "task_id": "TASK-BACKTEST-CORE-CURRENT-FORMAL-NEXT-DAY-REPLAY-201411-LATEST-001",
            "owner": "Core/Experiments",
            "action": "Run next-day full-rotation replay over 2014-11 to latest, split 2014-2023 and 2024-latest.",
            "acceptance": "Taiwan cost model, 1,000,000 TWD base, no same-day proof, caveat if price-only series is used.",
        },
    ]
    return pd.DataFrame(rows)


def _target_stream_stub() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "status": "blocked",
                "reason": "missing Pool1 daily ranking panel and Pool2 daily confirmation panel for 2014-2021",
                "can_emit_target_stream": False,
                "formal_model_target": FORMAL_MODEL_TARGET,
                "formal_model_route": FORMAL_MODEL_ROUTE,
            }
        ]
    )


def _source_dependency_snapshot(phase7: pd.DataFrame) -> pd.DataFrame:
    if phase7.empty:
        return pd.DataFrame(columns=["dependency_name", "can_reconstruct_now", "blocker_type", "notes"])
    cols = [column for column in ["dependency_name", "required_for", "can_reconstruct_now", "blocker_type", "notes"] if column in phase7.columns]
    return phase7[cols].copy()


def _summary_markdown(pit: dict[str, Any], price: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Current formal 長區間 replay readiness",
            "",
            f"- 目前正式模型：`{FORMAL_MODEL_TARGET}` / `{FORMAL_MODEL_ROUTE}`。",
            "- 結論：現在還不能直接做 2014/11 起的 current formal next-day replay。",
            "- 不是卡在 0050 PIT candidate 或 price-only coverage：0050 monthly anchor 已 110/110 months，PIT universe price-only coverage 已 76/76。",
            "- 真正 blocker 是 formal target/signal stream：2014-2021 尚缺 Pool1 每日排名票、Pool2 每日確認票，以及兩者合成的正式 target stream。",
            "",
            "## 已有資料",
            f"- 0050 PCF/Daily monthly anchor：covered_months={pit.get('covered_months', '')}，anchor_rows={pit.get('anchor_rows', '')}，formal_exact={pit.get('formal_exact', '')}。",
            f"- PIT universe price-only coverage：{price.get('price_only_ready_tickers', '')}/{price.get('pit_universe_tickers', '')}。",
            f"- adjusted close / total-return：adjusted_close_ready={price.get('adjusted_close_blocker_cleared', '')}；4 檔仍是 unadjusted-only caveat。",
            "- 00631L 真實價格 source 已延伸到 2014-11-03；不得使用 synthetic 0050x2 取代。",
            "",
            "## 下一步",
            "1. 先產 2014-2021 Pool1 / Pool2 daily signal panels。",
            "2. 再套 current formal `pool1_primary_pool2_confirmation` 產 target stream。",
            "3. 最後才跑 next-day full-rotation replay；不再研究 partial/staged/hold-old/cash buffer。",
            "",
            "本 package 不改正式模型、不改 trade decision。",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build current formal long replay readiness package.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    output = run_current_formal_long_replay_readiness(output_dir=args.output_dir)
    print(f"OUTPUT_DIR={output.resolve()}")


if __name__ == "__main__":
    main()
