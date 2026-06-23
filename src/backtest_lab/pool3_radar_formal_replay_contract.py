from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_OUTPUT_DIR = "outputs/pool3_radar_formal_replay_contract_20260623"
BASELINE_SCHEMA = (
    "date",
    "pool1_vote",
    "pool2_vote",
    "pool3_vote",
    "consensus_state",
    "winner_ticker",
    "position_ticker",
    "cash",
    "equity",
    "drawdown",
    "turnover",
    "transaction_cost",
)
OVERLAY_SCHEMA = (
    "date",
    "variant",
    "pool3_formal_vote",
    "holding_ticker",
    "holding_name",
    "theme",
    "weight",
    "shares",
    "fill_action",
    "fill_price",
    "cash",
    "position_value",
    "transaction_cost",
    "equity",
)
DECISION_DIFF_SCHEMA = (
    "date",
    "variant",
    "baseline_pool3_vote",
    "challenger_pool3_vote",
    "baseline_winner",
    "challenger_winner",
    "changed_final_consensus",
    "changed_reason",
    "change_source",
)


def run_pool3_radar_formal_replay_contract(
    *,
    output_dir: str | Path,
    baseline_daily: str | Path | None = None,
    overlay_daily: str | Path | None = None,
    readiness_manifest: str | Path | None = None,
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

    log("build_contract", "started", "")
    _write_schema_templates(output)
    baseline_checks = _validate_optional_frame(
        path=Path(baseline_daily) if baseline_daily else None,
        required_columns=BASELINE_SCHEMA,
        frame_name="baseline_three_pool_formal_daily",
        reject_proxy=True,
    )
    overlay_checks = _validate_optional_frame(
        path=Path(overlay_daily) if overlay_daily else None,
        required_columns=OVERLAY_SCHEMA,
        frame_name="pool3_radar_weighted_overlay_formal_daily",
        reject_proxy=False,
    )
    baseline_ready = all(row["passed"] for row in baseline_checks)
    overlay_ready = all(row["passed"] for row in overlay_checks)
    readiness = _read_json(Path(readiness_manifest) if readiness_manifest else None)
    engineering_checks = [
        {
            "check_id": "core_engineering:baseline_and_overlay_ready",
            "passed": baseline_ready and overlay_ready,
            "severity": "blocker",
            "detail": "formal baseline replay and weighted overlay accounting are available",
        }
    ]
    readiness_checks = _readiness_checks(
        readiness,
        baseline_ready=baseline_ready,
        overlay_accounting_ready=overlay_ready,
    )
    checks = pd.DataFrame(baseline_checks + overlay_checks + engineering_checks + readiness_checks)
    accepted = bool(not checks.empty and checks["passed"].all())
    manifest = {
        "model": "pool3_radar_formal_replay_contract_v1",
        "status": "contract_ready_inputs_pending" if not accepted else "inputs_pass_contract",
        "active_in_trade_decision": False,
        "formal_model_changed": False,
        "pool3_formal_vote_changed": False,
        "purpose": "Define the formal daily replay/accounting contract required before Pool3 Radar can be tested as a formal challenger.",
        "baseline_schema": list(BASELINE_SCHEMA),
        "overlay_schema": list(OVERLAY_SCHEMA),
        "decision_diff_schema": list(DECISION_DIFF_SCHEMA),
        "core_engineering_inputs_ready": baseline_ready and overlay_ready,
        "radar_data_readiness_ready": all(row["passed"] for row in readiness_checks),
        "accepted_for_formal_challenger_replay": accepted,
        "failed_checks": checks[checks["passed"] == False].to_dict(orient="records") if not checks.empty else [],  # noqa: E712
        "next_runner_expectation": (
            "Experiments should produce baseline and overlay daily files matching these schemas; "
            "Core can then compute formal decision diff without compressing the Radar basket into one ticker."
        ),
    }
    checks.to_csv(output / "formal_replay_contract_checks.csv", index=False, encoding="utf-8-sig")
    (output / "formal_replay_contract.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "formal_replay_contract.md").write_text(_markdown_contract(manifest), encoding="utf-8")
    (output / "completed.txt").write_text("completed\n", encoding="utf-8")
    log("completed", "completed", manifest["status"])
    (output / "current_step.txt").write_text("completed\n", encoding="utf-8")
    return output


def _write_schema_templates(output: Path) -> None:
    pd.DataFrame(columns=BASELINE_SCHEMA).to_csv(output / "required_baseline_three_pool_formal_daily_schema.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(columns=OVERLAY_SCHEMA).to_csv(output / "required_pool3_radar_weighted_overlay_formal_daily_schema.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(columns=DECISION_DIFF_SCHEMA).to_csv(output / "required_formal_decision_diff_schema.csv", index=False, encoding="utf-8-sig")


def _validate_optional_frame(
    *,
    path: Path | None,
    required_columns: tuple[str, ...],
    frame_name: str,
    reject_proxy: bool,
) -> list[dict[str, Any]]:
    if path is None:
        return [
            {
                "check_id": f"{frame_name}:provided",
                "passed": False,
                "severity": "blocker",
                "detail": "input file not provided",
            }
        ]
    if not path.exists():
        return [
            {
                "check_id": f"{frame_name}:exists",
                "passed": False,
                "severity": "blocker",
                "detail": str(path),
            }
        ]
    frame = pd.read_csv(path).fillna("")
    rows: list[dict[str, Any]] = []
    missing = [col for col in required_columns if col not in frame.columns]
    rows.append(
        {
            "check_id": f"{frame_name}:schema",
            "passed": not missing,
            "severity": "blocker",
            "detail": "missing=" + ",".join(missing) if missing else "ok",
        }
    )
    rows.append(
        {
            "check_id": f"{frame_name}:rows",
            "passed": len(frame) > 0,
            "severity": "blocker",
            "detail": f"rows={len(frame)}",
        }
    )
    if reject_proxy:
        proxy_cols = [col for col in ("data_status", "source_status", "status") if col in frame.columns]
        proxy_found = any(frame[col].astype(str).str.lower().str.contains("proxy|stride20|partial").any() for col in proxy_cols)
        rows.append(
            {
                "check_id": f"{frame_name}:not_proxy",
                "passed": not proxy_found,
                "severity": "blocker",
                "detail": "baseline daily replay must not be partial/stride20 proxy",
            }
        )
    if frame_name == "pool3_radar_weighted_overlay_formal_daily":
        rows.append(
            {
                "check_id": f"{frame_name}:basket_not_single_vote",
                "passed": {"holding_ticker", "weight"}.issubset(frame.columns),
                "severity": "blocker",
                "detail": "weighted basket must preserve holding_ticker and weight",
            }
        )
    return rows


def _readiness_checks(
    readiness: dict[str, Any],
    *,
    baseline_ready: bool,
    overlay_accounting_ready: bool,
) -> list[dict[str, Any]]:
    if not readiness:
        return [
            {
                "check_id": "radar_readiness:provided",
                "passed": False,
                "severity": "blocker",
                "detail": "readiness manifest not provided",
            }
        ]
    blockers = _normalize_readiness_blockers(
        readiness.get("blockers") or [],
        baseline_ready=baseline_ready,
        overlay_accounting_ready=overlay_accounting_ready,
    )
    return [
        {
            "check_id": "radar_readiness:formal_absorb_flag",
            "passed": readiness.get("can_core_absorb_as_formal_challenger") is True,
            "severity": "blocker",
            "detail": str(readiness.get("can_core_absorb_as_formal_challenger")),
        },
        {
            "check_id": "radar_readiness:no_blockers",
            "passed": len(blockers) == 0,
            "severity": "blocker",
            "detail": "; ".join(str(item) for item in blockers),
        },
    ]


def _normalize_readiness_blockers(
    blockers: list[Any],
    *,
    baseline_ready: bool,
    overlay_accounting_ready: bool,
) -> list[str]:
    normalized: list[str] = []
    for item in blockers:
        text = str(item)
        if baseline_ready and "baseline_three_pool_daily_equity is partial proxy" in text:
            continue
        if overlay_accounting_ready and "primary overlay daily basket is synthetic blend" in text:
            if "synthetic blend" in text:
                normalized.append("primary overlay daily basket is synthetic blend")
            continue
        normalized.append(text)
    return normalized


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _markdown_contract(manifest: dict[str, Any]) -> str:
    lines = [
        "# Pool3 Radar Formal Replay Contract",
        "",
        f"- status: `{manifest['status']}`",
        f"- active_in_trade_decision: `{manifest['active_in_trade_decision']}`",
        f"- core_engineering_inputs_ready: `{manifest.get('core_engineering_inputs_ready')}`",
        f"- radar_data_readiness_ready: `{manifest.get('radar_data_readiness_ready')}`",
        f"- accepted_for_formal_challenger_replay: `{manifest['accepted_for_formal_challenger_replay']}`",
        "",
        "## Required Baseline Daily Columns",
        "",
        ", ".join(manifest["baseline_schema"]),
        "",
        "## Required Pool3 Radar Overlay Daily Columns",
        "",
        ", ".join(manifest["overlay_schema"]),
        "",
        "## Required Decision Diff Columns",
        "",
        ", ".join(manifest["decision_diff_schema"]),
        "",
        "## Failed Checks",
        "",
    ]
    failed = manifest.get("failed_checks") or []
    if not failed:
        lines.append("- 無。")
    else:
        for row in failed:
            lines.append(f"- {row.get('check_id')}: {row.get('detail')}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Pool3 Radar formal daily replay contract outputs.")
    parser.add_argument("--baseline-daily", default="")
    parser.add_argument("--overlay-daily", default="")
    parser.add_argument("--readiness-manifest", default="")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    output = run_pool3_radar_formal_replay_contract(
        baseline_daily=args.baseline_daily or None,
        overlay_daily=args.overlay_daily or None,
        readiness_manifest=args.readiness_manifest or None,
        output_dir=args.output_dir,
    )
    print(f"OUTPUT_DIR={output.resolve()}")


if __name__ == "__main__":
    main()
