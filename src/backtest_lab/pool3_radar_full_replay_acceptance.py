from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_OUTPUT_DIR = "outputs/pool3_radar_full_replay_acceptance_20260623"
PRIMARY_VARIANT = "ma200_radar20_00631l80_else_top10"
REQUIRED_FILES = (
    "pool3_radar_weighted_basket_daily.csv",
    "baseline_three_pool_daily_equity.csv",
    "pool3_radar_full_replay_decision_diff.csv",
    "pool3_radar_full_replay_summary.csv",
    "concentration_by_ticker_theme_month_quarter.csv",
    "readiness_manifest.json",
    "formal_absorption_blocker_audit.md",
)


def run_pool3_radar_full_replay_acceptance(
    *,
    full_replay_dir: str | Path,
    output_dir: str | Path,
) -> Path:
    source = Path(full_replay_dir)
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

    log("load_full_replay_pack", "started", str(source))
    manifest = _read_json(source / "readiness_manifest.json")
    summary = _read_csv(source / "pool3_radar_full_replay_summary.csv")
    checks = _build_checks(source, manifest=manifest, summary=summary)
    decision = _build_acceptance_decision(manifest=manifest, checks=checks, summary=summary)
    checks.to_csv(output / "core_acceptance_checks.csv", index=False, encoding="utf-8-sig")
    (output / "core_acceptance_manifest.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output / "core_acceptance_summary.md").write_text(_markdown_summary(decision, checks), encoding="utf-8")
    (output / "completed.txt").write_text("completed\n", encoding="utf-8")
    log("completed", "completed", decision["core_decision"])
    (output / "current_step.txt").write_text("completed\n", encoding="utf-8")
    return output


def _build_checks(source: Path, *, manifest: dict[str, Any], summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for filename in REQUIRED_FILES:
        path = source / filename
        rows.append(
            {
                "check_id": f"file_exists:{filename}",
                "passed": path.exists(),
                "severity": "blocker",
                "detail": str(path),
            }
        )
    rows.append(
        {
            "check_id": "can_core_absorb_as_formal_challenger",
            "passed": manifest.get("can_core_absorb_as_formal_challenger") is True,
            "severity": "blocker",
            "detail": str(manifest.get("can_core_absorb_as_formal_challenger")),
        }
    )
    blockers = manifest.get("blockers") or []
    rows.append(
        {
            "check_id": "manifest_blockers_empty",
            "passed": len(blockers) == 0,
            "severity": "blocker",
            "detail": "; ".join(str(item) for item in blockers),
        }
    )
    rows.append(
        {
            "check_id": "baseline_is_formal_daily_replay",
            "passed": not any("partial proxy" in str(item).lower() or "stride20" in str(item).lower() for item in blockers),
            "severity": "blocker",
            "detail": "baseline three-pool daily equity must be formal daily replay, not proxy",
        }
    )
    rows.append(
        {
            "check_id": "overlay_has_transaction_cost_accounting",
            "passed": not any("transaction-cost realistic" in str(item).lower() for item in blockers),
            "severity": "blocker",
            "detail": "overlay basket must have trade/cost accounting before formal challenger",
        }
    )
    hard_gate_rows = _primary_2024_rows(summary)
    if hard_gate_rows.empty:
        rows.append(
            {
                "check_id": "hard_gate_2024_primary_present",
                "passed": False,
                "severity": "blocker",
                "detail": "primary 2024 row missing",
            }
        )
    else:
        row = hard_gate_rows.iloc[0]
        mdd = _number(row.get("max_drawdown_pct"))
        excess_0050 = _number(row.get("excess_vs_0050_pct"))
        excess_00631l = _number(row.get("excess_vs_00631l_pct"))
        rows.extend(
            [
                {
                    "check_id": "hard_gate_2024_mdd",
                    "passed": mdd > -25,
                    "severity": "blocker",
                    "detail": f"MDD={mdd:.4f}%",
                },
                {
                    "check_id": "hard_gate_2024_excess_vs_0050",
                    "passed": excess_0050 >= 5,
                    "severity": "blocker",
                    "detail": f"excess_vs_0050={excess_0050:.4f}pp",
                },
                {
                    "check_id": "hard_gate_2024_excess_vs_00631l",
                    "passed": excess_00631l >= 0,
                    "severity": "blocker",
                    "detail": f"excess_vs_00631l={excess_00631l:.4f}pp",
                },
            ]
        )
    return pd.DataFrame(rows)


def _build_acceptance_decision(
    *,
    manifest: dict[str, Any],
    checks: pd.DataFrame,
    summary: pd.DataFrame,
) -> dict[str, Any]:
    failed = checks[checks["passed"] == False]  # noqa: E712
    core_decision = "accept_formal_challenger" if failed.empty else "reject_formal_keep_report_only"
    return {
        "model": "pool3_radar_full_replay_acceptance_v1",
        "source_status": manifest.get("status", "unknown"),
        "core_decision": core_decision,
        "active_in_trade_decision": False,
        "formal_model_changed": False,
        "pool3_formal_vote_changed": False,
        "valuation_used": bool(manifest.get("valuation_used", False)),
        "h3_used": bool(manifest.get("h3_used", False)),
        "can_core_absorb_as_formal_challenger": core_decision == "accept_formal_challenger",
        "required_action": (
            "keep Pool3 Radar as report-only shadow/diagnostic"
            if core_decision != "accept_formal_challenger"
            else "open a separate Core formal challenger promotion task"
        ),
        "failed_checks": failed.to_dict(orient="records"),
        "rows": manifest.get("rows", {}),
        "hard_gate_2024": manifest.get("hard_gate_2024", []),
        "summary_rows": int(len(summary)),
    }


def _markdown_summary(decision: dict[str, Any], checks: pd.DataFrame) -> str:
    lines = [
        "# Pool3 Radar Full Replay Core Acceptance",
        "",
        f"- core_decision: `{decision['core_decision']}`",
        f"- active_in_trade_decision: `{decision['active_in_trade_decision']}`",
        f"- formal_model_changed: `{decision['formal_model_changed']}`",
        f"- pool3_formal_vote_changed: `{decision['pool3_formal_vote_changed']}`",
        f"- required_action: {decision['required_action']}",
        "",
        "## Failed Checks",
        "",
    ]
    failed = checks[checks["passed"] == False]  # noqa: E712
    if failed.empty:
        lines.append("- 無。")
    else:
        for row in failed.to_dict(orient="records"):
            lines.append(f"- {row['check_id']}: {row['detail']}")
    return "\n".join(lines) + "\n"


def _primary_2024_rows(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    return summary[
        (summary["period"].astype(str) == "2024")
        & (summary["variant"].astype(str) == PRIMARY_VARIANT)
    ].copy()


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path).fillna("")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _number(value: object) -> float:
    try:
        text = str(value).replace(",", "").replace("%", "").strip()
        return float(text) if text else 0.0
    except (TypeError, ValueError):
        return 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Accept or reject Pool3 Radar full replay for Core formal challenger use.")
    parser.add_argument("--full-replay-dir", required=True)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    output = run_pool3_radar_full_replay_acceptance(
        full_replay_dir=args.full_replay_dir,
        output_dir=args.output_dir,
    )
    print(f"OUTPUT_DIR={output.resolve()}")


if __name__ == "__main__":
    main()
