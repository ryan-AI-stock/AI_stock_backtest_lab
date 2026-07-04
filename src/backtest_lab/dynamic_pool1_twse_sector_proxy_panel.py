from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


TASK_ID = "TASK-BACKTEST-CORE-DYNAMIC-POOL1-TWSE-SECTOR-PROXY-DIAGNOSTIC-PANEL-001"
DEFAULT_SECTOR_OUTPUT = (
    "C:/Users/zergv/Documents/Codex/2026-05-23/ai-stock-rotation-radar-https-docs/outputs/"
    "radar_dynamic_pool1_sector_mainline_pit_full_sweep_and_tpex_reverse_20260703"
)
DEFAULT_CANDIDATE_PANEL = "outputs/candidate_ranking_score_contract_2015_2021_20260703/candidate_ranking_panel_2015_2021.csv"
DEFAULT_OUTPUT_DIR = "outputs/dynamic_pool1_twse_sector_proxy_diagnostic_panel_20260704"


def run_dynamic_pool1_twse_sector_proxy_panel(
    *,
    sector_output: str | Path = DEFAULT_SECTOR_OUTPUT,
    candidate_panel_path: str | Path = DEFAULT_CANDIDATE_PANEL,
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
        (output / "current_step.txt").write_text(f"{step}:{status}\n{detail}", encoding="utf-8")

    try:
        log("load_sector_membership", "started", str(sector_output))
        sector_root = Path(sector_output)
        readiness = _load_json(sector_root / "readiness_for_core.json")
        source_manifest = _load_json(sector_root / "source_manifest.json")
        membership = _load_membership_shards(sector_root)
        panel = _build_membership_panel(membership)

        log("build_breadth", "started", "")
        breadth = _build_breadth(panel)

        log("build_candidate_context", "started", str(candidate_panel_path))
        candidate_context = _build_candidate_context(panel, Path(candidate_panel_path))

        log("write_outputs", "started", str(output))
        future_audit = _future_data_violation_audit(sector_root, readiness)
        proxy_readiness = _proxy_readiness(readiness, panel, breadth, candidate_context)
        manifest = _manifest(output, sector_root, candidate_panel_path, readiness, proxy_readiness, source_manifest)

        panel.to_csv(output / "dynamic_pool1_twse_sector_proxy_panel.csv", index=False, encoding="utf-8-sig")
        breadth.to_csv(output / "dynamic_pool1_twse_sector_breadth_monthly_anchor.csv", index=False, encoding="utf-8-sig")
        candidate_context.to_csv(output / "dynamic_pool1_candidate_sector_context.csv", index=False, encoding="utf-8-sig")
        future_audit.to_csv(output / "future_data_violation_audit.csv", index=False, encoding="utf-8-sig")
        (output / "dynamic_pool1_sector_proxy_readiness.json").write_text(
            json.dumps(proxy_readiness, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

        pd.DataFrame([{"step": "build_twse_sector_proxy_panel", "status": "completed", "rows": len(panel)}]).to_csv(
            output / "completed.csv",
            index=False,
            encoding="utf-8-sig",
        )
        pd.DataFrame(columns=["step", "status", "error"]).to_csv(output / "failed.csv", index=False, encoding="utf-8-sig")
        log("completed", "completed", str(output))
        return output
    except Exception as exc:
        pd.DataFrame([{"step": "build_twse_sector_proxy_panel", "status": "failed", "error": str(exc)}]).to_csv(
            output / "failed.csv",
            index=False,
            encoding="utf-8-sig",
        )
        log("failed", "failed", str(exc))
        raise


def _load_membership_shards(sector_root: Path) -> pd.DataFrame:
    manifest_path = sector_root / "twse_sector_membership_pit_daily_manifest.csv"
    if not manifest_path.exists():
        manifest_path = sector_root / "accepted_sector_membership_rows_manifest.csv"
    manifest = pd.read_csv(manifest_path).fillna("")
    frames: list[pd.DataFrame] = []
    for item in manifest.to_dict(orient="records"):
        rel_path = str(item.get("file") or "").replace("\\", "/")
        if not rel_path:
            continue
        path = sector_root / rel_path
        if path.exists():
            frames.append(pd.read_csv(path).fillna(""))
    if not frames:
        return pd.DataFrame(columns=_membership_source_columns())
    frame = pd.concat(frames, ignore_index=True)
    for column in _membership_source_columns():
        if column not in frame.columns:
            frame[column] = ""
    return frame[_membership_source_columns()]


def _build_membership_panel(membership: pd.DataFrame) -> pd.DataFrame:
    if membership.empty:
        return pd.DataFrame(columns=_panel_columns())
    frame = membership.copy()
    frame["ticker"] = frame["ticker"].astype(str).str.strip()
    frame["ticker_with_suffix"] = frame["ticker"].where(frame["ticker"].str.endswith(".TW"), frame["ticker"] + ".TW")
    frame["source_date"] = pd.to_datetime(frame["source_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    frame["effective_date"] = pd.to_datetime(frame["effective_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    frame["as_of_date"] = pd.to_datetime(frame["as_of_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    frame["membership_policy"] = "monthly_anchor"
    frame["daily_exact"] = False
    frame["twse_only"] = True
    frame["tpex_included"] = False
    frame["mainline_theme_ready"] = False
    frame["diagnostic_only"] = True
    frame["active_in_trade_decision"] = False
    frame["accepted_for_formal"] = False
    frame["future_data_violation"] = False
    return frame[_panel_columns()].sort_values(["as_of_date", "sector_code", "ticker"]).reset_index(drop=True)


def _build_breadth(panel: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "as_of_date",
        "source_date",
        "effective_date",
        "market",
        "sector_code",
        "sector_name",
        "constituent_count",
        "membership_policy",
        "daily_exact",
        "twse_only",
        "tpex_included",
        "mainline_theme_ready",
        "diagnostic_only",
        "active_in_trade_decision",
    ]
    if panel.empty:
        return pd.DataFrame(columns=columns)
    grouped = (
        panel.groupby(["as_of_date", "source_date", "effective_date", "market", "sector_code", "sector_name"], dropna=False)
        .agg(constituent_count=("ticker", "nunique"))
        .reset_index()
    )
    grouped["membership_policy"] = "monthly_anchor"
    grouped["daily_exact"] = False
    grouped["twse_only"] = True
    grouped["tpex_included"] = False
    grouped["mainline_theme_ready"] = False
    grouped["diagnostic_only"] = True
    grouped["active_in_trade_decision"] = False
    return grouped[columns].sort_values(["as_of_date", "sector_code"]).reset_index(drop=True)


def _build_candidate_context(panel: pd.DataFrame, candidate_panel_path: Path) -> pd.DataFrame:
    columns = [
        "date",
        "candidate_ticker",
        "candidate_name",
        "candidate_source",
        "sector_code",
        "sector_name",
        "sector_source_date",
        "sector_effective_date",
        "sector_as_of_date",
        "membership_policy",
        "daily_exact",
        "twse_only",
        "tpex_included",
        "mainline_theme_ready",
        "diagnostic_only",
        "active_in_trade_decision",
        "context_status",
    ]
    if panel.empty or not candidate_panel_path.exists():
        return pd.DataFrame(columns=columns)
    candidates = pd.read_csv(candidate_panel_path).fillna("")
    required = {"date", "candidate_ticker", "candidate_name", "candidate_source"}
    if not required.issubset(candidates.columns):
        return pd.DataFrame(columns=columns)
    candidates = candidates[list(required)].copy()
    candidates["date"] = pd.to_datetime(candidates["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    candidates["candidate_ticker"] = candidates["candidate_ticker"].astype(str).str.strip()
    anchors = panel[
        [
            "ticker_with_suffix",
            "sector_code",
            "sector_name",
            "source_date",
            "effective_date",
            "as_of_date",
        ]
    ].copy()
    merged = candidates.merge(
        anchors,
        left_on=["date", "candidate_ticker"],
        right_on=["as_of_date", "ticker_with_suffix"],
        how="inner",
    )
    if merged.empty:
        return pd.DataFrame(columns=columns)
    merged["sector_source_date"] = merged["source_date"]
    merged["sector_effective_date"] = merged["effective_date"]
    merged["sector_as_of_date"] = merged["as_of_date"]
    merged["membership_policy"] = "monthly_anchor"
    merged["daily_exact"] = False
    merged["twse_only"] = True
    merged["tpex_included"] = False
    merged["mainline_theme_ready"] = False
    merged["diagnostic_only"] = True
    merged["active_in_trade_decision"] = False
    merged["context_status"] = "twse_monthly_anchor_exact_date_match"
    return merged[columns].sort_values(["date", "candidate_ticker"]).reset_index(drop=True)


def _proxy_readiness(
    readiness: dict[str, Any],
    panel: pd.DataFrame,
    breadth: pd.DataFrame,
    candidate_context: pd.DataFrame,
) -> dict[str, Any]:
    future_count = int(readiness.get("future_data_violation_count", 0) or 0)
    return {
        "schema_version": 1,
        "task_id": TASK_ID,
        "status": "completed_twse_only_sector_proxy_diagnostic_panel",
        "twse_only_sector_proxy_diagnostic_ready": bool(
            readiness.get("twse_sector_monthly_anchor_ready", False)
            and readiness.get("sector_membership_pit_partial_ready", False)
            and not readiness.get("tpex_included", False)
            and len(panel) > 0
        ),
        "ready_for_experiments_twse_only_diagnostic": bool(len(panel) > 0 and future_count == 0),
        "ready_for_strategy_replay": False,
        "dynamic_pool1_shadow_challenger_ready": False,
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "twse_only": True,
        "tpex_included": False,
        "mainline_theme_ready": False,
        "daily_exact": False,
        "diagnostic_only": True,
        "membership_policy": "monthly_anchor",
        "membership_rows": int(len(panel)),
        "breadth_rows": int(len(breadth)),
        "candidate_context_rows": int(len(candidate_context)),
        "future_data_violation_count": future_count,
        "remaining_blockers": [
            "TPEx sector membership route remains blocked.",
            "AI/mainline/theme taxonomy remains blocked.",
            "TWSE official industry monthly-anchor is not daily exact membership.",
            "Sector breadth is diagnostic context only and not a formal trading condition.",
        ],
    }


def _future_data_violation_audit(sector_root: Path, readiness: dict[str, Any]) -> pd.DataFrame:
    upstream = _read_csv_if_exists(sector_root / "future_data_violation_audit.csv")
    if not upstream.empty:
        return upstream
    return pd.DataFrame(
        [
            {
                "data_area": "twse_sector_membership_proxy",
                "future_data_violation": False,
                "future_data_violation_count": int(readiness.get("future_data_violation_count", 0) or 0),
                "audit_reason": "TWSE monthly-anchor rows retain source_date/effective_date/as_of_date and are diagnostic-only.",
            }
        ]
    )


def _manifest(
    output: Path,
    sector_root: Path,
    candidate_panel_path: str | Path,
    readiness: dict[str, Any],
    proxy_readiness: dict[str, Any],
    source_manifest: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "task_id": TASK_ID,
        "status": proxy_readiness["status"],
        "generated_at": pd.Timestamp.now(tz="Asia/Taipei").isoformat(),
        "output_dir": str(output),
        "sector_output": str(sector_root),
        "candidate_panel_path": str(candidate_panel_path),
        "source_task_id": readiness.get("task_id", ""),
        "twse_only": True,
        "tpex_included": False,
        "mainline_theme_ready": False,
        "daily_exact": False,
        "diagnostic_only": True,
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "ready_for_strategy_replay": False,
        "ready_for_experiments_twse_only_diagnostic": proxy_readiness[
            "ready_for_experiments_twse_only_diagnostic"
        ],
        "future_data_violation_count": proxy_readiness["future_data_violation_count"],
        "source_manifest_type": type(source_manifest).__name__,
        "source_manifest_entries": len(source_manifest) if isinstance(source_manifest, list) else len(source_manifest.keys()),
        "outputs": {
            "dynamic_pool1_twse_sector_proxy_panel": "dynamic_pool1_twse_sector_proxy_panel.csv",
            "dynamic_pool1_twse_sector_breadth_monthly_anchor": "dynamic_pool1_twse_sector_breadth_monthly_anchor.csv",
            "dynamic_pool1_candidate_sector_context": "dynamic_pool1_candidate_sector_context.csv",
            "dynamic_pool1_sector_proxy_readiness": "dynamic_pool1_sector_proxy_readiness.json",
            "future_data_violation_audit": "future_data_violation_audit.csv",
        },
    }


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path).fillna("")
    except Exception:
        return pd.DataFrame()


def _membership_source_columns() -> list[str]:
    return [
        "ticker",
        "name",
        "market",
        "sector_code",
        "sector_name",
        "mainline",
        "theme",
        "source_date",
        "effective_date",
        "as_of_date",
        "source_url",
        "source_type",
        "formal_exact",
        "accepted_for_diagnostic",
        "accepted_for_formal",
        "evidence",
        "notes",
    ]


def _panel_columns() -> list[str]:
    return [
        "as_of_date",
        "effective_date",
        "source_date",
        "ticker",
        "ticker_with_suffix",
        "name",
        "market",
        "sector_code",
        "sector_name",
        "mainline",
        "theme",
        "membership_policy",
        "daily_exact",
        "twse_only",
        "tpex_included",
        "mainline_theme_ready",
        "diagnostic_only",
        "active_in_trade_decision",
        "accepted_for_formal",
        "future_data_violation",
        "source_type",
        "formal_exact",
        "source_url",
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build TWSE-only Dynamic Pool1 sector proxy diagnostic panel.")
    parser.add_argument("--sector-output", default=DEFAULT_SECTOR_OUTPUT)
    parser.add_argument("--candidate-panel", default=DEFAULT_CANDIDATE_PANEL)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    run_dynamic_pool1_twse_sector_proxy_panel(
        sector_output=args.sector_output,
        candidate_panel_path=args.candidate_panel,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
