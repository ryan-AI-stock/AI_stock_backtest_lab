from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from backtest_lab.drive_publish import DEFAULT_FOLDER_ID, build_drive_service, upsert_pdf
from backtest_lab.frozen_report_pdf import _configure_chinese_font, _save_figure_as_raster_pdf_page


DEFAULT_OUTPUT_DIR = "outputs/live_path_tracker"
DEFAULT_REMOTE_NAME = "AI模型實戰路徑追蹤報告_最新版.pdf"
SCENARIO_LABELS = {
    "optimistic_p90": "p90",
    "upper_p75": "p75",
    "median_p50": "p50",
    "lower_p25": "p25",
    "stress_p10": "p10",
}


def run_live_path_tracker(
    *,
    scenario_dir: str | Path,
    output_dir: str | Path,
    report_date: str,
    actual_portfolio_value: float | None = None,
    actual_holding_ticker: str = "",
    actual_holding_name: str = "",
    model_target_ticker: str = "",
    model_target_name: str = "",
    actual_tracking_csv: str | Path | None = None,
    publish_drive: bool = False,
    drive_folder_id: str = DEFAULT_FOLDER_ID,
    remote_name: str = DEFAULT_REMOTE_NAME,
) -> dict[str, Any]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
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
        pd.DataFrame(run_log).to_csv(root / "run_log.csv", index=False, encoding="utf-8-sig")
        (root / "current_step.txt").write_text(step, encoding="utf-8")

    log("load_scenarios", "started", str(scenario_dir))
    scenario = load_scenario_bundle(scenario_dir)
    log("load_scenarios", "completed", f"paths={len(scenario['paths'])}")

    actual_source = "tracking_csv" if actual_tracking_csv else "manual"
    log("build_tracking", "started", actual_source)
    tracking = build_live_tracking_table(
        scenario=scenario,
        report_date=report_date,
        actual_portfolio_value=actual_portfolio_value,
        actual_holding_ticker=actual_holding_ticker,
        actual_holding_name=actual_holding_name,
        model_target_ticker=model_target_ticker,
        model_target_name=model_target_name,
        actual_tracking_csv=actual_tracking_csv,
    )
    latest = tracking.iloc[-1].to_dict()
    deviation_events = tracking[tracking["deviation_status"].astype(str) != "normal_range"].copy()
    log("build_tracking", "completed", f"rows={len(tracking)};latest={latest.get('deviation_status')}")

    log("write_reports", "started", "")
    tracking.to_csv(root / "live_path_tracking.csv", index=False, encoding="utf-8-sig")
    deviation_events.to_csv(root / "deviation_events.csv", index=False, encoding="utf-8-sig")
    report_md = markdown_report(scenario, tracking)
    (root / "report.md").write_text(report_md, encoding="utf-8")
    pdf_path = root / DEFAULT_REMOTE_NAME
    write_live_path_pdf(pdf_path, scenario, tracking)
    manifest = {
        "status": "completed",
        "report_date": report_date,
        "scenario_source": str(Path(scenario_dir)),
        "scenario_inputs": scenario["summary_json"].get("scenario_inputs", {}),
        "actual_source": actual_source,
        "latest_status": latest.get("deviation_status", ""),
        "latest_nearest_scenario": latest.get("nearest_scenario", ""),
        "model_changed": False,
        "active_in_trade_decision": False,
        "not_forecast": True,
        "not_investment_advice": True,
        "outputs": {
            "pdf": str(pdf_path),
            "markdown": str(root / "report.md"),
            "tracking_csv": str(root / "live_path_tracking.csv"),
            "deviation_events": str(root / "deviation_events.csv"),
            "run_log": str(root / "run_log.csv"),
        },
    }
    if publish_drive:
        log("publish_drive", "started", drive_folder_id)
        service, auth_mode = build_drive_service()
        file_id, action = upsert_pdf(service, drive_folder_id, pdf_path, remote_name)
        manifest["drive"] = {"file_id": file_id, "action": action, "auth_mode": auth_mode, "remote_name": remote_name}
        log("publish_drive", "completed", f"{action}:{file_id}")
    (root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (root / "completed.txt").write_text("completed\n", encoding="utf-8")
    (root / "current_step.txt").write_text("completed\n", encoding="utf-8")
    log("completed", "completed", str(root.resolve()))
    return manifest


def load_scenario_bundle(scenario_dir: str | Path) -> dict[str, Any]:
    root = Path(scenario_dir)
    paths = pd.read_csv(root / "scenario_paths.csv").fillna("")
    summary = pd.read_csv(root / "scenario_summary.csv").fillna("")
    analogs = pd.read_csv(root / "analog_cases.csv").fillna("")
    summary_json = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    paths["step"] = pd.to_numeric(paths["step"], errors="coerce").fillna(0).astype(int)
    paths["projected_value"] = pd.to_numeric(paths["projected_value"], errors="coerce").fillna(0.0)
    return {
        "root": root,
        "paths": paths,
        "summary": summary,
        "analogs": analogs,
        "summary_json": summary_json,
        "percentile_paths": build_percentile_paths(paths),
    }


def build_percentile_paths(paths: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for step, group in paths.groupby("step"):
        values = group["projected_value"].astype(float)
        rows.append(
            {
                "step": int(step),
                "p90": float(values.quantile(0.90)),
                "p75": float(values.quantile(0.75)),
                "p50": float(values.quantile(0.50)),
                "p25": float(values.quantile(0.25)),
                "p10": float(values.quantile(0.10)),
            }
        )
    return pd.DataFrame(rows).sort_values("step")


def build_live_tracking_table(
    *,
    scenario: dict[str, Any],
    report_date: str,
    actual_portfolio_value: float | None,
    actual_holding_ticker: str,
    actual_holding_name: str,
    model_target_ticker: str,
    model_target_name: str,
    actual_tracking_csv: str | Path | None,
) -> pd.DataFrame:
    existing = _read_existing_tracking(actual_tracking_csv)
    scenario_inputs = scenario["summary_json"].get("scenario_inputs", {})
    if actual_portfolio_value is None:
        if existing.empty:
            raise ValueError("actual_portfolio_value is required when no actual_tracking_csv is provided.")
        current = existing[existing["report_date"].astype(str) == report_date]
        if current.empty:
            raise ValueError("actual_portfolio_value is required for a new report_date.")
        actual_portfolio_value = float(current.iloc[-1]["actual_portfolio_value"])
    scenario_start = str(scenario_inputs.get("scenario_start") or "")
    step = scenario_step(scenario_start, report_date)
    percentile_row = scenario_percentile_at_step(scenario["percentile_paths"], step)
    row = {
        "report_date": report_date,
        "scenario_step": step,
        "actual_portfolio_value": round(float(actual_portfolio_value), 2),
        "actual_holding_ticker": actual_holding_ticker,
        "actual_holding_name": actual_holding_name,
        "model_target_ticker": model_target_ticker or str(scenario_inputs.get("target_ticker") or ""),
        "model_target_name": model_target_name or str(scenario_inputs.get("target_label") or ""),
        "model_regime": str(scenario_inputs.get("current_regime") or ""),
        "model_mode": str(scenario_inputs.get("current_mode") or ""),
        **percentile_row,
    }
    row.update(classify_deviation(row, existing))
    current = pd.DataFrame([row])
    if existing.empty:
        combined = current
    else:
        combined = pd.concat(
            [existing[existing["report_date"].astype(str) != report_date], current],
            ignore_index=True,
            sort=False,
        )
    combined["report_date"] = combined["report_date"].astype(str)
    return combined.sort_values("report_date").reset_index(drop=True)


def classify_deviation(row: dict[str, Any], existing: pd.DataFrame) -> dict[str, Any]:
    actual = float(row["actual_portfolio_value"])
    distances = {
        "distance_to_p90": actual - float(row["p90"]),
        "distance_to_p75": actual - float(row["p75"]),
        "distance_to_p50": actual - float(row["p50"]),
        "distance_to_p25": actual - float(row["p25"]),
        "distance_to_p10": actual - float(row["p10"]),
    }
    nearest = min(("p90", "p75", "p50", "p25", "p10"), key=lambda key: abs(actual - float(row[key])))
    if actual <= float(row["p10"]):
        status = "stress_warning"
        reason = "actual path is at or below p10 scenario line"
    elif actual < float(row["p25"]):
        below_count = _consecutive_below(existing, "p25") + 1
        if below_count >= 5:
            status = "model_assumption_warning"
            reason = f"actual path stayed below p25 for {below_count} tracking rows"
        else:
            status = "weak_warning"
            reason = "actual path is below p25 but not persistent yet"
    elif actual > float(row["p75"]):
        status = "bullish_outperform"
        reason = "actual path is above p75 scenario line"
    else:
        status = "normal_range"
        reason = "actual path remains between p25 and p75"
    live_values = [float(value) for value in existing.get("actual_portfolio_value", pd.Series(dtype=float)).tolist()]
    live_values.append(actual)
    live_high = max(live_values) if live_values else actual
    drawdown = actual / live_high - 1 if live_high else 0.0
    return {
        **{key: round(value, 2) for key, value in distances.items()},
        "nearest_scenario": nearest,
        "scenario_percentile_band": status,
        "drawdown_from_live_high": round(drawdown, 6),
        "deviation_status": status,
        "deviation_reason_placeholder": reason,
    }


def scenario_step(scenario_start: str, report_date: str) -> int:
    if not scenario_start:
        return 0
    dates = pd.bdate_range(pd.Timestamp(scenario_start), pd.Timestamp(report_date))
    return max(0, len(dates) - 1)


def scenario_percentile_at_step(percentile_paths: pd.DataFrame, step: int) -> dict[str, float]:
    if percentile_paths.empty:
        raise ValueError("scenario percentile paths are empty")
    frame = percentile_paths[percentile_paths["step"] <= step]
    selected = frame.iloc[-1] if not frame.empty else percentile_paths.iloc[0]
    return {key: round(float(selected[key]), 2) for key in ("p90", "p75", "p50", "p25", "p10")}


def markdown_report(scenario: dict[str, Any], tracking: pd.DataFrame) -> str:
    latest = tracking.iloc[-1]
    inputs = scenario["summary_json"].get("scenario_inputs", {})
    lines = [
        "# AI模型實戰路徑追蹤報告",
        "",
        "本報告是歷史類比情境追蹤，不是預測，不是投資建議，也不會自動修改正式模型。",
        "",
        "## 今日狀態",
        "",
        f"- 報告日：{latest['report_date']}",
        f"- 實際資產淨值：{float(latest['actual_portfolio_value']):,.0f} 元",
        f"- 實際持倉：{latest.get('actual_holding_name', '')} ({latest.get('actual_holding_ticker', '')})",
        f"- 模型目標：{latest.get('model_target_name', '')} ({latest.get('model_target_ticker', '')})",
        f"- 偏離狀態：{latest['deviation_status']}",
        f"- 最近情境線：{latest['nearest_scenario']}",
        f"- 診斷原因：{latest['deviation_reason_placeholder']}",
        "",
        "## 情境來源",
        "",
        f"- signal date：{inputs.get('signal_date', '')}",
        f"- scenario window：{inputs.get('scenario_start', '')} 到 {inputs.get('scenario_end', '')}",
        f"- analog count：{scenario['summary_json'].get('analog_count', '')}",
        "",
        "## 最新距離",
        "",
        "| line | distance |",
        "| --- | ---: |",
    ]
    for key in ("distance_to_p90", "distance_to_p75", "distance_to_p50", "distance_to_p25", "distance_to_p10"):
        lines.append(f"| {key.replace('distance_to_', '')} | {float(latest[key]):+,.0f} |")
    lines.extend(["", "## 情境摘要", "", "| scenario | final value | total return | max drawdown |", "| --- | ---: | ---: | ---: |"])
    for _, row in scenario["summary"].iterrows():
        lines.append(
            "| "
            f"{row.get('scenario', '')} | "
            f"{float(row.get('final_value', 0)):,.0f} | "
            f"{float(row.get('total_return_pct', 0)):+.2%} | "
            f"{float(row.get('max_drawdown_pct', 0)):+.2%} |"
        )
    return "\n".join(lines) + "\n"


def write_live_path_pdf(path: Path, scenario: dict[str, Any], tracking: pd.DataFrame) -> None:
    _configure_chinese_font()
    with PdfPages(path) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69), facecolor="#f4f6f8")
        ax = fig.add_axes((0, 0, 1, 1))
        ax.axis("off")
        _draw_summary_page(ax, scenario, tracking)
        _save_figure_as_raster_pdf_page(pdf, fig)

        fig = plt.figure(figsize=(8.27, 11.69), facecolor="#f4f6f8")
        ax = fig.add_axes((0, 0, 1, 1))
        ax.axis("off")
        _draw_path_page(ax, scenario, tracking)
        _save_figure_as_raster_pdf_page(pdf, fig)

        fig = plt.figure(figsize=(8.27, 11.69), facecolor="#f4f6f8")
        ax = fig.add_axes((0, 0, 1, 1))
        ax.axis("off")
        _draw_diagnostic_page(ax, scenario, tracking)
        _save_figure_as_raster_pdf_page(pdf, fig)


def _draw_summary_page(ax, scenario: dict[str, Any], tracking: pd.DataFrame) -> None:
    latest = tracking.iloc[-1]
    ax.add_patch(plt.Rectangle((0, 0.86), 1, 0.14, color="#17212a", transform=ax.transAxes))
    ax.text(0.06, 0.935, "AI模型實戰路徑追蹤報告", color="white", fontsize=20, fontweight="bold", transform=ax.transAxes)
    ax.text(0.06, 0.895, f"報告日 {latest['report_date']} · 歷史類比情境追蹤", color="#c8d5df", fontsize=10.5, transform=ax.transAxes)
    cards = [
        ("實際淨值", f"{float(latest['actual_portfolio_value']):,.0f} 元", "#13795b"),
        ("偏離狀態", str(latest["deviation_status"]), "#2457a7"),
        ("最近情境", str(latest["nearest_scenario"]), "#c77917"),
        ("模型目標", f"{latest.get('model_target_ticker', '')}", "#26323b"),
    ]
    for index, (label, value, color) in enumerate(cards):
        x = 0.06 + index * 0.225
        ax.add_patch(plt.Rectangle((x, 0.74), 0.2, 0.08, facecolor="white", edgecolor="#d9e0e5", transform=ax.transAxes))
        ax.text(x + 0.012, 0.787, label, color="#66737d", fontsize=9, transform=ax.transAxes)
        ax.text(x + 0.012, 0.758, str(value), color=color, fontsize=11, fontweight="bold", transform=ax.transAxes)
    ax.text(0.06, 0.67, "與情境線距離", color="#17212a", fontsize=14, fontweight="bold", transform=ax.transAxes)
    y = 0.62
    for key in ("distance_to_p90", "distance_to_p75", "distance_to_p50", "distance_to_p25", "distance_to_p10"):
        ax.text(0.08, y, key.replace("distance_to_", "").upper(), color="#52616b", fontsize=10, transform=ax.transAxes)
        ax.text(0.27, y, f"{float(latest[key]):+,.0f} 元", color="#26323b", fontsize=10, transform=ax.transAxes)
        y -= 0.04
    ax.text(0.06, 0.34, "使用邊界", color="#17212a", fontsize=14, fontweight="bold", transform=ax.transAxes)
    notes = [
        "這是歷史類比情境追蹤，不是預測。",
        "偏離診斷只用來判斷是否需要 Research 提出 shadow 假設。",
        "本報告不產生交易指令，也不自動修改正式模型。",
    ]
    for idx, note in enumerate(notes):
        ax.text(0.08, 0.30 - idx * 0.035, f"• {note}", color="#4d5b66", fontsize=9.5, transform=ax.transAxes)
    _footer(ax, tracking)


def _draw_path_page(ax, scenario: dict[str, Any], tracking: pd.DataFrame) -> None:
    ax.text(0.06, 0.94, "路徑圖", color="#17212a", fontsize=18, fontweight="bold", transform=ax.transAxes)
    chart_ax = ax.inset_axes([0.08, 0.22, 0.84, 0.62])
    paths = scenario["percentile_paths"]
    for key, color in [("p90", "#13795b"), ("p75", "#5aa469"), ("p50", "#2457a7"), ("p25", "#c77917"), ("p10", "#b42318")]:
        chart_ax.plot(paths["step"], paths[key] / 10_000, label=key, linewidth=1.7, color=color)
    best_path = _extreme_analog_path(scenario["paths"], highest=True)
    worst_path = _extreme_analog_path(scenario["paths"], highest=False)
    if not best_path.empty:
        chart_ax.plot(best_path["step"], best_path["projected_value"] / 10_000, label="best", linewidth=1.0, color="#13795b", alpha=0.32, linestyle="--")
    if not worst_path.empty:
        chart_ax.plot(worst_path["step"], worst_path["projected_value"] / 10_000, label="worst", linewidth=1.0, color="#b42318", alpha=0.32, linestyle="--")
    chart_ax.plot(tracking["scenario_step"], tracking["actual_portfolio_value"] / 10_000, label="actual", color="#111827", linewidth=2.4, marker="o")
    chart_ax.set_ylabel("資產淨值（萬元）")
    chart_ax.set_xlabel("scenario trading day")
    chart_ax.grid(True, alpha=0.22)
    chart_ax.legend(loc="best", fontsize=8)
    ax.text(0.08, 0.13, "圖中 p90/p75/p50/p25/p10 是歷史類比路徑分位線，actual 是實際追蹤值。", color="#52616b", fontsize=9, transform=ax.transAxes)
    _footer(ax, tracking)


def _extreme_analog_path(paths: pd.DataFrame, *, highest: bool) -> pd.DataFrame:
    if paths.empty or "analog_start" not in paths.columns:
        return pd.DataFrame()
    final_rows = paths.sort_values("step").groupby("analog_start").tail(1)
    if final_rows.empty:
        return pd.DataFrame()
    selected = final_rows.sort_values("projected_value", ascending=not highest).iloc[0]["analog_start"]
    return paths[paths["analog_start"] == selected].sort_values("step")


def _draw_diagnostic_page(ax, scenario: dict[str, Any], tracking: pd.DataFrame) -> None:
    latest = tracking.iloc[-1]
    ax.text(0.06, 0.94, "診斷與下一步", color="#17212a", fontsize=18, fontweight="bold", transform=ax.transAxes)
    ax.add_patch(plt.Rectangle((0.06, 0.82), 0.88, 0.08, facecolor="white", edgecolor="#d9e0e5", transform=ax.transAxes))
    ax.text(0.08, 0.865, f"目前狀態：{latest['deviation_status']}", color="#2457a7", fontsize=14, fontweight="bold", transform=ax.transAxes)
    ax.text(0.08, 0.835, str(latest["deviation_reason_placeholder"]), color="#52616b", fontsize=9.5, transform=ax.transAxes)
    recent = tracking.tail(20)
    counts = recent["deviation_status"].value_counts().to_dict()
    ax.text(0.06, 0.75, "最近追蹤狀態統計", color="#17212a", fontsize=13, fontweight="bold", transform=ax.transAxes)
    y = 0.70
    for status, count in counts.items():
        ax.text(0.08, y, f"{status}: {count}", color="#26323b", fontsize=10, transform=ax.transAxes)
        y -= 0.035
    trigger = latest["deviation_status"] in {"model_assumption_warning", "stress_warning"}
    ax.text(0.06, 0.48, "Research 觸發狀態", color="#17212a", fontsize=13, fontweight="bold", transform=ax.transAxes)
    ax.text(
        0.08,
        0.435,
        "需要 Research2 檢查 shadow 假設" if trigger else "未觸發 Research2 強制檢查；持續追蹤。",
        color="#b42318" if trigger else "#13795b",
        fontsize=11,
        fontweight="bold",
        transform=ax.transAxes,
    )
    ax.text(0.06, 0.30, "下一步邊界", color="#17212a", fontsize=13, fontweight="bold", transform=ax.transAxes)
    notes = [
        "每日只更新追蹤，不因單日偏離改模型。",
        "連續偏離或壓力警示時，由 Research2 另開 shadow 回測任務。",
        "正式模型更新仍需回測、壓力測試與使用者確認。",
    ]
    for idx, note in enumerate(notes):
        ax.text(0.08, 0.255 - idx * 0.035, f"• {note}", color="#4d5b66", fontsize=9.5, transform=ax.transAxes)
    _footer(ax, tracking)


def _footer(ax, tracking: pd.DataFrame) -> None:
    ax.text(0.06, 0.04, f"AI模型實戰路徑追蹤 · {tracking.iloc[-1]['report_date']}", color="#9aa7b1", fontsize=8.5, transform=ax.transAxes)
    ax.text(0.94, 0.04, "AI_stock_backtest_lab", color="#9aa7b1", fontsize=8.5, ha="right", transform=ax.transAxes)


def _read_existing_tracking(path: str | Path | None) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    target = Path(path)
    if not target.exists():
        return pd.DataFrame()
    return pd.read_csv(target).fillna("")


def _consecutive_below(existing: pd.DataFrame, percentile_col: str) -> int:
    if existing.empty or percentile_col not in existing.columns:
        return 0
    count = 0
    for _, row in existing.sort_values("report_date", ascending=False).iterrows():
        try:
            if float(row["actual_portfolio_value"]) < float(row[percentile_col]):
                count += 1
            else:
                break
        except (TypeError, ValueError):
            break
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Build live path tracking report from historical analog scenarios.")
    parser.add_argument("--scenario-dir", required=True)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-date", required=True)
    parser.add_argument("--actual-value", type=float, default=None)
    parser.add_argument("--actual-holding-ticker", default="")
    parser.add_argument("--actual-holding-name", default="")
    parser.add_argument("--model-target-ticker", default="")
    parser.add_argument("--model-target-name", default="")
    parser.add_argument("--actual-tracking-csv", default="")
    parser.add_argument("--publish-drive", action="store_true")
    parser.add_argument("--drive-folder-id", default=DEFAULT_FOLDER_ID)
    parser.add_argument("--remote-name", default=DEFAULT_REMOTE_NAME)
    args = parser.parse_args()
    manifest = run_live_path_tracker(
        scenario_dir=args.scenario_dir,
        output_dir=args.output_dir,
        report_date=args.report_date,
        actual_portfolio_value=args.actual_value,
        actual_holding_ticker=args.actual_holding_ticker,
        actual_holding_name=args.actual_holding_name,
        model_target_ticker=args.model_target_ticker,
        model_target_name=args.model_target_name,
        actual_tracking_csv=args.actual_tracking_csv or None,
        publish_drive=args.publish_drive,
        drive_folder_id=args.drive_folder_id,
        remote_name=args.remote_name,
    )
    print(f"LIVE_PATH_TRACKER_DIR={Path(args.output_dir).resolve()}")
    if "drive" in manifest:
        print(f"DRIVE_FILE_ID={manifest['drive']['file_id']}")


if __name__ == "__main__":
    main()
