from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from backtest_lab.decision_layers import DIAGNOSTIC, SHADOW_OVERLAY
from backtest_lab.institutional_flow_overlay_shadow import load_institutional_flows


RADAR_FACTOR_ROOT = (
    "C:/Users/zergv/Documents/Codex/2026-05-23/ai-stock-rotation-radar-https-docs/"
    "data/formal_sources/backtest_factor_2024_2026"
)
DEFAULT_INSTITUTIONAL_SOURCE = f"{RADAR_FACTOR_ROOT}/institutional_flows_daily_20240102_20260526.csv"
DEFAULT_MARGIN_SOURCE = f"{RADAR_FACTOR_ROOT}/margin_short_daily_20240102_20260526.csv"
DEFAULT_DAY_TRADING_SOURCE = f"{RADAR_FACTOR_ROOT}/day_trading_daily_20240102_20260526.csv"
DEFAULT_OUTPUT_DIR = "outputs/chip_shadow_diagnostic_adapter_20260620"
DEFAULT_START_DATE = "2024-01-02"
DEFAULT_END_DATE = "2026-05-26"
SUPPORTED_SIGNALS = (
    "inst_total_net_positive",
    "foreign_trust_sync_buy",
    "foreign_sell_ge3",
    "foreign_sell_ge5",
    "foreign_trust_sync_sell",
    "inst_total_net_negative",
    "h1_negative_or_h2_sell_pressure",
)
EXCLUDED_SIGNALS = (
    "day_ratio_top10",
    "margin_and_day_overheat_flag",
    "valuation_entry_block",
)
FORBIDDEN_WORDING = ("買進", "賣出", "明牌", "保證", "穩賺", "必買")


@dataclass(frozen=True)
class ChipDiagnosticSources:
    institutional_source: str
    margin_source: str | None = None
    day_trading_source: str | None = None
    valuation_source: str | None = None


def run_chip_shadow_diagnostic_adapter(
    *,
    institutional_source: str,
    margin_source: str | None,
    day_trading_source: str | None,
    valuation_source: str | None,
    output_dir: str,
    start_date: str,
    end_date: str,
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
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
        pd.DataFrame(run_log).to_csv(output_path / "run_log.csv", index=False, encoding="utf-8-sig")
        _write_text(output_path / "current_step.txt", step)

    log("load_sources", "started", institutional_source)
    sources = ChipDiagnosticSources(
        institutional_source=institutional_source,
        margin_source=margin_source,
        day_trading_source=day_trading_source,
        valuation_source=valuation_source,
    )
    if valuation_source:
        raise ValueError("Valuation source is PIT-blocked for this adapter and must not be provided.")
    _assert_source_exists(institutional_source, "institutional_source")
    _assert_source_exists(margin_source, "margin_source")
    _assert_source_exists(day_trading_source, "day_trading_source")
    institutional = load_institutional_flows(institutional_source)
    institutional = _filter_period(institutional, start_date=start_date, end_date=end_date)
    log("load_sources", "completed", f"institutional_rows={len(institutional)}")

    log("build_panel", "started", "")
    panel = build_chip_diagnostic_panel(institutional)
    summary = build_signal_summary(panel)
    manifest = build_manifest(sources=sources, panel=panel, summary=summary, start_date=start_date, end_date=end_date)
    log("build_panel", "completed", f"panel_rows={len(panel)}")

    panel.to_csv(output_path / "chip_diagnostic_panel.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output_path / "chip_diagnostic_summary.csv", index=False, encoding="utf-8-sig")
    (output_path / "chip_shadow_diagnostic_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_report(output_path / "chip_shadow_diagnostic_report.md", manifest, summary)
    _write_text(output_path / "completed.txt", "completed")
    _write_text(output_path / "current_step.txt", "completed")
    log("completed", "completed", str(output_path.resolve()))
    return output_path


def build_chip_diagnostic_panel(institutional: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for row in institutional.itertuples(index=False):
        ticker = str(row.ticker)
        if ticker in {"0050.TW", "00631L.TW"}:
            continue
        foreign = _number(getattr(row, "foreign_net_buy_shares", 0.0))
        trust = _number(getattr(row, "investment_trust_net_buy_shares", 0.0))
        dealer = _number(getattr(row, "dealer_net_buy_shares", 0.0))
        total = foreign + trust + dealer
        foreign_buy_days = _number(getattr(row, "foreign_consecutive_buy_days", 0.0))
        foreign_sell_days = _number(getattr(row, "foreign_consecutive_sell_days", 0.0))
        trust_buy_days = _number(getattr(row, "trust_consecutive_buy_days", 0.0))
        trust_sell_days = _number(getattr(row, "trust_consecutive_sell_days", 0.0))

        inst_total_net_positive = total > 0
        foreign_trust_sync_buy = foreign > 0 and trust > 0
        foreign_sell_ge3 = foreign_sell_days >= 3
        foreign_sell_ge5 = foreign_sell_days >= 5
        foreign_trust_sync_sell = foreign < 0 and trust < 0
        inst_total_net_negative = total < 0

        h1_positive = total > 0 or foreign_buy_days >= 2 or trust_buy_days >= 2
        h2_sell = total < 0 or foreign_sell_days >= 2 or trust_sell_days >= 2
        h1_negative_or_h2_sell_pressure = (not h1_positive) or h2_sell

        attack_flags = [inst_total_net_positive, foreign_trust_sync_buy]
        risk_flags = [foreign_sell_ge3, foreign_sell_ge5, foreign_trust_sync_sell, inst_total_net_negative]
        panel_row = {
            "date": pd.Timestamp(row.date).strftime("%Y-%m-%d"),
            "ticker": ticker,
            "name": str(getattr(row, "name", "")),
            "decision_layer": SHADOW_OVERLAY if h1_negative_or_h2_sell_pressure else DIAGNOSTIC,
            "active_in_trade_decision": False,
            "source_module": "chip_shadow_diagnostic_adapter",
            "inst_total_net_positive": inst_total_net_positive,
            "foreign_trust_sync_buy": foreign_trust_sync_buy,
            "foreign_sell_ge3": foreign_sell_ge3,
            "foreign_sell_ge5": foreign_sell_ge5,
            "foreign_trust_sync_sell": foreign_trust_sync_sell,
            "inst_total_net_negative": inst_total_net_negative,
            "h1_negative_or_h2_sell_pressure": h1_negative_or_h2_sell_pressure,
            "attack_confirmation_score": int(inst_total_net_positive) + int(foreign_trust_sync_buy),
            "sell_pressure_warning_score": sum(int(flag) for flag in risk_flags),
            "shadow_risk_group": "sell_pressure_or_attack_absent" if h1_negative_or_h2_sell_pressure else "none",
            "total_institutional_net_buy_shares": total,
            "foreign_net_buy_shares": foreign,
            "investment_trust_net_buy_shares": trust,
            "dealer_net_buy_shares": dealer,
            "foreign_consecutive_sell_days": foreign_sell_days,
            "trust_consecutive_sell_days": trust_sell_days,
            "diagnostic_wording": diagnostic_wording(
                attack_confirmation_score=sum(int(flag) for flag in attack_flags),
                sell_pressure_warning_score=sum(int(flag) for flag in risk_flags),
                shadow_risk=h1_negative_or_h2_sell_pressure,
            ),
            "excluded_h3_signals": "day_ratio_top10; margin_and_day_overheat_flag",
            "valuation_used": False,
        }
        validate_wording(panel_row["diagnostic_wording"])
        rows.append(panel_row)
    return pd.DataFrame(rows).sort_values(["date", "ticker"]).reset_index(drop=True)


def build_signal_summary(panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for signal in SUPPORTED_SIGNALS:
        triggered = int(panel[signal].sum()) if signal in panel.columns else 0
        rows.append(
            {
                "signal_id": signal,
                "decision_layer": SHADOW_OVERLAY if signal == "h1_negative_or_h2_sell_pressure" else DIAGNOSTIC,
                "active_in_trade_decision": False,
                "triggered_rows": triggered,
                "triggered_ratio": round(triggered / len(panel), 6) if len(panel) else 0.0,
                "status": "included_shadow_or_diagnostic",
            }
        )
    for signal in EXCLUDED_SIGNALS:
        rows.append(
            {
                "signal_id": signal,
                "decision_layer": DIAGNOSTIC,
                "active_in_trade_decision": False,
                "triggered_rows": 0,
                "triggered_ratio": 0.0,
                "status": "excluded_from_core_adapter",
            }
        )
    return pd.DataFrame(rows)


def build_manifest(
    *,
    sources: ChipDiagnosticSources,
    panel: pd.DataFrame,
    summary: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> dict:
    return {
        "model": "chip_shadow_diagnostic_adapter_v1",
        "decision_layer": "shadow_or_diagnostic",
        "active_in_trade_decision": False,
        "start_date": start_date,
        "end_date": end_date,
        "institutional_source": str(Path(sources.institutional_source).resolve()),
        "margin_source": str(Path(sources.margin_source).resolve()) if sources.margin_source else "",
        "day_trading_source": str(Path(sources.day_trading_source).resolve()) if sources.day_trading_source else "",
        "valuation_source": "",
        "valuation_status": "excluded_pit_blocked",
        "panel_rows": int(len(panel)),
        "signal_summary": summary.to_dict(orient="records"),
        "included_signals": list(SUPPORTED_SIGNALS),
        "excluded_signals": list(EXCLUDED_SIGNALS),
        "formal_trade_decision_changed": False,
        "frozen_baseline_changed": False,
        "wording_boundary": "AI輔助市場觀察與風險診斷；不是交易指令或績效承諾。",
        "next_step": "若要測 formal challenger，需另開任務並經 out-of-sample/walk-forward 驗證。",
    }


def diagnostic_wording(*, attack_confirmation_score: int, sell_pressure_warning_score: int, shadow_risk: bool) -> str:
    parts: list[str] = []
    if attack_confirmation_score > 0:
        parts.append("進攻確認觀察：法人資金條件偏正，僅作旁路加分觀察。")
    else:
        parts.append("進攻確認觀察：法人資金條件未形成明確正向確認。")
    if sell_pressure_warning_score > 0:
        parts.append("風險診斷：法人賣壓或同步轉弱條件出現，需列為風險警訊。")
    if shadow_risk:
        parts.append("Shadow 分組：攻擊確認不足或賣壓條件出現，僅供後續觀察與研究。")
    return " ".join(parts)


def validate_wording(text: str) -> None:
    for word in FORBIDDEN_WORDING:
        if word in text:
            raise ValueError(f"Diagnostic wording contains forbidden investment wording: {word}")


def _filter_period(frame: pd.DataFrame, *, start_date: str, end_date: str) -> pd.DataFrame:
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    return frame.loc[(frame["date"] >= start) & (frame["date"] <= end)].copy()


def _number(value: object) -> float:
    try:
        text = str(value).replace(",", "").replace("%", "").strip()
        return float(text) if text else 0.0
    except (TypeError, ValueError):
        return 0.0


def _write_report(path: Path, manifest: dict, summary: pd.DataFrame) -> None:
    lines = [
        "# Chip Shadow Diagnostic Adapter",
        "",
        f"- decision_layer: `{manifest['decision_layer']}`",
        f"- active_in_trade_decision: `{manifest['active_in_trade_decision']}`",
        f"- period: `{manifest['start_date']}` ~ `{manifest['end_date']}`",
        f"- panel_rows: `{manifest['panel_rows']}`",
        f"- valuation_status: `{manifest['valuation_status']}`",
        "",
        "## Boundary",
        "",
        "- 本輸出只做 AI 輔助市場觀察與風險診斷。",
        "- H1/H2/H4 皆為 shadow/diagnostic，不改正式交易目標、權重、閘門或 frozen baseline。",
        "- H3 當沖/融資過熱與 valuation 不納入本 adapter。",
        "- 本檔案只保留中性觀察語氣，不提供交易指令或績效承諾。",
        "",
        "## Signal Summary",
        "",
        "| signal | status | layer | triggered rows | ratio |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for row in summary.to_dict(orient="records"):
        lines.append(
            f"| {row['signal_id']} | {row['status']} | {row['decision_layer']} | "
            f"{row['triggered_rows']} | {row['triggered_ratio']:.2%} |"
        )
    content = "\n".join(lines) + "\n"
    validate_wording(content)
    path.write_text(content, encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _assert_source_exists(source: str | None, label: str) -> None:
    if not source:
        return
    if not Path(source).exists():
        raise FileNotFoundError(f"{label} does not exist: {source}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build chip-factor shadow/diagnostic adapter outputs.")
    parser.add_argument("--institutional-source", default=DEFAULT_INSTITUTIONAL_SOURCE)
    parser.add_argument("--margin-source", default=DEFAULT_MARGIN_SOURCE)
    parser.add_argument("--day-trading-source", default=DEFAULT_DAY_TRADING_SOURCE)
    parser.add_argument("--valuation-source", default="")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    args = parser.parse_args()
    run_chip_shadow_diagnostic_adapter(
        institutional_source=args.institutional_source,
        margin_source=args.margin_source or None,
        day_trading_source=args.day_trading_source or None,
        valuation_source=args.valuation_source or None,
        output_dir=args.output_dir,
        start_date=args.start_date,
        end_date=args.end_date,
    )


if __name__ == "__main__":
    main()
