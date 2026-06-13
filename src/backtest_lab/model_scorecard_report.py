from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from backtest_lab.config import BacktestConfig, load_config
from backtest_lab.data import download_yfinance_prices
from backtest_lab.frozen_strategy_engine import build_frozen_strategy_context, simulate_frozen_baseline
from backtest_lab.frozen_report_pdf import _configure_chinese_font, _save_figure_as_raster_pdf_page
from backtest_lab.simulation import simulate_buy_and_hold


REPORT_NAME = "AI模型延遲公開成績單"
REPORT_VERSION = "v20260612"
DEFAULT_OUTPUT_ROOT = "outputs/model_scorecard_report"
DEFAULT_DRIVE_FOLDER_ID = "1NDqeKNo3Sa08t0PUqWiSkCLQTZGfKHIe"
DEFAULT_TRACKING_START = "2026-05-29"
DEFAULT_REPORT_DATE = "2026-06-12"
DEFAULT_INITIAL_CASH = 1_328_709
DEFAULT_DELAY_DAYS = 7
NO_DATA_EXIT_CODE = 3


@dataclass(frozen=True)
class ScorecardRow:
    name: str
    ticker: str
    final_value_twd: float
    total_return_pct: float
    max_drawdown_pct: float
    trade_count: int


@dataclass(frozen=True)
class ScorecardReport:
    report_name: str
    report_version: str
    report_date: str
    public_cutoff_date: str
    data_end_date: str
    tracking_start_date: str
    initial_cash_twd: float
    delay_days: int
    model_name: str
    tracking_case_ticker: str
    tracking_case_label: str
    model_tracking_label: str
    model_holding_records: list[dict]
    rows: list[ScorecardRow]
    equity_curves: dict[str, list[dict]]
    disclaimer: str = "本報告為 AI 輔助回測/觀察與延遲公開模型成績單，不是投資建議。"

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["rows"] = [asdict(row) for row in self.rows]
        return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Generate {REPORT_NAME}.")
    parser.add_argument("--config", default="configs/ep05_universe.json")
    parser.add_argument("--strategy-config", default="configs/frozen_cycle_proven_top1_v1.json")
    parser.add_argument("--group-id", default="group_c_0050_00631l_plus_mega_caps")
    parser.add_argument("--report-date", default=DEFAULT_REPORT_DATE)
    parser.add_argument("--tracking-start", default=DEFAULT_TRACKING_START)
    parser.add_argument("--tracking-case-ticker", default="auto")
    parser.add_argument("--initial-cash", type=float, default=DEFAULT_INITIAL_CASH)
    parser.add_argument("--delay-days", type=int, default=DEFAULT_DELAY_DAYS)
    parser.add_argument("--cache-dir", default="backtest_cache/model_scorecard_report")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--warmup-start", default="2020-01-02")
    args = parser.parse_args()

    config = load_config(args.config)
    frozen_config = json.loads(Path(args.strategy_config).read_text(encoding="utf-8"))
    if frozen_config["strategy_id"] != "frozen_cycle_proven_top1_v1" or frozen_config["status"] != "frozen_baseline":
        raise ValueError("Scorecard must use the frozen baseline strategy config.")

    group = config.group_by_id(args.group_id)
    labels = {asset.ticker: asset.label for asset in group.assets}
    asset_types = {asset.ticker: asset.asset_type for asset in group.assets}
    tickers = sorted(labels)
    public_cutoff = public_cutoff_date(args.report_date, args.delay_days)
    prices = download_yfinance_prices(
        tickers=tickers,
        start_date=args.warmup_start,
        end_date=public_cutoff,
        cache_dir=args.cache_dir,
        allow_edge_gap=False,
    )
    data_end = resolve_scorecard_data_end(prices, args.tracking_start, public_cutoff)
    report = build_scorecard_report(
        prices_by_ticker=prices,
        labels=labels,
        asset_types=asset_types,
        manual_splits=config.manual_splits,
        cost_model=config.cost_model,
        config=config,
        report_date=args.report_date,
        tracking_start=args.tracking_start,
        data_end=data_end,
        initial_cash=args.initial_cash,
        delay_days=args.delay_days,
        tracking_case_ticker=args.tracking_case_ticker,
    )
    output_dir = Path(args.output_root) / args.report_date.replace("-", "")
    write_scorecard_outputs(output_dir, report)
    print(f"SCORECARD_REPORT_DIR={output_dir.resolve()}")
    print(f"SCORECARD_LATEST_PDF={(output_dir / report_filename('pdf', latest=True)).resolve()}")


def public_cutoff_date(report_date: str, delay_days: int = DEFAULT_DELAY_DAYS) -> str:
    return (pd.Timestamp(report_date) - pd.Timedelta(days=delay_days)).strftime("%Y-%m-%d")


def resolve_scorecard_data_end(
    prices_by_ticker: dict[str, pd.DataFrame],
    tracking_start: str,
    public_cutoff: str,
) -> str:
    start = pd.Timestamp(tracking_start)
    cutoff = pd.Timestamp(public_cutoff)
    common = None
    for frame in prices_by_ticker.values():
        dates = set(frame.index[(frame.index >= start) & (frame.index <= cutoff)])
        common = dates if common is None else common & dates
    if not common:
        raise ValueError(f"No common scorecard dates between {tracking_start} and {public_cutoff}")
    return max(common).strftime("%Y-%m-%d")


def build_scorecard_report(
    *,
    prices_by_ticker: dict[str, pd.DataFrame],
    labels: dict[str, str],
    asset_types: dict[str, str],
    manual_splits: dict[str, tuple[dict[str, float | str], ...]] | None,
    cost_model,
    report_date: str,
    tracking_start: str,
    data_end: str,
    initial_cash: float,
    delay_days: int,
    tracking_case_ticker: str,
    group_id: str = "group_c_0050_00631l_plus_mega_caps",
    config: BacktestConfig | None = None,
) -> ScorecardReport:
    prices_by_ticker = _truncate_prices(prices_by_ticker, data_end)
    frozen_config = replace(
        config or load_config("configs/ep05_universe.json"),
        cost_model=cost_model,
        manual_splits=manual_splits or {},
    )
    frozen_context = build_frozen_strategy_context(
        config=frozen_config,
        group_id=group_id,
        prices_by_ticker=prices_by_ticker,
    )
    dividends = frozen_context.dividends_by_ticker
    model_result = simulate_frozen_baseline(
        context=frozen_context,
        name="scorecard_best_v20260605",
        start_date=tracking_start,
        end_date=data_end,
        initial_cash=initial_cash,
    )
    holding_records = _model_holding_records(model_result.equity_curve, labels)
    current_model_ticker = _current_model_ticker(model_result.equity_curve)
    resolved_tracking_ticker = current_model_ticker if tracking_case_ticker == "auto" else tracking_case_ticker
    resolved_tracking_label = labels.get(resolved_tracking_ticker, resolved_tracking_ticker)
    model_series_name = f"AI模型追蹤：{resolved_tracking_label}"
    benchmarks = [
        ("0050買進持有", "0050.TW"),
        ("0050正二買進持有", "00631L.TW"),
    ]
    results = [(model_series_name, "model", model_result)]
    for label, ticker in benchmarks:
        results.append(
            (
                label,
                ticker,
                simulate_buy_and_hold(
                    name=label,
                    ticker=ticker,
                    asset_type=asset_types[ticker],
                    prices=prices_by_ticker[ticker],
                    start_date=tracking_start,
                    end_date=data_end,
                    initial_cash=initial_cash,
                    cost_model=cost_model,
                    dividend_series=dividends[ticker],
                ),
            )
        )
    rows = [
        ScorecardRow(
            name=label,
            ticker=ticker,
            final_value_twd=round(result.final_value, 2),
            total_return_pct=round(result.total_return * 100, 4),
            max_drawdown_pct=round(result.max_drawdown * 100, 4),
            trade_count=len([trade for trade in result.trades if trade.action in {"buy", "sell"}]),
        )
        for label, ticker, result in results
    ]
    equity_curves = {
        label: _equity_rows(result.equity_curve)
        for label, _, result in results
    }
    return ScorecardReport(
        report_name=REPORT_NAME,
        report_version=REPORT_VERSION,
        report_date=report_date,
        public_cutoff_date=public_cutoff_date(report_date, delay_days),
        data_end_date=data_end,
        tracking_start_date=tracking_start,
        initial_cash_twd=initial_cash,
        delay_days=delay_days,
        model_name="AI大型權值股最佳版 v20260605",
        tracking_case_ticker=resolved_tracking_ticker,
        tracking_case_label=resolved_tracking_label,
        model_tracking_label=model_series_name,
        model_holding_records=holding_records,
        rows=rows,
        equity_curves=equity_curves,
    )


def _truncate_prices(prices_by_ticker: dict[str, pd.DataFrame], end_date: str) -> dict[str, pd.DataFrame]:
    cutoff = pd.Timestamp(end_date)
    return {
        ticker: frame.loc[frame.index <= cutoff].copy()
        for ticker, frame in prices_by_ticker.items()
    }


def write_scorecard_outputs(output_dir: Path, report: ScorecardReport) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "model_scorecard_report.json").write_text(
        json.dumps({"status": "ready", "report": report.to_dict()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame([asdict(row) for row in report.rows]).to_csv(
        output_dir / "model_scorecard_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(_flat_equity_rows(report)).to_csv(
        output_dir / "model_scorecard_equity_curve.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(_flat_normalized_equity_rows(report)).to_csv(
        output_dir / "model_scorecard_chart_curve.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(report.model_holding_records).to_csv(
        output_dir / "model_scorecard_holding_records.csv",
        index=False,
        encoding="utf-8-sig",
    )
    markdown = markdown_report(report)
    (output_dir / report_filename("md", report.report_date)).write_text(markdown, encoding="utf-8")
    (output_dir / report_filename("md", latest=True)).write_text(markdown, encoding="utf-8")
    write_scorecard_pdf(output_dir / report_filename("pdf", report.report_date), report)
    write_scorecard_pdf(output_dir / report_filename("pdf", latest=True), report)


def markdown_report(report: ScorecardReport) -> str:
    lines = [
        f"# {report.report_name}",
        "",
        f"- 報告日期：{report.report_date}",
        f"- 延遲公開截止日：{report.data_end_date}",
        f"- 模擬起點：{report.tracking_start_date}",
        f"- 初始資金：{report.initial_cash_twd:,.0f} 元",
        f"- 模型：{report.model_name}",
        "",
        "| 項目 | 期末淨值 | 報酬率 | 最大回撤 | 交易次數 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(report.rows, key=lambda item: item.final_value_twd, reverse=True):
        lines.append(
            f"| {row.name} | {row.final_value_twd:,.0f} | {row.total_return_pct:+.2f}% | "
            f"{row.max_drawdown_pct:+.2f}% | {row.trade_count} |"
        )
    lines.extend(["", report.disclaimer])
    return "\n".join(lines)


def write_scorecard_pdf(path: Path, report: ScorecardReport) -> None:
    _configure_chinese_font()
    with PdfPages(path) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69), facecolor="#f4f6f8")
        ax = fig.add_axes((0, 0, 1, 1))
        ax.axis("off")
        _draw_scorecard_cover(ax, report)
        _save_figure_as_raster_pdf_page(pdf, fig)

        fig = plt.figure(figsize=(8.27, 11.69), facecolor="#f4f6f8")
        ax = fig.add_axes((0, 0, 1, 1))
        ax.axis("off")
        _draw_method_page(ax, report)
        _save_figure_as_raster_pdf_page(pdf, fig)


def report_filename(extension: str, report_date: str | None = None, *, latest: bool = False) -> str:
    suffix = "最新版" if latest else report_date
    if not suffix:
        raise ValueError("report_date is required when latest is false.")
    return f"{REPORT_NAME}_{suffix}_{REPORT_VERSION}.{extension}"


def _draw_scorecard_cover(ax, report: ScorecardReport) -> None:
    ax.add_patch(plt.Rectangle((0, 0.86), 1, 0.14, color="#17212a", transform=ax.transAxes))
    ax.text(0.06, 0.94, report.report_name, color="white", fontsize=20, fontweight="bold", transform=ax.transAxes)
    ax.text(
        0.06,
        0.895,
        f"報告日 {report.report_date} · 公開資料到 {report.data_end_date} · {report.report_version}",
        color="#c8d5df",
        fontsize=11,
        transform=ax.transAxes,
    )
    best = max(report.rows, key=lambda row: row.final_value_twd)
    model = next(row for row in report.rows if row.ticker == "model")
    cards = [
        ("目前第一名", best.name, "#2457a7"),
        ("模型淨值", f"{model.final_value_twd:,.0f} 元", "#13795b"),
        ("模型報酬率", f"{model.total_return_pct:+.2f}%", "#13795b" if model.total_return_pct >= 0 else "#b42318"),
        ("公開延遲", f"{report.delay_days} 天", "#17212a"),
    ]
    for index, (label, value, color) in enumerate(cards):
        x = 0.06 + index * 0.225
        ax.add_patch(plt.Rectangle((x, 0.75), 0.2, 0.08, facecolor="white", edgecolor="#d9e0e5", transform=ax.transAxes))
        ax.text(x + 0.014, 0.802, label, color="#66737d", fontsize=9.5, transform=ax.transAxes)
        ax.text(x + 0.014, 0.775, value, color=color, fontsize=11.2, fontweight="bold", transform=ax.transAxes)
    _draw_equity_chart(ax, report)
    _draw_summary_table(ax, report)
    _draw_footer(ax, report.disclaimer)


def _draw_equity_chart(ax, report: ScorecardReport) -> None:
    chart_ax = ax.inset_axes((0.08, 0.42, 0.84, 0.25))
    styles = {
        "0050買進持有": {"color": "#a15c00", "linewidth": 1.9, "linestyle": "-", "zorder": 2},
        "0050正二買進持有": {"color": "#b42318", "linewidth": 1.9, "linestyle": "-", "zorder": 2},
    }
    for name, rows in report.equity_curves.items():
        if not rows:
            continue
        style = (
            {"color": "#2457a7", "linewidth": 2.6, "linestyle": "-", "zorder": 5}
            if name.startswith("AI模型追蹤：")
            else styles.get(name, {"color": "#52616b", "linewidth": 1.8, "linestyle": "-", "zorder": 1})
        )
        dates = [pd.Timestamp(row["date"]) for row in rows]
        values = _normalized_equity_values(rows, report.initial_cash_twd)
        chart_ax.plot(
            dates,
            values,
            label=name,
            linewidth=style["linewidth"],
            color=style["color"],
            linestyle=style["linestyle"],
            zorder=style["zorder"],
        )
    chart_ax.set_title(f"{report.initial_cash_twd:,.0f} 元模擬資金曲線（起點統一）", fontsize=12)
    chart_ax.grid(True, alpha=0.22)
    chart_ax.legend(fontsize=8, loc="best")
    chart_ax.tick_params(axis="x", labelrotation=20, labelsize=8)
    chart_ax.tick_params(axis="y", labelsize=8)


def _draw_summary_table(ax, report: ScorecardReport) -> None:
    ax.text(0.06, 0.36, "延遲公開成績比較", fontsize=15, fontweight="bold", color="#17212a", transform=ax.transAxes)
    headers = ("排名", "項目", "期末淨值", "報酬率", "最大回撤", "交易")
    widths = (0.08, 0.27, 0.18, 0.14, 0.14, 0.08)
    x0, y, row_h = 0.06, 0.31, 0.045
    ax.add_patch(plt.Rectangle((x0, y), 0.88, row_h, facecolor="#e8eef3", edgecolor="#d9e0e5", transform=ax.transAxes))
    x = x0
    for header, width in zip(headers, widths):
        ax.text(x + 0.008, y + 0.016, header, fontsize=9.8, fontweight="bold", color="#31414d", transform=ax.transAxes)
        x += width
    for rank, row in enumerate(sorted(report.rows, key=lambda item: item.final_value_twd, reverse=True), start=1):
        y -= row_h
        fill = "#fff7e6" if row.name == "AI大型權值股最佳版" else "white"
        ax.add_patch(plt.Rectangle((x0, y), 0.88, row_h, facecolor=fill, edgecolor="#d9e0e5", linewidth=0.8, transform=ax.transAxes))
        values = (
            str(rank),
            row.name,
            f"{row.final_value_twd:,.0f}",
            f"{row.total_return_pct:+.2f}%",
            f"{row.max_drawdown_pct:+.2f}%",
            str(row.trade_count),
        )
        x = x0
        for value, width in zip(values, widths):
            ax.text(x + 0.008, y + 0.016, value, fontsize=9.3, color="#1f2d36", transform=ax.transAxes)
            x += width


def _draw_method_page(ax, report: ScorecardReport) -> None:
    ax.add_patch(plt.Rectangle((0, 0.9), 1, 0.1, color="#17212a", transform=ax.transAxes))
    ax.text(0.06, 0.94, "方法與使用邊界", color="white", fontsize=18, fontweight="bold", transform=ax.transAxes)
    sections = [
        (
            "資料口徑",
            [
                f"報告日期：{report.report_date}",
                f"延遲公開規則：報告日期往前 {report.delay_days} 天，實際使用最近可用共同交易日 {report.data_end_date}。",
                f"模擬起點：{report.tracking_start_date}，初始資金 {report.initial_cash_twd:,.0f} 元。",
            ],
        ),
        (
            "比較組",
            [
                f"模型組：{report.model_tracking_label}，依 AI大型權值股最佳版 v20260605 的歷史模型狀態重建。",
                "對照組：0050買進持有、0050正二買進持有。",
            ],
        ),
        (
            "模型追蹤紀錄",
            [
                f"{record['start_date']} 起：{record['label']}({record['ticker'].replace('.TW', '')})，曝險約 {record['exposure_pct']:.0f}%"
                for record in report.model_holding_records
            ] or ["此區間沒有可追蹤持股紀錄。"],
        ),
        (
            "公開邊界",
            [
                "本報告只呈現延遲後的歷史觀察結果，不公布即時訊號。",
                "成績單用語不得寫成買進、賣出、推薦或明牌。",
                "歷史績效不能保證未來結果，模型也可能失效。",
            ],
        ),
    ]
    y = 0.82
    for title, lines in sections:
        ax.add_patch(plt.Rectangle((0.06, y - 0.015), 0.88, 0.035, facecolor="#e8eef3", edgecolor="#d9e0e5", transform=ax.transAxes))
        ax.text(0.075, y - 0.004, title, fontsize=12.5, fontweight="bold", color="#17212a", transform=ax.transAxes)
        y -= 0.05
        for line in lines:
            ax.text(0.075, y, f"• {line}", fontsize=10.2, color="#1f2d36", transform=ax.transAxes)
            y -= 0.036
        y -= 0.025
    _draw_footer(ax, report.disclaimer)


def _draw_footer(ax, text: str) -> None:
    ax.plot([0.06, 0.94], [0.078, 0.078], color="#d9e0e5", linewidth=0.8, transform=ax.transAxes)
    ax.text(0.06, 0.032, text, color="#73818b", fontsize=8.2, transform=ax.transAxes)
    ax.text(0.94, 0.032, "被AI研究所", color="#73818b", fontsize=8.2, ha="right", transform=ax.transAxes)


def _equity_rows(equity_curve: pd.DataFrame) -> list[dict]:
    return [
        {"date": index.strftime("%Y-%m-%d"), "total_value_twd": round(float(row["total_value"]), 2)}
        for index, row in equity_curve.iterrows()
    ]


def _normalized_equity_values(rows: list[dict], initial_cash: float) -> list[float]:
    if not rows:
        return []
    first_value = float(rows[0]["total_value_twd"])
    if first_value <= 0:
        return [float(row["total_value_twd"]) / 10_000 for row in rows]
    values = [float(row["total_value_twd"]) / first_value * initial_cash / 10_000 for row in rows]
    values[0] = initial_cash / 10_000
    return values


def _series_overlap(report: ScorecardReport, left: str, right: str) -> bool:
    left_rows = report.equity_curves.get(left, [])
    right_rows = report.equity_curves.get(right, [])
    if len(left_rows) != len(right_rows) or not left_rows:
        return False
    return all(
        left_row["date"] == right_row["date"]
        and abs(float(left_row["total_value_twd"]) - float(right_row["total_value_twd"])) < 0.01
        for left_row, right_row in zip(left_rows, right_rows)
    )


def _flat_equity_rows(report: ScorecardReport) -> list[dict]:
    rows = []
    for name, curve_rows in report.equity_curves.items():
        for row in curve_rows:
            rows.append({"series": name, **row})
    return rows


def _flat_normalized_equity_rows(report: ScorecardReport) -> list[dict]:
    rows = []
    for name, curve_rows in report.equity_curves.items():
        normalized_values = _normalized_equity_values(curve_rows, report.initial_cash_twd)
        for row, normalized in zip(curve_rows, normalized_values):
            rows.append(
                {
                    "series": name,
                    "date": row["date"],
                    "chart_value_twd": round(normalized * 10_000, 2),
                    "chart_value_10k_twd": round(normalized, 6),
                }
            )
    return rows


def _current_model_ticker(equity_curve: pd.DataFrame) -> str:
    if "current_ticker" not in equity_curve.columns:
        return "cash"
    for ticker in reversed(equity_curve["current_ticker"].astype(str).tolist()):
        if ticker and ticker != "cash":
            return ticker
    return "cash"


def _model_holding_records(equity_curve: pd.DataFrame, labels: dict[str, str]) -> list[dict]:
    if "current_ticker" not in equity_curve.columns:
        return []
    records = []
    last_ticker = None
    for date, row in equity_curve.iterrows():
        ticker = str(row.get("current_ticker") or "cash")
        if ticker == last_ticker:
            continue
        last_ticker = ticker
        records.append(
            {
                "start_date": pd.Timestamp(date).strftime("%Y-%m-%d"),
                "ticker": ticker,
                "label": "現金" if ticker == "cash" else labels.get(ticker, ticker),
                "exposure_pct": round(float(row.get("current_exposure", 0.0)) * 100, 2),
            }
        )
    return records


if __name__ == "__main__":
    main()
