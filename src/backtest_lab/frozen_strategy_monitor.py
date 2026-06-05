from __future__ import annotations

import argparse
import json
import textwrap
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from backtest_lab.config import load_config
from backtest_lab.data import download_yfinance_prices, split_adjusted_dividends
from backtest_lab.market_regime import classify_market_regime, latest_available_date
from backtest_lab.regime_mode_switch import (
    frozen_cycle_proven_top1_v1_variant,
    simulate_regime_mode_switch,
)
from backtest_lab.strategies import relative_strength_scores


STRATEGY_ID = "frozen_cycle_proven_top1_v1"
REPORT_VERSION = "v20260605"
REPORT_NAME = "AI股票最佳策略每日觀察報告"
REPORT_VARIANT_LABEL = f"最佳版 {REPORT_VERSION}"
DRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1O6Se-HfI7ZDTQ-LWeAO6f8vtvoLcCzIj"
DEFAULT_REPLAY_START = "2020-01-02"
NO_DATA_EXIT_CODE = 3


@dataclass(frozen=True)
class FrozenStrategySignal:
    strategy_id: str
    signal_date: str
    execution_timing: str
    market_regime: str
    market_regime_label: str
    current_ticker: str
    current_label: str
    current_exposure: float
    target_ticker: str
    target_label: str
    target_exposure: float
    action: str
    target_is_actionable: bool
    model_target_status: str
    cash_account_reference: str
    attack_gate_active: bool
    attack_gate_ever_activated: bool
    risk_off_active: bool
    model_total_value_twd: float
    close_prices: dict[str, float]
    ranking: list[dict]
    projected_trades: list[dict]

    def to_dict(self) -> dict:
        return asdict(self)


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Generate {REPORT_NAME}.")
    parser.add_argument("--config", default="configs/ep05_universe.json")
    parser.add_argument("--strategy-config", default="configs/frozen_cycle_proven_top1_v1.json")
    parser.add_argument("--group-id", default="group_c_0050_00631l_plus_mega_caps")
    parser.add_argument("--signal-date", required=True, help="YYYY-MM-DD Taiwan market signal date.")
    parser.add_argument("--cache-dir", default="backtest_cache/frozen_strategy_monitor")
    parser.add_argument("--output-root", default="outputs/frozen_strategy_monitor")
    parser.add_argument("--replay-start", default=DEFAULT_REPLAY_START)
    args = parser.parse_args()

    output_dir = Path(args.output_root) / args.signal_date.replace("-", "")
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    frozen_config = json.loads(Path(args.strategy_config).read_text(encoding="utf-8"))
    if frozen_config["strategy_id"] != STRATEGY_ID or frozen_config["status"] != "frozen_baseline":
        raise ValueError("The daily report must use the frozen baseline strategy config.")

    group = config.group_by_id(args.group_id)
    labels = {asset.ticker: asset.label for asset in group.assets}
    asset_types = {asset.ticker: asset.asset_type for asset in group.assets}
    tickers = sorted(labels)
    try:
        prices = download_yfinance_prices(
            tickers=tickers,
            start_date=(pd.Timestamp(args.replay_start) - pd.DateOffset(years=2)).strftime("%Y-%m-%d"),
            end_date=args.signal_date,
            cache_dir=args.cache_dir,
            allow_edge_gap=False,
        )
    except ValueError as error:
        _write_download_waiting(output_dir, args.signal_date, error)
        print(f"WAITING_FOR_DOWNLOAD={error}")
        raise SystemExit(NO_DATA_EXIT_CODE) from error
    incomplete = _incomplete_tickers(prices, args.signal_date)
    if incomplete:
        prices = fill_signal_date_from_twse(prices, args.signal_date, incomplete)
        _write_price_cache(Path(args.cache_dir), prices, incomplete)
        incomplete = _incomplete_tickers(prices, args.signal_date)
    if incomplete:
        _write_skip(output_dir, args.signal_date, prices, incomplete)
        print(f"WAITING_FOR_DATA={','.join(incomplete)}")
        raise SystemExit(NO_DATA_EXIT_CODE)

    signal = build_frozen_strategy_signal(
        signal_date=args.signal_date,
        replay_start=args.replay_start,
        prices_by_ticker=prices,
        labels=labels,
        asset_types=asset_types,
        initial_cash=config.initial_cash_twd,
        cost_model=config.cost_model,
        manual_splits=config.manual_splits,
    )
    _write_outputs(output_dir, signal)
    print(f"REPORT_DIR={output_dir.resolve()}")
    print(f"LATEST_PDF={(output_dir / _report_filename('pdf', latest=True)).resolve()}")


def build_frozen_strategy_signal(
    *,
    signal_date: str,
    replay_start: str,
    prices_by_ticker: dict[str, pd.DataFrame],
    labels: dict[str, str],
    asset_types: dict[str, str],
    initial_cash: float,
    cost_model,
    manual_splits: dict[str, tuple[dict[str, float | str], ...]] | None = None,
) -> FrozenStrategySignal:
    signal_ts = pd.Timestamp(signal_date)
    projection_ts = signal_ts + pd.offsets.BDay(1)
    projected_prices = {
        ticker: _append_projection_row(frame, signal_ts, projection_ts)
        for ticker, frame in prices_by_ticker.items()
    }
    splits = manual_splits or {}
    dividends = {
        ticker: split_adjusted_dividends(frame, splits.get(ticker, ()))
        for ticker, frame in projected_prices.items()
    }
    result = simulate_regime_mode_switch(
        name=STRATEGY_ID,
        prices_by_ticker=projected_prices,
        asset_types=asset_types,
        market_prices=projected_prices["0050.TW"],
        start_date=replay_start,
        end_date=projection_ts.strftime("%Y-%m-%d"),
        initial_cash=initial_cash,
        cost_model=cost_model,
        variant=frozen_cycle_proven_top1_v1_variant(),
        dividend_series_by_ticker=dividends,
    )
    current = result.equity_curve.loc[signal_ts]
    target = result.equity_curve.loc[projection_ts]
    current_ticker = str(current["current_ticker"])
    target_ticker = str(target["current_ticker"])
    current_exposure = float(current["current_exposure"])
    target_exposure = float(target["current_exposure"])
    projected_trades = [
        {
            "ticker": trade.ticker,
            "label": labels.get(trade.ticker, trade.ticker),
            "action": trade.action,
            "shares": trade.shares,
            "reference_price": round(trade.price, 4),
            "gross_amount_twd": round(trade.gross_amount, 2),
            "estimated_costs_twd": trade.costs,
            "reason": trade.reason,
        }
        for trade in result.trades
        if trade.date == projection_ts.strftime("%Y-%m-%d") and trade.action in {"buy", "sell"}
    ]
    scores = relative_strength_scores(prices_by_ticker, signal_ts)
    ranking = _ranking_rows(scores, labels)
    regime = classify_market_regime(
        prices_by_ticker["0050.TW"],
        signal_ts,
        universe_prices=prices_by_ticker,
    )
    return FrozenStrategySignal(
        strategy_id=STRATEGY_ID,
        signal_date=signal_date,
        execution_timing="下一個台股交易日，由投資人自行決定是否執行",
        market_regime=regime.regime,
        market_regime_label=regime.regime_label,
        current_ticker=current_ticker,
        current_label=_label(current_ticker, labels),
        current_exposure=current_exposure,
        target_ticker=target_ticker,
        target_label=_label(target_ticker, labels),
        target_exposure=target_exposure,
        action=_action(current_ticker, target_ticker, current_exposure, target_exposure),
        target_is_actionable=_target_is_actionable(target_ticker, target_exposure),
        model_target_status=_model_target_status(target_ticker, target_exposure),
        cash_account_reference=_cash_account_reference(target_ticker, _label(target_ticker, labels), target_exposure),
        attack_gate_active=bool(target["attack_gate_active"]),
        attack_gate_ever_activated=bool(target["attack_gate_ever_activated"]),
        risk_off_active=bool(target["risk_off_active"]),
        model_total_value_twd=float(current["total_value"]),
        close_prices={
            ticker: round(float(frame.loc[signal_ts, "close"]), 4)
            for ticker, frame in prices_by_ticker.items()
        },
        ranking=ranking,
        projected_trades=projected_trades,
    )


def _append_projection_row(frame: pd.DataFrame, signal_date: pd.Timestamp, projection_date: pd.Timestamp) -> pd.DataFrame:
    projected = frame.loc[frame.index <= signal_date].copy()
    source = projected.loc[signal_date].copy()
    for column in ("dividend", "stock_split"):
        if column in source.index:
            source[column] = 0.0
    projected.loc[projection_date] = source
    return projected.sort_index()


def _incomplete_tickers(prices_by_ticker: dict[str, pd.DataFrame], signal_date: str) -> list[str]:
    signal_ts = pd.Timestamp(signal_date)
    incomplete: list[str] = []
    for ticker, frame in prices_by_ticker.items():
        if signal_ts not in frame.index:
            incomplete.append(ticker)
            continue
        row = frame.loc[signal_ts]
        if pd.isna(row.get("open")) or pd.isna(row.get("close")) or pd.isna(row.get("adj_close")):
            incomplete.append(ticker)
    return sorted(incomplete)


def fill_signal_date_from_twse(
    prices_by_ticker: dict[str, pd.DataFrame],
    signal_date: str,
    tickers: list[str],
    fetcher=None,
) -> dict[str, pd.DataFrame]:
    filled = dict(prices_by_ticker)
    fetch = fetcher or _fetch_twse_stock_day
    for ticker in tickers:
        try:
            row = fetch(ticker, signal_date)
        except (OSError, urllib.error.URLError, TimeoutError):
            row = None
        if row is None:
            continue
        frame = filled[ticker].copy()
        frame.loc[pd.Timestamp(signal_date)] = row
        filled[ticker] = frame.sort_index()
    return filled


def _write_price_cache(cache_dir: Path, prices_by_ticker: dict[str, pd.DataFrame], tickers: list[str]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    for ticker in tickers:
        if ticker not in prices_by_ticker:
            continue
        path = cache_dir / f"{ticker.replace('.', '_')}.csv"
        frame = prices_by_ticker[ticker].copy()
        frame.index.name = "date"
        frame.reset_index().to_csv(path, index=False)


def _fetch_twse_stock_day(ticker: str, signal_date: str) -> dict[str, float] | None:
    stock_no = ticker.split(".")[0]
    query = urllib.parse.urlencode(
        {
            "date": pd.Timestamp(signal_date).strftime("%Y%m%d"),
            "stockNo": stock_no,
            "response": "json",
        }
    )
    url = f"https://www.twse.com.tw/rwd/zh/afterTrading/STOCK_DAY?{query}"
    request = urllib.request.Request(url, headers={"User-Agent": "AI_stock_backtest_lab/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if payload.get("stat") != "OK":
        return None
    target = pd.Timestamp(signal_date)
    fields = payload.get("fields", [])
    data = payload.get("data", [])
    try:
        date_i = fields.index("日期")
        volume_i = fields.index("成交股數")
        open_i = fields.index("開盤價")
        high_i = fields.index("最高價")
        low_i = fields.index("最低價")
        close_i = fields.index("收盤價")
    except ValueError:
        return None
    for item in data:
        if _roc_date_to_timestamp(str(item[date_i])) != target:
            continue
        close = _twse_float(item[close_i])
        return {
            "open": _twse_float(item[open_i]),
            "high": _twse_float(item[high_i]),
            "low": _twse_float(item[low_i]),
            "close": close,
            "adj_close": close,
            "volume": _twse_float(item[volume_i]),
            "dividend": 0.0,
            "stock_split": 0.0,
        }
    return None


def _roc_date_to_timestamp(value: str) -> pd.Timestamp:
    year, month, day = [int(part) for part in value.split("/")]
    return pd.Timestamp(year + 1911, month, day)


def _twse_float(value: str) -> float:
    cleaned = str(value).replace(",", "").replace("--", "").strip()
    return float(cleaned) if cleaned else 0.0


def _ranking_rows(scores: dict[str, float], labels: dict[str, str]) -> list[dict]:
    ordered = sorted(scores.items(), key=lambda item: (item[1], item[0]), reverse=True)
    return [
        {
            "rank": rank,
            "ticker": ticker,
            "label": labels.get(ticker, ticker),
            "score": round(float(score), 6),
            "score_band": _score_band(float(score)),
            "role": "市場訊號/等待工具" if ticker in {"0050.TW", "00631L.TW"} else "進攻候選",
        }
        for rank, (ticker, score) in enumerate(ordered, start=1)
    ]


def _score_band(score: float) -> str:
    if score >= 0.8:
        return "極強勢"
    if score >= 0.5:
        return "強勢觀察"
    if score >= 0.25:
        return "中性偏強"
    if score >= 0:
        return "弱勢或未明顯領先"
    return "弱勢/風險偏高"


def _score_guide_lines() -> list[str]:
    return [
        "可執行門檻不是單看分數，而是該標的同時成為「下一交易日模型目標」，且「模型目標狀態」為有合格模型目標。",
        "0.80 以上：極強勢。若同時通過模型目標條件，才列為可執行參考。",
        "0.50 到 0.80：強勢觀察。可進入候選，但不代表立刻轉入。",
        "0.25 到 0.50：中性偏強。需要等待更明確趨勢或模型目標確認。",
        "0 到 0.25：弱勢或未明顯領先。通常不是進攻優先。",
        "0 以下：弱勢/風險偏高。通常不作為進攻優先。",
    ]


def _action(current_ticker: str, target_ticker: str, current_exposure: float, target_exposure: float) -> str:
    if current_ticker == target_ticker and abs(current_exposure - target_exposure) < 0.02:
        return "維持目前模型部位"
    if target_ticker == "cash":
        return "模型轉為現金觀察"
    if current_ticker == "cash":
        return "模型建立新部位"
    if current_ticker == target_ticker:
        return "模型調整同一標的曝險"
    return "模型輪動至新標的"


def _model_target_status(target_ticker: str, target_exposure: float) -> str:
    if not _target_is_actionable(target_ticker, target_exposure):
        return "沒有合格持股目標，模型目標為現金觀察"
    return "有合格模型目標"


def _cash_account_reference(target_ticker: str, target_label: str, target_exposure: float) -> str:
    if not _target_is_actionable(target_ticker, target_exposure):
        return "若目前全現金，模型不建立新部位，只保留觀察。"
    return f"若目前全現金且選擇跟隨模型，模型目標是{target_label}，目標曝險約 {target_exposure:.0%}。"


def _target_is_actionable(target_ticker: str, target_exposure: float) -> bool:
    return target_ticker != "cash" and target_exposure > 0


def _label(ticker: str, labels: dict[str, str]) -> str:
    return "現金" if ticker == "cash" else labels.get(ticker, ticker)


def _write_outputs(output_dir: Path, signal: FrozenStrategySignal) -> None:
    waiting_status = output_dir / "waiting_status.json"
    if waiting_status.exists():
        waiting_status.unlink()
    payload = {
        "status": "ready",
        "report_name": REPORT_NAME,
        "drive_folder_url": DRIVE_FOLDER_URL,
        "signal": signal.to_dict(),
        "disclaimer": "AI 輔助市場觀察、回測與紀律提醒，不是投資建議。",
    }
    (output_dir / "frozen_strategy_signal.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame([{
        "strategy_id": signal.strategy_id,
        "signal_date": signal.signal_date,
        "market_regime": signal.market_regime,
        "market_regime_label": signal.market_regime_label,
        "action": signal.action,
        "current_ticker": signal.current_ticker,
        "current_label": signal.current_label,
        "current_exposure": signal.current_exposure,
        "target_ticker": signal.target_ticker,
        "target_label": signal.target_label,
        "target_exposure": signal.target_exposure,
        "target_is_actionable": signal.target_is_actionable,
        "model_target_status": signal.model_target_status,
        "cash_account_reference": signal.cash_account_reference,
        "attack_gate_active": signal.attack_gate_active,
        "attack_gate_ever_activated": signal.attack_gate_ever_activated,
        "risk_off_active": signal.risk_off_active,
    }]).to_csv(output_dir / "frozen_strategy_daily_status.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(signal.ranking).to_csv(output_dir / "frozen_strategy_ranking.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(signal.projected_trades).to_csv(
        output_dir / "frozen_strategy_projected_trades.csv",
        index=False,
        encoding="utf-8-sig",
    )
    report = _markdown_report(signal)
    dated_md = output_dir / _report_filename("md", signal.signal_date)
    latest_md = output_dir / _report_filename("md", latest=True)
    dated_md.write_text(report, encoding="utf-8")
    latest_md.write_text(report, encoding="utf-8")
    _write_signal_pdf(output_dir / _report_filename("pdf", signal.signal_date), signal)
    _write_signal_pdf(output_dir / _report_filename("pdf", latest=True), signal)


def _report_filename(extension: str, signal_date: str | None = None, *, latest: bool = False) -> str:
    if latest:
        suffix = "最新版"
    elif signal_date:
        suffix = signal_date
    else:
        raise ValueError("signal_date is required when latest is false.")
    return f"{REPORT_NAME}_{suffix}_{REPORT_VERSION}.{extension}"


def _markdown_report(signal: FrozenStrategySignal) -> str:
    ranking_lines = [
        f"{row['rank']}. {row['label']} ({row['ticker']})，分數 {row['score']:.4f}（{row['score_band']}），角色：{row['role']}"
        for row in signal.ranking
    ]
    score_guide_lines = [f"- {line}" for line in _score_guide_lines()]
    trade_lines = [
        f"- {row['action']} {row['label']}，模型參考股數 {row['shares']}，參考價 {row['reference_price']}"
        for row in signal.projected_trades
    ] or ["- 模型目標未改變，沒有模擬換倉動作。"]
    return "\n".join(
        [
            f"# {REPORT_NAME}",
            "",
            "## 摘要",
            "",
            f"- 策略版本：{REPORT_VARIANT_LABEL}",
            f"- 訊號日期：{signal.signal_date}",
            f"- 執行時點：{signal.execution_timing}",
            "- 定位：每日 AI 輔助操作建議，投資人自行判斷，不是自動下單，也不是投資建議。",
            "",
            "## 今日結論",
            "",
            f"- 市場環境：{signal.market_regime_label}",
            f"- 模型動作：{signal.action}",
            f"- 模型目標狀態：{signal.model_target_status}",
            f"- 今日收盤後模型部位：{signal.current_label}，曝險約 {signal.current_exposure:.0%}",
            f"- 下一交易日模型目標：{signal.target_label}，曝險約 {signal.target_exposure:.0%}",
            f"- 全現金帳戶參考：{signal.cash_account_reference}",
            f"- 進攻閘門：{'已開啟' if signal.attack_gate_active else '尚未開啟'}",
            f"- 風險關閉狀態：{'啟動' if signal.risk_off_active else '未啟動'}",
            "",
            "## 模型模擬動作",
            "",
            *trade_lines,
            "",
            "上述股數與價格只用來重建模型狀態，不是針對使用者資產的實際下單建議。",
            "",
            "## 九標的強弱排名",
            "",
            "注意：排名只代表相對強弱，不等於買入資格；可執行參考請看「下一交易日模型目標」與「模型目標狀態」。",
            "",
            "### 分數解讀",
            "",
            *score_guide_lines,
            "",
            *ranking_lines,
            "",
            "## 角色說明",
            "",
            "- 0050 是市場代理、循環判斷基準與比較基準。",
            "- 0050正二是槓桿大盤訊號、積極等待與曝險工具，不是唯一趨勢判斷依據。",
            "- 七檔指定個股才是進攻模式的持股候選。",
            "",
            "## 風險聲明",
            "",
            "歷史回測與 shadow mode 都不能保證未來績效。本報告只供 AI 輔助市場觀察、回測與紀律提醒。",
            "",
        ]
    )


def _write_skip(
    output_dir: Path,
    signal_date: str,
    prices_by_ticker: dict[str, pd.DataFrame],
    incomplete: list[str],
) -> None:
    payload = {
        "status": "waiting_for_complete_market_data",
        "signal_date": signal_date,
        "incomplete_tickers": incomplete,
        "latest_available_dates": {
            ticker: (
                latest_available_date(frame).strftime("%Y-%m-%d")
                if latest_available_date(frame) is not None
                else ""
            )
            for ticker, frame in prices_by_ticker.items()
        },
    }
    (output_dir / "waiting_status.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _write_download_waiting(output_dir: Path, signal_date: str, error: Exception) -> None:
    payload = {
        "status": "waiting_for_market_data_download",
        "signal_date": signal_date,
        "error": str(error),
    }
    (output_dir / "waiting_status.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _write_signal_pdf(path: Path, signal: FrozenStrategySignal) -> None:
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK TC", "Microsoft JhengHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    with PdfPages(path) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69), facecolor="#f4f6f8")
        ax = fig.add_axes((0, 0, 1, 1))
        ax.axis("off")
        _draw_header(ax, signal)
        _draw_metric_cards(ax, signal)
        _draw_ranking_table(ax, signal)
        _draw_footer(ax, "本報告為 AI 輔助市場觀察與紀律提醒，不是投資建議。")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig = plt.figure(figsize=(8.27, 11.69), facecolor="#f4f6f8")
        ax = fig.add_axes((0, 0, 1, 1))
        ax.axis("off")
        _draw_second_page(ax, signal)
        _draw_footer(ax, f"{REPORT_NAME} · {signal.signal_date}")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def _draw_header(ax, signal: FrozenStrategySignal) -> None:
    ax.add_patch(plt.Rectangle((0, 0.86), 1, 0.14, color="#17212a", transform=ax.transAxes))
    ax.text(0.06, 0.94, REPORT_NAME, color="white", fontsize=20, fontweight="bold", transform=ax.transAxes)
    ax.text(
        0.06,
        0.895,
        f"訊號日 {signal.signal_date} · {REPORT_VARIANT_LABEL} · 人工決策",
        color="#c8d5df",
        fontsize=11,
        transform=ax.transAxes,
    )
    ax.text(
        0.94,
        0.925,
        signal.market_regime_label,
        color="white",
        fontsize=15,
        ha="right",
        fontweight="bold",
        transform=ax.transAxes,
    )


def _draw_metric_cards(ax, signal: FrozenStrategySignal) -> None:
    cards = [
        ("模型動作", signal.action, "#2457a7"),
        ("下一交易日目標", f"{signal.target_label} · {signal.target_exposure:.0%}", "#13795b"),
        ("目標狀態", signal.model_target_status, "#17212a"),
        ("風險狀態", "風險關閉" if signal.risk_off_active else "風控未觸發", "#b42318" if signal.risk_off_active else "#13795b"),
    ]
    for index, (label, value, color) in enumerate(cards):
        x = 0.06 + index * 0.225
        ax.add_patch(
            plt.Rectangle((x, 0.73), 0.2, 0.09, facecolor="white", edgecolor="#d9e0e5", linewidth=1, transform=ax.transAxes)
        )
        ax.text(x + 0.014, 0.79, label, color="#66737d", fontsize=9.5, transform=ax.transAxes)
        ax.text(x + 0.014, 0.755, _fit_card_text(value), color=color, fontsize=11.5, fontweight="bold", transform=ax.transAxes)


def _draw_ranking_table(ax, signal: FrozenStrategySignal) -> None:
    ax.text(0.06, 0.68, "九標的強弱排名", color="#17212a", fontsize=15, fontweight="bold", transform=ax.transAxes)
    ax.add_patch(
        plt.Rectangle((0.06, 0.61), 0.88, 0.052, facecolor="#fff8e8", edgecolor="#ead8ac", linewidth=0.8, transform=ax.transAxes)
    )
    ax.text(
        0.075,
        0.642,
        "分數解讀：0.80+ 極強勢；0.50-0.80 強勢觀察；0.25-0.50 中性偏強；0 以下偏弱。",
        color="#624711",
        fontsize=8.8,
        transform=ax.transAxes,
    )
    ax.text(
        0.075,
        0.622,
        "可執行參考仍以「下一交易日模型目標」與「模型目標狀態」為準，不能只看排名。",
        color="#624711",
        fontsize=8.8,
        transform=ax.transAxes,
    )
    headers = ("名次", "標的", "角色", "分數", "收盤價")
    widths = (0.08, 0.24, 0.24, 0.15, 0.16)
    x0 = 0.06
    y = 0.575
    row_h = 0.039
    ax.add_patch(plt.Rectangle((x0, y), 0.88, row_h, facecolor="#e8eef3", edgecolor="#d9e0e5", transform=ax.transAxes))
    x = x0
    for header, width in zip(headers, widths):
        ax.text(x + 0.01, y + 0.014, header, color="#31414d", fontsize=10, fontweight="bold", transform=ax.transAxes)
        x += width
    for row in signal.ranking:
        y -= row_h
        is_target = row["ticker"] == signal.target_ticker
        fill = "#fff7e6" if is_target else "white"
        ax.add_patch(plt.Rectangle((x0, y), 0.88, row_h, facecolor=fill, edgecolor="#d9e0e5", linewidth=0.8, transform=ax.transAxes))
        values = (
            str(row["rank"]),
            f"{row['label']} ({row['ticker'].replace('.TW', '')})",
            row["role"],
            f"{row['score']:.4f}",
            _format_price(signal.close_prices.get(row["ticker"])),
        )
        x = x0
        for value, width in zip(values, widths):
            weight = "bold" if is_target else "normal"
            color = "#a15c00" if is_target else "#1f2d36"
            ax.text(x + 0.01, y + 0.014, value, color=color, fontsize=9.5, fontweight=weight, transform=ax.transAxes)
            x += width
    ax.text(
        0.06,
        0.19,
        "重點：排名只代表相對強弱，不等於買入資格；是否建立部位，以「下一交易日模型目標」與「模型目標狀態」為準。",
        color="#52616b",
        fontsize=9.5,
        transform=ax.transAxes,
    )


def _draw_second_page(ax, signal: FrozenStrategySignal) -> None:
    ax.add_patch(plt.Rectangle((0, 0.9), 1, 0.1, color="#17212a", transform=ax.transAxes))
    ax.text(0.06, 0.94, "操作摘要與風險說明", color="white", fontsize=18, fontweight="bold", transform=ax.transAxes)
    sections = [
        (
            "模型模擬動作",
            [
                f"{row['action']} {row['label']}，模型參考股數 {row['shares']}，參考價 {row['reference_price']}"
                for row in signal.projected_trades
            ] or ["模型目標未改變，沒有模擬換倉動作。"],
        ),
        (
            "狀態解讀",
            [
                f"模型目標狀態：{signal.model_target_status}",
                f"全現金帳戶參考：{signal.cash_account_reference}",
                f"進攻閘門：{'已開啟' if signal.attack_gate_active else '尚未開啟'}",
                f"歷史循環是否已證明：{'是' if signal.attack_gate_ever_activated else '否'}",
                f"風險關閉狀態：{'啟動' if signal.risk_off_active else '未啟動'}",
                f"模型重建淨值：{signal.model_total_value_twd:,.0f} 元",
            ],
        ),
        (
            "使用邊界",
            [
                "強弱排名是觀察清單，不是買入資格清單。",
                "可執行參考只看模型目標、目標曝險與持倉工作台依個人現金/持股計算出的參考股數。",
                "本報告是每日 AI 輔助操作建議，投資人隔日自行決定是否執行。",
                "模型參考股數只用來重建策略狀態，不等於個人帳戶的實際下單股數。",
                "實際下單前仍需確認可用現金、零股成交、滑價、交易成本與個人風險承受度。",
            ],
        ),
        (
            "風險聲明",
            [
                "歷史回測與 shadow mode 都不能保證未來績效。",
                "本策略可能在風格轉換、資料延遲、極端跳空或流動性不足時失效。",
                "本報告只能作為 AI 輔助市場觀察、回測與紀律提醒，不是投資建議。",
            ],
        ),
    ]
    y = 0.83
    for title, lines in sections:
        ax.add_patch(plt.Rectangle((0.06, y - 0.015), 0.88, 0.035, facecolor="#e8eef3", edgecolor="#d9e0e5", transform=ax.transAxes))
        ax.text(0.075, y - 0.004, title, fontsize=12.5, fontweight="bold", color="#17212a", transform=ax.transAxes)
        y -= 0.055
        for line in lines:
            wrapped = textwrap.wrap(line, width=48)
            for text in wrapped:
                ax.text(0.075, y, f"• {text}", fontsize=10.5, color="#1f2d36", transform=ax.transAxes)
                y -= 0.032
        y -= 0.028


def _draw_footer(ax, text: str) -> None:
    ax.text(0.06, 0.055, text, color="#73818b", fontsize=8.5, transform=ax.transAxes)
    ax.text(0.94, 0.055, "AI_stock_backtest_lab", color="#73818b", fontsize=8.5, ha="right", transform=ax.transAxes)


def _format_price(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:,.2f}".rstrip("0").rstrip(".")


def _fit_card_text(value: str) -> str:
    return value if len(value) <= 12 else textwrap.shorten(value, width=15, placeholder="...")


def _write_pdf(path: Path, markdown_text: str) -> None:
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK TC", "Microsoft JhengHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    lines: list[str] = []
    for raw_line in markdown_text.splitlines():
        clean = raw_line.replace("#", "").replace("`", "").strip()
        if not clean:
            lines.append("")
            continue
        lines.extend(textwrap.wrap(clean, width=42) or [""])
    with PdfPages(path) as pdf:
        for start in range(0, len(lines), 38):
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.text(0.08, 0.95, "\n".join(lines[start : start + 38]), va="top", ha="left", fontsize=11, linespacing=1.35)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


if __name__ == "__main__":
    main()
