from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import pandas as pd

from backtest_lab.config import load_config
from backtest_lab.data import download_yfinance_prices, split_adjusted_dividends
from backtest_lab.frozen_report_pdf import (
    DETAIL_BOTTOM_Y,
    DETAIL_LINE_HEIGHT,
    DETAIL_SECTION_GAP,
    DETAIL_START_Y,
    DETAIL_TITLE_STEP,
    DETAIL_WRAP_WIDTH,
    detail_section_height as _pdf_detail_section_height,
    detail_sections as _pdf_detail_sections,
    paginate_detail_sections as _pdf_paginate_detail_sections,
    write_signal_pdf as _pdf_write_signal_pdf,
)
from backtest_lab.market_regime import classify_market_regime, latest_available_date
from backtest_lab.regime_mode_switch import (
    frozen_cycle_proven_top1_v1_variant,
    simulate_regime_mode_switch,
)
from backtest_lab.portfolio_app import DEFAULT_STORE_PATH, PortfolioStore, build_dashboard
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
    personal_portfolio: dict | None = None
    personal_recommendations: list[dict] | None = None

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
    parser.add_argument("--portfolio-store", default=DEFAULT_STORE_PATH)
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
    signal = attach_personal_portfolio(
        signal,
        portfolio_store=args.portfolio_store,
        asset_types=asset_types,
        cost_model=config.cost_model,
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


def attach_personal_portfolio(
    signal: FrozenStrategySignal,
    *,
    portfolio_store: str | Path,
    asset_types: dict[str, str],
    cost_model,
) -> FrozenStrategySignal:
    store_path = Path(portfolio_store)
    if not store_path.exists():
        return signal
    user = PortfolioStore(store_path).get_user()
    dashboard = build_dashboard(user, signal.to_dict(), asset_types, cost_model)
    return replace(
        signal,
        personal_portfolio=dashboard["portfolio"],
        personal_recommendations=dashboard["recommendations"],
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
        "report_mode": _report_mode(signal),
        "drive_folder_url": DRIVE_FOLDER_URL,
        "signal": signal.to_dict(),
        "disclaimer": "AI 輔助市場觀察、回測與紀律提醒，不是投資建議。",
    }
    (output_dir / "frozen_strategy_signal.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    personal_summary = _personal_exposure_summary(signal) if signal.personal_portfolio else None
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
        "report_mode": _report_mode(signal),
        "personal_portfolio_attached": signal.personal_portfolio is not None,
        "personal_total_value_twd": personal_summary["total_value_twd"] if personal_summary else "",
        "personal_cash_exposure": personal_summary["cash_exposure"] if personal_summary else "",
        "personal_market_exposure": personal_summary["market_exposure"] if personal_summary else "",
        "personal_target_actual_exposure": personal_summary["target_actual_exposure"] if personal_summary else "",
        "personal_target_gap_exposure": personal_summary["target_gap_exposure"] if personal_summary else "",
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
    personal_lines = _personal_markdown_lines(signal)
    return "\n".join(
        [
            f"# {REPORT_NAME}",
            "",
            "## 摘要",
            "",
            f"- 策略版本：{REPORT_VARIANT_LABEL}",
            f"- 報告模式：{_report_mode_label(signal)}",
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
            *personal_lines,
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


def _personal_markdown_lines(signal: FrozenStrategySignal) -> list[str]:
    if not signal.personal_portfolio:
        return [
            "## 個人持倉參考",
            "",
            "- 尚未連結個人持倉檔；本報告為一般版，只顯示模型帳戶狀態。",
        ]
    summary = _personal_exposure_summary(signal)
    recommendation_lines = [
        f"- {_display_action(row['action'])} {row['ticker']}，參考股數 {row['shares']}，目標比例 {row['target_exposure']:.0%}，原因：{row['reason']}"
        for row in signal.personal_recommendations or []
    ] or ["- 依目前持倉與模型目標，暫無可計算的個人調整參考。"]
    return [
        "## 個人持倉參考",
        "",
        f"- 個人組合估值：約 {summary['total_value_twd']:,.2f} 元",
        f"- 可用現金：約 {summary['cash_twd']:,.2f} 元，現金水位約 {summary['cash_exposure']:.2%}",
        f"- 目前持股水位：約 {summary['market_exposure']:.2%}",
        f"- 模型目標標的：{signal.target_label}，模型目標曝險 {signal.target_exposure:.0%}",
        f"- 個人目前目標標的曝險：約 {summary['target_actual_exposure']:.2%}，與模型目標差距約 {summary['target_gap_exposure']:+.2%}",
        "",
        "### 個人帳戶參考調整",
        "",
        *recommendation_lines,
        "",
        "個人持倉參考只依目前手動輸入資料估算，不是自動下單或投資建議。",
    ]


def _personal_exposure_summary(signal: FrozenStrategySignal) -> dict:
    portfolio = signal.personal_portfolio or {}
    total = float(portfolio.get("total_value_twd") or 0)
    cash = float(portfolio.get("cash_twd") or 0)
    market = float(portfolio.get("market_value_twd") or 0)
    target_value = 0.0
    for row in portfolio.get("positions", []):
        if row.get("ticker") == signal.target_ticker:
            target_value += float(row.get("market_value_twd") or 0)
    target_actual = target_value / total if total > 0 else 0.0
    return {
        "total_value_twd": total,
        "cash_twd": cash,
        "cash_exposure": cash / total if total > 0 else 0.0,
        "market_exposure": market / total if total > 0 else 0.0,
        "target_actual_exposure": target_actual,
        "target_gap_exposure": float(signal.target_exposure) - target_actual,
    }


def _report_mode(signal: FrozenStrategySignal) -> str:
    return "personalized" if signal.personal_portfolio else "general"


def _report_mode_label(signal: FrozenStrategySignal) -> str:
    if signal.personal_portfolio:
        return "個人化版，已結合目前持倉檔"
    return "一般版，未連結個人持倉檔"


def _display_action(action: str) -> str:
    return {"buy": "買進", "sell": "賣出", "hold": "維持"}.get(action, action)


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
    _pdf_write_signal_pdf(
        path,
        signal,
        report_name=REPORT_NAME,
        report_variant_label=REPORT_VARIANT_LABEL,
        report_mode_label=_report_mode_label,
        personal_exposure_summary=_personal_exposure_summary,
        personal_pdf_section=_personal_pdf_section,
    )


def _detail_sections(signal: FrozenStrategySignal) -> list[tuple[str, list[str]]]:
    return _pdf_detail_sections(signal, personal_pdf_section=_personal_pdf_section)


def _paginate_detail_sections(sections: list[tuple[str, list[str]]]) -> list[list[tuple[str, list[str]]]]:
    return _pdf_paginate_detail_sections(sections)


def _detail_section_height(lines: list[str]) -> float:
    return _pdf_detail_section_height(lines)


def _personal_pdf_section(signal: FrozenStrategySignal) -> tuple[str, list[str]]:
    if not signal.personal_portfolio:
        return ("個人持倉參考", ["尚未連結個人持倉檔；本報告為一般版，只顯示模型帳戶狀態。"])
    summary = _personal_exposure_summary(signal)
    lines = [
        f"個人組合估值：約 {summary['total_value_twd']:,.2f} 元；可用現金：約 {summary['cash_twd']:,.2f} 元。",
        f"現金水位約 {summary['cash_exposure']:.2%}；持股水位約 {summary['market_exposure']:.2%}。",
        f"模型目標：{signal.target_label} {signal.target_exposure:.0%}；個人目前目標標的曝險約 {summary['target_actual_exposure']:.2%}。",
        f"與模型目標差距約 {summary['target_gap_exposure']:+.2%}。",
    ]
    for row in (signal.personal_recommendations or [])[:4]:
        lines.append(
            f"個人參考：{_display_action(row['action'])} {row['ticker']} {row['shares']} 股，參考價 {row['reference_price']:.2f}。"
        )
    return ("個人持倉參考", lines)


if __name__ == "__main__":
    main()
