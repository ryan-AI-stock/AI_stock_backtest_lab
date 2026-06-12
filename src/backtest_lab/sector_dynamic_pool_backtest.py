from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import pandas as pd

from backtest_lab.config import load_config
from backtest_lab.data import download_yfinance_prices, split_adjusted_dividends
from backtest_lab.radar_snapshot_v2_source import RadarSnapshotCandidateSet, select_radar_snapshot_candidates
from backtest_lab.regime_mode_switch import frozen_cycle_proven_top1_v1_variant, simulate_regime_mode_switch
from backtest_lab.simulation import BacktestResult, _common_trade_dates, _date_str, _max_drawdown, simulate_buy_and_hold


REPORT_NAME = "雷達動態題材池個股輪動回測"
REPORT_VERSION = "v0_記憶體題材池"
REPORT_REMOTE_NAME = f"{REPORT_NAME}_最新版_{REPORT_VERSION}.pdf"
DEFAULT_RADAR_ROOT = r"C:\Users\zergv\Documents\Codex\2026-05-23\ai-stock-rotation-radar-https-docs"
DEFAULT_OUTPUT_ROOT = "outputs/sector_dynamic_pool"
DEFAULT_CACHE_DIR = "backtest_cache/sector_dynamic_pool"


@dataclass(frozen=True)
class ThemeMember:
    theme: str
    ticker: str
    symbol: str
    name: str
    role: str
    conviction: str


@dataclass(frozen=True)
class SectorPoolVariant:
    variant_id: str
    label: str
    top_n: int
    max_single_weight: float
    rebalance_frequency: str = "weekly"
    min_avg_turnover_twd: float = 50_000_000
    trend_window: int = 60
    score_windows: tuple[int, int, int] = (20, 60, 126)
    volatility_window: int = 20


@dataclass(frozen=True)
class RadarSnapshotPoolVariant:
    variant_id: str
    label: str
    top_n: int
    max_single_weight: float
    rebalance_frequency: str = "weekly"
    empty_candidate_policy: str = "cash"
    min_theme_score: float | None = None
    min_stock_score: float | None = None
    min_fundamental_score: float | None = None
    max_risk_heat: float | None = None
    min_turnover_share_in_theme: float | None = None
    price_trend_rule: str | None = None
    recent_candidate_lookback_days: int = 0
    min_recent_candidate_days: int = 0


@dataclass
class SectorPoolResult:
    result: BacktestResult
    variant: SectorPoolVariant
    holdings: pd.DataFrame
    score_log: pd.DataFrame


class MultiPositionAccount:
    def __init__(self, cash: float, cost_model) -> None:
        self.cash = float(cash)
        self.cost_model = cost_model
        self.positions: dict[str, int] = {}
        self.trades = []

    def value(self, prices: dict[str, float]) -> float:
        return self.cash + sum(shares * prices[ticker] for ticker, shares in self.positions.items() if shares > 0)

    def shares(self, ticker: str) -> int:
        return int(self.positions.get(ticker, 0))

    def rebalance(
        self,
        *,
        date: pd.Timestamp,
        target_weights: dict[str, float],
        open_prices: dict[str, float],
        close_prices: dict[str, float],
        asset_types: dict[str, str],
        reason: str,
    ) -> None:
        total_value = self.value(close_prices)
        target_values = {ticker: total_value * weight for ticker, weight in target_weights.items() if weight > 0}
        for ticker, shares in list(self.positions.items()):
            if shares <= 0:
                continue
            current_value = shares * open_prices[ticker]
            target_value = target_values.get(ticker, 0.0)
            if current_value <= target_value:
                continue
            sell_shares = int((current_value - target_value) // open_prices[ticker])
            if sell_shares > 0:
                self.sell(date, ticker, sell_shares, open_prices[ticker], asset_types[ticker], reason)
        for ticker, target_value in target_values.items():
            current_value = self.shares(ticker) * open_prices[ticker]
            if target_value <= current_value:
                continue
            buy_cash = min(self.cash, target_value - current_value)
            shares = int(buy_cash // open_prices[ticker])
            while shares > 0:
                gross = shares * open_prices[ticker]
                if gross + self.cost_model.buy_cost(gross) <= self.cash:
                    break
                shares -= 1
            if shares > 0:
                self.buy(date, ticker, shares, open_prices[ticker], reason)

    def buy(self, date: pd.Timestamp, ticker: str, shares: int, price: float, reason: str) -> None:
        gross = shares * price
        costs = self.cost_model.buy_cost(gross)
        self.cash -= gross + costs
        self.positions[ticker] = self.shares(ticker) + shares
        self.trades.append(_trade(date, ticker, "buy", shares, price, gross, costs, self.cash, reason))

    def sell(self, date: pd.Timestamp, ticker: str, shares: int, price: float, asset_type: str, reason: str) -> None:
        shares = min(shares, self.shares(ticker))
        if shares <= 0:
            return
        gross = shares * price
        costs = self.cost_model.sell_cost(gross, asset_type)
        self.cash += gross - costs
        self.positions[ticker] = self.shares(ticker) - shares
        self.trades.append(_trade(date, ticker, "sell", shares, price, gross, costs, self.cash, reason))

    def credit_dividend(self, date: pd.Timestamp, ticker: str, dividend_per_share: float) -> None:
        shares = self.shares(ticker)
        if shares <= 0 or dividend_per_share <= 0:
            return
        amount = shares * dividend_per_share
        self.cash += amount
        self.trades.append(_trade(date, ticker, "dividend", shares, dividend_per_share, amount, 0, self.cash, "cash_dividend"))


def _trade(date, ticker, action, shares, price, gross, costs, cash_after, reason):
    from backtest_lab.portfolio import Trade

    return Trade(
        date=_date_str(date),
        ticker=ticker,
        action=action,
        shares=int(shares),
        price=float(price),
        gross_amount=float(gross),
        costs=int(costs),
        cash_after=float(cash_after),
        reason=reason,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build radar-driven sector pool backtest v0.")
    parser.add_argument("--config", default="configs/ep05_universe.json")
    parser.add_argument("--strategy-config", default="configs/frozen_cycle_proven_top1_v1.json")
    parser.add_argument("--radar-root", default=DEFAULT_RADAR_ROOT)
    parser.add_argument("--theme", default="記憶體")
    parser.add_argument("--start-date", default="2024-01-02")
    parser.add_argument("--end-date", default="2026-05-26")
    parser.add_argument("--warmup-start", default="2022-01-01")
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()

    config = load_config(args.config)
    radar_root = Path(args.radar_root)
    members = load_theme_members(radar_root / "data" / "theme_map.csv", args.theme)
    if not members:
        raise ValueError(f"No theme members found for {args.theme}")
    sector_metrics = _read_optional_csv(radar_root / "data" / "sector_metrics.csv")
    stock_metrics = _read_optional_csv(radar_root / "data" / "stock_metrics.csv")

    benchmark_tickers = ["0050.TW", "00631L.TW"]
    group = config.group_by_id("group_c_0050_00631l_plus_mega_caps")
    frozen_tickers = sorted({asset.ticker for asset in group.assets})
    prices = download_yfinance_prices(
        sorted(set(benchmark_tickers + frozen_tickers)),
        start_date=args.warmup_start,
        end_date=args.end_date,
        cache_dir=args.cache_dir,
    )
    members, member_prices = download_theme_member_prices(
        members,
        start_date=args.warmup_start,
        end_date=args.end_date,
        cache_dir=args.cache_dir,
    )
    prices.update(member_prices)
    member_tickers = [member.ticker for member in members]
    all_tickers = sorted(prices)
    asset_types = {ticker: "stock" for ticker in all_tickers}
    asset_types.update({"0050.TW": "etf", "00631L.TW": "etf"})
    labels = {member.ticker: member.name for member in members}
    labels.update({asset.ticker: asset.label for asset in group.assets})

    dividends = {
        ticker: split_adjusted_dividends(prices[ticker], tuple(config.manual_splits.get(ticker, ())))
        for ticker in all_tickers
    }
    variants = (
        SectorPoolVariant("memory_top3_cap30_weekly", "記憶體題材 Top3 單檔30% 週輪動", top_n=3, max_single_weight=0.30),
        SectorPoolVariant("memory_top4_cap25_weekly", "記憶體題材 Top4 單檔25% 週輪動", top_n=4, max_single_weight=0.25),
        SectorPoolVariant("memory_top5_cap20_weekly", "記憶體題材 Top5 單檔20% 週輪動", top_n=5, max_single_weight=0.20),
    )
    sector_results = [
        simulate_sector_pool(
            variant=variant,
            prices_by_ticker={ticker: prices[ticker] for ticker in member_tickers},
            labels=labels,
            asset_types=asset_types,
            start_date=args.start_date,
            end_date=args.end_date,
            initial_cash=config.initial_cash_twd,
            cost_model=config.cost_model,
            dividend_series_by_ticker={ticker: dividends[ticker] for ticker in member_tickers},
        )
        for variant in variants
    ]
    benchmarks = build_benchmarks(
        config=config,
        prices=prices,
        asset_types=asset_types,
        start_date=args.start_date,
        end_date=args.end_date,
        dividends=dividends,
    )

    output_dir = Path(args.output_root) / "latest"
    output_dir.mkdir(parents=True, exist_ok=True)
    history_dir = Path(args.output_root) / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    write_outputs(
        output_dir=output_dir,
        history_dir=history_dir,
        theme=args.theme,
        members=members,
        sector_metrics=sector_metrics,
        stock_metrics=stock_metrics,
        sector_results=sector_results,
        benchmarks=benchmarks,
        labels=labels,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    print(f"REPORT_DIR={output_dir.resolve()}")
    print(f"LATEST_PDF={(output_dir / (REPORT_NAME + '_最新版_' + REPORT_VERSION + '.pdf')).resolve()}")


def load_theme_members(path: Path, theme: str) -> list[ThemeMember]:
    frame = pd.read_csv(path, dtype={"symbol": str}).fillna("")
    rows = frame[(frame["theme"] == theme) & (frame.get("primary", "yes").astype(str).str.lower() != "no")]
    members = []
    for _, row in rows.iterrows():
        symbol = str(row["symbol"]).strip()
        members.append(
            ThemeMember(
                theme=str(row["theme"]).strip(),
                ticker=f"{symbol}.TW",
                symbol=symbol,
                name=str(row["name"]).strip(),
                role=str(row.get("role", "")).strip(),
                conviction=str(row.get("conviction", "")).strip(),
            )
        )
    return members


def download_theme_member_prices(
    members: list[ThemeMember],
    *,
    start_date: str,
    end_date: str,
    cache_dir: str | Path,
) -> tuple[list[ThemeMember], dict[str, pd.DataFrame]]:
    resolved_members: list[ThemeMember] = []
    prices: dict[str, pd.DataFrame] = {}
    skipped: list[str] = []
    for member in members:
        candidates = _exchange_candidates(member.symbol, cache_dir)
        last_error: Exception | None = None
        for ticker in candidates:
            try:
                downloaded = download_yfinance_prices(
                    [ticker],
                    start_date=start_date,
                    end_date=end_date,
                    cache_dir=cache_dir,
                )
            except ValueError as error:
                last_error = error
                continue
            prices[ticker] = downloaded[ticker]
            resolved_members.append(replace(member, ticker=ticker))
            break
        else:
            skipped.append(f"{member.symbol} {member.name}: {last_error}")
    if skipped:
        message = "; ".join(skipped)
        raise ValueError(f"Unable to resolve theme member prices: {message}")
    return resolved_members, prices


def _exchange_candidates(symbol: str, cache_dir: str | Path) -> list[str]:
    cache_path = Path(cache_dir)
    tw = f"{symbol}.TW"
    two = f"{symbol}.TWO"
    two_cache = cache_path / f"{two.replace('.', '_')}.csv"
    tw_cache = cache_path / f"{tw.replace('.', '_')}.csv"
    if two_cache.exists() and not tw_cache.exists():
        return [two, tw]
    return [tw, two]


def simulate_sector_pool(
    *,
    variant: SectorPoolVariant,
    prices_by_ticker: dict[str, pd.DataFrame],
    labels: dict[str, str],
    asset_types: dict[str, str],
    start_date: str,
    end_date: str,
    initial_cash: float,
    cost_model,
    dividend_series_by_ticker: dict[str, pd.Series] | None = None,
) -> SectorPoolResult:
    trade_dates = _common_trade_dates(prices_by_ticker, start_date, end_date)
    if not trade_dates:
        raise ValueError(f"No common trade dates between {start_date} and {end_date}")
    account = MultiPositionAccount(initial_cash, cost_model)
    equity_rows = []
    holding_rows = []
    score_rows = []

    for index, trade_date in enumerate(trade_dates):
        if dividend_series_by_ticker is not None:
            for ticker in list(account.positions):
                account.credit_dividend(trade_date, ticker, float(dividend_series_by_ticker[ticker].get(trade_date, 0.0)))

        signal_date = _previous_common_signal_date(prices_by_ticker, trade_date)
        should_rebalance = index == 0 or _is_rebalance_date(trade_dates, index, variant.rebalance_frequency)
        scores = score_candidates(prices_by_ticker, signal_date, variant)
        for rank, (ticker, score) in enumerate(scores, start=1):
            score_rows.append(
                {
                    "date": _date_str(trade_date),
                    "signal_date": _date_str(signal_date),
                    "rank": rank,
                    "ticker": ticker,
                    "label": labels.get(ticker, ticker),
                    "score": score,
                }
            )
        if should_rebalance:
            targets = target_weights_from_scores(scores, variant)
            open_prices = {ticker: float(prices_by_ticker[ticker].loc[trade_date, "open"]) for ticker in prices_by_ticker}
            close_prices = {ticker: float(prices_by_ticker[ticker].loc[trade_date, "close"]) for ticker in prices_by_ticker}
            account.rebalance(
                date=trade_date,
                target_weights=targets,
                open_prices=open_prices,
                close_prices=close_prices,
                asset_types=asset_types,
                reason=variant.variant_id,
            )
        close_prices = {ticker: float(prices_by_ticker[ticker].loc[trade_date, "close"]) for ticker in prices_by_ticker}
        total_value = account.value(close_prices)
        equity_rows.append(
            {
                "date": trade_date,
                "total_value": total_value,
                "cash": account.cash,
                "market_exposure": 1 - account.cash / total_value if total_value else 0.0,
                "current_ticker": "|".join(sorted(t for t, shares in account.positions.items() if shares > 0)) or "cash",
            }
        )
        for ticker, shares in sorted(account.positions.items()):
            if shares <= 0:
                continue
            holding_rows.append(
                {
                    "date": _date_str(trade_date),
                    "ticker": ticker,
                    "label": labels.get(ticker, ticker),
                    "shares": shares,
                    "close": close_prices[ticker],
                    "market_value": shares * close_prices[ticker],
                    "weight": shares * close_prices[ticker] / total_value if total_value else 0.0,
                }
            )

    equity_curve = pd.DataFrame(equity_rows).set_index("date")
    result = BacktestResult(
        name=variant.label,
        final_value=float(equity_curve["total_value"].iloc[-1]),
        total_return=float(equity_curve["total_value"].iloc[-1] / initial_cash - 1),
        max_drawdown=_max_drawdown(equity_curve["total_value"]),
        trades=account.trades,
        equity_curve=equity_curve,
    )
    return SectorPoolResult(result=result, variant=variant, holdings=pd.DataFrame(holding_rows), score_log=pd.DataFrame(score_rows))


def score_candidates(
    prices_by_ticker: dict[str, pd.DataFrame],
    signal_date: pd.Timestamp,
    variant: SectorPoolVariant,
) -> list[tuple[str, float]]:
    scored = []
    required = max(max(variant.score_windows), variant.trend_window, variant.volatility_window) + 1
    for ticker, prices in prices_by_ticker.items():
        history = prices.loc[prices.index <= signal_date].dropna(subset=["adj_close"])
        if len(history) < required:
            continue
        close = float(history["adj_close"].iloc[-1])
        ma = float(history["adj_close"].iloc[-variant.trend_window :].mean())
        if close <= ma:
            continue
        avg_turnover = float((history["close"] * history.get("volume", 0)).iloc[-20:].mean())
        if avg_turnover < variant.min_avg_turnover_twd:
            continue
        returns = {
            window: float(history["adj_close"].iloc[-1] / history["adj_close"].iloc[-window] - 1)
            for window in variant.score_windows
        }
        if returns[20] <= 0 or returns[60] <= 0:
            continue
        volatility = float(history["adj_close"].pct_change().dropna().iloc[-variant.volatility_window :].std() * (252**0.5))
        score = (0.45 * returns[20]) + (0.35 * returns[60]) + (0.20 * returns[126]) - (0.08 * volatility)
        scored.append((ticker, score))
    return sorted(scored, key=lambda item: (item[1], item[0]), reverse=True)


def target_weights_from_scores(scores: list[tuple[str, float]], variant: SectorPoolVariant) -> dict[str, float]:
    chosen = scores[: variant.top_n]
    if not chosen:
        return {}
    weight = min(variant.max_single_weight, 1.0 / len(chosen))
    return {ticker: weight for ticker, _ in chosen}


def simulate_radar_snapshot_pool(
    *,
    variant: RadarSnapshotPoolVariant,
    snapshot_history: pd.DataFrame,
    prices_by_ticker: dict[str, pd.DataFrame],
    symbol_to_ticker: dict[str, str],
    labels: dict[str, str],
    asset_types: dict[str, str],
    start_date: str,
    end_date: str,
    initial_cash: float,
    cost_model,
    dividend_series_by_ticker: dict[str, pd.Series] | None = None,
) -> SectorPoolResult:
    trade_dates = _common_trade_dates(prices_by_ticker, start_date, end_date)
    if not trade_dates:
        raise ValueError(f"No common trade dates between {start_date} and {end_date}")
    account = MultiPositionAccount(initial_cash, cost_model)
    equity_rows = []
    holding_rows = []
    score_rows = []

    for index, trade_date in enumerate(trade_dates):
        if dividend_series_by_ticker is not None:
            for ticker in list(account.positions):
                account.credit_dividend(trade_date, ticker, float(dividend_series_by_ticker[ticker].get(trade_date, 0.0)))

        signal_date = _previous_common_signal_date(prices_by_ticker, trade_date)
        candidates = select_radar_snapshot_candidates(snapshot_history, signal_date)
        ranked = _snapshot_ranked_tickers(candidates, symbol_to_ticker, prices_by_ticker)
        ranked = _filter_snapshot_ranked(
            ranked,
            variant=variant,
            prices_by_ticker=prices_by_ticker,
            snapshot_history=snapshot_history,
            signal_date=signal_date,
            symbol_to_ticker=symbol_to_ticker,
        )
        for rank, item in enumerate(ranked, start=1):
            score_rows.append(
                {
                    "date": _date_str(trade_date),
                    "signal_date": _date_str(signal_date),
                    "snapshot_date": _date_str(candidates.snapshot_date),
                    "rank": rank,
                    "ticker": item["ticker"],
                    "symbol": item["symbol"],
                    "label": labels.get(item["ticker"], item["name"]),
                    "theme": item["theme"],
                    "bucket": item["bucket"],
                    "theme_score": item["theme_score"],
                    "stock_score": item["stock_score"],
                    "fundamental_score": item["fundamental_score"],
                    "risk_heat": item["risk_heat"],
                    "stock_turnover_share_in_theme": item["stock_turnover_share_in_theme"],
                }
            )
        should_rebalance = index == 0 or _is_rebalance_date(trade_dates, index, variant.rebalance_frequency)
        if should_rebalance:
            targets = target_weights_from_snapshot_ranked(ranked, variant)
            if not targets and variant.empty_candidate_policy == "hold":
                close_prices = {ticker: float(prices_by_ticker[ticker].loc[trade_date, "close"]) for ticker in prices_by_ticker}
                total_value = account.value(close_prices)
                equity_rows.append(
                    {
                        "date": trade_date,
                        "total_value": total_value,
                        "cash": account.cash,
                        "market_exposure": 1 - account.cash / total_value if total_value else 0.0,
                        "current_ticker": "|".join(sorted(t for t, shares in account.positions.items() if shares > 0)) or "cash",
                    }
                )
                for ticker, shares in sorted(account.positions.items()):
                    if shares <= 0:
                        continue
                    holding_rows.append(
                        {
                            "date": _date_str(trade_date),
                            "ticker": ticker,
                            "label": labels.get(ticker, ticker),
                            "shares": shares,
                            "close": close_prices[ticker],
                            "market_value": shares * close_prices[ticker],
                            "weight": shares * close_prices[ticker] / total_value if total_value else 0.0,
                        }
                    )
                continue
            if not targets and variant.empty_candidate_policy != "cash":
                raise ValueError(f"Unsupported empty_candidate_policy: {variant.empty_candidate_policy}")
            open_prices = {ticker: float(prices_by_ticker[ticker].loc[trade_date, "open"]) for ticker in prices_by_ticker}
            close_prices = {ticker: float(prices_by_ticker[ticker].loc[trade_date, "close"]) for ticker in prices_by_ticker}
            account.rebalance(
                date=trade_date,
                target_weights=targets,
                open_prices=open_prices,
                close_prices=close_prices,
                asset_types=asset_types,
                reason=variant.variant_id,
            )
        close_prices = {ticker: float(prices_by_ticker[ticker].loc[trade_date, "close"]) for ticker in prices_by_ticker}
        total_value = account.value(close_prices)
        equity_rows.append(
            {
                "date": trade_date,
                "total_value": total_value,
                "cash": account.cash,
                "market_exposure": 1 - account.cash / total_value if total_value else 0.0,
                "current_ticker": "|".join(sorted(t for t, shares in account.positions.items() if shares > 0)) or "cash",
            }
        )
        for ticker, shares in sorted(account.positions.items()):
            if shares <= 0:
                continue
            holding_rows.append(
                {
                    "date": _date_str(trade_date),
                    "ticker": ticker,
                    "label": labels.get(ticker, ticker),
                    "shares": shares,
                    "close": close_prices[ticker],
                    "market_value": shares * close_prices[ticker],
                    "weight": shares * close_prices[ticker] / total_value if total_value else 0.0,
                }
            )

    equity_curve = pd.DataFrame(equity_rows).set_index("date")
    result = BacktestResult(
        name=variant.label,
        final_value=float(equity_curve["total_value"].iloc[-1]),
        total_return=float(equity_curve["total_value"].iloc[-1] / initial_cash - 1),
        max_drawdown=_max_drawdown(equity_curve["total_value"]),
        trades=account.trades,
        equity_curve=equity_curve,
    )
    return SectorPoolResult(result=result, variant=variant, holdings=pd.DataFrame(holding_rows), score_log=pd.DataFrame(score_rows))


def target_weights_from_snapshot_ranked(ranked: list[dict[str, object]], variant: RadarSnapshotPoolVariant) -> dict[str, float]:
    chosen = ranked[: variant.top_n]
    if not chosen:
        return {}
    weight = min(variant.max_single_weight, 1.0 / len(chosen))
    return {str(item["ticker"]): weight for item in chosen}


def _snapshot_ranked_tickers(
    candidates: RadarSnapshotCandidateSet,
    symbol_to_ticker: dict[str, str],
    prices_by_ticker: dict[str, pd.DataFrame],
) -> list[dict[str, object]]:
    ranked: list[dict[str, object]] = []
    for _, row in candidates.rows.iterrows():
        symbol = str(row["symbol"]).strip()
        ticker = symbol_to_ticker.get(symbol, f"{symbol}.TW")
        if ticker not in prices_by_ticker:
            continue
        ranked.append(
            {
                "ticker": ticker,
                "symbol": symbol,
                "name": str(row["name"]),
                "theme": str(row["theme"]),
                "bucket": str(row["bucket"]),
                "theme_score": float(row["theme_score"]),
                "stock_score": float(row["stock_score"]),
                "fundamental_score": float(row["fundamental_score"]),
                "risk_heat": float(row["risk_heat"]),
                "stock_turnover_share_in_theme": float(row["stock_turnover_share_in_theme"]),
            }
        )
    return ranked


def _filter_snapshot_ranked(
    ranked: list[dict[str, object]],
    *,
    variant: RadarSnapshotPoolVariant,
    prices_by_ticker: dict[str, pd.DataFrame],
    snapshot_history: pd.DataFrame,
    signal_date: pd.Timestamp,
    symbol_to_ticker: dict[str, str],
) -> list[dict[str, object]]:
    if not ranked:
        return ranked
    recent_counts = _recent_candidate_counts(
        snapshot_history,
        signal_date=signal_date,
        lookback_days=variant.recent_candidate_lookback_days,
        symbol_to_ticker=symbol_to_ticker,
    )
    filtered = []
    for item in ranked:
        ticker = str(item["ticker"])
        if variant.min_theme_score is not None and float(item["theme_score"]) < variant.min_theme_score:
            continue
        if variant.min_stock_score is not None and float(item["stock_score"]) < variant.min_stock_score:
            continue
        if variant.min_fundamental_score is not None and float(item["fundamental_score"]) < variant.min_fundamental_score:
            continue
        if variant.max_risk_heat is not None and float(item["risk_heat"]) > variant.max_risk_heat:
            continue
        if (
            variant.min_turnover_share_in_theme is not None
            and float(item["stock_turnover_share_in_theme"]) < variant.min_turnover_share_in_theme
        ):
            continue
        if variant.min_recent_candidate_days > 0 and recent_counts.get(ticker, 0) < variant.min_recent_candidate_days:
            continue
        if variant.price_trend_rule is not None and not _passes_price_trend(
            prices_by_ticker[ticker],
            signal_date,
            variant.price_trend_rule,
        ):
            continue
        filtered.append(item)
    return filtered


def _recent_candidate_counts(
    snapshot_history: pd.DataFrame,
    *,
    signal_date: pd.Timestamp,
    lookback_days: int,
    symbol_to_ticker: dict[str, str],
) -> dict[str, int]:
    if lookback_days <= 0:
        return {}
    valid_dates = (
        snapshot_history.loc[snapshot_history["date"] <= signal_date, "date"]
        .dropna()
        .drop_duplicates()
        .sort_values()
        .tail(lookback_days)
    )
    counts: dict[str, int] = {}
    for snapshot_date in valid_dates:
        candidates = select_radar_snapshot_candidates(snapshot_history, snapshot_date)
        for symbol in candidates.rows["symbol"].astype(str):
            ticker = symbol_to_ticker.get(symbol)
            if ticker is not None:
                counts[ticker] = counts.get(ticker, 0) + 1
    return counts


def _passes_price_trend(prices: pd.DataFrame, signal_date: pd.Timestamp, rule: str) -> bool:
    history = prices.loc[prices.index <= signal_date].dropna(subset=["adj_close"])
    if rule == "ma20":
        if len(history) < 21:
            return False
        close = float(history["adj_close"].iloc[-1])
        return close > float(history["adj_close"].iloc[-20:].mean()) and close > float(history["adj_close"].iloc[-20])
    if rule == "ma60":
        if len(history) < 61:
            return False
        close = float(history["adj_close"].iloc[-1])
        return close > float(history["adj_close"].iloc[-60:].mean()) and close > float(history["adj_close"].iloc[-60])
    if rule == "ma20_ma60":
        if len(history) < 61:
            return False
        close = float(history["adj_close"].iloc[-1])
        ma20 = float(history["adj_close"].iloc[-20:].mean())
        ma60 = float(history["adj_close"].iloc[-60:].mean())
        return close > ma20 > ma60
    raise ValueError(f"Unsupported price_trend_rule: {rule}")


def build_benchmarks(
    *,
    config,
    prices: dict[str, pd.DataFrame],
    asset_types: dict[str, str],
    start_date: str,
    end_date: str,
    dividends: dict[str, pd.Series],
) -> list[BacktestResult]:
    results = [
        simulate_buy_and_hold(
            "0050 買進持有",
            "0050.TW",
            "etf",
            prices["0050.TW"],
            start_date,
            end_date,
            config.initial_cash_twd,
            config.cost_model,
            dividend_series=dividends["0050.TW"],
        ),
        simulate_buy_and_hold(
            "0050正二 買進持有",
            "00631L.TW",
            "etf",
            prices["00631L.TW"],
            start_date,
            end_date,
            config.initial_cash_twd,
            config.cost_model,
            dividend_series=dividends["00631L.TW"],
        ),
    ]
    group = config.group_by_id("group_c_0050_00631l_plus_mega_caps")
    frozen_prices = {asset.ticker: prices[asset.ticker] for asset in group.assets}
    frozen_dividends = {ticker: dividends[ticker] for ticker in frozen_prices}
    results.append(
        simulate_regime_mode_switch(
            name="最佳版 v20260605",
            prices_by_ticker=frozen_prices,
            asset_types={asset.ticker: asset.asset_type for asset in group.assets},
            market_prices=frozen_prices["0050.TW"],
            start_date=start_date,
            end_date=end_date,
            initial_cash=config.initial_cash_twd,
            cost_model=config.cost_model,
            variant=frozen_cycle_proven_top1_v1_variant(),
            dividend_series_by_ticker=frozen_dividends,
        )
    )
    return results


def write_outputs(
    *,
    output_dir: Path,
    history_dir: Path,
    theme: str,
    members: list[ThemeMember],
    sector_metrics: pd.DataFrame,
    stock_metrics: pd.DataFrame,
    sector_results: list[SectorPoolResult],
    benchmarks: list[BacktestResult],
    labels: dict[str, str],
    start_date: str,
    end_date: str,
) -> None:
    all_results = [sector.result for sector in sector_results] + benchmarks
    summary = pd.DataFrame([_summary_row(result) for result in all_results])
    summary.to_csv(output_dir / "sector_dynamic_pool_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([asdict(member) for member in members]).to_csv(output_dir / "sector_dynamic_pool_members.csv", index=False, encoding="utf-8-sig")
    _trade_rows(all_results, labels).to_csv(output_dir / "sector_dynamic_pool_trades.csv", index=False, encoding="utf-8-sig")
    _equity_rows(all_results).to_csv(output_dir / "sector_dynamic_pool_equity_curve.csv", index=False, encoding="utf-8-sig")
    pd.concat([sector.holdings.assign(strategy=sector.result.name) for sector in sector_results], ignore_index=True).to_csv(
        output_dir / "sector_dynamic_pool_holdings.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.concat([sector.score_log.assign(strategy=sector.result.name) for sector in sector_results], ignore_index=True).to_csv(
        output_dir / "sector_dynamic_pool_score_log.csv",
        index=False,
        encoding="utf-8-sig",
    )
    metadata = _metadata(theme, members, sector_metrics, stock_metrics, start_date, end_date)
    (output_dir / "sector_dynamic_pool_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    report = _markdown_report(summary, metadata)
    latest_md = output_dir / f"{REPORT_NAME}_最新版_{REPORT_VERSION}.md"
    latest_md.write_text(report, encoding="utf-8")
    _write_pdf(output_dir / f"{REPORT_NAME}_最新版_{REPORT_VERSION}.pdf", summary, metadata)
    _append_history(history_dir / "sector_dynamic_pool_run_history.csv", summary, metadata)


def _summary_row(result: BacktestResult) -> dict:
    return {
        "strategy": result.name,
        "final_value_twd": round(result.final_value, 2),
        "total_return_pct": round(result.total_return * 100, 2),
        "max_drawdown_pct": round(result.max_drawdown * 100, 2),
        "trade_count": len([trade for trade in result.trades if trade.action in {"buy", "sell"}]),
        "first_holding": str(result.equity_curve["current_ticker"].iloc[0]),
        "last_holding": str(result.equity_curve["current_ticker"].iloc[-1]),
    }


def _trade_rows(results: list[BacktestResult], labels: dict[str, str]) -> pd.DataFrame:
    rows = []
    for result in results:
        for trade in result.trades:
            rows.append(
                {
                    "strategy": result.name,
                    "date": trade.date,
                    "ticker": trade.ticker,
                    "label": labels.get(trade.ticker, trade.ticker),
                    "action": trade.action,
                    "shares": trade.shares,
                    "price": trade.price,
                    "gross_amount_twd": trade.gross_amount,
                    "costs_twd": trade.costs,
                    "cash_after": trade.cash_after,
                    "reason": trade.reason,
                }
            )
    return pd.DataFrame(rows)


def _equity_rows(results: list[BacktestResult]) -> pd.DataFrame:
    rows = []
    for result in results:
        frame = result.equity_curve.reset_index()
        for _, row in frame.iterrows():
            rows.append(
                {
                    "strategy": result.name,
                    "date": row["date"].strftime("%Y-%m-%d"),
                    "total_value_twd": row["total_value"],
                    "current_ticker": row.get("current_ticker", ""),
                    "market_exposure": row.get("market_exposure", ""),
                }
            )
    return pd.DataFrame(rows)


def _metadata(theme: str, members: list[ThemeMember], sector_metrics: pd.DataFrame, stock_metrics: pd.DataFrame, start_date: str, end_date: str) -> dict:
    current_theme_row = {}
    if not sector_metrics.empty and "name" in sector_metrics.columns:
        match = sector_metrics[sector_metrics["name"] == theme]
        if not match.empty:
            current_theme_row = match.iloc[0].to_dict()
    covered_stock_metrics = []
    if not stock_metrics.empty and "symbol" in stock_metrics.columns:
        symbols = {member.symbol for member in members}
        covered_stock_metrics = stock_metrics[stock_metrics["symbol"].astype(str).isin(symbols)].to_dict("records")
    return {
        "report_name": REPORT_NAME,
        "report_version": REPORT_VERSION,
        "theme": theme,
        "period": {"start_date": start_date, "end_date": end_date},
        "member_count": len(members),
        "current_theme_metrics": current_theme_row,
        "current_stock_metrics_rows": covered_stock_metrics,
        "limitations": [
            "v0 使用雷達 theme_map 作為靜態題材分類字典，歷史回測訊號只使用每個交易日前一日以前的價格資料。",
            "目前雷達 repo 沒有完整逐日歷史雷達快照，因此 sector_metrics 與 stock_metrics 只作為目前雷達背景說明，不回推成歷史每日訊號。",
            "本報告是 AI 輔助市場觀察、回測與策略驗證，不是投資建議或交易指令。",
        ],
    }


def _markdown_report(summary: pd.DataFrame, metadata: dict) -> str:
    lines = [
        f"# {REPORT_NAME}（最新版 {REPORT_VERSION}）",
        "",
        f"- 題材池：{metadata['theme']}，成員數：{metadata['member_count']}",
        f"- 回測期間：{metadata['period']['start_date']} 到 {metadata['period']['end_date']}",
        "- 資料口徑：雷達題材分類字典 + 歷史價格回測；不宣稱為歷史實盤雷達紀錄。",
        "",
        "## 結果摘要",
        "",
        _markdown_table(summary),
        "",
        "## 使用限制",
        "",
    ]
    lines.extend(f"- {item}" for item in metadata["limitations"])
    return "\n".join(lines) + "\n"


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in frame.iterrows():
        rows.append("| " + " | ".join(_markdown_cell(row[column]) for column in columns) + " |")
    return "\n".join(rows)


def _markdown_cell(value: object) -> str:
    return str(value).replace("|", "<br>")


def _write_pdf(path: Path, summary: pd.DataFrame, metadata: dict) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Noto Sans CJK TC", "Noto Sans CJK JP", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    with PdfPages(path) as pdf:
        fig = plt.figure(figsize=(11.69, 8.27), dpi=180)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")
        ax.add_patch(plt.Rectangle((0.04, 0.82), 0.92, 0.12, color="#16212b"))
        ax.text(0.07, 0.89, REPORT_NAME, fontsize=22, fontweight="bold", color="white", va="center")
        ax.text(0.07, 0.845, f"最新版 {REPORT_VERSION} · {metadata['period']['start_date']} - {metadata['period']['end_date']}", fontsize=10, color="#cbd5df")
        ax.text(0.07, 0.76, f"題材池：{metadata['theme']} · 成員數：{metadata['member_count']}", fontsize=14, fontweight="bold", color="#15202b")
        y = 0.70
        headers = ["策略", "期末淨值", "報酬率", "最大回撤", "交易次數"]
        widths = [0.36, 0.17, 0.13, 0.13, 0.12]
        x0 = 0.07
        ax.add_patch(plt.Rectangle((x0, y), 0.86, 0.04, color="#e8eff5"))
        x = x0
        for header, width in zip(headers, widths):
            ax.text(x + 0.005, y + 0.025, header, fontsize=10, fontweight="bold", va="center")
            x += width
        y -= 0.045
        for _, row in summary.iterrows():
            ax.add_patch(plt.Rectangle((x0, y), 0.86, 0.038, color="#fff8e8" if "記憶體" in row["strategy"] else "#ffffff", ec="#dde5ed", lw=0.5))
            values = [
                row["strategy"],
                f"{row['final_value_twd']:,.0f}",
                f"{row['total_return_pct']:.2f}%",
                f"{row['max_drawdown_pct']:.2f}%",
                f"{int(row['trade_count'])}",
            ]
            x = x0
            for value, width in zip(values, widths):
                ax.text(x + 0.005, y + 0.022, str(value), fontsize=9, va="center", color="#263646")
                x += width
            y -= 0.04
        y -= 0.03
        ax.text(0.07, y, "限制：v0 使用靜態題材分類字典；歷史訊號只使用交易日前已知價格資料。不是投資建議。", fontsize=10, color="#6b7280")
        pdf.savefig(fig)
        plt.close(fig)


def _append_history(path: Path, summary: pd.DataFrame, metadata: dict) -> None:
    rows = []
    run_ts = pd.Timestamp.now(tz="Asia/Taipei").strftime("%Y-%m-%d %H:%M:%S%z")
    for _, row in summary.iterrows():
        payload = row.to_dict()
        payload.update(
            {
                "run_timestamp_taipei": run_ts,
                "report_version": metadata["report_version"],
                "theme": metadata["theme"],
                "period_start": metadata["period"]["start_date"],
                "period_end": metadata["period"]["end_date"],
            }
        )
        rows.append(payload)
    frame = pd.DataFrame(rows)
    if path.exists():
        previous = pd.read_csv(path)
        frame = pd.concat([previous, frame], ignore_index=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _read_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _previous_common_signal_date(prices_by_ticker: dict[str, pd.DataFrame], trade_date: pd.Timestamp) -> pd.Timestamp:
    common_dates: set[pd.Timestamp] | None = None
    for prices in prices_by_ticker.values():
        dates = set(prices.index[prices.index < trade_date])
        common_dates = dates if common_dates is None else common_dates & dates
    if not common_dates:
        raise ValueError(f"No common signal date before {trade_date.date()}")
    return max(common_dates)


def _is_rebalance_date(trade_dates: list[pd.Timestamp], index: int, frequency: str) -> bool:
    if index == 0 or frequency == "daily":
        return True
    if frequency == "weekly":
        current = trade_dates[index].isocalendar()
        previous = trade_dates[index - 1].isocalendar()
        return (current.year, current.week) != (previous.year, previous.week)
    if frequency == "biweekly":
        current = trade_dates[index].isocalendar()
        previous = trade_dates[index - 1].isocalendar()
        current_key = current.year * 100 + (current.week // 2)
        previous_key = previous.year * 100 + (previous.week // 2)
        return current_key != previous_key
    if frequency == "monthly":
        return trade_dates[index].month != trade_dates[index - 1].month
    raise ValueError(f"Unsupported rebalance frequency: {frequency}")


if __name__ == "__main__":
    main()
