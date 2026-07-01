from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from backtest_lab.costs import COST_MODEL_VERSION, cost_model_metadata
from backtest_lab.costs import TaiwanCostModel
from backtest_lab.data import load_price_csv
from backtest_lab.portfolio import Portfolio


DEFAULT_OUTPUT_DIR = "outputs/stock_pool_formal_daily_replay_20260623"


def run_stock_pool_formal_daily_replay(
    *,
    replay_panel_path: str | Path,
    price_cache_dir: str | Path,
    output_dir: str | Path,
    initial_cash: float = 1_000_000,
) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    replay_panel = pd.read_csv(replay_panel_path).fillna("")
    price_cache = Path(price_cache_dir)
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

    log("build_decisions", "started", "")
    decision_panel = _build_decision_panel(replay_panel)
    tickers = sorted({ticker for ticker in decision_panel["winner_ticker"].astype(str).tolist() if ticker})
    prices = _load_prices(tickers, price_cache)
    daily = _simulate_decision_panel(decision_panel, prices, initial_cash=initial_cash)
    summary = _summary(daily)
    decision_panel.to_csv(output / "formal_three_pool_decision_panel.csv", index=False, encoding="utf-8-sig")
    daily.to_csv(output / "baseline_three_pool_formal_daily_equity.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([summary]).to_csv(output / "formal_three_pool_summary.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "model": "stock_pool_formal_daily_replay_v1",
        "status": "completed",
        "active_in_trade_decision": False,
        "formal_model_changed": False,
        "input_replay_panel": str(replay_panel_path),
        "price_cache_dir": str(price_cache),
        "initial_cash": initial_cash,
        "rows": {"decision_panel": len(decision_panel), "daily_equity": len(daily)},
        "policy": "trade only when 2/3 consensus winner exists; otherwise keep existing position or cash",
        "cost_model_version": COST_MODEL_VERSION,
        "cost_model": cost_model_metadata(),
        "cost_fields": {
            "daily_equity": ["transaction_cost"],
            "trade_ledger": "not_written_by_this_legacy_runner",
            "boundary": "daily transaction_cost is netted in portfolio cash/equity; no buy_fee/sell_fee/tax split in legacy daily output",
        },
        "outputs": {
            "decision_panel": "formal_three_pool_decision_panel.csv",
            "daily_equity": "baseline_three_pool_formal_daily_equity.csv",
            "summary": "formal_three_pool_summary.csv",
        },
    }
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "completed.txt").write_text("completed\n", encoding="utf-8")
    log("completed", "completed", str(output.resolve()))
    (output / "current_step.txt").write_text("completed\n", encoding="utf-8")
    return output


def _build_decision_panel(replay_panel: pd.DataFrame) -> pd.DataFrame:
    required = {"period", "signal_date", "pool_id", "top_ticker", "eligible_for_pool_selection"}
    missing = required - set(replay_panel.columns)
    if missing:
        raise ValueError("missing replay panel columns: " + ",".join(sorted(missing)))
    rows: list[dict[str, Any]] = []
    frame = replay_panel.copy()
    frame["signal_date"] = frame["signal_date"].astype(str)
    frame = frame[frame["signal_date"].str.strip().ne("")]
    frame = frame[pd.to_datetime(frame["signal_date"], errors="coerce").notna()]
    for (period, signal_date), group in frame.groupby(["period", "signal_date"], dropna=False):
        eligible = group[group["eligible_for_pool_selection"].map(_truthy)].copy()
        votes = [str(value).strip() for value in eligible["top_ticker"].tolist() if str(value).strip()]
        counts = Counter(votes)
        winner = ""
        state = "no_vote"
        if len(votes) < 2:
            state = "insufficient_votes" if votes else "no_vote"
        else:
            ticker, count = counts.most_common(1)[0]
            if count >= 2:
                winner = ticker
                state = "consensus"
            else:
                state = "divergent"
        rows.append(
            {
                "period": period,
                "date": signal_date,
                "pool1_vote": _vote_for_pool(group, "ai_theme_large_cap"),
                "pool2_vote": _vote_for_pool(group, "tw50_dynamic_constituents"),
                "pool3_vote": _vote_for_pool(group, "large_core_bluechip"),
                "consensus_state": state,
                "winner_ticker": winner,
                "eligible_vote_count": len(votes),
            }
        )
    panel = pd.DataFrame(rows)
    panel["date"] = pd.to_datetime(panel["date"])
    return panel.sort_values("date").reset_index(drop=True)


def _simulate_decision_panel(
    decision_panel: pd.DataFrame,
    prices: dict[str, pd.Series],
    *,
    initial_cash: float,
) -> pd.DataFrame:
    portfolio = Portfolio(initial_cash, TaiwanCostModel())
    rows: list[dict[str, Any]] = []
    running_max = initial_cash
    previous_equity = initial_cash
    for item in decision_panel.to_dict(orient="records"):
        date = pd.Timestamp(item["date"]).strftime("%Y-%m-%d")
        winner = str(item.get("winner_ticker") or "").strip()
        current = portfolio.current_ticker()
        transaction_cost = 0
        turnover = 0.0
        action = "hold"
        if winner and winner in prices:
            price = _price_on_or_before(prices[winner], date)
            if price is not None and current != winner:
                if current:
                    current_price = _price_on_or_before(prices.get(current, pd.Series(dtype=float)), date)
                    if current_price is not None:
                        trade = portfolio.sell_all(date, current, _asset_type(current), current_price, "formal_three_pool_switch")
                        if trade:
                            transaction_cost += trade.costs
                            turnover += trade.gross_amount
                trade = portfolio.buy_max(date, winner, _asset_type(winner), price, "formal_three_pool_consensus")
                if trade:
                    transaction_cost += trade.costs
                    turnover += trade.gross_amount
                    action = "switch" if current else "buy"
        close_prices = {
            ticker: price
            for ticker, series in prices.items()
            if (price := _price_on_or_before(series, date)) is not None
        }
        equity = portfolio.market_value(close_prices) if close_prices else previous_equity
        previous_equity = equity
        running_max = max(running_max, equity)
        rows.append(
            {
                "date": date,
                "period": item.get("period", ""),
                "pool1_vote": item.get("pool1_vote", ""),
                "pool2_vote": item.get("pool2_vote", ""),
                "pool3_vote": item.get("pool3_vote", ""),
                "consensus_state": item.get("consensus_state", ""),
                "winner_ticker": winner,
                "position_ticker": portfolio.current_ticker() or "cash",
                "cash": round(portfolio.cash, 2),
                "equity": round(equity, 2),
                "drawdown": round(equity / running_max - 1, 8) if running_max else 0.0,
                "turnover": round(turnover, 2),
                "transaction_cost": transaction_cost,
                "action": action,
                "data_status": "formal_daily_replay",
            }
        )
    return pd.DataFrame(rows)


def _load_prices(tickers: list[str], cache_dir: Path) -> dict[str, pd.Series]:
    prices: dict[str, pd.Series] = {}
    for ticker in tickers:
        path = cache_dir / f"{ticker.replace('.', '_')}.csv"
        if not path.exists():
            continue
        frame = load_price_csv(path)
        close = pd.to_numeric(frame["adj_close"], errors="coerce").dropna()
        prices[ticker] = close
    return prices


def _price_on_or_before(series: pd.Series, date: str) -> float | None:
    if series.empty:
        return None
    ts = pd.Timestamp(date)
    clipped = series.loc[series.index <= ts]
    if clipped.empty:
        return None
    return float(clipped.iloc[-1])


def _vote_for_pool(group: pd.DataFrame, pool_id_fragment: str) -> str:
    subset = group[group["pool_id"].astype(str).str.contains(pool_id_fragment, na=False)]
    subset = subset[subset["eligible_for_pool_selection"].map(_truthy)]
    if subset.empty:
        return ""
    return str(subset.iloc[0].get("top_ticker") or "").strip()


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _asset_type(ticker: str) -> str:
    symbol = ticker.split(".")[0]
    return "etf" if symbol in {"0050", "00631L"} else "stock"


def _summary(daily: pd.DataFrame) -> dict[str, Any]:
    if daily.empty:
        return {"status": "empty"}
    start = float(daily["equity"].iloc[0])
    end = float(daily["equity"].iloc[-1])
    return {
        "status": "completed",
        "start_date": daily["date"].iloc[0],
        "end_date": daily["date"].iloc[-1],
        "start_equity": start,
        "final_equity": end,
        "total_return_pct": round((end / start - 1) * 100, 4) if start else 0.0,
        "max_drawdown_pct": round(float(pd.to_numeric(daily["drawdown"], errors="coerce").min()) * 100, 4),
        "trade_days": int((daily["action"].astype(str) != "hold").sum()),
        "total_transaction_cost": float(pd.to_numeric(daily["transaction_cost"], errors="coerce").sum()),
        "total_turnover": float(pd.to_numeric(daily["turnover"], errors="coerce").sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build formal daily replay/equity for three-pool consensus decisions.")
    parser.add_argument("--replay-panel", required=True)
    parser.add_argument("--price-cache-dir", required=True)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--initial-cash", type=float, default=1_000_000)
    args = parser.parse_args()
    output = run_stock_pool_formal_daily_replay(
        replay_panel_path=args.replay_panel,
        price_cache_dir=args.price_cache_dir,
        output_dir=args.output_dir,
        initial_cash=args.initial_cash,
    )
    print(f"OUTPUT_DIR={output.resolve()}")


if __name__ == "__main__":
    main()
