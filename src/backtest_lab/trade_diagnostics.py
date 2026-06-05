from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class _Lot:
    date: str
    ticker: str
    label: str
    shares: int
    unit_cost: float


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FIFO closed-trade diagnostics from a trade log.")
    parser.add_argument("--trade-log", required=True)
    parser.add_argument("--strategy-name", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trades = pd.read_csv(args.trade_log)
    diagnostics = build_closed_trade_diagnostics(trades, args.strategy_name)
    summary = summarize_closed_trades(diagnostics)
    diagnostics.to_csv(output_dir / "closed_trade_diagnostics.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output_dir / "closed_trade_summary.csv", index=False, encoding="utf-8-sig")
    _write_report(output_dir / "closed_trade_diagnostics.md", args.strategy_name, diagnostics, summary)
    print(f"OUTPUT_DIR={output_dir.resolve()}")


def build_closed_trade_diagnostics(trades: pd.DataFrame, strategy_name: str) -> pd.DataFrame:
    rows = trades.loc[trades["strategy_name"] == strategy_name].copy()
    if rows.empty:
        raise ValueError(f"No trades for strategy_name={strategy_name}")
    output: list[dict] = []
    for period_id, period_rows in rows.groupby("period_id", sort=False):
        lots: list[_Lot] = []
        for trade in period_rows.sort_values("sequence").itertuples(index=False):
            action = str(trade.action)
            shares = int(trade.shares)
            gross = float(trade.gross_amount_twd)
            costs = float(trade.costs_twd)
            if action == "buy":
                lots.append(
                    _Lot(
                        date=str(trade.date),
                        ticker=str(trade.ticker),
                        label=str(trade.label),
                        shares=shares,
                        unit_cost=(gross + costs) / shares,
                    )
                )
                continue
            if action != "sell":
                continue
            remaining = shares
            cost_basis = 0.0
            buy_dates: list[str] = []
            while remaining > 0 and lots:
                lot = lots[0]
                take = min(remaining, lot.shares)
                cost_basis += take * lot.unit_cost
                buy_dates.append(lot.date)
                lot.shares -= take
                remaining -= take
                if lot.shares == 0:
                    lots.pop(0)
            proceeds = gross - costs
            pnl = proceeds - cost_basis
            output.append(
                {
                    "period_id": period_id,
                    "strategy_name": strategy_name,
                    "sell_date": trade.date,
                    "ticker": trade.ticker,
                    "label": trade.label,
                    "shares": shares,
                    "proceeds_twd": round(proceeds, 2),
                    "cost_basis_twd": round(cost_basis, 2),
                    "pnl_twd": round(pnl, 2),
                    "pnl_pct": round(pnl / cost_basis * 100, 2) if cost_basis else 0.0,
                    "reason": trade.reason,
                    "buy_dates": ",".join(sorted(set(buy_dates))),
                    "unmatched_sell_shares": remaining,
                }
            )
    return pd.DataFrame(output)


def summarize_closed_trades(diagnostics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for period_id, period_rows in diagnostics.groupby("period_id", sort=False):
        pnl = period_rows["pnl_twd"]
        rows.append(
            {
                "period_id": period_id,
                "sell_count": len(period_rows),
                "win_rate_pct": round(float((pnl > 0).mean() * 100), 2) if len(period_rows) else 0.0,
                "realized_pnl_twd": round(float(pnl.sum()), 2),
                "average_pnl_pct": round(float(period_rows["pnl_pct"].mean()), 2) if len(period_rows) else 0.0,
                "worst_pnl_twd": round(float(pnl.min()), 2) if len(period_rows) else 0.0,
                "best_pnl_twd": round(float(pnl.max()), 2) if len(period_rows) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _write_report(path: Path, strategy_name: str, diagnostics: pd.DataFrame, summary: pd.DataFrame) -> None:
    worst = diagnostics.sort_values("pnl_twd").head(8)
    lines = [
        "# Closed Trade Diagnostics",
        "",
        f"Strategy: `{strategy_name}`",
        "",
        "This is an AI-assisted backtest diagnostic, not investment advice.",
        "",
        "## Summary",
        "",
        _markdown_table(summary),
        "",
        "## Worst Sells",
        "",
        _markdown_table(worst),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _markdown_table(frame: pd.DataFrame) -> str:
    headers = list(frame.columns)
    rows = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in frame.iterrows():
        rows.append("| " + " | ".join(str(row[col]) for col in headers) + " |")
    return "\n".join(rows)


if __name__ == "__main__":
    main()
