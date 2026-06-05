import unittest

import pandas as pd

from backtest_lab.trade_diagnostics import build_closed_trade_diagnostics, summarize_closed_trades


class TradeDiagnosticsTest(unittest.TestCase):
    def test_fifo_partial_sells(self) -> None:
        trades = pd.DataFrame(
            [
                {
                    "period_id": "p1",
                    "strategy_name": "s",
                    "sequence": 1,
                    "date": "2024-01-01",
                    "ticker": "A",
                    "label": "A",
                    "action": "buy",
                    "shares": 100,
                    "gross_amount_twd": 1000,
                    "costs_twd": 10,
                    "reason": "buy",
                },
                {
                    "period_id": "p1",
                    "strategy_name": "s",
                    "sequence": 2,
                    "date": "2024-01-02",
                    "ticker": "A",
                    "label": "A",
                    "action": "sell",
                    "shares": 40,
                    "gross_amount_twd": 480,
                    "costs_twd": 5,
                    "reason": "partial_sell",
                },
                {
                    "period_id": "p1",
                    "strategy_name": "s",
                    "sequence": 3,
                    "date": "2024-01-03",
                    "ticker": "A",
                    "label": "A",
                    "action": "sell",
                    "shares": 60,
                    "gross_amount_twd": 660,
                    "costs_twd": 6,
                    "reason": "final_sell",
                },
            ]
        )
        diagnostics = build_closed_trade_diagnostics(trades, "s")
        self.assertEqual(len(diagnostics), 2)
        self.assertEqual(diagnostics.iloc[0]["cost_basis_twd"], 404.0)
        self.assertEqual(diagnostics.iloc[0]["pnl_twd"], 71.0)
        summary = summarize_closed_trades(diagnostics)
        self.assertEqual(summary.iloc[0]["sell_count"], 2)
        self.assertEqual(summary.iloc[0]["win_rate_pct"], 100.0)


if __name__ == "__main__":
    unittest.main()
