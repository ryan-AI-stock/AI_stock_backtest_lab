from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import test_paths  # noqa: F401

from backtest_lab.twse_institutional_flow_refresh import parse_twse_t86_payload, write_twse_institutional_flows


class TwseInstitutionalFlowRefreshTest(unittest.TestCase):
    def test_parses_t86_payload_to_risk_factor_contract(self) -> None:
        payload = {
            "stat": "OK",
            "fields": [
                "證券代號",
                "證券名稱",
                "外陸資買賣超股數(不含外資自營商)",
                "投信買賣超股數",
                "自營商買賣超股數",
            ],
            "data": [["2330", "台積電", "1,000", "2,000", "-500"]],
        }

        frame = parse_twse_t86_payload(payload, signal_date="2026-06-26")

        self.assertEqual(len(frame), 1)
        row = frame.iloc[0].to_dict()
        self.assertEqual(row["date"], "2026-06-26")
        self.assertEqual(row["symbol"], "2330")
        self.assertEqual(row["ticker"], "2330.TW")
        self.assertEqual(row["name"], "台積電")
        self.assertEqual(row["foreign_net_buy_shares"], 1000)
        self.assertEqual(row["investment_trust_net_buy_shares"], 2000)
        self.assertEqual(row["dealer_net_buy_shares"], -500)
        self.assertEqual(row["total_institutional_net_buy_shares"], 2500)

    def test_writes_csv_for_stock_pool_observation_runner(self) -> None:
        frame = parse_twse_t86_payload(
            {
                "stat": "OK",
                "fields": ["證券代號", "證券名稱", "外資買賣超股數", "投信買賣超股數", "自營商買賣超股數"],
                "data": [["2303", "聯電", "-1,000", "0", "100"]],
            },
            signal_date="2026-06-26",
        )
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "institutional_flows.latest.csv"
            write_twse_institutional_flows(frame, output)

            text = output.read_text(encoding="utf-8-sig")

        self.assertIn("2303.TW", text)
        self.assertIn("foreign_net_buy_shares", text)


if __name__ == "__main__":
    unittest.main()
