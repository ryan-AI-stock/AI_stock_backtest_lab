from __future__ import annotations

import tempfile
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import patch

import test_paths  # noqa: F401

from backtest_lab.twse_institutional_flow_refresh import (
    _read_json_with_redirects,
    parse_twse_t86_payload,
    refresh_twse_institutional_flows,
    write_twse_institutional_flows,
)


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

    def test_follows_twse_temporary_redirect(self) -> None:
        request = urllib.request.Request("https://example.test/original", headers={"User-Agent": "test"})
        redirect = urllib.error.HTTPError(
            request.full_url,
            307,
            "Temporary Redirect",
            {"Location": "https://example.test/redirected"},
            None,
        )
        response = _FakeResponse(b'{"stat":"OK","fields":[],"data":[]}')

        with patch("urllib.request.urlopen", side_effect=[redirect, response]) as mocked:
            payload = _read_json_with_redirects(request)

        self.assertEqual(payload["stat"], "OK")
        self.assertEqual(mocked.call_count, 2)
        second_request = mocked.call_args_list[1].args[0]
        self.assertEqual(second_request.full_url, "https://example.test/redirected")

    def test_no_rows_writes_header_only_status_and_does_not_fail_by_default(self) -> None:
        empty = parse_twse_t86_payload(
            {"stat": "很抱歉，沒有符合條件的資料!", "fields": [], "data": []},
            signal_date="2026-07-02",
        )
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "institutional_flows.latest.csv"
            status_json = Path(tmp) / "institutional_flows.status.json"
            with patch("backtest_lab.twse_institutional_flow_refresh.fetch_twse_institutional_flows", return_value=empty):
                status = refresh_twse_institutional_flows(
                    signal_date="2026-07-02",
                    output=output,
                    status_json=status_json,
                )

            csv_text = output.read_text(encoding="utf-8-sig")
            status_text = status_json.read_text(encoding="utf-8")

        self.assertIn("foreign_net_buy_shares", csv_text)
        self.assertEqual(status["flow_data_status"], "empty_no_rows")
        self.assertEqual(status["row_count"], 0)
        self.assertIn("optional_report_only_risk_factor_source", status_text)


class _FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.payload


if __name__ == "__main__":
    unittest.main()
