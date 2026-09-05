"""Focused regression tests for C6 frozen-action accounting event semantics."""
from __future__ import annotations

import csv
import importlib.util
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_c6_64_start_monthly_withdrawal_rechain.py"
SPEC = importlib.util.spec_from_file_location("c6_rechain", SCRIPT)
assert SPEC and SPEC.loader
RUNNER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RUNNER
SPEC.loader.exec_module(RUNNER)


class C6WithdrawalEventTests(unittest.TestCase):
    def test_missing_mark_does_not_silently_carry(self):
        position = RUNNER._Position(slot=1, ticker="9999", units=10)
        self.assertIsNone(RUNNER._mark(position, "2026-09-04", {}, {"9999": 100}))

    def test_valuation_carry_requires_exact_official_no_trade_key(self):
        position = RUNNER._Position(slot=1, ticker="9999", units=10)
        self.assertEqual(RUNNER._mark(position, "2026-09-04", {}, {"9999": 100}, {("9999", "2026-09-04")}), 1000)
        self.assertIsNone(RUNNER._mark(position, "2026-09-04", {}, {"9999": 100}, {("9999", "2026-09-03")}))

    def test_ex_date_snapshots_units_before_record_date_and_payment(self) -> None:
        position = RUNNER._Position(slot=1, ticker="9999", units=10, entry_date="2023-01-01", episode_key="1|9999|2023-01-01")
        event = {"ticker": "9999", "event_type": "cash_dividend", "ex_date": "2023-01-02", "record_date": "2023-01-05", "payment_date": "2023-01-07", "cash_dividend_per_share": "2"}
        cash, entitlement, rows = [0.0, 0.0, 0.0], {}, []
        cash = RUNNER._apply_event("R", date(2023, 1, 2), [position], cash, [event], entitlement, rows)
        position.units = 0  # Selling after ex-date cannot remove the entitlement.
        cash = RUNNER._apply_event("R", date(2023, 1, 7), [position], cash, [event], entitlement, rows)
        self.assertEqual(cash[0], 20.0)
        self.assertEqual(rows[0]["entitlement_units"], 10)

    def test_terminal_partial_month_is_not_a_withdrawal_month(self) -> None:
        market = [date(2026, 7, 31), date(2026, 8, 12)]
        end = date(2026, 8, 12)
        withdrawal_days = {item for item in RUNNER._month_ends(market) if (item.year, item.month) < (end.year, end.month)}
        self.assertEqual(withdrawal_days, {date(2026, 7, 31)})

    def test_conditional_payment_term_is_not_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "terms.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["ticker", "status"])
                writer.writeheader()
                writer.writerows([{"ticker": "1111", "status": "accepted_complete"}, {"ticker": "2222", "status": "accepted_conditional_payment_terms"}])
            accepted = RUNNER._load_events([path])
        self.assertEqual([row["ticker"] for row in accepted], ["1111"])

    def test_conditional_payment_override_accepts_only_listed_alternative(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "terms.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["event_id", "ticker", "status", "payment_date", "payment_date_alternatives"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "event_id": "2421_2023-07-05_cash_dividend",
                        "ticker": "2421",
                        "status": "accepted_conditional_payment_terms",
                        "payment_date": "",
                        "payment_date_alternatives": "2023-07-27|2023-07-28",
                    }
                )
            accepted = RUNNER._load_events(
                [path],
                {"2421_2023-07-05_cash_dividend": "2023-07-28"},
            )
            self.assertEqual(len(accepted), 1)
            self.assertEqual(accepted[0]["payment_date"], "2023-07-28")
            self.assertEqual(accepted[0]["status"], "accepted_complete_sensitivity_override")
            with self.assertRaises(ValueError):
                RUNNER._load_events(
                    [path],
                    {"2421_2023-07-05_cash_dividend": "2023-07-29"},
                )


if __name__ == "__main__":
    unittest.main()
