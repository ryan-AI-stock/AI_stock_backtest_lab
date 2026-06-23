from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

import test_paths  # noqa: F401

from backtest_lab.tw50_technical_notice_events import (
    build_tw50_pit_intervals_from_events,
    build_tw50_pit_snapshots_from_current_and_events,
    _expand_input_paths,
    parse_tw50_technical_notice_text,
    run_tw50_technical_notice_ingestion,
)


class Tw50TechnicalNoticeEventsTest(unittest.TestCase):
    def test_parses_review_notice_into_add_delete_events(self) -> None:
        text = """
        「臺灣50指數」成分股審核結果
        2023年9月1日
        成分股納入和刪除之變動將自2023年9月15日交易結束後生效
        (亦即自2023年9月18日起生效)。
        成分股納入(4)：
        光寶科 2301
        智邦 2345
        緯創 3231
        緯穎 6669
        成分股刪除(4)：
        遠東新 1402
        華新 1605
        陽明 2609
        萬海 2615
        """

        rows = parse_tw50_technical_notice_text(text, source_url="https://example.test/notice.pdf")

        self.assertEqual(len(rows), 8)
        self.assertEqual({row["event_type"] for row in rows}, {"add", "delete"})
        self.assertEqual({row["effective_date"] for row in rows}, {"2023-09-18"})
        self.assertIn("3231.TW", {row["ticker"] for row in rows})
        self.assertTrue(all(row["accepted"] for row in rows))
        self.assertTrue(all(row["exact_or_proxy"] == "exact_candidate" for row in rows))

    def test_postponement_notice_is_blocked_not_accepted(self) -> None:
        text = """
        技術通知
        臺灣證券交易所與富時國際有限公司合編之臺灣指數系列成分股變動：禾伸堂延期納入指數成分股
        2026年6月18日
        指數 變動 生效日
        臺灣50指數 禾伸堂納入指數成分股之異動將延期。 2026年6月22日
        """

        rows = parse_tw50_technical_notice_text(text)

        self.assertEqual(len(rows), 1)
        self.assertFalse(rows[0]["accepted"])
        self.assertEqual(rows[0]["event_type"], "blocked_notice")
        self.assertEqual(rows[0]["blocked_reason"], "postponed_or_non_constituent_change_notice")

    def test_ingestion_outputs_events_and_keeps_formal_not_ready(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            notice = root / "notice.txt"
            notice.write_text(
                """
                「臺灣50指數」成分股審核結果
                2024年6月7日
                變動將自2024年6月21日交易結束後生效(亦即自2024年6月24日起生效)。
                成分股納入(1)：奇鋐 3017
                成分股刪除(1)：彰銀 2801
                """,
                encoding="utf-8",
            )

            result = run_tw50_technical_notice_ingestion(input_paths=[notice], output_dir=root / "out")

            self.assertEqual(result.accepted_event_rows, 2)
            self.assertEqual(result.readiness_status, "events_ready_pending_baseline_snapshot")
            metadata = (root / "out" / "metadata.json").read_text(encoding="utf-8")
            self.assertIn('"formal_ready": false', metadata)
            self.assertTrue((root / "out" / "tw50_technical_notice_events.csv").exists())

    def test_expands_glob_input_paths_for_windows_cli(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = root / "a.pdf"
            second = root / "b.pdf"
            first.write_bytes(b"%PDF-1")
            second.write_bytes(b"%PDF-2")

            paths = _expand_input_paths([str(root / "*.pdf")])

        self.assertEqual([path.name for path in paths], ["a.pdf", "b.pdf"])

    def test_interval_builder_requires_baseline_before_first_event(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            baseline = root / "baseline.csv"
            events = root / "events.csv"
            pd.DataFrame(
                [{"effective_date": "2025-06-23", "ticker": "2330.TW", "name": "台積電"}]
            ).to_csv(baseline, index=False)
            pd.DataFrame(
                [
                    {
                        "effective_date": "2024-06-24",
                        "event_type": "add",
                        "ticker": "3017.TW",
                        "name": "奇鋐",
                        "index_name": "臺灣50指數",
                        "source_date": "2024-06-07",
                        "source_title": "臺灣50指數成分股審核結果",
                        "source_url": "",
                        "source_type": "official_technical_notice",
                        "exact_or_proxy": "exact_candidate",
                        "accepted": True,
                        "blocked_reason": "",
                    }
                ]
            ).to_csv(events, index=False)

            with self.assertRaisesRegex(ValueError, "baseline snapshot starts after the first accepted event"):
                build_tw50_pit_intervals_from_events(
                    baseline_snapshot_path=baseline,
                    event_rows_path=events,
                    output_path=root / "tw50.csv",
                    source_updated_at="2026-06-23",
                )

    def test_interval_builder_applies_forward_events(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            baseline = root / "baseline.csv"
            events = root / "events.csv"
            pd.DataFrame(
                [
                    {"effective_date": "2024-01-01", "ticker": "2801.TW", "name": "彰銀"},
                    {"effective_date": "2024-01-01", "ticker": "2330.TW", "name": "台積電"},
                ]
            ).to_csv(baseline, index=False)
            pd.DataFrame(
                [
                    {
                        "effective_date": "2024-06-24",
                        "event_type": "delete",
                        "ticker": "2801.TW",
                        "name": "彰銀",
                        "index_name": "臺灣50指數",
                        "source_date": "2024-06-07",
                        "source_title": "臺灣50指數成分股審核結果",
                        "source_url": "",
                        "source_type": "official_technical_notice",
                        "exact_or_proxy": "exact_candidate",
                        "accepted": True,
                        "blocked_reason": "",
                    },
                    {
                        "effective_date": "2024-06-24",
                        "event_type": "add",
                        "ticker": "3017.TW",
                        "name": "奇鋐",
                        "index_name": "臺灣50指數",
                        "source_date": "2024-06-07",
                        "source_title": "臺灣50指數成分股審核結果",
                        "source_url": "",
                        "source_type": "official_technical_notice",
                        "exact_or_proxy": "exact_candidate",
                        "accepted": True,
                        "blocked_reason": "",
                    },
                ]
            ).to_csv(events, index=False)

            result = build_tw50_pit_intervals_from_events(
                baseline_snapshot_path=baseline,
                event_rows_path=events,
                output_path=root / "tw50.csv",
                source_updated_at="2026-06-23",
            )
            frame = pd.read_csv(root / "tw50.csv")

        self.assertTrue(result["formal_ready"])
        self.assertEqual(frame.loc[frame["ticker"] == "2801.TW", "end_date"].iloc[0], "2024-06-23")
        self.assertIn("3017.TW", set(frame["ticker"]))

    def test_reverse_snapshot_builder_reconstructs_prior_baskets(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            current = root / "current.csv"
            events = root / "events.csv"
            pd.DataFrame(
                [
                    {"ticker": "2330.TW", "name": "台積電"},
                    {"ticker": "3017.TW", "name": "奇鋐"},
                ]
            ).to_csv(current, index=False)
            pd.DataFrame(
                [
                    {
                        "effective_date": "2024-06-24",
                        "event_type": "delete",
                        "ticker": "2801.TW",
                        "name": "彰銀",
                        "index_name": "臺灣50指數",
                        "source_date": "2024-06-07",
                        "source_title": "臺灣50指數成分股審核結果",
                        "source_url": "",
                        "source_type": "official_technical_notice",
                        "exact_or_proxy": "exact_candidate",
                        "accepted": True,
                        "blocked_reason": "",
                    },
                    {
                        "effective_date": "2024-06-24",
                        "event_type": "add",
                        "ticker": "3017.TW",
                        "name": "奇鋐",
                        "index_name": "臺灣50指數",
                        "source_date": "2024-06-07",
                        "source_title": "臺灣50指數成分股審核結果",
                        "source_url": "",
                        "source_type": "official_technical_notice",
                        "exact_or_proxy": "exact_candidate",
                        "accepted": True,
                        "blocked_reason": "",
                    },
                ]
            ).to_csv(events, index=False)

            result = build_tw50_pit_snapshots_from_current_and_events(
                current_snapshot_path=current,
                snapshot_as_of="2024-07-01",
                event_rows_path=events,
                output_path=root / "tw50.csv",
                history_start="2024-01-01",
                source_updated_at="2026-06-23",
            )
            frame = pd.read_csv(root / "tw50.csv")
            initial = frame[frame["effective_date"] == "2024-01-01"]
            after_event = frame[frame["effective_date"] == "2024-06-24"]

        self.assertEqual(result["snapshot_count"], 2)
        self.assertIn("2801.TW", set(initial["ticker"]))
        self.assertNotIn("3017.TW", set(initial["ticker"]))
        self.assertIn("3017.TW", set(after_event["ticker"]))
        self.assertNotIn("2801.TW", set(after_event["ticker"]))


if __name__ == "__main__":
    unittest.main()
