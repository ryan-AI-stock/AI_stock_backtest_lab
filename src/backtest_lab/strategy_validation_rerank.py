from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from backtest_lab.strategy_validation_matrix import _rank_candidates, _top_candidate_diagnostics, _write_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Rerank an existing strategy validation summary.")
    parser.add_argument("--summary-csv", required=True)
    parser.add_argument("--trades-csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-diagnostics", type=int, default=12)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(args.summary_csv)
    ranking = _rank_candidates(summary)
    diagnostics = pd.DataFrame()
    if args.trades_csv:
        trades = pd.read_csv(args.trades_csv)
        diagnostics = _top_candidate_diagnostics(trades, ranking, args.top_diagnostics)
    summary.to_csv(output_dir / "strategy_validation_summary.csv", index=False, encoding="utf-8-sig")
    ranking.to_csv(output_dir / "strategy_validation_ranking.csv", index=False, encoding="utf-8-sig")
    diagnostics.to_csv(output_dir / "top_closed_trade_summary.csv", index=False, encoding="utf-8-sig")
    _write_report(output_dir / "strategy_validation_report.md", summary, ranking, diagnostics)
    print(f"OUTPUT_DIR={output_dir.resolve()}")


if __name__ == "__main__":
    main()
