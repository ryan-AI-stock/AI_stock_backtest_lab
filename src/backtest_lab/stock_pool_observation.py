from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from backtest_lab.data import download_yfinance_prices
from backtest_lab.frozen_report_pdf import _configure_chinese_font, _save_figure_as_raster_pdf_page
from backtest_lab.radar_snapshot_v2_source import load_radar_snapshot_history, select_radar_snapshot_candidates
from backtest_lab.stock_pool_store import StockPoolStore
from backtest_lab.universal_pool_strategy import (
    PoolProfile,
    UniversalCandidateScore,
    UniversalPoolParameters,
    default_parameters_for_profile,
    infer_pool_profile,
    score_universal_candidates,
)


DEFAULT_OUTPUT_ROOT = "outputs/stock_pool_observations"
REPORT_NAME = "AI股票池觀察總覽"
REPORT_VERSION = "v20260612"
REPORT_LATEST_FILENAME = f"{REPORT_NAME}_最新版_{REPORT_VERSION}.pdf"


@dataclass(frozen=True)
class StockPoolObservation:
    schema_version: int
    pool_id: str
    pool_name: str
    strategy_preset: str
    signal_date: str
    data_end_date: str
    candidate_count: int
    passed_count: int
    pool_profile: PoolProfile
    parameters: UniversalPoolParameters
    top_ticker: str | None
    top_display: str | None
    top_score: float | None
    action_state: str
    candidates: list[UniversalCandidateScore]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["pool_profile"] = asdict(self.pool_profile)
        payload["parameters"] = asdict(self.parameters)
        payload["candidates"] = [asdict(candidate) for candidate in self.candidates]
        return payload


def build_stock_pool_observation(
    *,
    pool: dict[str, Any],
    prices_by_ticker: dict[str, pd.DataFrame],
    signal_date: str | pd.Timestamp,
    theme_by_ticker: dict[str, str] | None = None,
    conviction_by_ticker: dict[str, float] | None = None,
) -> StockPoolObservation:
    signal_ts = _resolve_signal_date(prices_by_ticker, pd.Timestamp(signal_date))
    available_symbols = [
        symbol
        for symbol in pool.get("resolved_symbols") or pool.get("symbols") or []
        if symbol.get("ticker") in prices_by_ticker
    ]
    candidate_prices = {
        symbol["ticker"]: prices_by_ticker[symbol["ticker"]]
        for symbol in available_symbols
    }
    profile = infer_pool_profile(candidate_prices, signal_ts, theme_by_ticker=theme_by_ticker)
    params = default_parameters_for_profile(profile)
    scored = score_universal_candidates(
        candidate_prices,
        signal_ts,
        params,
        conviction_by_ticker=conviction_by_ticker,
    )
    candidates = sorted(
        scored.values(),
        key=lambda item: (item.passed, item.score, item.ret20, item.ticker),
        reverse=True,
    )
    top = next((candidate for candidate in candidates if candidate.passed), None)
    display_by_ticker = {
        symbol["ticker"]: symbol.get("display") or symbol["ticker"]
        for symbol in available_symbols
    }
    return StockPoolObservation(
        schema_version=1,
        pool_id=str(pool["pool_id"]),
        pool_name=str(pool["name"]),
        strategy_preset=str(pool.get("strategy_preset") or "universal_pool_custom"),
        signal_date=signal_ts.strftime("%Y-%m-%d"),
        data_end_date=signal_ts.strftime("%Y-%m-%d"),
        candidate_count=len(candidates),
        passed_count=sum(1 for candidate in candidates if candidate.passed),
        pool_profile=profile,
        parameters=params,
        top_ticker=top.ticker if top else None,
        top_display=display_by_ticker.get(top.ticker, top.ticker) if top else None,
        top_score=round(top.score, 6) if top else None,
        action_state="watch_candidate" if top else "no_valid_candidate",
        candidates=candidates,
    )


def write_stock_pool_observation(output_dir: Path, observation: StockPoolObservation) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = observation.to_dict()
    (output_dir / "stock_pool_observation.json").write_text(
        json.dumps({"status": "ready", "observation": payload}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(payload["candidates"]).to_csv(
        output_dir / "stock_pool_observation_candidates.csv",
        index=False,
        encoding="utf-8-sig",
    )


def run_stock_pool_observation_batch(
    *,
    pools: list[dict[str, Any]],
    signal_date: str,
    warmup_start: str,
    cache_dir: str | Path,
    output_root: str | Path,
    radar_snapshot_dir: str | Path | None = None,
    radar_top_n: int = 20,
) -> dict[str, Any]:
    date_key = pd.Timestamp(signal_date).strftime("%Y%m%d")
    root = Path(output_root) / date_key
    root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "status": "ready",
        "signal_date": signal_date,
        "output_root": str(root),
        "generated": [],
        "skipped": [],
    }
    for pool in pools:
        pool = _resolve_dynamic_observation_pool(
            pool,
            signal_date=signal_date,
            radar_snapshot_dir=radar_snapshot_dir,
            radar_top_n=radar_top_n,
        )
        tickers = [symbol["ticker"] for symbol in pool.get("resolved_symbols", [])]
        if not tickers:
            reason = (
                "missing_radar_snapshot_dir"
                if pool.get("strategy_preset") == "radar_core_mid_small_calibrated_v1" and not radar_snapshot_dir
                else "no_resolved_symbols"
            )
            manifest["skipped"].append(
                {
                    "pool_id": pool.get("pool_id"),
                    "pool_name": pool.get("name"),
                    "reason": reason,
                }
            )
            continue
        try:
            prices, missing_price_tickers = _load_observation_price_frames(
                tickers=tickers,
                start_date=warmup_start,
                end_date=signal_date,
                cache_dir=cache_dir,
            )
            if not prices:
                raise ValueError(f"No price data available for pool tickers: {', '.join(tickers)}")
            observation = build_stock_pool_observation(
                pool=pool,
                prices_by_ticker=prices,
                signal_date=signal_date,
            )
            pool_dir = root / str(pool["pool_id"])
            write_stock_pool_observation(pool_dir, observation)
            manifest["generated"].append(
                {
                    "pool_id": observation.pool_id,
                    "pool_name": observation.pool_name,
                    "signal_date": observation.signal_date,
                    "top_ticker": observation.top_ticker,
                    "top_display": observation.top_display,
                    "action_state": observation.action_state,
                    "missing_price_tickers": missing_price_tickers,
                    "output_dir": str(pool_dir),
                }
            )
        except Exception as error:  # pragma: no cover - defensive batch manifest path
            manifest["skipped"].append(
                {
                    "pool_id": pool.get("pool_id"),
                    "pool_name": pool.get("name"),
                    "reason": str(error),
                }
            )
    (root / "stock_pool_observation_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_stock_pool_observation_batch_summary(root, manifest)
    return manifest


def write_stock_pool_observation_batch_summary(root: Path, manifest: dict[str, Any]) -> None:
    rows = []
    for item in manifest.get("generated", []):
        rows.append(
            {
                "status": "generated",
                "pool_id": item.get("pool_id", ""),
                "pool_name": item.get("pool_name", ""),
                "signal_date": item.get("signal_date", manifest.get("signal_date", "")),
                "top_display": item.get("top_display", ""),
                "top_ticker": item.get("top_ticker", ""),
                "action_state": item.get("action_state", ""),
                "missing_price_tickers": ",".join(item.get("missing_price_tickers") or []),
                "reason": "",
                "output_dir": item.get("output_dir", ""),
            }
        )
    for item in manifest.get("skipped", []):
        rows.append(
            {
                "status": "skipped",
                "pool_id": item.get("pool_id", ""),
                "pool_name": item.get("pool_name", ""),
                "signal_date": manifest.get("signal_date", ""),
                "top_display": "",
                "top_ticker": "",
                "action_state": "",
                "missing_price_tickers": "",
                "reason": item.get("reason", ""),
                "output_dir": "",
            }
        )
    pd.DataFrame(rows).to_csv(root / "stock_pool_observation_summary.csv", index=False, encoding="utf-8-sig")
    (root / "stock_pool_observation_report.md").write_text(
        markdown_observation_batch_report(manifest, rows),
        encoding="utf-8",
    )
    write_stock_pool_observation_batch_pdf(root / REPORT_LATEST_FILENAME, manifest, rows)


def markdown_observation_batch_report(manifest: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = [
        "# 股票池觀察摘要",
        "",
        f"- 訊號日：{manifest.get('signal_date', '')}",
        f"- 已產出股票池：{len(manifest.get('generated', []))}",
        f"- 跳過股票池：{len(manifest.get('skipped', []))}",
        "",
        "| 狀態 | 股票池 | 第一名/原因 | 缺價股票 |",
        "| --- | --- | --- | --- |",
    ]
    for row in rows:
        if row["status"] == "generated":
            target = row["top_display"] or row["top_ticker"] or row["action_state"] or "無合格候選"
            missing = row["missing_price_tickers"] or "-"
        else:
            target = row["reason"] or "skipped"
            missing = "-"
        lines.append(f"| {row['status']} | {row['pool_name']} | {target} | {missing} |")
    lines.extend(
        [
            "",
            "本摘要為 AI 輔助股票池觀察輸出，不是投資建議；正式用途仍需搭配策略規則、交易成本、資料完整性與風險檢查。",
        ]
    )
    return "\n".join(lines)


def write_stock_pool_observation_batch_pdf(path: Path, manifest: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    _configure_chinese_font()
    with PdfPages(path) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69), facecolor="#f4f6f8")
        ax = fig.add_axes((0, 0, 1, 1))
        ax.axis("off")
        _draw_observation_pdf_page(ax, manifest, rows)
        _save_figure_as_raster_pdf_page(pdf, fig)


def _draw_observation_pdf_page(ax, manifest: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    generated_count = len(manifest.get("generated", []))
    skipped_count = len(manifest.get("skipped", []))
    first_generated = next((row for row in rows if row["status"] == "generated"), None)
    top_label = (first_generated or {}).get("top_display") or (first_generated or {}).get("top_ticker") or "無"

    ax.add_patch(plt.Rectangle((0, 0.86), 1, 0.14, color="#17212a", transform=ax.transAxes))
    ax.text(0.06, 0.94, REPORT_NAME, color="white", fontsize=20, fontweight="bold", transform=ax.transAxes)
    ax.text(
        0.06,
        0.895,
        f"訊號日 {manifest.get('signal_date', '')} · {REPORT_VERSION}",
        color="#c8d5df",
        fontsize=11,
        transform=ax.transAxes,
    )
    cards = [
        ("已產出股票池", f"{generated_count}", "#13795b"),
        ("跳過股票池", f"{skipped_count}", "#b42318" if skipped_count else "#13795b"),
        ("第一個池第一名", top_label, "#2457a7"),
        ("使用邊界", "觀察，不是建議", "#17212a"),
    ]
    for index, (label, value, color) in enumerate(cards):
        x = 0.06 + index * 0.225
        ax.add_patch(
            plt.Rectangle((x, 0.74), 0.2, 0.085, facecolor="white", edgecolor="#d9e0e5", linewidth=1, transform=ax.transAxes)
        )
        ax.text(x + 0.014, 0.795, label, color="#66737d", fontsize=9.5, transform=ax.transAxes)
        ax.text(x + 0.014, 0.767, str(value)[:18], color=color, fontsize=11.2, fontweight="bold", transform=ax.transAxes)

    ax.text(0.06, 0.675, "股票池觀察結果", color="#17212a", fontsize=16, fontweight="bold", transform=ax.transAxes)
    _draw_observation_table(ax, rows)
    ax.text(0.06, 0.18, "使用邊界", color="#17212a", fontsize=14, fontweight="bold", transform=ax.transAxes)
    notes = [
        "本報告用同一套觀察框架讀取不同股票池，方便比較各池目前第一順位與資料缺口。",
        "強弱排名是觀察清單，不是買入資格清單；實際操作仍需搭配策略規則、交易成本與風險承受度。",
        "若出現缺價股票，代表該檔未納入當次分數計算，後續需補資料或確認資料源。",
        "本報告為 AI 輔助市場觀察與回測工作流輸出，不是投資建議。",
    ]
    for index, note in enumerate(notes):
        ax.text(0.075, 0.15 - index * 0.028, f"• {note}", color="#4d5b66", fontsize=10.2, transform=ax.transAxes)
    ax.text(0.06, 0.04, f"{REPORT_NAME} · {manifest.get('signal_date', '')}", color="#9aa7b1", fontsize=8.5, transform=ax.transAxes)
    ax.text(0.94, 0.04, "AI_stock_backtest_lab", color="#9aa7b1", fontsize=8.5, ha="right", transform=ax.transAxes)


def _draw_observation_table(ax, rows: list[dict[str, Any]]) -> None:
    x0, y0 = 0.06, 0.62
    widths = [0.12, 0.31, 0.2, 0.22]
    headers = ["狀態", "股票池", "第一名/原因", "缺價股票"]
    ax.add_patch(plt.Rectangle((x0, y0), sum(widths), 0.044, facecolor="#e9f0f5", edgecolor="#d7e0e7", transform=ax.transAxes))
    x = x0
    for header, width in zip(headers, widths):
        ax.text(x + 0.01, y0 + 0.015, header, color="#34424d", fontsize=10, fontweight="bold", transform=ax.transAxes)
        x += width
    max_rows = rows[:8]
    for index, row in enumerate(max_rows):
        y = y0 - (index + 1) * 0.047
        color = "#ffffff" if index % 2 == 0 else "#f8fafc"
        ax.add_patch(plt.Rectangle((x0, y), sum(widths), 0.047, facecolor=color, edgecolor="#e1e7ec", transform=ax.transAxes))
        status_color = "#13795b" if row["status"] == "generated" else "#b42318"
        target = (
            row["top_display"]
            or row["top_ticker"]
            or row["reason"]
            or row["action_state"]
            or "-"
        )
        cells = [
            row["status"],
            row["pool_name"],
            target,
            row["missing_price_tickers"] or "-",
        ]
        x = x0
        for cell_index, (cell, width) in enumerate(zip(cells, widths)):
            text_color = status_color if cell_index == 0 else "#26323b"
            ax.text(x + 0.01, y + 0.016, str(cell)[:24], color=text_color, fontsize=9.2, transform=ax.transAxes)
            x += width


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a unified stock-pool observation snapshot.")
    parser.add_argument("--pool-store", default="work/stock_pools/stock_pools.json")
    parser.add_argument("--pool-id", default="large_cap_best_v20260605")
    parser.add_argument("--signal-date", required=True)
    parser.add_argument("--cache-dir", default="backtest_cache/stock_pool_observations")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--warmup-start", default="2020-01-02")
    parser.add_argument("--radar-snapshot-dir", default=os.getenv("RADAR_SNAPSHOT_DIR", ""))
    parser.add_argument("--radar-top-n", type=int, default=20)
    args = parser.parse_args()

    store = StockPoolStore(args.pool_store)
    pools = store.list_pools()
    if args.pool_id == "all":
        manifest = run_stock_pool_observation_batch(
            pools=pools,
            signal_date=args.signal_date,
            warmup_start=args.warmup_start,
            cache_dir=args.cache_dir,
            output_root=args.output_root,
            radar_snapshot_dir=args.radar_snapshot_dir or None,
            radar_top_n=args.radar_top_n,
        )
        print(f"STOCK_POOL_OBSERVATION_MANIFEST={Path(manifest['output_root']).resolve() / 'stock_pool_observation_manifest.json'}")
        return
    pool = next((item for item in pools if item["pool_id"] == args.pool_id), None)
    if pool is None:
        raise ValueError(f"Unknown pool_id: {args.pool_id}")
    pool = _resolve_dynamic_observation_pool(
        pool,
        signal_date=args.signal_date,
        radar_snapshot_dir=args.radar_snapshot_dir or None,
        radar_top_n=args.radar_top_n,
    )
    tickers = [symbol["ticker"] for symbol in pool.get("resolved_symbols", [])]
    if not tickers:
        raise ValueError(f"Pool has no resolved tickers: {args.pool_id}")
    prices, missing_price_tickers = _load_observation_price_frames(
        tickers=tickers,
        start_date=args.warmup_start,
        end_date=args.signal_date,
        cache_dir=args.cache_dir,
    )
    if not prices:
        raise ValueError(f"No price data available for pool tickers: {', '.join(tickers)}")
    observation = build_stock_pool_observation(
        pool=pool,
        prices_by_ticker=prices,
        signal_date=args.signal_date,
    )
    if missing_price_tickers:
        print(f"STOCK_POOL_OBSERVATION_MISSING_PRICE_TICKERS={','.join(missing_price_tickers)}")
    output_dir = Path(args.output_root) / args.pool_id / args.signal_date.replace("-", "")
    write_stock_pool_observation(output_dir, observation)
    print(f"STOCK_POOL_OBSERVATION_DIR={output_dir.resolve()}")


def _resolve_signal_date(prices_by_ticker: dict[str, pd.DataFrame], requested: pd.Timestamp) -> pd.Timestamp:
    common = None
    for frame in prices_by_ticker.values():
        dates = set(frame.index[frame.index <= requested])
        common = dates if common is None else common & dates
    if not common:
        raise ValueError(f"No common signal date on or before {requested.strftime('%Y-%m-%d')}")
    return max(common)


def _load_observation_price_frames(
    *,
    tickers: list[str],
    start_date: str,
    end_date: str,
    cache_dir: str | Path,
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    prices: dict[str, pd.DataFrame] = {}
    missing: list[str] = []
    for ticker in tickers:
        try:
            loaded = download_yfinance_prices(
                tickers=[ticker],
                start_date=start_date,
                end_date=end_date,
                cache_dir=cache_dir,
                allow_edge_gap=False,
            )
            prices.update(loaded)
        except Exception:
            missing.append(ticker)
    return prices, missing


def _resolve_dynamic_observation_pool(
    pool: dict[str, Any],
    *,
    signal_date: str,
    radar_snapshot_dir: str | Path | None,
    radar_top_n: int,
) -> dict[str, Any]:
    if pool.get("resolved_symbols") or pool.get("strategy_preset") != "radar_core_mid_small_calibrated_v1":
        return pool
    if not radar_snapshot_dir:
        return pool
    history = load_radar_snapshot_history(radar_snapshot_dir)
    candidates = select_radar_snapshot_candidates(history, signal_date, top_n=radar_top_n)
    resolved = []
    for _, row in candidates.rows.iterrows():
        symbol = str(row.get("symbol") or "").strip()
        if not symbol:
            continue
        ticker = f"{symbol}.TW"
        name = str(row.get("name") or symbol).strip() or symbol
        resolved.append(
            {
                "ticker": ticker,
                "symbol": symbol,
                "name": name,
                "display": f"{name}({symbol})",
                "asset_type": "stock",
                "source": "radar_snapshot_v2",
                "theme": str(row.get("theme") or ""),
                "snapshot_date": candidates.snapshot_date.strftime("%Y-%m-%d"),
                "candidate_rank": int(row.get("candidate_rank") or len(resolved) + 1),
            }
        )
    updated = json.loads(json.dumps(pool, ensure_ascii=False))
    updated["resolved_symbols"] = resolved
    updated["radar_snapshot_date"] = candidates.snapshot_date.strftime("%Y-%m-%d")
    return updated


if __name__ == "__main__":
    main()
