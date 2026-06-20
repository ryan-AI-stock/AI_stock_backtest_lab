from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from backtest_lab.decision_layers import CANDIDATE_SOURCE


DEFAULT_VOTE_GROUP = "three_perspective_v1"


def build_consensus(manifest: dict[str, Any], *, vote_group: str = DEFAULT_VOTE_GROUP) -> dict[str, Any]:
    voters = [
        item
        for item in manifest.get("generated", [])
        if item.get("vote_group") == vote_group and item.get("top_ticker") and _eligible_vote_item(item)
    ]
    votes = Counter(str(item["top_ticker"]) for item in voters)
    displays: dict[str, str] = {}
    pools_by_ticker: dict[str, list[str]] = defaultdict(list)
    for item in voters:
        ticker = str(item["top_ticker"])
        displays[ticker] = str(item.get("top_display") or ticker)
        pools_by_ticker[ticker].append(str(item.get("pool_name") or item.get("pool_id") or ""))

    if not voters:
        result_state = "no_vote"
        winner_ticker = None
        winner_display = None
        reason = "沒有可用的三立場股票池輸出。"
    elif len(voters) < 2:
        result_state = "insufficient_votes"
        winner_ticker = None
        winner_display = None
        reason = f"只有 {len(voters)} 個股票池有可投票入選標的，未形成 2/3 共識。"
    else:
        winner_ticker, winner_votes = votes.most_common(1)[0]
        if winner_votes >= 2:
            result_state = "consensus"
            winner_display = displays.get(winner_ticker, winner_ticker)
            reason = f"{winner_display} 取得 {winner_votes}/{len(voters)} 票。"
        else:
            result_state = "divergent"
            winner_ticker = None
            winner_display = None
            reason = "三個立場沒有形成 2:1 以上共識，應視為模型分歧。"

    vote_rows = [
        {
            "pool_id": item.get("pool_id", ""),
            "pool_name": item.get("pool_name", ""),
            "top_ticker": item.get("top_ticker", ""),
            "top_display": item.get("top_display", ""),
            "action_state": item.get("action_state", ""),
            "selection_layer": item.get("selection_layer", ""),
            "eligible_for_pool_selection": bool(item.get("eligible_for_pool_selection", True)),
            "attack_gate_open": item.get("attack_gate_open", ""),
            "top_asset_type": item.get("top_asset_type", ""),
            "decision_layer": item.get("decision_layer", CANDIDATE_SOURCE),
            "active_in_trade_decision": bool(item.get("active_in_trade_decision", False)),
            "source_module": item.get("source_module", ""),
        }
        for item in voters
    ]
    return {
        "schema_version": 1,
        "vote_group": vote_group,
        "signal_date": manifest.get("signal_date", ""),
        "decision_layer": CANDIDATE_SOURCE,
        "active_in_trade_decision": False,
        "consensus_type": "consensus_observation",
        "formal_trade_target": None,
        "result_state": result_state,
        "winner_ticker": winner_ticker,
        "winner_display": winner_display,
        "reason": reason,
        "votes": [
            {
                "ticker": ticker,
                "display": displays.get(ticker, ticker),
                "vote_count": count,
                "pools": pools_by_ticker.get(ticker, []),
            }
            for ticker, count in votes.most_common()
        ],
        "voters": vote_rows,
        "skipped_vote_pools": [
            {
                "pool_id": item.get("pool_id", ""),
                "pool_name": item.get("pool_name", ""),
                "reason": item.get("reason", "") or item.get("selection_reason", ""),
                "selection_layer": item.get("selection_layer", ""),
                "eligible_for_pool_selection": bool(item.get("eligible_for_pool_selection", False)),
                "top_ticker": item.get("top_ticker", ""),
                "top_display": item.get("top_display", ""),
            }
            for item in manifest.get("skipped", [])
            if (item.get("dispatch") or {}).get("operational_observation")
        ] + [
            {
                "pool_id": item.get("pool_id", ""),
                "pool_name": item.get("pool_name", ""),
                "reason": item.get("selection_reason", "未通過池內入選條件。"),
                "selection_layer": item.get("selection_layer", ""),
                "eligible_for_pool_selection": bool(item.get("eligible_for_pool_selection", False)),
                "top_ticker": item.get("top_ticker", ""),
                "top_display": item.get("top_display", ""),
            }
            for item in manifest.get("generated", [])
            if item.get("vote_group") == vote_group and item.get("top_ticker") and not _eligible_vote_item(item)
        ],
        "boundary": "AI 輔助市場觀察與模型表決，不是投資建議。",
    }


def write_consensus_outputs(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    consensus = build_consensus(manifest)
    (root / "stock_pool_consensus.json").write_text(
        json.dumps(consensus, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(consensus["voters"]).to_csv(root / "stock_pool_consensus_votes.csv", index=False, encoding="utf-8-sig")
    (root / "stock_pool_consensus_report.md").write_text(markdown_consensus_report(consensus), encoding="utf-8")
    return consensus


def markdown_consensus_report(consensus: dict[str, Any]) -> str:
    lines = [
        "# 三立場股票池表決摘要",
        "",
        f"- 訊號日：{consensus.get('signal_date', '')}",
        f"- 狀態：{consensus.get('result_state', '')}",
        f"- 結論：{consensus.get('winner_display') or '沒有形成明確共識'}",
        f"- 原因：{consensus.get('reason', '')}",
        f"- 決策層：{consensus.get('consensus_type', 'consensus_observation')}；正式交易目標：未設定",
        "",
        "| 股票池 | 第一順位 | 入選層級 | 狀態 |",
        "| --- | --- | --- | --- |",
    ]
    for row in consensus.get("voters", []):
        lines.append(
            f"| {row.get('pool_name', '')} | {row.get('top_display') or row.get('top_ticker') or '-'} | "
            f"{row.get('selection_layer', '')} | {row.get('action_state', '')} |"
        )
    for row in consensus.get("skipped_vote_pools", []):
        if not row.get("top_ticker"):
            continue
        lines.append(
            f"| {row.get('pool_name', '')} | {row.get('top_display') or row.get('top_ticker') or '-'} | "
            f"{row.get('selection_layer', '') or 'no_selection'} | 不投票：{row.get('reason', '')} |"
        )
    lines.extend(
        [
            "",
            "使用邊界：這是 AI 輔助市場觀察與模型表決，不是投資建議。若三池分歧，不應硬解讀成明確換倉訊號。",
        ]
    )
    return "\n".join(lines)


def _eligible_vote_item(item: dict[str, Any]) -> bool:
    if "eligible_for_pool_selection" not in item:
        return True
    return bool(item.get("eligible_for_pool_selection"))
