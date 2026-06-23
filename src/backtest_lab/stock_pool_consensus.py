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
            "rank_score": item.get("rank_score", item.get("score", "")),
            "base_pool_passed": bool(item.get("base_pool_passed", False)),
            "selection_layer": item.get("selection_layer", ""),
            "eligible_for_pool_selection": bool(item.get("eligible_for_pool_selection", True)),
            "attack_gate_open": item.get("attack_gate_open", ""),
            "gate_rule_id": item.get("gate_rule_id", ""),
            "gate_reason": item.get("gate_reason", ""),
            "top_asset_type": item.get("top_asset_type", ""),
            "decision_layer": item.get("decision_layer", CANDIDATE_SOURCE),
            "active_in_trade_decision": bool(item.get("active_in_trade_decision", False)),
            "source_module": item.get("source_module", ""),
        }
        for item in voters
    ]
    skipped_vote_pools = [
        {
            "pool_id": item.get("pool_id", ""),
            "pool_name": item.get("pool_name", ""),
            "reason": item.get("reason", "") or item.get("selection_reason", ""),
            "selection_layer": item.get("selection_layer", ""),
            "eligible_for_pool_selection": bool(item.get("eligible_for_pool_selection", False)),
            "gate_rule_id": item.get("gate_rule_id", ""),
            "gate_reason": item.get("gate_reason", ""),
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
    ]
    health_diagnostic = _build_health_diagnostic(
        result_state=result_state,
        winner_ticker=winner_ticker,
        voters=vote_rows,
        skipped_vote_pools=skipped_vote_pools,
        votes=votes,
    )
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
        "health_diagnostic": health_diagnostic,
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
        "skipped_vote_pools": skipped_vote_pools,
        "boundary": "AI 輔助市場觀察與模型表決，不是投資建議。",
    }


def write_consensus_outputs(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    consensus = build_consensus(manifest)
    (root / "stock_pool_consensus.json").write_text(
        json.dumps(consensus, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(consensus["voters"]).to_csv(root / "stock_pool_consensus_votes.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([consensus["health_diagnostic"]]).to_csv(
        root / "stock_pool_consensus_health.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (root / "stock_pool_consensus_report.md").write_text(markdown_consensus_report(consensus), encoding="utf-8")
    return consensus


def markdown_consensus_report(consensus: dict[str, Any]) -> str:
    health = consensus.get("health_diagnostic") or {}
    lines = [
        "# 三立場股票池表決摘要",
        "",
        f"- 訊號日：{consensus.get('signal_date', '')}",
        f"- 狀態：{consensus.get('result_state', '')}",
        f"- 結論：{consensus.get('winner_display') or '沒有形成明確共識'}",
        f"- 原因：{consensus.get('reason', '')}",
        f"- 決策層：{consensus.get('consensus_type', 'consensus_observation')}；正式交易目標：未設定",
        "",
        "## 共識健康診斷",
        "",
        f"- decision_state：{health.get('decision_state', '')}",
        f"- 共識強度：{health.get('consensus_strength', '')}",
        f"- exact_ticker_consensus_rate：{health.get('exact_ticker_consensus_rate', 0)}",
        f"- direction_consensus_rate：{health.get('direction_consensus_rate', 0)}",
        f"- divergent_rate：{health.get('divergent_rate', 0)}",
        f"- no_vote_or_data_insufficient_rate：{health.get('no_vote_or_data_insufficient_rate', 0)}",
        f"- actionable_decision_rate：{health.get('actionable_decision_rate', 0)}",
        f"- decision_protocol_used_rate：{health.get('decision_protocol_used_rate', 0)}",
        f"- 診斷：{health.get('health_note', '')}",
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


def _build_health_diagnostic(
    *,
    result_state: str,
    winner_ticker: str | None,
    voters: list[dict[str, Any]],
    skipped_vote_pools: list[dict[str, Any]],
    votes: Counter,
) -> dict[str, Any]:
    total_considered = len(voters) + len(skipped_vote_pools)
    eligible_count = len(voters)
    winner_vote_count = int(votes.get(str(winner_ticker), 0)) if winner_ticker else 0
    max_vote_count = max(votes.values(), default=0)
    direction_counts = Counter(_direction_key(row) for row in voters)
    max_direction_count = max(direction_counts.values(), default=0)
    no_vote = result_state in {"no_vote", "insufficient_votes"} or total_considered == 0
    divergent = result_state == "divergent"
    protocol_used = False
    consensus_strength = _consensus_strength(
        result_state=result_state,
        winner_vote_count=winner_vote_count,
        eligible_count=eligible_count,
        total_considered=total_considered,
    )
    decision_state = _decision_state(
        result_state=result_state,
        consensus_strength=consensus_strength,
        winner_ticker=winner_ticker,
        voters=voters,
        skipped_vote_pools=skipped_vote_pools,
    )
    return {
        "schema_version": 1,
        "decision_layer": CANDIDATE_SOURCE,
        "active_in_trade_decision": False,
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "decision_state": decision_state,
        "consensus_strength": consensus_strength,
        "pool_count_considered": total_considered,
        "eligible_vote_count": eligible_count,
        "skipped_or_ineligible_pool_count": len(skipped_vote_pools),
        "winner_vote_count": winner_vote_count,
        "exact_ticker_consensus_rate": _rate(max_vote_count, total_considered),
        "direction_consensus_rate": _rate(max_direction_count, total_considered),
        "divergent_rate": 1.0 if divergent else 0.0,
        "no_vote_or_data_insufficient_rate": 1.0 if no_vote else 0.0,
        "actionable_decision_rate": 1.0 if result_state == "consensus" and winner_ticker else 0.0,
        "decision_protocol_used_rate": 1.0 if protocol_used else 0.0,
        "health_note": _health_note(decision_state, divergent=divergent, no_vote=no_vote),
        "boundary": "report-only diagnostic；不是正式交易決策。",
    }


def _consensus_strength(
    *,
    result_state: str,
    winner_vote_count: int,
    eligible_count: int,
    total_considered: int,
) -> str:
    if result_state != "consensus":
        return "none"
    if winner_vote_count >= 3 or (total_considered <= 2 and winner_vote_count == eligible_count):
        return "strong"
    if winner_vote_count >= 2:
        return "weak"
    return "none"


def _decision_state(
    *,
    result_state: str,
    consensus_strength: str,
    winner_ticker: str | None,
    voters: list[dict[str, Any]],
    skipped_vote_pools: list[dict[str, Any]],
) -> str:
    if _has_forced_stop(voters, skipped_vote_pools):
        return "forced_stop"
    if result_state in {"no_vote", "insufficient_votes"}:
        return "data_insufficient"
    if result_state == "divergent":
        return "divergent_observe"
    winner_rows = [row for row in voters if row.get("top_ticker") == winner_ticker]
    if any(_direction_key(row) in {"market_exposure", "defensive"} for row in winner_rows):
        return "defensive_or_market_exposure"
    if consensus_strength == "strong":
        return "strong_consensus"
    if consensus_strength == "weak":
        return "weak_consensus"
    return "divergent_observe"


def _direction_key(row: dict[str, Any]) -> str:
    asset_type = str(row.get("top_asset_type") or "").lower()
    selection_layer = str(row.get("selection_layer") or "").lower()
    action_state = str(row.get("action_state") or "").lower()
    ticker = str(row.get("top_ticker") or "")
    if asset_type in {"etf", "cash"} or ticker.startswith(("0050", "00631L")):
        return "market_exposure"
    if "market_exposure" in selection_layer:
        return "market_exposure"
    if "defensive" in selection_layer or "防守" in action_state:
        return "defensive"
    if "observation" in selection_layer:
        return "observation"
    return "stock_attack"


def _has_forced_stop(voters: list[dict[str, Any]], skipped_vote_pools: list[dict[str, Any]]) -> bool:
    stop_tokens = ("forced_stop", "stop_latch", "停損", "強制")
    for row in voters + skipped_vote_pools:
        text = " ".join(str(row.get(key, "")) for key in ("action_state", "selection_layer", "reason", "gate_reason"))
        lowered = text.lower()
        if any(token in lowered for token in stop_tokens):
            return True
    return False


def _rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 4)


def _health_note(decision_state: str, *, divergent: bool, no_vote: bool) -> str:
    if divergent:
        return "三池分歧，應檢查池角色與候選設計，不應包裝成明確行動訊號。"
    if no_vote:
        return "可投票資料不足，應先檢查資料完整性與池內 gate。"
    if decision_state == "defensive_or_market_exposure":
        return "共識偏向市場曝險或防守工具，代表模型目前不偏向單一個股攻擊。"
    if decision_state == "strong_consensus":
        return "三池高度一致；仍需保留資料日與風險邊界。"
    if decision_state == "weak_consensus":
        return "形成 2/3 共識；少數池意見仍應保留為風險觀察。"
    if decision_state == "forced_stop":
        return "偵測到強制防守或停損語意，僅作 report-only 風險診斷。"
    return "report-only 共識健康診斷。"
