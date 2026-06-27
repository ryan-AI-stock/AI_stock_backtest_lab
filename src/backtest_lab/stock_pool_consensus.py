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
        _pool_diagnostic_row(item, eligible_vote=True)
        for item in voters
    ]
    skipped_vote_pools = [
        _pool_diagnostic_row(item, eligible_vote=False, reason=item.get("reason", "") or item.get("selection_reason", ""))
        for item in manifest.get("skipped", [])
        if (item.get("dispatch") or {}).get("operational_observation")
    ] + [
        _pool_diagnostic_row(item, eligible_vote=False, reason=item.get("selection_reason", "未通過池內入選條件。"))
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
        "pool_diagnostics": vote_rows + skipped_vote_pools,
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
        "boundary": "AI 輔助市場觀察與模型診斷，不是正式交易指令。",
    }


def write_consensus_outputs(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    consensus = build_consensus(manifest)
    (root / "stock_pool_consensus.json").write_text(
        json.dumps(consensus, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(consensus["voters"]).to_csv(root / "stock_pool_consensus_votes.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(consensus["pool_diagnostics"]).to_csv(
        root / "stock_pool_consensus_pool_diagnostics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame([consensus["health_diagnostic"]]).to_csv(
        root / "stock_pool_consensus_health.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (root / "stock_pool_consensus_report.md").write_text(markdown_consensus_report(consensus), encoding="utf-8")
    return consensus


def markdown_consensus_report(consensus: dict[str, Any]) -> str:
    health = consensus.get("health_diagnostic") or {}
    visible_voters = [
        row for row in consensus.get("voters", [])
        if not _hide_from_visible_consensus_report(row)
    ]
    visible_skipped = [
        row for row in consensus.get("skipped_vote_pools", [])
        if not _hide_from_visible_consensus_report(row)
    ]
    lines = [
        "# 候選分歧診斷摘要",
        "",
        f"- 訊號日：{consensus.get('signal_date', '')}",
        f"- 狀態：{_user_facing_state(consensus.get('result_state', ''))}",
        f"- 結論：{consensus.get('winner_display') or '沒有形成明確共識'}",
        f"- 原因：{_user_facing_text(consensus.get('reason', ''))}",
        f"- 診斷層：候選分歧觀察；正式交易目標：未設定",
        "",
        "## 共識健康診斷",
        "",
        f"- 決策狀態：{_user_facing_state(health.get('decision_state', ''))}",
        f"- 共識強度：{_user_facing_state(health.get('consensus_strength', ''))}",
        f"- 同一標的共識率：{health.get('exact_ticker_consensus_rate', 0)}",
        f"- 方向共識率：{health.get('direction_consensus_rate', 0)}",
        f"- 分歧比例：{health.get('divergent_rate', 0)}",
        f"- 無訊號或資料不足比例：{health.get('no_vote_or_data_insufficient_rate', 0)}",
        f"- 可形成行動觀察比例：{health.get('actionable_decision_rate', 0)}",
        f"- 需要人工判讀比例：{health.get('decision_protocol_used_rate', 0)}",
        f"- 原始共識狀態：{_user_facing_state(health.get('raw_consensus_state', ''))}",
        f"- 同一標的共識：{_user_facing_text(health.get('exact_ticker_consensus', ''))} / {_user_facing_text(health.get('exact_ticker_consensus_group', ''))}",
        f"- 方向共識：{_user_facing_text(health.get('direction_consensus', ''))} / {_user_facing_text(health.get('direction_consensus_group', ''))} / {_user_facing_state(health.get('direction_consensus_strength', ''))}",
        f"- 可行動狀態：{_user_facing_state(health.get('actionable_decision_state', ''))}",
        f"- 判斷來源：{_user_facing_state(health.get('decision_source', ''))}",
        f"- 是否需要人工判讀：{_user_facing_bool(health.get('decision_protocol_used', ''))}",
        f"- 人工判讀類型：{_user_facing_state(health.get('protocol_usage_category', ''))}",
        f"- 假共識提醒：{_user_facing_flags(health.get('fake_consensus_flags') or [])}",
        f"- 健康分層：{_user_facing_state(health.get('consensus_health_bucket', ''))}",
        f"- 診斷：{_user_facing_text(health.get('health_note', ''))}",
        "",
        "| 股票池 | 第一順位 | 入選層級 | 狀態 |",
        "| --- | --- | --- | --- |",
    ]
    for row in visible_voters:
        lines.append(
            f"| {row.get('pool_name', '')} | {row.get('top_display') or row.get('top_ticker') or '-'} | "
            f"{_user_facing_state(row.get('selection_layer', ''))} | {_user_facing_state(row.get('action_state', ''))} |"
        )
    for row in visible_skipped:
        if not row.get("top_ticker"):
            continue
        lines.append(
            f"| {row.get('pool_name', '')} | {row.get('top_display') or row.get('top_ticker') or '-'} | "
            f"{_user_facing_state(row.get('selection_layer', '') or 'no_selection')} | 不納入：{_user_facing_text(row.get('reason', ''))} |"
        )
    return "\n".join(lines)


def _hide_from_visible_consensus_report(row: dict[str, Any]) -> bool:
    text = " ".join(
        str(row.get(key) or "")
        for key in ("pool_id", "pool_name", "pool_role", "top_display", "reason", "blocked_reason")
    )
    return any(marker in text for marker in ("large_core_bluechip_v0", "風格補強", "Pool3", "pool3", "Radar"))


def _user_facing_text(value: object) -> str:
    text = str(value or "")
    replacements = {
        "combined_cap40_confirmation1_base": "目前正式模型",
        "pool1_primary_pool2_confirmation_cap40": "主攻池優先、確認池風險確認",
        "Pool1+Pool2 formal baseline": "正式模型基準",
        "Pool1+Pool2": "主攻池 + 確認池",
        "PIT-ready Pool2": "已通過歷史成分檢查的確認池",
        "selector": "選股邏輯",
        "formal target": "正式採用版本",
        "正式 target": "正式採用版本",
        "no_selection": "未形成正式觀察",
        "none": "無",
    }
    for raw, translated in replacements.items():
        text = text.replace(raw, translated)
    return text


def _user_facing_state(value: object) -> str:
    text = _user_facing_text(value)
    labels = {
        "consensus": "形成觀察共識",
        "divergent": "候選分歧",
        "no_vote": "無正式觀察",
        "no_selection": "未形成正式觀察",
        "formal_vote": "正式觀察",
        "observation_only": "僅供觀察",
        "diagnostic_only": "僅供診斷",
        "report_only": "僅供報告說明",
        "attack": "偏攻擊",
        "risk_off": "偏防守",
        "data_insufficient": "資料不足",
        "strong": "強",
        "weak": "弱",
        "weak_consensus": "弱共識",
        "strong_consensus": "強共識",
        "divergent_observe": "分歧觀察",
        "defensive_or_market_exposure": "防守或市場曝險",
        "protocol_candidate_diagnostic": "人工判讀候選，僅供診斷",
        "protocol_resolved_divergence": "分歧情境，需人工判讀",
        "candidate_not_applied": "僅列候選，未套用",
        "exact_2_of_3_ticker": "兩個觀察池指向同一標的",
        "exact_3_of_3_ticker": "全部觀察池指向同一標的",
        "consensus_with_ineligible_pool": "包含未合格觀察池的假共識風險",
        "observation_only_excluded": "僅供觀察項目已排除",
        "acceptable": "可接受",
        "healthy": "健康",
        "warning": "警示",
        "none": "無",
        "False": "否",
        "True": "是",
        "false": "否",
        "true": "是",
    }
    return labels.get(text, text)


def _user_facing_bool(value: object) -> str:
    return "是" if str(value).lower() == "true" or value is True else "否"


def _user_facing_flags(flags: list[object]) -> str:
    if not flags:
        return "無"
    return "、".join(_user_facing_state(flag) for flag in flags)


def _eligible_vote_item(item: dict[str, Any]) -> bool:
    if "eligible_for_pool_selection" not in item:
        return True
    return bool(item.get("eligible_for_pool_selection"))


def _pool_diagnostic_row(item: dict[str, Any], *, eligible_vote: bool, reason: str = "") -> dict[str, Any]:
    ticker = str(item.get("top_ticker") or "")
    selection_layer = str(item.get("selection_layer") or "")
    gate_reason = str(item.get("gate_reason") or "")
    blocked_reason = reason or str(item.get("selection_reason") or "") or gate_reason
    direction_state = _direction_key(
        {
            "top_ticker": ticker,
            "top_asset_type": item.get("top_asset_type", ""),
            "selection_layer": selection_layer,
            "action_state": item.get("action_state", ""),
        }
    )
    return {
        "pool_id": item.get("pool_id", ""),
        "pool_name": item.get("pool_name", ""),
        "pool_role": item.get("role_name", item.get("pool_name", "")),
        "top_ticker": ticker,
        "top_display": item.get("top_display", ""),
        "top_asset_type": item.get("top_asset_type", ""),
        "top_score": item.get("rank_score", item.get("score", "")),
        "rank_score": item.get("rank_score", item.get("score", "")),
        "base_pool_passed": bool(item.get("base_pool_passed", False)),
        "selection_layer": selection_layer,
        "eligible_for_pool_selection": bool(eligible_vote),
        "eligible_vote": bool(eligible_vote),
        "vote_target": ticker if eligible_vote else "",
        "direction_state": direction_state,
        "direction_confidence": _direction_confidence(item, eligible_vote=eligible_vote),
        "data_readiness_state": _data_readiness_state(item, eligible_vote=eligible_vote),
        "blocked_reason": "" if eligible_vote else blocked_reason,
        "shadow_or_diagnostic_flags": _shadow_or_diagnostic_flags(item),
        "attack_gate_open": item.get("attack_gate_open", ""),
        "gate_rule_id": item.get("gate_rule_id", ""),
        "gate_reason": gate_reason,
        "action_state": item.get("action_state", ""),
        "decision_layer": item.get("decision_layer", CANDIDATE_SOURCE),
        "active_in_trade_decision": bool(item.get("active_in_trade_decision", False)),
        "source_module": item.get("source_module", ""),
        "reason": blocked_reason,
    }


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
    direction_support_rows = [
        row
        for row in skipped_vote_pools
        if str(row.get("selection_layer") or "") == "direction_support_only"
    ]
    direction_items = voters + direction_support_rows
    direction_counts = Counter(_direction_key(row) for row in direction_items)
    max_direction_count = max(direction_counts.values(), default=0)
    exact_ticker_consensus, exact_group, exact_count = _consensus_group(row.get("vote_target") or row.get("top_ticker") for row in voters)
    direction_consensus, direction_group, direction_count = _consensus_group(row.get("direction_state") for row in direction_items)
    no_vote = result_state in {"no_vote", "insufficient_votes"} or total_considered == 0
    divergent = result_state == "divergent"
    protocol_candidate = divergent and direction_consensus and direction_group not in {"", "observation"}
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
        "raw_consensus_state": result_state,
        "exact_ticker_consensus": exact_ticker_consensus,
        "exact_ticker_consensus_group": exact_group,
        "direction_consensus": direction_consensus,
        "direction_consensus_group": direction_group,
        "direction_consensus_strength": _direction_consensus_strength(
            direction_count=direction_count,
            eligible_count=len(direction_items),
            total_considered=total_considered,
        ),
        "actionable_decision_state": _actionable_decision_state(
            result_state=result_state,
            winner_ticker=winner_ticker,
            protocol_candidate=protocol_candidate,
            no_vote=no_vote,
        ),
        "actionable_decision_reason": _actionable_decision_reason(
            result_state=result_state,
            winner_ticker=winner_ticker,
            protocol_candidate=protocol_candidate,
            no_vote=no_vote,
        ),
        "decision_source": _decision_source(
            result_state=result_state,
            winner_vote_count=winner_vote_count,
            eligible_count=eligible_count,
            exact_ticker_consensus=exact_ticker_consensus,
            direction_consensus=direction_consensus,
            protocol_candidate=protocol_candidate,
            no_vote=no_vote,
        ),
        "decision_protocol_used": protocol_used,
        "decision_protocol_reason": _decision_protocol_reason(protocol_candidate=protocol_candidate),
        "protocol_usage_category": _protocol_usage_category(protocol_candidate=protocol_candidate),
        "fake_consensus_flags": _fake_consensus_flags(voters=voters, skipped_vote_pools=skipped_vote_pools, result_state=result_state),
        "consensus_health_bucket": _consensus_health_bucket(
            result_state=result_state,
            consensus_strength=consensus_strength,
            protocol_candidate=protocol_candidate,
            no_vote=no_vote,
            skipped_count=len(skipped_vote_pools),
            total_considered=total_considered,
        ),
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


def _direction_confidence(item: dict[str, Any], *, eligible_vote: bool) -> str:
    if not eligible_vote:
        return "blocked"
    if _number(item.get("rank_score", item.get("score", ""))) >= 0.8:
        return "high"
    if item.get("attack_gate_open") is True or str(item.get("selection_layer", "")).lower() in {"formal_candidate", "market_exposure_tool"}:
        return "medium"
    return "low"


def _data_readiness_state(item: dict[str, Any], *, eligible_vote: bool) -> str:
    missing = item.get("missing_price_tickers") or []
    if missing:
        return "partial"
    decision_layer = str(item.get("decision_layer") or "").lower()
    if "data_readiness" in decision_layer:
        return "blocked"
    return "ready" if eligible_vote else "blocked"


def _shadow_or_diagnostic_flags(item: dict[str, Any]) -> str:
    flags: list[str] = []
    selection_layer = str(item.get("selection_layer") or "").lower()
    decision_layer = str(item.get("decision_layer") or "").lower()
    if "shadow" in selection_layer or "shadow" in decision_layer:
        flags.append("shadow")
    if "diagnostic" in selection_layer or "diagnostic" in decision_layer:
        flags.append("diagnostic")
    if "observation" in selection_layer:
        flags.append("observation_only")
    if item.get("active_in_trade_decision") is False:
        flags.append("not_active_trade_decision")
    return ",".join(dict.fromkeys(flags))


def _consensus_group(values: Any) -> tuple[bool, str, int]:
    cleaned = [str(value or "").strip() for value in values if str(value or "").strip()]
    if not cleaned:
        return False, "", 0
    group, count = Counter(cleaned).most_common(1)[0]
    return count >= 2, group, count


def _direction_consensus_strength(*, direction_count: int, eligible_count: int, total_considered: int) -> str:
    if direction_count <= 0:
        return "none"
    if direction_count >= 3 or (total_considered <= 2 and direction_count == eligible_count):
        return "strong"
    if direction_count >= 2:
        return "weak"
    return "none"


def _actionable_decision_state(*, result_state: str, winner_ticker: str | None, protocol_candidate: bool, no_vote: bool) -> str:
    if result_state == "consensus" and winner_ticker:
        return "exact_consensus_observation"
    if protocol_candidate:
        return "protocol_candidate_diagnostic"
    if no_vote:
        return "data_blocked"
    return "observe_only"


def _actionable_decision_reason(*, result_state: str, winner_ticker: str | None, protocol_candidate: bool, no_vote: bool) -> str:
    if result_state == "consensus" and winner_ticker:
        return "至少 2/3 股票池形成相同標的共識。"
    if protocol_candidate:
        return "標的分歧但方向一致；僅標示為後續決策協議研究候選。"
    if no_vote:
        return "可投票資料不足或池內條件未通過。"
    return "候選標的與方向均未形成可行動共識。"


def _decision_source(
    *,
    result_state: str,
    winner_vote_count: int,
    eligible_count: int,
    exact_ticker_consensus: bool,
    direction_consensus: bool,
    protocol_candidate: bool,
    no_vote: bool,
) -> str:
    if no_vote:
        return "data_blocked"
    if result_state == "consensus" and exact_ticker_consensus:
        return "exact_3_of_3_ticker" if winner_vote_count >= 3 else "exact_2_of_3_ticker"
    if protocol_candidate:
        return "protocol_resolved_divergence"
    if direction_consensus:
        return "direction_3_of_3" if eligible_count >= 3 else "direction_2_of_3"
    return "diagnostic_only"


def _decision_protocol_reason(*, protocol_candidate: bool) -> str:
    if protocol_candidate:
        return "僅供後續 Experiments 驗證；不得取代 2/3 正式表決。"
    return "未啟用決策協議；維持原 2/3 表決輸出。"


def _protocol_usage_category(*, protocol_candidate: bool) -> str:
    return "candidate_not_applied" if protocol_candidate else "not_used"


def _fake_consensus_flags(
    *,
    voters: list[dict[str, Any]],
    skipped_vote_pools: list[dict[str, Any]],
    result_state: str,
) -> list[str]:
    flags: list[str] = []
    if result_state == "consensus" and skipped_vote_pools:
        flags.append("consensus_with_ineligible_pool")
    if result_state == "consensus" and len(voters) < 3:
        flags.append("consensus_from_less_than_three_pools")
    if any(row.get("selection_layer") == "observation_only" for row in skipped_vote_pools):
        flags.append("observation_only_excluded")
    if any(row.get("data_readiness_state") in {"blocked", "partial"} for row in voters + skipped_vote_pools):
        flags.append("data_readiness_issue")
    return flags


def _consensus_health_bucket(
    *,
    result_state: str,
    consensus_strength: str,
    protocol_candidate: bool,
    no_vote: bool,
    skipped_count: int,
    total_considered: int,
) -> str:
    if no_vote or total_considered == 0:
        return "not_evaluable"
    if result_state == "divergent" and not protocol_candidate:
        return "unhealthy"
    if skipped_count:
        return "warning"
    if consensus_strength == "strong":
        return "healthy"
    if consensus_strength == "weak":
        return "acceptable"
    return "warning"


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


def _number(value: object) -> float:
    try:
        text = str(value).replace(",", "").strip()
        return float(text) if text else 0.0
    except (TypeError, ValueError):
        return 0.0


def _health_note(decision_state: str, *, divergent: bool, no_vote: bool) -> str:
    if divergent:
        return "候選分歧，應檢查模型角色與候選設計，不應包裝成明確行動訊號。"
    if no_vote:
        return "可投票資料不足，應先檢查資料完整性與池內 gate。"
    if decision_state == "defensive_or_market_exposure":
        return "共識偏向市場曝險或防守工具，代表模型目前不偏向單一個股攻擊。"
    if decision_state == "strong_consensus":
        return "候選高度一致；仍需保留資料日與風險邊界。"
    if decision_state == "weak_consensus":
        return "形成 2/3 共識；少數池意見仍應保留為風險觀察。"
    if decision_state == "forced_stop":
        return "偵測到強制防守或停損語意，僅作 report-only 風險診斷。"
    return "report-only 共識健康診斷。"
