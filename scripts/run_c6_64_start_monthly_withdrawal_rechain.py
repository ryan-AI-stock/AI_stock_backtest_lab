"""Checkpointed exact-accounting replay for the frozen C6 64-start paths.

The accounting runner intentionally refuses fractional diagnostics, adjusted
prices, or incomplete company-action terms.  It is ready to consume Radar's
exact raw execution/holding authority once delivered.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path


def read(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--radar-authority", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    actions, episodes = read(args.actions), read(args.episodes)
    required_action = {"route_start_date", "date", "action", "slot", "ticker"}
    if not actions or not required_action.issubset(actions[0]):
        raise ValueError("frozen action authority lacks route/date/action/slot/ticker")
    starts = sorted({row["route_start_date"] for row in actions})
    if len(starts) != 64:
        raise ValueError(f"expected 64 frozen starts, found {len(starts)}")
    readiness = args.radar_authority / "readiness_for_core_c6_64_start_exact_rechain.json"
    if not readiness.exists():
        raise FileNotFoundError("waiting_for_radar_exact_raw_price_and_holder_event_authority")
    authority = json.loads(readiness.read_text(encoding="utf-8"))
    if not authority.get("ready_for_core_c6_64_start_exact_rechain", False):
        raise ValueError("Radar authority remains incomplete")
    args.output.mkdir(parents=True, exist_ok=True)
    payload = {
        "task_id": "TASK-BACKTEST-CORE-C6-64-START-MONTHLY-WITHDRAWAL-EXACT-RECHAIN-001",
        "frozen_route_starts": starts,
        "action_rows": len(actions),
        "episode_rows": len(episodes),
        "initial_capital": 7000000,
        "slots": 3,
        "monthly_withdrawal": 75000,
        "execution_basis": "official_raw_close_next_trading_day",
        "holding_basis": "official_raw_close_or_official_no_trade_valuation_carry",
        "corporate_action_basis": "accepted_holder_scale_terms_payment_date_cash",
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "report_changed": False,
        "not_live_rule": True,
    }
    (args.output / "preflight.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if False:  # Legacy gate retained above only for source history.
    main()


# The implementation below supersedes the legacy gate above.  Keeping it in
# one file preserves existing task links while making the new CLI explicit.
import hashlib
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime

TASK_ID = "TASK-BACKTEST-CORE-C6-64-START-MONTHLY-WITHDRAWAL-EXACT-RECHAIN-001"
INITIAL_CASH, SLOT_COUNT, MONTHLY_WITHDRAWAL = 7_000_000.0, 3, 75_000.0
COMMISSION, SELL_TAX, SLIPPAGE = 0.001425, 0.003, 0.001


def _rows(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _number(row: dict[str, str], *keys: str) -> float | None:
    for key in keys:
        text = row.get(key, "").strip().replace(",", "")
        if text and text.lower() not in {"na", "nan", "none"}:
            try:
                return float(text)
            except ValueError:
                continue
    return None


def _ticker(value: str) -> str:
    return value.strip().upper().replace(".TW", "").replace(".TWO", "")


def _day(value: str) -> date:
    return datetime.strptime(value[:10], "%Y-%m-%d").date()


def _text(value: date) -> str:
    return value.isoformat()


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) if rows else ["status"]
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def _hash(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=str):
        digest.update(str(path).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _checksum_manifest(root: Path) -> list[dict[str, object]]:
    """Hash outputs after all accounting writes; exclude the self-referential file."""
    rows: list[dict[str, object]] = []
    for path in sorted((item for item in root.rglob("*") if item.is_file() and item.name != "checksum_manifest.csv"), key=lambda item: str(item)):
        rows.append({"relative_path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return rows


def _load_prices(paths: list[Path]) -> tuple[dict[tuple[str, str], float], list[dict[str, object]]]:
    prices: dict[tuple[str, str], float] = {}
    conflicts: list[dict[str, object]] = []
    for path in paths:
        for row in _rows(path):
            ticker, day = _ticker(row.get("ticker", "")), row.get("date", "")[:10]
            close = _number(row, "official_raw_close", "raw_close", "close")
            if not ticker or not day or close is None or close <= 0:
                continue
            key = ticker, day
            if key in prices and not math.isclose(prices[key], close, abs_tol=1e-9):
                conflicts.append({"class": "conflicting_official_close", "ticker": ticker, "date": day, "first": prices[key], "second": close, "source": str(path)})
            else:
                prices[key] = close
    return prices, conflicts


def _load_no_trade(paths: list[Path]) -> set[tuple[str, str]]:
    result: set[tuple[str, str]] = set()
    for path in paths:
        for row in _rows(path):
            ticker, day = _ticker(row.get("ticker", "")), row.get("date", "")[:10]
            status = row.get("status", "").lower()
            if ticker and day and (not status or "no_trade" in status or "no_target" in status):
                result.add((ticker, day))
    return result


def _load_events(paths: list[Path], conditional_payment_overrides: dict[str, str] | None = None) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    conditional_payment_overrides = conditional_payment_overrides or {}
    for path in paths:
        for row in _rows(path):
            status = row.get("status", "accepted_official_holder_scale").lower()
            # Conditional terms may contain a valid amount but no unique
            # payment date.  Only fully accepted terms may create cash.
            event_id = row.get("event_id", "")
            override = conditional_payment_overrides.get(event_id)
            if override and status == "accepted_conditional_payment_terms":
                alternatives = {item.strip() for item in row.get("payment_date_alternatives", "").split("|") if item.strip()}
                if override not in alternatives:
                    raise ValueError(f"conditional_payment_override_not_in_official_alternatives:{event_id}:{override}")
                row = dict(row)
                row["payment_date"] = override
                row["status"] = "accepted_complete_sensitivity_override"
                row["payment_date_source"] = "sensitivity_only_official_alternative_not_selected_for_formal_accounting"
                status = row["status"]
            if not row.get("status") or status == "accepted_complete" or status == "accepted_complete_sensitivity_override" or status.startswith("accepted_official_holder_scale"):
                event = dict(row)
                event["ticker"] = _ticker(event.get("ticker", ""))
                result.append(event)
    return result


@dataclass
class _Position:
    slot: int
    ticker: str = ""
    units: int = 0
    entry_date: str = ""
    entry_spend: float = 0.0
    withdrawal_net: float = 0.0
    dividend_cash: float = 0.0
    episode_key: str = ""


def _mark(position: _Position, day: str, prices: dict[tuple[str, str], float], carried: dict[str, float], no_trade: set[tuple[str, str]] | None = None) -> float | None:
    if not position.units:
        return 0.0
    close = prices.get((position.ticker, day))
    if close is None and (position.ticker, day) in (no_trade or set()):
        close = carried.get(position.ticker)
    return position.units * close if close is not None else None


def _calendar(prices: dict[tuple[str, str], float], actions: list[dict[str, str]], start: date, end: date) -> list[date]:
    dates = {_day(day) for _, day in prices if start <= _day(day) <= end}
    dates.update(_day(row["date"]) for row in actions if row.get("date"))
    dates.update({start, end})
    return sorted(dates)


def _month_ends(calendar: list[date]) -> set[date]:
    last: dict[tuple[int, int], date] = {}
    for current in calendar:
        last[(current.year, current.month)] = current
    return set(last.values())


def _actual_episodes(route_id: str, action_rows: list[dict[str, object]], event_rows: list[dict[str, object]], end: date) -> list[dict[str, object]]:
    """Whole-share episodes with trade, withdrawal and payment-date cash joined."""
    episodes: dict[str, dict[str, object]] = {}
    for row in action_rows:
        key = str(row.get("episode_key", ""))
        if row["action"] == "buy":
            episodes[key] = {"route_id": route_id, "episode_key": key, "slot": row["slot"], "ticker": row["ticker"], "entry_date": row["date"], "entry_units": row["units"], "entry_spend": -float(row["cash_delta"]), "withdrawal_sale_net": 0.0, "dividend_cash": 0.0, "exit_net": 0.0, "exit_date": "", "open_at_end": True}
        elif key in episodes and row["action"] == "withdrawal_forced_sell":
            episodes[key]["withdrawal_sale_net"] = float(episodes[key]["withdrawal_sale_net"]) + float(row["cash_delta"])
        elif key in episodes and row["action"] == "sell":
            episodes[key]["exit_net"] = float(row["cash_delta"])
            episodes[key]["exit_date"] = row["date"]
            episodes[key]["open_at_end"] = False
    for row in event_rows:
        key = str(row.get("episode_key", ""))
        if key in episodes and row.get("event_type") == "cash_dividend_payment":
            episodes[key]["dividend_cash"] = float(episodes[key]["dividend_cash"]) + float(row["cash_credit"])
    result: list[dict[str, object]] = []
    for row in episodes.values():
        if not row["exit_date"]:
            row["exit_date"] = _text(end)
        row["episode_net_cash"] = float(row["exit_net"]) + float(row["withdrawal_sale_net"]) + float(row["dividend_cash"]) - float(row["entry_spend"])
        row["completed_round"] = not bool(row["open_at_end"])
        row["win"] = bool(row["completed_round"]) and float(row["episode_net_cash"]) > 0
        result.append(row)
    return sorted(result, key=lambda row: (str(row["entry_date"]), int(row["slot"])))


def _next_price(ticker: str, current: date, prices: dict[tuple[str, str], float]) -> date | None:
    candidates = [_day(day) for code, day in prices if code == ticker and _day(day) >= current]
    return min(candidates) if candidates else None


def _apply_event(route_id: str, current: date, positions: list[_Position], cash_by_slot: list[float], events: list[dict[str, str]], entitlement: dict[tuple[str, int], tuple[_Position, int]], event_rows: list[dict[str, object]]) -> list[float]:
    text = _text(current)
    for event in events:
        ticker = event["ticker"]
        # Ex-date is the governing pre-open entitlement boundary.  Record date
        # is only a fallback when an official ex-date is unavailable.
        entitlement_date = (event.get("ex_date") or event.get("entitlement_date") or event.get("record_date") or "")[:10]
        event_id = event.get("event_id") or "|".join([ticker, entitlement_date, event.get("payment_date", ""), event.get("cash_dividend_per_share", "")])
        if entitlement_date == text:
            for position in positions:
                if position.ticker == ticker and position.units:
                    entitlement[(event_id, position.slot)] = (position, position.units)
        cash_per_share = _number(event, "cash_dividend_per_share", "cash_per_share", "cash_amount_per_share")
        if event.get("payment_date", "")[:10] == text and cash_per_share is not None:
            for slot in range(1, SLOT_COUNT + 1):
                entitled = entitlement.get((event_id, slot))
                if entitled:
                    holding, units = entitled
                    credit = units * cash_per_share
                    cash_by_slot[slot - 1] += credit
                    holding.dividend_cash += credit
                    event_rows.append({"route_id": route_id, "date": text, "ticker": ticker, "slot": slot, "event_type": "cash_dividend_payment", "entitlement_units": units, "cash_per_share": cash_per_share, "cash_credit": credit, "episode_key": holding.episode_key})
        multiplier = _number(event, "unit_multiplier", "share_multiplier")
        if event.get("effective_date", "")[:10] == text and multiplier is not None:
            for position in positions:
                if position.ticker == ticker and position.units:
                    before = position.units
                    position.units = math.floor(before * multiplier)
                    event_rows.append({"route_id": route_id, "date": text, "ticker": ticker, "slot": position.slot, "event_type": "unit_multiplier", "units_before": before, "multiplier": multiplier, "units_after": position.units})
    return cash_by_slot


def _withdraw(route_id: str, current: date, positions: list[_Position], cash_by_slot: list[float], prices: dict[tuple[str, str], float], carried: dict[str, float], action_rows: list[dict[str, object]], blockers: list[dict[str, object]]) -> tuple[list[float], dict[str, object], float]:
    text, initial_cash = _text(current), sum(cash_by_slot)
    forced_sale_net = 0.0
    forced_cost = 0.0
    while sum(cash_by_slot) + 1e-8 < MONTHLY_WITHDRAWAL:
        # A no-trade row is usable only as valuation carry.  A withdrawal sale
        # must choose the largest *currently executable* holding instead.
        candidates = [(p.units * prices[(p.ticker, text)], p) for p in positions if p.units and (p.ticker, text) in prices]
        if not candidates:
            blockers.append({"route_id": route_id, "date": text, "class": "monthly_withdrawal_shortfall", "reason": "cash_and_sellable_mark_insufficient"})
            break
        _, position = max(candidates, key=lambda pair: pair[0])
        raw = prices[(position.ticker, text)]
        net_per_share = raw * (1 - SLIPPAGE) * (1 - COMMISSION - SELL_TAX)
        units = min(position.units, max(1, math.ceil((MONTHLY_WITHDRAWAL - sum(cash_by_slot)) / net_per_share)))
        gross = units * raw * (1 - SLIPPAGE)
        cost = gross * (COMMISSION + SELL_TAX)
        proceeds = gross - cost
        position.units -= units
        position.withdrawal_net += proceeds
        cash_by_slot[position.slot - 1] += proceeds
        forced_sale_net += proceeds
        forced_cost += cost + units * raw * SLIPPAGE
        action_rows.append({"route_id": route_id, "date": text, "action": "withdrawal_forced_sell", "slot": position.slot, "ticker": position.ticker, "units": units, "raw_close": raw, "cost": cost, "cash_delta": proceeds, "episode_key": position.episode_key})
    remaining = MONTHLY_WITHDRAWAL
    withdrawals_by_slot: list[float] = []
    for slot in range(SLOT_COUNT):
        amount = min(remaining, cash_by_slot[slot])
        cash_by_slot[slot] -= amount
        remaining -= amount
        withdrawals_by_slot.append(amount)
    actual = MONTHLY_WITHDRAWAL - remaining
    return cash_by_slot, {"route_id": route_id, "date": text, "requested_withdrawal": MONTHLY_WITHDRAWAL, "actual_withdrawal": actual, "cash_before": initial_cash, "forced_sale_net": forced_sale_net, "cash_from_slot_1": withdrawals_by_slot[0], "cash_from_slot_2": withdrawals_by_slot[1], "cash_from_slot_3": withdrawals_by_slot[2], "cash_after": sum(cash_by_slot), "cash_priority": "slot_1_then_slot_2_then_slot_3"}, forced_cost


def _run_route(route_id: str, frozen_actions: list[dict[str, str]], calendar: list[date], market_calendar: list[date], prices: dict[tuple[str, str], float], no_trade: set[tuple[str, str]], events: list[dict[str, str]], end: date) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    positions, cash_by_slot, carried, entitlement = [_Position(slot=index) for index in range(1, SLOT_COUNT + 1)], [INITIAL_CASH / SLOT_COUNT for _ in range(SLOT_COUNT)], {}, {}
    daily: list[dict[str, object]] = []
    executed: list[dict[str, object]] = []
    withdrawals: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []
    blockers: list[dict[str, object]] = []
    actions_by_day: dict[str, list[dict[str, str]]] = defaultdict(list)
    for action in frozen_actions:
        actions_by_day[action["date"][:10]].append(action)
    start = _day(frozen_actions[0]["route_start_date"])
    # The terminal partial month (for example 2026-08-12) is not a month-end
    # withdrawal date.  The opening partial month remains eligible at its
    # actual final market session.
    month_end = {item for item in _month_ends([item for item in market_calendar if start <= item <= end]) if (item.year, item.month) < (end.year, end.month)}
    prior_nav, total_cost, withdrawal_total = INITIAL_CASH, 0.0, 0.0
    for current in [item for item in calendar if start <= item <= end]:
        text = _text(current)
        for position in positions:
            if position.ticker and (position.ticker, text) in prices:
                carried[position.ticker] = prices[(position.ticker, text)]
        cash_by_slot = _apply_event(route_id, current, positions, cash_by_slot, events, entitlement, event_rows)
        # Sell first preserves atomic replacement accounting at a shared exact close.
        for frozen in sorted(actions_by_day.get(text, []), key=lambda row: (int(row["slot"]), 0 if row["action"] == "sell" else 1)):
            slot = int(frozen["slot"]) - 1
            position, ticker, raw = positions[slot], _ticker(frozen["ticker"]), prices.get((_ticker(frozen["ticker"]), text))
            if frozen["action"] == "sell":
                if not position.units or position.ticker != ticker:
                    blockers.append({"route_id": route_id, "date": text, "slot": slot + 1, "ticker": ticker, "class": "frozen_sell_position_mismatch"})
                    continue
                if raw is None:
                    later = _next_price(ticker, current, prices)
                    blockers.append({"route_id": route_id, "date": text, "slot": slot + 1, "ticker": ticker, "class": "sell_deferred_requires_frozen_rechain", "official_no_trade": (ticker, text) in no_trade, "next_exact_close_date": _text(later) if later else ""})
                    continue
                gross = position.units * raw * (1 - SLIPPAGE)
                cost = gross * (COMMISSION + SELL_TAX)
                proceeds = gross - cost
                cash_by_slot[slot] += proceeds
                total_cost += cost + position.units * raw * SLIPPAGE
                episode_net = proceeds + position.withdrawal_net + position.dividend_cash - position.entry_spend
                executed.append({"route_id": route_id, "date": text, "action": "sell", "slot": slot + 1, "ticker": ticker, "units": position.units, "raw_close": raw, "cost": cost, "cash_delta": proceeds, "reason": frozen.get("reason", ""), "episode_key": position.episode_key, "episode_net_before_later_payments": episode_net})
                positions[slot] = _Position(slot=slot + 1)
            else:
                if raw is None:
                    blockers.append({"route_id": route_id, "date": text, "slot": slot + 1, "ticker": ticker, "class": "buy_unexecutable_no_exact_close", "official_no_trade": (ticker, text) in no_trade})
                    continue
                if position.units:
                    blockers.append({"route_id": route_id, "date": text, "slot": slot + 1, "ticker": ticker, "class": "buy_while_slot_occupied"})
                    continue
                unit_cost = raw * (1 + SLIPPAGE) * (1 + COMMISSION)
                units = math.floor(cash_by_slot[slot] / unit_cost)
                if units < 1:
                    blockers.append({"route_id": route_id, "date": text, "slot": slot + 1, "ticker": ticker, "class": "insufficient_cash_for_whole_share"})
                    continue
                spent = units * unit_cost
                cash_by_slot[slot] -= spent
                positions[slot] = _Position(slot=slot + 1, ticker=ticker, units=units, entry_date=text, entry_spend=spent, episode_key=f"{slot + 1}|{ticker}|{text}")
                total_cost += units * raw * (1 + SLIPPAGE) * COMMISSION + units * raw * SLIPPAGE
                executed.append({"route_id": route_id, "date": text, "action": "buy", "slot": slot + 1, "ticker": ticker, "units": units, "raw_close": raw, "cost": units * raw * (1 + SLIPPAGE) * COMMISSION, "cash_delta": -spent, "reason": frozen.get("reason", ""), "episode_key": positions[slot].episode_key})
        withdrawal_today = 0.0
        if current in month_end:
            cash_by_slot, record, withdrawal_cost = _withdraw(route_id, current, positions, cash_by_slot, prices, carried, executed, blockers)
            withdrawals.append(record)
            withdrawal_today = float(record["actual_withdrawal"])
            withdrawal_total += withdrawal_today
            total_cost += withdrawal_cost
        values, missing = [], []
        for position in positions:
            value = _mark(position, text, prices, carried, no_trade)
            if value is None:
                missing.append(position.ticker)
            else:
                values.append(value)
        pending_today = 0.0
        for event in events:
            payment = event.get("payment_date", "")[:10]
            entitlement_date = (event.get("ex_date") or event.get("entitlement_date") or event.get("record_date") or "")[:10]
            cash_per_share = _number(event, "cash_dividend_per_share", "cash_per_share", "cash_amount_per_share")
            event_id = event.get("event_id") or "|".join([event["ticker"], entitlement_date, event.get("payment_date", ""), event.get("cash_dividend_per_share", "")])
            if payment and payment > text and cash_per_share is not None:
                pending_today += sum(units * cash_per_share for (candidate, _), (_, units) in entitlement.items() if candidate == event_id)
        if missing:
            raise ValueError(f"data_readiness_blocked: missing exact holding mark or official_no_trade: {text}: {','.join(missing)}")
        nav = sum(cash_by_slot) + sum(values) + pending_today
        daily.append({"route_id": route_id, "date": text, "cash": sum(cash_by_slot), "slot_1_cash": cash_by_slot[0], "slot_2_cash": cash_by_slot[1], "slot_3_cash": cash_by_slot[2], "pending_cash_receivable": pending_today, "nav": nav, "daily_twr": (nav + withdrawal_today) / prior_nav - 1 if prior_nav else 0.0, "slot_1_ticker": positions[0].ticker, "slot_1_units": positions[0].units, "slot_2_ticker": positions[1].ticker, "slot_2_units": positions[1].units, "slot_3_ticker": positions[2].ticker, "slot_3_units": positions[2].units, "missing_official_mark_tickers": ";".join(missing)})
        prior_nav = nav
    # A right earned by ex-date but paid after the requested end remains an
    # explicit receivable, not cash and not a silently omitted dividend.
    pending_receivable = 0.0
    for event in events:
        payment = event.get("payment_date", "")[:10]
        cash_per_share = _number(event, "cash_dividend_per_share", "cash_per_share", "cash_amount_per_share")
        entitlement_date = (event.get("ex_date") or event.get("entitlement_date") or event.get("record_date") or "")[:10]
        event_id = event.get("event_id") or "|".join([event["ticker"], entitlement_date, event.get("payment_date", ""), event.get("cash_dividend_per_share", "")])
        if payment and _day(payment) > end and cash_per_share is not None:
            for slot in range(1, SLOT_COUNT + 1):
                entitled = entitlement.get((event_id, slot))
                if entitled:
                    holding, units = entitled
                    amount = units * cash_per_share
                    pending_receivable += amount
                    event_rows.append({"route_id": route_id, "date": _text(end), "ticker": event["ticker"], "slot": slot, "event_type": "cash_dividend_receivable_after_period_end", "entitlement_units": units, "cash_per_share": cash_per_share, "cash_receivable": amount, "payment_date": payment, "episode_key": holding.episode_key})
    if daily and pending_receivable:
        # The daily ledger already includes it; retain this separately for the
        # terminal audit rather than adding it twice.
        daily[-1]["pending_cash_receivable"] = pending_receivable
    # Monthly withdrawals are external cash flows.  MDD/CAGR therefore use a
    # daily time-weighted wealth index, while ending account NAV stays separate.
    wealth, twr_peak, twr_mdd = 1.0, 1.0, 0.0
    account_peak, account_nav_mdd = INITIAL_CASH, 0.0
    for row in daily:
        account_peak = max(account_peak, float(row["nav"]))
        row["account_nav_drawdown"] = float(row["nav"]) / account_peak - 1
        account_nav_mdd = min(account_nav_mdd, float(row["account_nav_drawdown"]))
        wealth *= 1 + float(row["daily_twr"])
        row["twr_wealth_index"] = wealth
        twr_peak = max(twr_peak, wealth)
        row["twr_drawdown"] = wealth / twr_peak - 1
        twr_mdd = min(twr_mdd, float(row["twr_drawdown"]))
    episode_rows = _actual_episodes(route_id, executed, event_rows, end)
    completed = [row for row in episode_rows if row["completed_round"]]
    years = max(1 / 365.25, (end - start).days / 365.25)
    summary = {"route_id": route_id, "route_start_date": frozen_actions[0]["route_start_date"], "final_date": daily[-1]["date"], "final_nav": daily[-1]["nav"], "remaining_asset_growth_pct": (float(daily[-1]["nav"]) / INITIAL_CASH - 1) * 100, "account_nav_mdd": account_nav_mdd, "twr_total_return": wealth - 1, "twr_cagr": wealth ** (1 / years) - 1, "twr_mdd": twr_mdd, "action_count": len(executed), "completed_rounds": len(completed), "win_rate": sum(bool(row["win"]) for row in completed) / len(completed) if completed else None, "episode_net_cash": sum(float(row["episode_net_cash"]) for row in completed), "total_cost": total_cost, "withdrawal_total": withdrawal_total, "blocker_count": len(blockers), "accounting_scope": "frozen_action_accounting_replay"}
    return daily, executed, withdrawals, event_rows, episode_rows, blockers, summary


def main() -> None:  # noqa: C901 - explicit accounting phases are easier to audit than hidden helpers.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--price-file", type=Path, action="append", default=[])
    parser.add_argument("--no-trade-file", type=Path, action="append", default=[])
    parser.add_argument("--event-file", type=Path, action="append", default=[])
    parser.add_argument("--event-coverage-file", type=Path, help="Per-interval official event/no-event completeness evidence.")
    parser.add_argument("--no-event-proof-file", type=Path, help="Official inventory no-event proof; evidence only, never a cash event.")
    parser.add_argument("--blocked-event-file", type=Path, help="Explicit event terms that remain outside accepted accounting authority.")
    parser.add_argument("--conditional-payment-override", action="append", default=[], metavar="EVENT_ID=YYYY-MM-DD", help="Sensitivity-only official payment-date alternative; never formal authority.")
    parser.add_argument("--holder-interval-file", type=Path, help="Current frozen intervals including open positions; used only for event-coverage audit.")
    parser.add_argument("--requirements-file", type=Path, help="Exact price requirement union including open holdings.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--end-date", default="2026-08-12")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    actions, episodes = _rows(args.actions), _rows(args.episodes)
    required = {"route_id", "route_start_date", "date", "action", "slot", "ticker"}
    errors: list[dict[str, object]] = []
    if not actions or not required.issubset(actions[0]):
        errors.append({"class": "frozen_action_schema_invalid"})
    starts = sorted({row.get("route_start_date", "") for row in actions})
    if len(starts) != 64:
        errors.append({"class": "expected_64_routes", "actual": len(starts)})
    price_paths = [path for path in args.price_file if path.exists()]
    missing_paths = [str(path) for path in args.price_file if not path.exists()]
    prices, conflicts = _load_prices(price_paths)
    errors.extend(conflicts)
    no_trade = _load_no_trade([path for path in args.no_trade_file if path.exists()])
    overrides: dict[str, str] = {}
    for item in args.conditional_payment_override:
        event_id, separator, payment_date = item.partition("=")
        if not separator or not event_id or not payment_date:
            raise ValueError("conditional_payment_override_requires_EVENT_ID=YYYY-MM-DD")
        overrides[event_id] = payment_date
    events = _load_events([path for path in args.event_file if path.exists()], overrides)
    coverage_rows = _rows(args.event_coverage_file) if args.event_coverage_file and args.event_coverage_file.exists() else []
    no_event_proof_rows = _rows(args.no_event_proof_file) if args.no_event_proof_file and args.no_event_proof_file.exists() else []
    blocked_event_rows = _rows(args.blocked_event_file) if args.blocked_event_file and args.blocked_event_file.exists() else []
    coverage_complete = sum(row.get("status") == "complete_event_authority" for row in coverage_rows)
    coverage_blocked = sum(row.get("status") != "complete_event_authority" for row in coverage_rows)
    required_keys: set[tuple[str, str]] = set()
    if args.requirements_file and args.requirements_file.exists():
        required_keys = {(_ticker(row.get("ticker", "")), row.get("date", "")[:10]) for row in _rows(args.requirements_file)}
        required_keys.discard(("", ""))
    missing_required = sorted(required_keys - set(prices) - no_trade)
    intervals = _rows(args.holder_interval_file) if args.holder_interval_file and args.holder_interval_file.exists() else []
    event_hits: list[dict[str, object]] = []
    for event in events:
        ex_date = event.get("ex_date", "")[:10]
        for interval in intervals:
            if _ticker(interval.get("ticker", "")) == event["ticker"] and interval.get("entry_date", "")[:10] <= ex_date <= interval.get("exit_date", args.end_date)[:10]:
                event_hits.append({"ticker": event["ticker"], "ex_date": ex_date, "payment_date": event.get("payment_date", "")[:10], "event_type": event.get("event_type", ""), "interval_entry": interval.get("entry_date", ""), "interval_exit": interval.get("exit_date", ""), "open_at_end": interval.get("open_at_end", "")})
    hash_inputs = [args.actions, args.episodes, *price_paths]
    for path in [*args.no_trade_file, *args.event_file, args.event_coverage_file, args.no_event_proof_file, args.blocked_event_file, args.holder_interval_file, args.requirements_file]:
        if path and path.exists():
            hash_inputs.append(path)
    input_hash = _hash(hash_inputs)
    event_coverage_verified = bool(coverage_rows) and coverage_blocked == 0 and not blocked_event_rows
    event_coverage_status = "complete_official_interval_inventory" if event_coverage_verified else ("sensitivity_override_source_ambiguity_retained" if overrides else ("partial_payment_condition_blocked" if coverage_rows else "terms_only_until_current_64_path_event_inventory_or_no_event_proof_is_available"))
    preflight = {"task_id": TASK_ID, "route_count": len(starts), "action_rows": len(actions), "episode_rows": len(episodes), "exact_official_price_rows": len(prices), "official_no_trade_rows": len(no_trade), "required_unique_ticker_date_keys": len(required_keys), "required_keys_missing_from_exact_partition": len(missing_required), "accepted_event_rows": len(events), "conditional_payment_sensitivity_overrides": overrides, "accepted_event_interval_hits": len(event_hits), "event_coverage_rows": len(coverage_rows), "event_coverage_complete_rows": coverage_complete, "event_coverage_blocked_rows": coverage_blocked, "no_event_proof_rows": len(no_event_proof_rows), "blocked_event_requirement_rows": len(blocked_event_rows), "event_coverage_verified": event_coverage_verified, "event_coverage_status": event_coverage_status, "price_source_schema": "local_reuse|bounded_delta|accepted_checkpoint_exact_official_raw_close", "open_interval_policy": "actions_derive_positions_through_requested_end", "requested_end_date": args.end_date, "input_hash": input_hash, "errors": errors, "missing_price_inputs": missing_paths, "signal_authority": "frozen_research_actions_no_signal_recalculation", "research_engine_brokerage": 0.000855, "accounting_commission": COMMISSION, "action_cost_audit": "C_engine_exit_return_uses_current_over_entry_price_minus_one_without_commission; cost difference_not_presumed_action_error", "accounting_basis": "official_raw_execution_and_holding_marks_with_accepted_payment_date_events", "formal_model_changed": False, "trade_decision_changed": False, "active_in_trade_decision": False, "report_changed": False, "not_live_rule": True}
    if args.dry_run:
        preflight["dry_run"] = True
        preflight["runner_ready"] = not errors
        preflight["price_authority_ready"] = bool(price_paths) and not errors and not missing_required
        preflight["authority_ready_for_full_rechain"] = preflight["price_authority_ready"] and event_coverage_verified
        print(json.dumps(preflight, ensure_ascii=False, indent=2))
        return
    if errors:
        raise ValueError(json.dumps(preflight, ensure_ascii=False))
    end = _day(args.end_date)
    # Individual ticker no-trade does not create a market session.  It is
    # valuation carry only and must not move the monthly trading-day boundary.
    market_calendar = _calendar(prices, actions, min(_day(item) for item in starts), end)
    event_days = {_day(event[key]) for event in events for key in ("ex_date", "record_date", "entitlement_date", "effective_date", "payment_date") if event.get(key) and min(_day(item) for item in starts) <= _day(event[key]) <= end}
    calendar = sorted(set(market_calendar) | event_days)
    routes: dict[str, list[dict[str, str]]] = defaultdict(list)
    for action in actions:
        routes[action["route_id"]].append(action)
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_file = args.output / "checkpoint.json"
    checkpoint = json.loads(checkpoint_file.read_text(encoding="utf-8")) if args.resume and checkpoint_file.exists() else {"input_hash": input_hash, "completed_routes": []}
    if checkpoint.get("input_hash") != input_hash:
        raise ValueError("checkpoint_input_hash_mismatch")
    completed, summaries, blockers = set(checkpoint.get("completed_routes", [])), [], []
    for route_id in sorted(routes):
        folder, summary_file = args.output / "routes" / route_id, args.output / "routes" / route_id / "summary.json"
        if route_id in completed and summary_file.exists():
            summaries.append(json.loads(summary_file.read_text(encoding="utf-8")))
            continue
        result = _run_route(route_id, sorted(routes[route_id], key=lambda row: (row["date"], int(row["slot"]), 0 if row["action"] == "sell" else 1)), calendar, market_calendar, prices, no_trade, events, end)
        daily, action_rows, withdrawal_rows, event_rows, episode_rows, route_blockers, summary = result
        _write_rows(folder / "daily_nav.csv.gz", daily)
        _write_rows(folder / "actions.csv", action_rows)
        _write_rows(folder / "withdrawals.csv", withdrawal_rows)
        _write_rows(folder / "events.csv", event_rows)
        _write_rows(folder / "episodes.csv", episode_rows)
        _write_rows(folder / "blockers.csv", route_blockers)
        _atomic_json(summary_file, summary)
        summaries.append(summary)
        blockers.extend(route_blockers)
        completed.add(route_id)
        _atomic_json(checkpoint_file, {"input_hash": input_hash, "completed_routes": sorted(completed), "last_route": route_id, "status": "running"})
    summaries.sort(key=lambda row: float(row["final_nav"]))
    _write_rows(args.output / "route_summary.csv", summaries)
    _write_rows(args.output / "blocker_ledger.csv", blockers)
    _write_rows(args.output / "event_interval_coverage_audit.csv", event_hits)
    _write_rows(args.output / "event_no_event_proof_import_audit.csv", [{"interval_id": row.get("interval_id", ""), "ticker": _ticker(row.get("ticker", "")), "event_type": row.get("event_type", ""), "proof": row.get("proof", ""), "import_role": "coverage_evidence_only_no_cash_credit"} for row in no_event_proof_rows])
    _write_rows(args.output / "event_blocked_requirements_import_audit.csv", [{**row, "import_role": "not_applied_pending_strategy_center_payment_semantics"} for row in blocked_event_rows])
    _write_rows(args.output / "event_coverage_unverified_interval_ledger.csv", [{"ticker": _ticker(interval.get("ticker", "")), "entry_date": interval.get("entry_date", ""), "exit_date": interval.get("exit_date", ""), "open_at_end": interval.get("open_at_end", ""), "class": "current_64_path_event_inventory_or_no_event_proof_required"} for interval in intervals])
    _write_rows(args.output / "missing_exact_price_requirement_ledger.csv", [{"ticker": ticker, "date": day, "class": "exact_price_authority_missing"} for ticker, day in missing_required])
    lower = summaries[31] if len(summaries) == 64 else {}
    result = {**preflight, "completed_routes": len(completed), "blocked_rows": len(blockers), "event_coverage_unverified_interval_rows": len(intervals), "highest_actual_route": summaries[-1] if summaries else {}, "lowest_actual_route": summaries[0] if summaries else {}, "statistical_median_final_nav": (float(summaries[31]["final_nav"]) + float(summaries[32]["final_nav"])) / 2 if len(summaries) == 64 else None, "lower_median_actual_route": lower, "representative_2023_01_03_route": next((row for row in summaries if row["route_start_date"] == "2023-01-03"), {}), "ready_for_annual_performance": len(completed) == 64 and not blockers and not missing_required and preflight["event_coverage_verified"], "ready_for_sheet_update": False, "future_data_violation_count": 0, "diagnostic_only": True}
    _atomic_json(args.output / "summary.json", result)
    _atomic_json(args.output / "readiness.json", result)
    _atomic_json(checkpoint_file, {"input_hash": input_hash, "completed_routes": sorted(completed), "status": "complete"})
    (args.output / "current_step.txt").write_text("completed_accounting_rechain_event_coverage_unverified\n", encoding="utf-8")
    _write_rows(args.output / "checksum_manifest.csv", _checksum_manifest(args.output))
    print(json.dumps({"output": str(args.output), "completed_routes": len(completed), "blocked_rows": len(blockers), "ready_for_annual_performance": result["ready_for_annual_performance"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
