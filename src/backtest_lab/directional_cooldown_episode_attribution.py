"""Attribute the P2 CD7/CD10 result against the current CD7/CD7 reference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


TASK_ID = "TASK-BACKTEST-EXPERIMENTS-P1-P2-DIRECTIONAL-COOLDOWN-EPISODE-ATTRIBUTION-001"
PERIOD = "P2"
SIGNAL_PAIR = "S0_RETURN_BASE"
COST_BASIS = "after_cost_10bp_side"
REFERENCE_SCENARIO = "L7_7_REFERENCE"
CHALLENGER_SCENARIO = "L7_10_REENTRY10"
INITIAL_CAPITAL = 1_000_000.0
MATERIAL_LOG_DELTA = 1e-5


def closed_episodes(trades: pd.DataFrame, scenario: str) -> pd.DataFrame:
    selected = trades.loc[
        (trades["period"] == PERIOD)
        & (trades["signal_pair"] == SIGNAL_PAIR)
        & (trades["cooldown_scenario"] == scenario)
        & (trades["cost_basis"] == COST_BASIS)
    ].sort_values("execution_index")
    if selected.empty:
        raise ValueError(f"No trades found for {scenario}")
    sides = selected["side"].str.lower().tolist()
    if sides != [side for _ in range(len(sides) // 2) for side in ("buy", "sell")]:
        raise ValueError(f"Trades for {scenario} are not alternating closed episodes")

    buys = selected.loc[selected["side"].str.lower() == "buy"].reset_index(drop=True)
    sells = selected.loc[selected["side"].str.lower() == "sell"].reset_index(drop=True)
    if len(buys) != len(sells):
        raise ValueError(f"Open episode found for {scenario}")

    start_nav = pd.to_numeric(buys["cash_before"], errors="raise")
    end_nav = pd.to_numeric(sells["cash_after"], errors="raise")
    factor = end_nav / start_nav
    return pd.DataFrame(
        {
            "episode": np.arange(1, len(buys) + 1),
            "buy_signal_date": buys["signal_date"].to_numpy(),
            "buy_execution_date": buys["execution_date"].to_numpy(),
            "buy_execution_index": pd.to_numeric(buys["execution_index"], errors="raise").to_numpy(),
            "sell_signal_date": sells["signal_date"].to_numpy(),
            "sell_execution_date": sells["execution_date"].to_numpy(),
            "sell_execution_index": pd.to_numeric(sells["execution_index"], errors="raise").to_numpy(),
            "holding_td": (
                pd.to_numeric(sells["execution_index"], errors="raise").to_numpy()
                - pd.to_numeric(buys["execution_index"], errors="raise").to_numpy()
            ),
            "start_nav": start_nav.to_numpy(),
            "end_nav": end_nav.to_numpy(),
            "wealth_factor": factor.to_numpy(),
            "episode_return_pct": (factor.to_numpy() - 1.0) * 100.0,
        }
    )


def pair_episode_attribution(
    reference: pd.DataFrame,
    challenger: pd.DataFrame,
    *,
    reference_final_equity: float,
    initial_capital: float = INITIAL_CAPITAL,
) -> pd.DataFrame:
    if len(reference) != len(challenger):
        raise ValueError("Episode counts differ; ordinal attribution would be invalid")
    paired = reference.add_prefix("reference_").merge(
        challenger.add_prefix("challenger_"),
        left_on="reference_episode",
        right_on="challenger_episode",
        validate="one_to_one",
    )
    paired["buy_delay_td"] = (
        paired["challenger_buy_execution_index"] - paired["reference_buy_execution_index"]
    )
    paired["sell_delay_td"] = (
        paired["challenger_sell_execution_index"] - paired["reference_sell_execution_index"]
    )
    paired["wealth_factor_ratio"] = (
        paired["challenger_wealth_factor"] / paired["reference_wealth_factor"]
    )
    paired["log_wealth_delta"] = np.log(paired["wealth_factor_ratio"])
    paired["single_episode_substitution_delta_twd"] = reference_final_equity * (
        paired["wealth_factor_ratio"] - 1.0
    )
    paired["single_episode_substitution_delta_return_pp"] = (
        paired["single_episode_substitution_delta_twd"] / initial_capital * 100.0
    )
    paired["challenger_minus_reference_episode_return_pp"] = (
        paired["challenger_episode_return_pct"] - paired["reference_episode_return_pct"]
    )
    paired["effect_direction"] = np.select(
        [
            paired["log_wealth_delta"] < -MATERIAL_LOG_DELTA,
            paired["log_wealth_delta"] > MATERIAL_LOG_DELTA,
        ],
        ["challenger_hurt", "challenger_helped"],
        default="unchanged",
    )
    return paired


def reconciliation(
    reference: pd.DataFrame,
    challenger: pd.DataFrame,
    paired: pd.DataFrame,
    *,
    reference_final_equity: float,
    challenger_final_equity: float,
    initial_capital: float = INITIAL_CAPITAL,
) -> pd.DataFrame:
    ref_product = float(reference["wealth_factor"].prod())
    alt_product = float(challenger["wealth_factor"].prod())
    rebuilt_ref = initial_capital * ref_product
    rebuilt_alt = initial_capital * alt_product
    rows = [
        {"check": "reference_episode_product_to_final_nav", "reported": reference_final_equity, "rebuilt": rebuilt_ref},
        {"check": "challenger_episode_product_to_final_nav", "reported": challenger_final_equity, "rebuilt": rebuilt_alt},
        {
            "check": "challenger_reference_final_nav_ratio",
            "reported": challenger_final_equity / reference_final_equity,
            "rebuilt": alt_product / ref_product,
        },
        {
            "check": "sum_log_episode_delta",
            "reported": float(np.log(challenger_final_equity / reference_final_equity)),
            "rebuilt": float(paired["log_wealth_delta"].sum()),
        },
    ]
    out = pd.DataFrame(rows)
    out["absolute_error"] = (out["reported"] - out["rebuilt"]).abs()
    out["pass"] = out["absolute_error"] < 1e-6
    return out


def path_divergence(daily: pd.DataFrame) -> pd.DataFrame:
    selected = daily.loc[
        (daily["signal_pair"] == SIGNAL_PAIR)
        & (daily["cost_basis"] == COST_BASIS)
        & daily["cooldown_scenario"].isin([REFERENCE_SCENARIO, CHALLENGER_SCENARIO])
    ].copy()
    selected["date"] = pd.to_datetime(selected["date"])
    piv = selected.pivot(index=["period", "date"], columns="cooldown_scenario", values=["stock_exposure", "action", "equity"])
    rows = []
    for period, frame in piv.groupby(level=0):
        frame = frame.droplevel(0)
        exposure_diff = frame[("stock_exposure", REFERENCE_SCENARIO)] != frame[("stock_exposure", CHALLENGER_SCENARIO)]
        action_diff = frame[("action", REFERENCE_SCENARIO)] != frame[("action", CHALLENGER_SCENARIO)]
        equity_delta = frame[("equity", CHALLENGER_SCENARIO)] - frame[("equity", REFERENCE_SCENARIO)]
        rows.append(
            {
                "period": period,
                "different_exposure_days": int(exposure_diff.sum()),
                "different_action_days": int(action_diff.sum()),
                "first_exposure_divergence_date": frame.index[exposure_diff][0].date().isoformat() if exposure_diff.any() else "",
                "last_exposure_divergence_date": frame.index[exposure_diff][-1].date().isoformat() if exposure_diff.any() else "",
                "final_equity_delta_twd": float(equity_delta.iloc[-1]),
                "final_return_delta_pp": float(equity_delta.iloc[-1] / INITIAL_CAPITAL * 100.0),
            }
        )
    return pd.DataFrame(rows)


def run(input_dir: Path, output_dir: Path) -> None:
    trades = pd.read_csv(input_dir / "p1_p2_directional_cd_trades.csv")
    summary = pd.read_csv(input_dir / "p1_p2_directional_cd_summary.csv")
    daily = pd.read_csv(input_dir / "p1_p2_directional_cd_daily_nav.csv")
    target = summary.loc[
        (summary["period"] == PERIOD)
        & (summary["signal_pair"] == SIGNAL_PAIR)
        & (summary["cost_basis"] == COST_BASIS)
        & summary["cooldown_scenario"].isin([REFERENCE_SCENARIO, CHALLENGER_SCENARIO])
    ].set_index("cooldown_scenario")
    if set(target.index) != {REFERENCE_SCENARIO, CHALLENGER_SCENARIO}:
        raise ValueError("Expected exactly one reference and challenger summary row")

    reference = closed_episodes(trades, REFERENCE_SCENARIO)
    challenger = closed_episodes(trades, CHALLENGER_SCENARIO)
    ref_final = float(target.loc[REFERENCE_SCENARIO, "final_equity"])
    alt_final = float(target.loc[CHALLENGER_SCENARIO, "final_equity"])
    paired = pair_episode_attribution(reference, challenger, reference_final_equity=ref_final)
    checks = reconciliation(
        reference,
        challenger,
        paired,
        reference_final_equity=ref_final,
        challenger_final_equity=alt_final,
    )
    if not checks["pass"].all():
        raise ValueError("Episode attribution does not reconcile to final NAV")

    output_dir.mkdir(parents=True, exist_ok=True)
    paired.to_csv(output_dir / "p2_cd7_vs_cd7_10_episode_factor_attribution.csv", index=False)
    checks.to_csv(output_dir / "p2_cd7_vs_cd7_10_reconciliation.csv", index=False)
    path_divergence(daily).to_csv(output_dir / "p1_p2_path_divergence.csv", index=False)

    harmful = paired.loc[paired["effect_direction"] == "challenger_hurt"].sort_values("log_wealth_delta")
    helpful = paired.loc[paired["effect_direction"] == "challenger_helped"].sort_values("log_wealth_delta", ascending=False)
    top_harm = harmful.head(5)
    gross_harm_log = float(harmful["log_wealth_delta"].sum())
    top3_harm_share = float(top_harm.head(3)["log_wealth_delta"].sum() / gross_harm_log * 100.0)
    lines = [
        "# CD7/CD10 相對 CD7/CD7 逐段歸因",
        "",
        "## 規則白話",
        "",
        "MA4＋7日正斜率買入／MA10＋20日負斜率賣出／買後CD7＋賣後CD10。",
        "買進後7個交易日不能賣；賣出後10個交易日不能再買。訊號消失時不補做交易。",
        "",
        "## P2 結論",
        "",
        f"- CD7/CD7 最終資產：{ref_final:,.2f} 元。",
        f"- CD7/CD10 最終資產：{alt_final:,.2f} 元。",
        f"- 差額：{alt_final - ref_final:,.2f} 元，等於 {(alt_final-ref_final)/INITIAL_CAPITAL*100:.2f} 個百分點。",
        f"- CD7/CD10最終資產是CD7/CD7的 {alt_final/ref_final*100:.2f}%，相對少 {100-alt_final/ref_final*100:.2f}%。",
        f"- 18段中，CD10改善 {len(helpful)} 段、傷害 {len(harmful)} 段、其餘 {len(paired)-len(helpful)-len(harmful)} 段不變。",
        f"- 傷害最大的3段占全部實質負向log影響約 {top3_harm_share:.1f}%。",
        "- 18段wealth factor乘積與最終NAV完全對齊；這不是平均報酬或區間相減推估。",
        "",
        "## 傷害最大的五段",
        "",
    ]
    for _, row in top_harm.iterrows():
        lines.append(
            f"- 第{int(row['reference_episode'])}段：買點 {row['reference_buy_execution_date']} 延至 "
            f"{row['challenger_buy_execution_date']}（+{int(row['buy_delay_td'])}TD），"
            f"該段報酬 {row['reference_episode_return_pct']:.2f}% -> {row['challenger_episode_return_pct']:.2f}%，"
            f"單段替換對最終報酬約 {row['single_episode_substitution_delta_return_pp']:.2f}pp。"
        )
    lines.extend(
        [
            "",
            "## 邊界",
            "",
            "此輸出只解釋固定訊號下CD7/CD10為何不同，不新增策略、不調參，也不把事後單段排名用作live rule。",
        ]
    )
    (output_dir / "final_summary_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {
        "task_id": TASK_ID,
        "period": PERIOD,
        "signal_pair": SIGNAL_PAIR,
        "cost_basis": COST_BASIS,
        "reference": REFERENCE_SCENARIO,
        "challenger": CHALLENGER_SCENARIO,
        "episode_count": len(paired),
        "reconciliation_all_pass": bool(checks["pass"].all()),
        "future_return_used_as_rule": False,
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "report_changed": False,
        "not_live_rule": True,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
