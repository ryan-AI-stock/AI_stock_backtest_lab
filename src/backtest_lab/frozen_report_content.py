from __future__ import annotations


def markdown_report(signal, *, report_name: str, report_variant_label: str, score_guide_lines: list[str]) -> str:
    ranking_lines = [
        f"{row['rank']}. {row['label']} ({row['ticker']})，分數 {row['score']:.4f}（{row['score_band']}），角色：{row['role']}"
        for row in signal.ranking
    ]
    score_lines = [f"- {line}" for line in score_guide_lines]
    trade_lines = [
        f"- {row['action']} {row['label']}，模型參考股數 {row['shares']}，參考價 {row['reference_price']}"
        for row in signal.projected_trades
    ] or ["- 模型目標未改變，沒有模擬換倉動作。"]
    shadow_lines = shadow_mode_markdown_lines(signal)
    personal_lines = personal_markdown_lines(signal)
    return "\n".join(
        [
            f"# {report_name}",
            "",
            "## 摘要",
            "",
            f"- 策略版本：{report_variant_label}",
            f"- 報告模式：{report_mode_label(signal)}",
            f"- 訊號日期：{signal.signal_date}",
            f"- 執行時點：{signal.execution_timing}",
            "- 定位：每日 AI 輔助操作建議，投資人自行判斷，不是自動下單，也不是投資建議。",
            "",
            "## 今日結論",
            "",
            f"- 市場環境：{signal.market_regime_label}",
            f"- 模型動作：{signal.action}",
            f"- 模型目標狀態：{signal.model_target_status}",
            f"- 今日收盤後模型部位：{signal.current_label}，曝險約 {signal.current_exposure:.0%}",
            f"- 下一交易日模型目標：{signal.target_label}，曝險約 {signal.target_exposure:.0%}",
            f"- 全現金帳戶參考：{signal.cash_account_reference}",
            f"- 進攻閘門：{'已開啟' if signal.attack_gate_active else '尚未開啟'}",
            f"- 風險關閉狀態：{'啟動' if signal.risk_off_active else '未啟動'}",
            "",
            "## 模型模擬動作",
            "",
            *trade_lines,
            "",
            "上述股數與價格只用來重建模型狀態，不是針對使用者資產的實際下單建議。",
            "",
            *shadow_lines,
            "",
            *personal_lines,
            "",
            "## 九標的強弱排名",
            "",
            "注意：排名只代表相對強弱，不等於買入資格；可執行參考請看「下一交易日模型目標」與「模型目標狀態」。",
            "",
            "### 分數解讀",
            "",
            *score_lines,
            "",
            *ranking_lines,
            "",
            "## 角色說明",
            "",
            "- 0050 是市場代理、循環判斷基準與比較基準。",
            "- 0050正二是槓桿大盤訊號、積極等待與曝險工具，不是唯一趨勢判斷依據。",
            "- 七檔指定個股才是進攻模式的持股候選。",
            "",
            "## 風險聲明",
            "",
            "歷史回測與 shadow mode 都不能保證未來績效。本報告只供 AI 輔助市場觀察、回測與紀律提醒。",
            "",
        ]
    )


def shadow_mode_markdown_lines(signal) -> list[str]:
    shadows = list(signal.shadow_modes or [])
    if not shadows:
        return [
            "## Shadow Mode 對照",
            "",
            "- 目前沒有啟用 shadow mode；每日報告只追蹤正式最佳版。",
        ]
    lines = [
        "## Shadow Mode 對照",
        "",
        "Shadow mode 是用未來實際行情驗證候選模型，不會取代正式最佳版，也不是投資建議。",
        "",
        f"**正式最佳版同本金重建淨值：約 {signal.model_total_value_twd:,.0f} 元**",
        "",
        "| 類型 | 模型 | 同本金淨值 | 與最佳版差距 | 下一交易日目標 | 模型動作 | 觀察重點 |",
        "| --- | --- | ---: | ---: | --- | --- | --- |",
    ]
    for shadow in shadows[:3]:
        role, focus = _shadow_role_and_focus(shadow)
        lines.append(
            "| "
            + " | ".join(
                [
                    role,
                    _compact_shadow_label(shadow.shadow_label),
                    f"{shadow.model_total_value_twd:,.0f} 元",
                    _shadow_diff_text(shadow),
                    f"{shadow.target_label} {shadow.target_exposure:.0%}",
                    shadow.action,
                    focus,
                ]
            )
            + " |"
        )
    lines.extend(["", "### Shadow 模擬換倉明細", ""])
    for shadow in shadows[:3]:
        trade_text = _shadow_trade_text(shadow.projected_trades)
        role, _ = _shadow_role_and_focus(shadow)
        lines.append(f"- {role}／{_compact_shadow_label(shadow.shadow_label)}：{trade_text}")
        if shadow.target_ticker != signal.target_ticker:
            lines.append(
                f"  - 與最佳版不同：最佳版目標是 {signal.target_label}，此 shadow 目標是 {shadow.target_label}。後續實際差異以每日同本金淨值追蹤。"
            )
    return lines


def _compact_shadow_label(label: str) -> str:
    if "：" in label:
        return label.split("：", 1)[1]
    return label


def _shadow_role_and_focus(shadow) -> tuple[str, str]:
    shadow_id = getattr(shadow, "shadow_id", "")
    label = getattr(shadow, "shadow_label", "")
    if "attack_hybrid" in shadow_id or "攻擊型" in label:
        return ("攻擊型", "看能否提升報酬")
    if "risk_overlay" in shadow_id or "風控型" in label:
        return ("風控型", "看降低回撤的代價")
    if "challenger" in shadow_id or "挑戰版" in label:
        return ("對照型", "延續舊候選追蹤")
    return ("候選型", "觀察是否優於最佳版")


def _shadow_diff_text(shadow) -> str:
    sign = "+" if shadow.value_diff_twd >= 0 else "-"
    return f"{sign}{abs(shadow.value_diff_twd):,.0f} 元（{shadow.value_diff_pct:+.2%}）"


def _shadow_trade_text(projected_trades: list[dict]) -> str:
    if not projected_trades:
        return "目標未改變，沒有模擬換倉動作。"
    return "；".join(
        f"{row['action']} {row['label']} {row['shares']} 股，參考價 {row['reference_price']}"
        for row in projected_trades
    )


def personal_markdown_lines(signal) -> list[str]:
    if not signal.personal_portfolio:
        return [
            "## 個人持倉參考",
            "",
            "- 尚未連結個人持倉檔；本報告為一般版，只顯示模型帳戶狀態。",
        ]
    summary = personal_exposure_summary(signal)
    recommendation_lines = [
        f"- {display_action(row['action'])} {row['ticker']}，參考股數 {row['shares']}，目標比例 {row['target_exposure']:.0%}，原因：{row['reason']}"
        for row in signal.personal_recommendations or []
    ] or ["- 依目前持倉與模型目標，暫無可計算的個人調整參考。"]
    return [
        "## 個人持倉參考",
        "",
        f"- 個人組合估值：約 {summary['total_value_twd']:,.2f} 元",
        f"- 可用現金：約 {summary['cash_twd']:,.2f} 元，現金水位約 {summary['cash_exposure']:.2%}",
        f"- 目前持股水位：約 {summary['market_exposure']:.2%}",
        f"- 模型目標標的：{signal.target_label}，模型目標曝險 {signal.target_exposure:.0%}",
        f"- 個人目前目標標的曝險：約 {summary['target_actual_exposure']:.2%}，與模型目標差距約 {summary['target_gap_exposure']:+.2%}",
        "",
        "### 個人帳戶參考調整",
        "",
        *recommendation_lines,
        "",
        "個人持倉參考只依目前手動輸入資料估算，不是自動下單或投資建議。",
    ]


def personal_exposure_summary(signal) -> dict:
    portfolio = signal.personal_portfolio or {}
    total = float(portfolio.get("total_value_twd") or 0)
    cash = float(portfolio.get("cash_twd") or 0)
    market = float(portfolio.get("market_value_twd") or 0)
    target_value = 0.0
    for row in portfolio.get("positions", []):
        if row.get("ticker") == signal.target_ticker:
            target_value += float(row.get("market_value_twd") or 0)
    target_actual = target_value / total if total > 0 else 0.0
    return {
        "total_value_twd": total,
        "cash_twd": cash,
        "cash_exposure": cash / total if total > 0 else 0.0,
        "market_exposure": market / total if total > 0 else 0.0,
        "target_actual_exposure": target_actual,
        "target_gap_exposure": float(signal.target_exposure) - target_actual,
    }


def report_mode(signal) -> str:
    return "personalized" if signal.personal_portfolio else "general"


def report_mode_label(signal) -> str:
    if signal.personal_portfolio:
        return "個人化版，已結合目前持倉檔"
    return "一般版，未連結個人持倉檔"


def display_action(action: str) -> str:
    return {"buy": "買進", "sell": "賣出", "hold": "維持"}.get(action, action)


def personal_pdf_section(signal) -> tuple[str, list[str]]:
    if not signal.personal_portfolio:
        return ("個人持倉參考", ["尚未連結個人持倉檔；本報告為一般版，只顯示模型帳戶狀態。"])
    summary = personal_exposure_summary(signal)
    lines = [
        f"個人組合估值：約 {summary['total_value_twd']:,.2f} 元；可用現金：約 {summary['cash_twd']:,.2f} 元。",
        f"現金水位約 {summary['cash_exposure']:.2%}；持股水位約 {summary['market_exposure']:.2%}。",
        f"模型目標：{signal.target_label} {signal.target_exposure:.0%}；個人目前目標標的曝險約 {summary['target_actual_exposure']:.2%}。",
        f"與模型目標差距約 {summary['target_gap_exposure']:+.2%}。",
    ]
    for row in (signal.personal_recommendations or [])[:4]:
        lines.append(
            f"個人參考：{display_action(row['action'])} {row['ticker']} {row['shares']} 股，參考價 {row['reference_price']:.2f}。"
        )
    return ("個人持倉參考", lines)


def shadow_mode_pdf_section(signal) -> tuple[str, list[str]]:
    shadows = list(signal.shadow_modes or [])
    if not shadows:
        return ("Shadow Mode 對照", ["目前沒有啟用 shadow mode；每日報告只追蹤正式最佳版。"])
    lines = [
        "INFO|Shadow mode 是用未來實際行情驗證候選模型，不會取代正式最佳版。",
        f"BASE|正式最佳版同本金淨值|{signal.model_total_value_twd:,.0f} 元",
    ]
    for shadow in shadows[:3]:
        role, focus = _shadow_role_and_focus(shadow)
        lines.append(
            "CARD|"
            + "|".join(
                [
                    role,
                    _compact_shadow_label(shadow.shadow_label),
                    _shadow_diff_text(shadow),
                    f"{shadow.target_label} {shadow.target_exposure:.0%}",
                    shadow.action,
                    focus,
                    _shadow_trade_text(shadow.projected_trades),
                ]
            )
        )
        if shadow.target_ticker != signal.target_ticker:
            lines.append(f"NOTE|最佳版目標 {signal.target_label}；此 shadow 目標 {shadow.target_label}")
    return ("Shadow Mode 對照", lines)
