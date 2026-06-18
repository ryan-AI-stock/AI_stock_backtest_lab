from __future__ import annotations

import io
import textwrap
from pathlib import Path
from typing import Callable

from matplotlib import font_manager
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


FOOTER_Y = 0.032
FOOTER_SEPARATOR_Y = 0.078
DETAIL_START_Y = 0.83
DETAIL_BOTTOM_Y = 0.105
DETAIL_TITLE_STEP = 0.047
DETAIL_LINE_HEIGHT = 0.026
DETAIL_SECTION_GAP = 0.018
DETAIL_WRAP_WIDTH = 56


def write_signal_pdf(
    path: Path,
    signal,
    *,
    report_name: str,
    report_variant_label: str,
    report_mode_label: Callable[[object], str],
    personal_exposure_summary: Callable[[object], dict],
    personal_pdf_section: Callable[[object], tuple[str, list[str]]],
    shadow_pdf_section: Callable[[object], tuple[str, list[str]]],
) -> None:
    _configure_chinese_font()
    detail_pages = paginate_detail_sections(
        detail_sections(
            signal,
            personal_pdf_section=personal_pdf_section,
            shadow_pdf_section=shadow_pdf_section,
        )
    )
    total_pages = 1 + len(detail_pages)
    with PdfPages(path) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69), facecolor="#f4f6f8")
        ax = fig.add_axes((0, 0, 1, 1))
        ax.axis("off")
        _draw_header(ax, signal, report_name=report_name, report_variant_label=report_variant_label, report_mode_label=report_mode_label)
        _draw_metric_cards(ax, signal, personal_exposure_summary=personal_exposure_summary)
        _draw_ranking_table(ax, signal)
        _draw_footer(ax, "本報告為 AI 輔助市場觀察與紀律提醒，不是投資建議。", page_number=1, total_pages=total_pages)
        _save_figure_as_raster_pdf_page(pdf, fig)

        for index, sections in enumerate(detail_pages, start=2):
            fig = plt.figure(figsize=(8.27, 11.69), facecolor="#f4f6f8")
            ax = fig.add_axes((0, 0, 1, 1))
            ax.axis("off")
            _draw_detail_page(ax, sections, is_continued=index > 2)
            _draw_footer(ax, f"{report_name} · {signal.signal_date}", page_number=index, total_pages=total_pages)
            _save_figure_as_raster_pdf_page(pdf, fig)


def detail_sections(
    signal,
    *,
    personal_pdf_section: Callable[[object], tuple[str, list[str]]],
    shadow_pdf_section: Callable[[object], tuple[str, list[str]]],
) -> list[tuple[str, list[str]]]:
    return [
        (
            "模型模擬動作",
            [
                f"{row['action']} {row['label']}，模型參考股數 {row['shares']}，參考價 {row['reference_price']}"
                for row in signal.projected_trades
            ] or ["模型目標未改變，沒有模擬換倉動作。"],
        ),
        (
            "狀態解讀",
            [
                f"模型目標狀態：{signal.model_target_status}",
                f"全現金帳戶參考：{signal.cash_account_reference}",
                f"進攻閘門：{'已開啟' if signal.attack_gate_active else '尚未開啟'}",
                f"歷史循環是否已證明：{'是' if signal.attack_gate_ever_activated else '否'}",
                f"風險關閉狀態：{'啟動' if signal.risk_off_active else '未啟動'}",
                f"模型重建淨值：{signal.model_total_value_twd:,.0f} 元",
            ],
        ),
        shadow_pdf_section(signal),
        personal_pdf_section(signal),
        (
            "使用邊界",
            [
                "強弱排名是觀察清單，不是買入資格清單。",
                "可執行參考只看模型目標、目標曝險與持倉工作台依個人現金/持股計算出的參考股數。",
                "本報告是每日 AI 輔助市場觀察與紀律提醒，投資人隔日自行決定是否執行。",
                "模型參考股數只用來重建策略狀態，不等於個人帳戶的實際下單股數。",
                "實際下單前仍需確認可用現金、零股成交、滑價、交易成本與個人風險承受度。",
            ],
        ),
        (
            "風險聲明",
            [
                "歷史回測與 shadow mode 都不能保證未來績效。",
                "本策略可能在風格轉換、資料延遲、極端跳空或流動性不足時失效。",
                "本報告只能作為 AI 輔助市場觀察、回測與紀律提醒，不是投資建議。",
            ],
        ),
    ]


def paginate_detail_sections(sections: list[tuple[str, list[str]]]) -> list[list[tuple[str, list[str]]]]:
    pages: list[list[tuple[str, list[str]]]] = []
    current: list[tuple[str, list[str]]] = []
    used_height = 0.0
    available_height = DETAIL_START_Y - DETAIL_BOTTOM_Y
    for section in sections:
        title, lines = section
        if not title:
            continue
        height = detail_section_height(lines)
        if current and used_height + height > available_height:
            pages.append(current)
            current = []
            used_height = 0.0
        current.append(section)
        used_height += height
    if current:
        pages.append(current)
    return pages


def detail_section_height(lines: list[str]) -> float:
    if any(line.startswith("CARD|") for line in lines):
        card_count = sum(1 for line in lines if line.startswith("CARD|"))
        note_count = sum(1 for line in lines if line.startswith("NOTE|"))
        return DETAIL_TITLE_STEP + 0.064 + card_count * 0.116 + note_count * DETAIL_LINE_HEIGHT + DETAIL_SECTION_GAP
    wrapped_count = sum(max(1, len(textwrap.wrap(line, width=DETAIL_WRAP_WIDTH))) for line in lines)
    return DETAIL_TITLE_STEP + wrapped_count * DETAIL_LINE_HEIGHT + DETAIL_SECTION_GAP


def _configure_chinese_font() -> None:
    candidates = [
        Path("C:/Windows/Fonts/msjh.ttc"),
        Path("C:/Windows/Fonts/msjhbd.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"),
        Path("/usr/share/fonts/truetype/noto/NotoSansTC-Regular.ttf"),
    ]
    family = None
    for path in candidates:
        if path.exists():
            font_manager.fontManager.addfont(str(path))
            family = font_manager.FontProperties(fname=str(path)).get_name()
            break
    if family is None:
        family = "Microsoft JhengHei"
    plt.rcParams["font.sans-serif"] = [family, "Noto Sans CJK TC", "Microsoft JhengHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42


def _save_figure_as_raster_pdf_page(pdf: PdfPages, source_fig) -> None:
    buffer = io.BytesIO()
    source_fig.savefig(buffer, format="png", dpi=220, facecolor=source_fig.get_facecolor())
    plt.close(source_fig)
    buffer.seek(0)
    image = plt.imread(buffer, format="png")
    fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
    ax = fig.add_axes((0, 0, 1, 1))
    ax.axis("off")
    ax.imshow(image, aspect="auto", extent=(0, 1, 0, 1))
    pdf.savefig(fig)
    plt.close(fig)


def _draw_header(ax, signal, *, report_name: str, report_variant_label: str, report_mode_label: Callable[[object], str]) -> None:
    ax.add_patch(plt.Rectangle((0, 0.86), 1, 0.14, color="#17212a", transform=ax.transAxes))
    ax.text(0.06, 0.94, report_name, color="white", fontsize=20, fontweight="bold", transform=ax.transAxes)
    ax.text(
        0.06,
        0.895,
        f"訊號日 {signal.signal_date} · {report_variant_label} · {report_mode_label(signal)}",
        color="#c8d5df",
        fontsize=11,
        transform=ax.transAxes,
    )
    ax.text(
        0.94,
        0.925,
        signal.market_regime_label,
        color="white",
        fontsize=15,
        ha="right",
        fontweight="bold",
        transform=ax.transAxes,
    )


def _draw_metric_cards(ax, signal, *, personal_exposure_summary: Callable[[object], dict]) -> None:
    if signal.personal_portfolio:
        summary = personal_exposure_summary(signal)
        cards = [
            ("模型動作", signal.action, "#2457a7"),
            ("模型目標", f"{signal.target_label} · {signal.target_exposure:.0%}", "#13795b"),
            ("個人目標曝險", f"{summary['target_actual_exposure']:.2%}", "#17212a"),
            ("與模型差距", f"{summary['target_gap_exposure']:+.2%}", "#b42318" if summary["target_gap_exposure"] > 0 else "#13795b"),
        ]
    else:
        cards = [
            ("模型動作", signal.action, "#2457a7"),
            ("下一交易日目標", f"{signal.target_label} · {signal.target_exposure:.0%}", "#13795b"),
            ("目標狀態", signal.model_target_status, "#17212a"),
            ("風險狀態", "風險關閉" if signal.risk_off_active else "風控未觸發", "#b42318" if signal.risk_off_active else "#13795b"),
        ]
    for index, (label, value, color) in enumerate(cards):
        x = 0.06 + index * 0.225
        ax.add_patch(
            plt.Rectangle((x, 0.73), 0.2, 0.09, facecolor="white", edgecolor="#d9e0e5", linewidth=1, transform=ax.transAxes)
        )
        ax.text(x + 0.014, 0.79, label, color="#66737d", fontsize=9.5, transform=ax.transAxes)
        ax.text(x + 0.014, 0.755, _fit_card_text(value), color=color, fontsize=11.5, fontweight="bold", transform=ax.transAxes)


def _draw_ranking_table(ax, signal) -> None:
    ax.text(0.06, 0.68, "九標的強弱排名", color="#17212a", fontsize=15, fontweight="bold", transform=ax.transAxes)
    ax.add_patch(
        plt.Rectangle((0.06, 0.61), 0.88, 0.052, facecolor="#fff8e8", edgecolor="#ead8ac", linewidth=0.8, transform=ax.transAxes)
    )
    ax.text(
        0.075,
        0.642,
        "分數解讀：0.80+ 極強勢；0.50-0.80 強勢觀察；0.25-0.50 中性偏強；0 以下偏弱。",
        color="#624711",
        fontsize=8.8,
        transform=ax.transAxes,
    )
    ax.text(
        0.075,
        0.622,
        "可執行參考仍以「下一交易日模型目標」與「模型目標狀態」為準，不能只看排名。",
        color="#624711",
        fontsize=8.8,
        transform=ax.transAxes,
    )
    headers = ("名次", "標的", "角色", "分數", "收盤價")
    widths = (0.08, 0.24, 0.24, 0.15, 0.16)
    x0 = 0.06
    y = 0.575
    row_h = 0.039
    ax.add_patch(plt.Rectangle((x0, y), 0.88, row_h, facecolor="#e8eef3", edgecolor="#d9e0e5", transform=ax.transAxes))
    x = x0
    for header, width in zip(headers, widths):
        ax.text(x + 0.01, y + 0.014, header, color="#31414d", fontsize=10, fontweight="bold", transform=ax.transAxes)
        x += width
    for row in signal.ranking:
        y -= row_h
        is_target = row["ticker"] == signal.target_ticker
        fill = "#fff7e6" if is_target else "white"
        ax.add_patch(plt.Rectangle((x0, y), 0.88, row_h, facecolor=fill, edgecolor="#d9e0e5", linewidth=0.8, transform=ax.transAxes))
        values = (
            str(row["rank"]),
            f"{row['label']} ({row['ticker'].replace('.TW', '')})",
            row["role"],
            f"{row['score']:.4f}",
            _format_price(signal.close_prices.get(row["ticker"])),
        )
        x = x0
        for value, width in zip(values, widths):
            weight = "bold" if is_target else "normal"
            color = "#a15c00" if is_target else "#1f2d36"
            ax.text(x + 0.01, y + 0.014, value, color=color, fontsize=9.5, fontweight=weight, transform=ax.transAxes)
            x += width
    ax.text(
        0.06,
        0.19,
        "重點：排名只代表相對強弱，不等於買入資格；是否建立部位，以「下一交易日模型目標」與「模型目標狀態」為準。",
        color="#52616b",
        fontsize=9.5,
        transform=ax.transAxes,
    )


def _draw_detail_page(ax, sections: list[tuple[str, list[str]]], *, is_continued: bool) -> None:
    ax.add_patch(plt.Rectangle((0, 0.9), 1, 0.1, color="#17212a", transform=ax.transAxes))
    title = "操作摘要與風險說明（續）" if is_continued else "操作摘要與風險說明"
    ax.text(0.06, 0.94, title, color="white", fontsize=18, fontweight="bold", transform=ax.transAxes)
    y = DETAIL_START_Y
    for title, lines in sections:
        if not title:
            continue
        ax.add_patch(plt.Rectangle((0.06, y - 0.015), 0.88, 0.035, facecolor="#e8eef3", edgecolor="#d9e0e5", transform=ax.transAxes))
        ax.text(0.075, y - 0.004, title, fontsize=12.5, fontweight="bold", color="#17212a", transform=ax.transAxes)
        y -= DETAIL_TITLE_STEP
        if title == "Shadow Mode 對照" and any(line.startswith("CARD|") for line in lines):
            y = _draw_shadow_cards(ax, lines, y)
            y -= DETAIL_SECTION_GAP
            continue
        for line in lines:
            wrapped = textwrap.wrap(line, width=DETAIL_WRAP_WIDTH)
            for text in wrapped:
                ax.text(0.075, y, f"• {text}", fontsize=9.6, color="#1f2d36", transform=ax.transAxes)
                y -= DETAIL_LINE_HEIGHT
        y -= DETAIL_SECTION_GAP


def _draw_shadow_cards(ax, lines: list[str], start_y: float) -> float:
    y = start_y
    for line in lines:
        if line.startswith("INFO|"):
            ax.text(0.075, y, line.split("|", 1)[1], fontsize=9.4, color="#52616b", transform=ax.transAxes)
            y -= 0.026
        elif line.startswith("BASE|"):
            _, label, value = line.split("|", 2)
            ax.add_patch(
                plt.Rectangle((0.075, y - 0.02), 0.84, 0.036, facecolor="#f7fafc", edgecolor="#d9e0e5", linewidth=0.8, transform=ax.transAxes)
            )
            ax.text(0.09, y - 0.006, label, fontsize=9.4, color="#52616b", transform=ax.transAxes)
            ax.text(0.9, y - 0.006, value, fontsize=10.2, fontweight="bold", color="#17212a", ha="right", transform=ax.transAxes)
            y -= 0.05
        elif line.startswith("CARD|"):
            _, role, label, diff, target, action, focus, trade = line.split("|", 7)
            y = _draw_shadow_card(ax, y, role, label, diff, target, action, focus, trade)
        elif line.startswith("NOTE|"):
            ax.text(0.09, y, line.split("|", 1)[1], fontsize=8.8, color="#8a5b00", transform=ax.transAxes)
            y -= DETAIL_LINE_HEIGHT
    return y


def _draw_shadow_card(
    ax,
    y: float,
    role: str,
    label: str,
    diff: str,
    target: str,
    action: str,
    focus: str,
    trade: str,
) -> float:
    palette = {
        "攻擊型": ("#e8f5ef", "#13795b"),
        "風控型": ("#fff4e5", "#b86b00"),
        "對照型": ("#eef4ff", "#2457a7"),
    }
    fill, accent = palette.get(role, ("#f7fafc", "#52616b"))
    card_h = 0.1
    top = y - 0.006
    ax.add_patch(
        plt.Rectangle((0.075, top - card_h), 0.84, card_h, facecolor="white", edgecolor="#d9e0e5", linewidth=0.9, transform=ax.transAxes)
    )
    ax.add_patch(
        plt.Rectangle((0.075, top - card_h), 0.014, card_h, facecolor=accent, edgecolor=accent, linewidth=0, transform=ax.transAxes)
    )
    ax.add_patch(
        plt.Rectangle((0.095, top - 0.03), 0.085, 0.024, facecolor=fill, edgecolor=accent, linewidth=0.8, transform=ax.transAxes)
    )
    ax.text(0.1375, top - 0.022, role, fontsize=8.8, color=accent, fontweight="bold", ha="center", transform=ax.transAxes)
    ax.text(0.195, top - 0.022, label, fontsize=10.2, color="#17212a", fontweight="bold", transform=ax.transAxes)
    diff_color = "#13795b" if diff.startswith("+") else "#b42318"
    ax.text(0.9, top - 0.022, diff, fontsize=10.0, color=diff_color, fontweight="bold", ha="right", transform=ax.transAxes)
    ax.text(0.095, top - 0.055, f"目標：{target}", fontsize=9.0, color="#31414d", transform=ax.transAxes)
    ax.text(0.31, top - 0.055, f"動作：{action}", fontsize=9.0, color="#31414d", transform=ax.transAxes)
    ax.text(0.095, top - 0.079, f"觀察：{focus}", fontsize=8.8, color="#52616b", transform=ax.transAxes)
    ax.text(0.43, top - 0.079, f"換倉：{textwrap.shorten(trade, width=30, placeholder='...')}", fontsize=8.8, color="#52616b", transform=ax.transAxes)
    return y - 0.116


def _draw_footer(ax, text: str, *, page_number: int | None = None, total_pages: int | None = None) -> None:
    ax.plot([0.06, 0.94], [FOOTER_SEPARATOR_Y, FOOTER_SEPARATOR_Y], color="#d9e0e5", linewidth=0.8, transform=ax.transAxes)
    ax.text(0.06, FOOTER_Y, text, color="#73818b", fontsize=8.2, transform=ax.transAxes)
    right_text = f"第 {page_number}/{total_pages} 頁" if page_number and total_pages else "被AI研究所"
    ax.text(0.94, FOOTER_Y, right_text, color="#73818b", fontsize=8.2, ha="right", transform=ax.transAxes)


def _format_price(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:,.2f}".rstrip("0").rstrip(".")


def _fit_card_text(value: str) -> str:
    return value if len(value) <= 12 else textwrap.shorten(value, width=15, placeholder="...")
