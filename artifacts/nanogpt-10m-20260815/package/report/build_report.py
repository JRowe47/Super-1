#!/usr/bin/env python3
"""Build charts and a visually structured PDF from the locked experiment metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "report/metrics.json"
DEFAULT_MD = ROOT / "report/source_report.md"
DEFAULT_CHART_DIR = ROOT / "report/charts"
DEFAULT_PDF = ROOT / "report/generated/official_nanogpt_vs_proven_hierarchy_10m_report.pdf"

INK = "#0B1220"
SLATE = "#475569"
MUTED = "#64748B"
GRID = "#DCE4EC"
MIST = "#F3F6F9"
WHITE = "#FFFFFF"
TEAL = "#0F9D8A"
TEAL_DARK = "#087267"
ORANGE = "#F97316"
BLUE = "#2563EB"
BLUE_LIGHT = "#DBEAFE"
TEAL_LIGHT = "#DDF7F2"
ORANGE_LIGHT = "#FFEDD5"


def load_data(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_against_markdown(data: dict, md_path: Path) -> None:
    """Fail if the human-readable source report is not the same experiment."""
    text = md_path.read_text(encoding="utf-8")
    required = [
        f"{data['steps']:,}",
        f"{data['tokens']:,}",
        f"{data['parameters']['hierarchy']:,}",
        f"{data['parameters']['nanogpt']:,}",
        "5.126457",
        "5.260253",
        "5.261362",
        "5.070329",
        "5.988185",
        "5.291337",
        "14,002.60",
        "99.9929%",
    ]
    missing = [value for value in required if value not in text]
    if missing:
        raise RuntimeError(f"source report mismatch; missing: {missing}")


def pct_lower(a: float, b: float) -> float:
    """Percent by which a is lower than b."""
    return 100.0 * (b - a) / b


def human_bytes(value: int) -> str:
    if value >= 1024**2:
        return f"{value / 1024**2:g} MiB"
    return f"{value / 1024:g} KiB"


def chart_setup():
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 15,
            "axes.titleweight": "bold",
            "axes.labelcolor": SLATE,
            "axes.edgecolor": GRID,
            "axes.facecolor": WHITE,
            "figure.facecolor": WHITE,
            "xtick.color": SLATE,
            "ytick.color": SLATE,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "legend.frameon": False,
        }
    )
    return plt


def save_chart(fig, path: Path) -> None:
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor=WHITE)
    fig.clf()


def make_charts(data: dict, out_dir: Path) -> dict[str, Path]:
    import numpy as np

    plt = chart_setup()
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    contexts = np.array(data["fixed_eval"]["contexts"])
    nano_ce = np.array(data["fixed_eval"]["nanogpt_ce"])
    hier_ce = np.array(data["fixed_eval"]["hierarchy_ce"])

    fig, ax = plt.subplots(figsize=(10.5, 5.7))
    x = np.arange(len(contexts))
    width = 0.34
    bars_n = ax.bar(x - width / 2, nano_ce, width, color=BLUE, label="Official nanoGPT")
    bars_h = ax.bar(x + width / 2, hier_ce, width, color=TEAL, label="Pure hierarchy")
    ax.set_ylabel("Cross-entropy")
    ax.set_xticks(x, [f"{c:,} tokens" for c in contexts])
    ax.set_ylim(0, 6.55)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    for bars in (bars_n, bars_h):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.08, f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=10, color=INK)
    ax.legend(loc="upper left", bbox_to_anchor=(0, 0.96), ncol=2)
    fig.tight_layout()
    paths["fixed_ce"] = out_dir / "fixed_eval_ce.png"
    save_chart(fig, paths["fixed_ce"])

    nano_state = np.array(data["state"]["nanogpt_bytes"])
    hier_state = np.array(data["state"]["hierarchy_bytes"])
    ratios = np.array(data["state"]["nanogpt_to_hierarchy_ratio"])
    fig, ax = plt.subplots(figsize=(10.5, 5.9))
    ax.plot(contexts, nano_state, marker="o", markersize=9, linewidth=3, color=BLUE, label="nanoGPT theoretical KV state")
    ax.plot(contexts, hier_state, marker="o", markersize=9, linewidth=3, color=TEAL, label="Hierarchy recurrent state")
    ax.fill_between(contexts, hier_state, nano_state, color=BLUE_LIGHT, alpha=0.52)
    ax.set_yscale("log")
    ax.set_xlabel("Evaluation context")
    ax.set_ylabel("State bytes per sequence (log scale)")
    ax.set_xticks(contexts, [f"{c:,}" for c in contexts])
    ax.yaxis.grid(True, which="both")
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    for c, n, h, ratio in zip(contexts, nano_state, hier_state, ratios):
        ax.annotate(human_bytes(int(n)), (c, n), xytext=(0, 10), textcoords="offset points", ha="center", color=BLUE, fontsize=9, fontweight="bold")
        ax.annotate(f"{ratio:,.1f}x", (c, math.sqrt(n * h)), xytext=(0, -3), textcoords="offset points", ha="center", color=INK, fontsize=9, fontweight="bold")
    ax.annotate("5.375 KiB at every context", (contexts[-1], hier_state[-1]), xytext=(-12, 10), textcoords="offset points", ha="right", color=TEAL_DARK, fontsize=9, fontweight="bold")
    ax.legend(loc="upper left")
    fig.tight_layout()
    paths["state"] = out_dir / "state_scaling.png"
    save_chart(fig, paths["state"])

    train_steps = np.array(data["training_checkpoints"]["steps"])
    train_n = np.array(data["training_checkpoints"]["nanogpt_ce"])
    train_h = np.array(data["training_checkpoints"]["hierarchy_ce"])
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.plot(train_steps, train_n, marker="o", markersize=8, linewidth=2.5, color=BLUE, label="Official nanoGPT")
    ax.plot(train_steps, train_h, marker="o", markersize=8, linewidth=2.5, color=TEAL, label="Pure hierarchy")
    ax.set_xlabel("Optimizer step", fontsize=12)
    ax.set_ylabel("Training CE (zoomed axis)", fontsize=12)
    ax.set_xticks(train_steps, [f"{s:,}" for s in train_steps])
    ax.set_ylim(4.92, 5.48)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    for s, n, h in zip(train_steps, train_n, train_h):
        ax.annotate(f"{n:.3f}", (s, n), xytext=(0, -17), textcoords="offset points", ha="center", color=BLUE, fontsize=9)
        ax.annotate(f"{h:.3f}", (s, h), xytext=(0, 10), textcoords="offset points", ha="center", color=TEAL_DARK, fontsize=9)
    ax.tick_params(labelsize=11)
    ax.legend(loc="upper left", fontsize=10)
    fig.tight_layout()
    paths["training"] = out_dir / "matched_training_ce.png"
    save_chart(fig, paths["training"])

    tasks = data["benchmark"]["tasks"]
    bench_n = np.array(data["benchmark"]["nanogpt"])
    bench_h = np.array(data["benchmark"]["hierarchy"])
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    x = np.arange(len(tasks))
    bars_n = ax.bar(x - width / 2, bench_n, width, color=BLUE, label="Official nanoGPT")
    bars_h = ax.bar(x + width / 2, bench_h, width, color=TEAL, label="Pure hierarchy")
    ax.set_ylabel("Score", fontsize=12)
    ax.set_xticks(x, tasks)
    ax.set_ylim(0, 0.61)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    for bars in (bars_n, bars_h):
        for bar in bars:
            if bar.get_height() > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012, f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.tick_params(labelsize=10)
    ax.legend(loc="upper right", fontsize=10)
    fig.tight_layout()
    paths["benchmark"] = out_dir / "benchmark_scores.png"
    save_chart(fig, paths["benchmark"])

    return paths


def build_pdf(data: dict, chart_dir: Path, pdf_path: Path, source_md: Path) -> None:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch, mm
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.pdfbase import pdfmetrics
    from reportlab.platypus import (
        BaseDocTemplate,
        Frame,
        HRFlowable,
        Image,
        KeepTogether,
        PageBreak,
        PageTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
    )

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    width, height = A4
    margin_x = 17 * mm
    margin_top = 19 * mm
    margin_bottom = 16 * mm

    regular_font = "Helvetica"
    bold_font = "Helvetica-Bold"
    mono_font = "Courier"
    # Built-in PDF fonts keep the report portable across macOS and Linux.

    doc = BaseDocTemplate(
        str(pdf_path),
        pagesize=A4,
        rightMargin=margin_x,
        leftMargin=margin_x,
        topMargin=margin_top,
        bottomMargin=margin_bottom,
        title="Official nanoGPT vs. Proven Pure Hierarchy",
        author="Experiment report generated from immutable result artifacts",
        subject="Matched 10-million-token model comparison",
    )
    frame = Frame(margin_x, margin_bottom, width - 2 * margin_x, height - margin_top - margin_bottom, id="normal")

    def page_decor(canvas, doc_obj):
        page = canvas.getPageNumber()
        canvas.saveState()
        canvas.setFillColor(colors.HexColor(INK))
        canvas.rect(0, height - 8 * mm, width, 8 * mm, fill=1, stroke=0)
        canvas.setFillColor(colors.HexColor(TEAL))
        canvas.rect(0, height - 8 * mm, 38 * mm, 8 * mm, fill=1, stroke=0)
        if page > 1:
            canvas.setFont(bold_font, 7.5)
            canvas.setFillColor(colors.HexColor(MUTED))
            canvas.drawString(margin_x, 8 * mm, "OFFICIAL NANOGPT VS. PROVEN PURE HIERARCHY")
            canvas.drawRightString(width - margin_x, 8 * mm, f"{page}")
        canvas.restoreState()

    doc.addPageTemplates(PageTemplate(id="report", frames=[frame], onPage=page_decor))

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Kicker", fontName=bold_font, fontSize=9, leading=11, textColor=colors.HexColor(TEAL_DARK), spaceAfter=8, uppercase=True))
    styles.add(ParagraphStyle(name="TitleXL", fontName=bold_font, fontSize=33, leading=36, textColor=colors.HexColor(INK), spaceAfter=16))
    styles.add(ParagraphStyle(name="Deck", fontName=regular_font, fontSize=14, leading=20, textColor=colors.HexColor(SLATE), spaceAfter=18))
    styles.add(ParagraphStyle(name="H1x", fontName=bold_font, fontSize=23, leading=27, textColor=colors.HexColor(INK), spaceAfter=10))
    styles.add(ParagraphStyle(name="H2x", fontName=bold_font, fontSize=15, leading=18, textColor=colors.HexColor(INK), spaceBefore=4, spaceAfter=7))
    styles.add(ParagraphStyle(name="Bodyx", fontName=regular_font, fontSize=9.4, leading=13.5, textColor=colors.HexColor(SLATE), spaceAfter=7))
    styles.add(ParagraphStyle(name="Smallx", fontName=regular_font, fontSize=7.7, leading=10.5, textColor=colors.HexColor(MUTED), spaceAfter=4))
    styles.add(ParagraphStyle(name="Tinyx", fontName=regular_font, fontSize=6.8, leading=8.8, textColor=colors.HexColor(MUTED), spaceAfter=2))
    styles.add(ParagraphStyle(name="MetricBig", fontName=bold_font, fontSize=24, leading=26, textColor=colors.HexColor(INK), alignment=TA_LEFT))
    styles.add(ParagraphStyle(name="MetricLabel", fontName=bold_font, fontSize=8.3, leading=10.5, textColor=colors.HexColor(MUTED), alignment=TA_LEFT))
    styles.add(ParagraphStyle(name="Callout", fontName=bold_font, fontSize=14.5, leading=19, textColor=colors.HexColor(INK), alignment=TA_LEFT))
    styles.add(ParagraphStyle(name="Hash", fontName=mono_font, fontSize=5.7, leading=7.6, textColor=colors.HexColor(SLATE)))
    styles.add(ParagraphStyle(name="TableHead", fontName=bold_font, fontSize=7.4, leading=9, textColor=colors.white, alignment=TA_LEFT))
    styles.add(ParagraphStyle(name="TableCell", fontName=regular_font, fontSize=7.6, leading=9.6, textColor=colors.HexColor(SLATE), alignment=TA_LEFT))
    styles.add(ParagraphStyle(name="TableCellBold", fontName=bold_font, fontSize=7.6, leading=9.6, textColor=colors.HexColor(INK), alignment=TA_LEFT))

    def p(text: str, style="Bodyx"):
        return Paragraph(text, styles[style])

    def table(rows, widths, header=True, font_size=7.6, row_bgs=True):
        formatted = []
        for r_idx, row in enumerate(rows):
            formatted_row = []
            for c_idx, cell in enumerate(row):
                if header and r_idx == 0:
                    style_obj = styles["TableHead"]
                elif r_idx > 0 and c_idx == 0:
                    style_obj = styles["TableCellBold"]
                else:
                    style_obj = styles["TableCell"]
                formatted_row.append(Paragraph(str(cell), style_obj))
            formatted.append(formatted_row)
        tbl = Table(formatted, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
        cmds = [
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor(GRID)),
        ]
        if header:
            cmds += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(INK))]
        if row_bgs:
            for idx in range(1 if header else 0, len(rows)):
                if idx % 2 == 0:
                    cmds.append(("BACKGROUND", (0, idx), (-1, idx), colors.HexColor(MIST)))
        tbl.setStyle(TableStyle(cmds))
        return tbl

    def metric_card(value: str, label: str, bg: str):
        cell = Table([[p(value, "MetricBig")], [p(label, "MetricLabel")]], colWidths=[50 * mm])
        cell.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(bg)),
            ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor(GRID)),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ("TOPPADDING", (0, 0), (-1, 0), 10),
            ("BOTTOMPADDING", (0, -1), (-1, -1), 10),
        ]))
        return cell

    story = []
    contexts = data["fixed_eval"]["contexts"]
    n_ce = data["fixed_eval"]["nanogpt_ce"]
    h_ce = data["fixed_eval"]["hierarchy_ce"]
    ratios = data["state"]["nanogpt_to_hierarchy_ratio"]
    reductions = data["state"]["hierarchy_percent_fewer"]

    # Page 1 - cover and executive outcome.
    story += [Spacer(1, 27 * mm), p("MATCHED 10-MILLION-TOKEN STUDY", "Kicker"), p("Official nanoGPT vs.<br/>Proven Pure Hierarchy", "TitleXL")]
    story += [p("A controlled comparison of predictive quality, long-context behavior, and per-sequence inference state under the same data, teacher objective, training exposure, and evaluation protocol.", "Deck")]
    story += [Spacer(1, 5 * mm)]
    cards = Table(
        [[
            metric_card(f"{pct_lower(h_ce[2], n_ce[2]):.2f}%", "LOWER HIERARCHY CE AT 2,048", TEAL_LIGHT),
            metric_card(f"{ratios[2]:,.1f}x", "LARGER NANOGPT KV STATE AT 2,048", BLUE_LIGHT),
            metric_card(f"{reductions[2]:.4f}%", "FEWER HIERARCHY STATE BYTES", ORANGE_LIGHT),
        ]],
        colWidths=[54 * mm, 54 * mm, 54 * mm],
        hAlign="LEFT",
    )
    cards.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 5)]))
    story += [cards, Spacer(1, 12 * mm)]
    callout = Table([[p("At context 2,048, the hierarchy achieved CE 5.291337 versus nanoGPT 5.988185 while using 5.375 KiB rather than 73.5 MiB of analytic BF16 sequence state.", "Callout")]], colWidths=[162 * mm])
    callout.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(MIST)), ("BOX", (0, 0), (-1, -1), 1.2, colors.HexColor(TEAL)), ("LEFTPADDING", (0, 0), (-1, -1), 14), ("RIGHTPADDING", (0, 0), (-1, -1), 14), ("TOPPADDING", (0, 0), (-1, -1), 13), ("BOTTOMPADDING", (0, 0), (-1, -1), 13)]))
    story += [callout, Spacer(1, 9 * mm), p("Boundary of the result", "H2x"), p("This is not a universal quality win. nanoGPT is better at the trained 512-token context and has a slightly higher four-task benchmark mean. The hierarchy advantage appears at the 1,024- and 2,048-token evaluations, where its fixed recurrent state is also thousands of times smaller than nanoGPT's theoretical KV state.")]
    story += [Spacer(1, 8 * mm), p(f"Report date {data['report_date']}  |  Status: training, fixed evaluation, and benchmark complete", "Smallx"), PageBreak()]

    # Page 2 - design contract.
    story += [p("A matched experimental contract", "Kicker"), p("Same exposure. Same data. Near-identical scale.", "H1x"), p("The comparison keeps the proven hierarchy design unchanged and evaluates a pinned official nanoGPT comparator under the historical Aug-8 teacher-distillation protocol.")]
    contract_rows = [
        ["Variable", "Matched value", "Why it matters"],
        ["Teacher", "Qwen3-0.6B-Base", "Same target distributions"],
        ["Training / validation data", "Byte-identical train.bin / val.bin", "Same corpus bytes"],
        ["Training context", "512 tokens", "Same token window"],
        ["Batch / world size", "4 sequences / 1 process", "Same examples per step"],
        ["Optimization", "4,883 steps / 10,000,384 tokens", "Same exposure"],
        ["AdamW", "LR 0.0003; warmup 500; betas 0.9/0.95", "Same schedule"],
        ["Objective", "0.1 next-token CE + 1.0 forward KL", "Same loss contract"],
        ["Training / eval seeds", "12001 / 14001", "Same deterministic schedule"],
        ["Fixed evaluation", "512, 1,024, and 2,048 contexts", "Same held-out windows"],
    ]
    story += [table(contract_rows, [38 * mm, 70 * mm, 54 * mm]), Spacer(1, 8 * mm)]
    param_delta = data["parameters"]["hierarchy"] - data["parameters"]["nanogpt"]
    param_pct = 100 * param_delta / data["parameters"]["hierarchy"]
    param_rows = [
        ["Model", "Stored parameters", "Architecture"],
        ["Pure hierarchy", f"{data['parameters']['hierarchy']:,}", "Original width-688 recurrent hierarchy"],
        ["Official nanoGPT", f"{data['parameters']['nanogpt']:,}", "Width 448 / 21 layers / 7 heads"],
        ["Absolute difference", f"{param_delta:,} ({param_pct:.5f}%)", "Effectively parameter matched"],
    ]
    story += [p("Parameter match", "H2x"), table(param_rows, [44 * mm, 50 * mm, 68 * mm]), Spacer(1, 8 * mm)]
    story += [p("Design boundary", "H2x"), p("The comparator uses upstream nanoGPT model bytes pinned to commit 3adf61e154c3fe3fca428ad6bc3818b27a3b8291. No perturbation solver, post-training optimizer, extra objective, or alternate hierarchy design was introduced.")]
    story += [PageBreak()]

    # Page 3 - fixed evaluation.
    story += [p("PREDICTIVE QUALITY", "Kicker"), p("nanoGPT leads at 512; hierarchy leads at longer contexts", "H1x")]
    story += [p("Fixed-evaluation cross-entropy. Lower is better; bars use a zero baseline.", "Smallx"), Image(str(chart_dir / "fixed_eval_ce.png"), width=162 * mm, height=82 * mm), Spacer(1, 4 * mm)]
    ce_rows = [["Context", "nanoGPT CE", "Hierarchy CE", "Outcome", "Evaluation shape"]]
    for idx, c in enumerate(contexts):
        if n_ce[idx] < h_ce[idx]:
            outcome = f"nanoGPT {pct_lower(n_ce[idx], h_ce[idx]):.2f}% lower"
        else:
            outcome = f"Hierarchy {pct_lower(h_ce[idx], n_ce[idx]):.2f}% lower"
        ce_rows.append([f"{c:,}", f"{n_ce[idx]:.6f}", f"{h_ce[idx]:.6f}", outcome, data["fixed_eval"]["eval_shapes"][idx]])
    story += [table(ce_rows, [22 * mm, 30 * mm, 30 * mm, 39 * mm, 41 * mm]), Spacer(1, 7 * mm)]
    story += [p("Interpretation", "H2x"), p("nanoGPT's fixed CE is 2.54% lower at the trained 512-token context. The hierarchy reverses the result at 1,024 tokens and widens the advantage at 2,048 tokens. The long-context samples are small, so these points should be treated as strong directional evidence rather than a confidence-bounded population estimate.")]
    story += [PageBreak()]

    # Page 4 - state scaling.
    story += [p("STATE SCALING", "Kicker"), p("Fixed recurrent state versus linearly growing KV state", "H1x")]
    story += [p("Per-sequence BF16 state on a log scale. This is architecture accounting, not total GPU memory.", "Smallx"), Image(str(chart_dir / "state_scaling.png"), width=162 * mm, height=82 * mm), Spacer(1, 3 * mm)]
    state_rows = [["Context", "nanoGPT KV state", "Hierarchy state", "nanoGPT / hierarchy", "Hierarchy reduction"]]
    for idx, c in enumerate(contexts):
        state_rows.append([
            f"{c:,}",
            human_bytes(data["state"]["nanogpt_bytes"][idx]),
            human_bytes(data["state"]["hierarchy_bytes"][idx]),
            f"{ratios[idx]:,.2f}x",
            f"{reductions[idx]:.4f}% fewer",
        ])
    story += [table(state_rows, [22 * mm, 36 * mm, 32 * mm, 35 * mm, 37 * mm]), Spacer(1, 6 * mm)]
    formula_rows = [
        ["Architecture", "Analytic BF16 state formula"],
        ["Pure hierarchy", "4 recurrent tensors x width 688 x 2 bytes = 5,504 bytes"],
        ["Official nanoGPT", "K/V x 21 layers x 7 heads x 64 head dimension x context x 2 bytes"],
    ]
    story += [table(formula_rows, [42 * mm, 120 * mm]), Spacer(1, 6 * mm)]
    story += [p("What this does and does not mean", "H2x"), p("These are nominal per-sequence state values. They do not include parameters, activations, optimizer state, allocator overhead, or total runtime memory. The nanoGPT adapter did not directly materialize or measure an incremental KV cache; its values are architecture accounting.")]
    story += [PageBreak()]

    # Page 5 - training and benchmark charts.
    story += [p("SUPPORTING RESULTS", "Kicker"), p("Matched checkpoints and downstream benchmark", "H1x")]
    chart_table = Table(
        [
            [p("Matched training-minibatch CE", "TableCellBold"), p("Four-task benchmark (limit 1,000)", "TableCellBold")],
            [Image(str(chart_dir / "matched_training_ce.png"), width=79 * mm, height=50 * mm), Image(str(chart_dir / "benchmark_scores.png"), width=79 * mm, height=50 * mm)],
        ],
        colWidths=[81 * mm, 81 * mm],
    )
    chart_table.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 2)]))
    story += [chart_table, Spacer(1, 5 * mm)]
    training_rows = [["Step", "Tokens", "nanoGPT CE", "Hierarchy CE", "Lower CE"]]
    for idx, step in enumerate(data["training_checkpoints"]["steps"]):
        nn = data["training_checkpoints"]["nanogpt_ce"][idx]
        hh = data["training_checkpoints"]["hierarchy_ce"][idx]
        training_rows.append([f"{step:,}", f"{data['training_checkpoints']['tokens'][idx]:,}", f"{nn:.6f}", f"{hh:.6f}", f"nanoGPT {pct_lower(nn, hh):.2f}%"])
    story += [p("Matched deterministic training minibatches", "H2x"), table(training_rows, [25 * mm, 36 * mm, 32 * mm, 32 * mm, 37 * mm]), Spacer(1, 2 * mm), p("nanoGPT rows come from the included 246-record training log. Historical rows are preserved transcript values; the primary conclusion uses fixed evaluation.", "Tinyx"), Spacer(1, 4 * mm)]
    benchmark_rows = [["Task", "nanoGPT", "Hierarchy", "Higher score"]]
    for task, nn, hh in zip(data["benchmark"]["tasks"], data["benchmark"]["nanogpt"], data["benchmark"]["hierarchy"]):
        winner = "Tie" if nn == hh else "nanoGPT" if nn > hh else "Hierarchy"
        benchmark_rows.append([task, f"{nn:.6f}", f"{hh:.6f}", winner])
    story += [p("Four-task benchmark", "H2x"), table(benchmark_rows, [44 * mm, 38 * mm, 38 * mm, 42 * mm])]
    story += [Spacer(1, 5 * mm), p("nanoGPT's benchmark mean is 0.006132 points higher, a 2.19% relative advantage. The benchmark is one seed with no confidence interval.", "Smallx"), PageBreak()]

    # Page 6 - interpretation and limitations.
    story += [p("INTERPRETATION", "Kicker"), p("The result is a long-context state-efficiency result", "H1x")]
    conclusions = [
        ("1", "At 512 tokens", "nanoGPT has 2.54% lower fixed CE and is therefore the stronger model at the trained context."),
        ("2", "At 1,024 tokens", f"Hierarchy has {pct_lower(h_ce[1], n_ce[1]):.2f}% lower CE while nanoGPT's theoretical state is {ratios[1]:,.1f}x as large."),
        ("3", "At 2,048 tokens", f"Hierarchy has {pct_lower(h_ce[2], n_ce[2]):.2f}% lower CE while nanoGPT's theoretical state is {ratios[2]:,.1f}x as large."),
        ("4", "Across the benchmark", "nanoGPT has a modest 2.19% relative mean advantage; PIQA favors hierarchy, ARC-Easy and HellaSwag favor nanoGPT, and LAMBADA ties at zero."),
    ]
    conclusion_rows = []
    for number, title, body in conclusions:
        badge = Paragraph(number, ParagraphStyle(name=f"Badge{number}", fontName=bold_font, fontSize=15, textColor=colors.white, alignment=TA_CENTER))
        conclusion_rows.append([badge, Paragraph(f"<b>{title}</b><br/>{body}", styles["Bodyx"])])
    conclusion_table = Table(conclusion_rows, colWidths=[12 * mm, 150 * mm], rowHeights=[21 * mm] * len(conclusion_rows))
    conclusion_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor(TEAL)),
        ("BACKGROUND", (1, 0), (1, -1), colors.HexColor(MIST)),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor(GRID)),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor(GRID)),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (1, 0), (1, -1), 10),
        ("RIGHTPADDING", (1, 0), (1, -1), 10),
    ]))
    story += [conclusion_table, Spacer(1, 8 * mm), p("Bottom line", "H2x"), p("At context 2,048, the proven hierarchy combines lower CE with 99.9929% fewer analytic state bytes. nanoGPT remains the better result at the trained 512-token context and on the downstream benchmark mean. Both facts are necessary for an accurate conclusion.")]
    story += [PageBreak()]

    # Page 7 - provenance.
    story += [p("PROVENANCE", "Kicker"), p("Content-addressed result evidence", "H1x"), p("The hashes below bind the report to the historical hierarchy artifacts, the frozen official nanoGPT package, the completed training output, and the downstream benchmark output.")]
    prov = data["provenance"]
    prov_rows = [["Artifact", "SHA-256 / commit"]]
    labels = [
        ("Hierarchy config", "hierarchy_config_sha256"),
        ("Hierarchy result", "hierarchy_result_sha256"),
        ("Hierarchy benchmark", "hierarchy_benchmark_sha256"),
        ("Hierarchy checkpoint", "hierarchy_checkpoint_sha256"),
        ("nanoGPT frozen manifest", "nanogpt_manifest_sha256"),
        ("nanoGPT result", "nanogpt_result_sha256"),
        ("nanoGPT training completion", "nanogpt_training_completion_sha256"),
        ("nanoGPT final model", "nanogpt_final_model_sha256"),
        ("nanoGPT training log", "nanogpt_training_log_sha256"),
        ("nanoGPT benchmark", "nanogpt_benchmark_sha256"),
        ("nanoGPT benchmark completion", "nanogpt_benchmark_completion_sha256"),
        ("Upstream nanoGPT commit", "nanogpt_upstream_commit"),
        ("Pinned nanoGPT model.py", "nanogpt_model_sha256"),
        ("Training data", "train_data_sha256"),
        ("Validation data", "val_data_sha256"),
        ("Teacher weights", "teacher_weights_sha256"),
    ]
    for label, key in labels:
        prov_rows.append([label, prov[key]])
    formatted = []
    for r_idx, row in enumerate(prov_rows):
        if r_idx == 0:
            formatted.append([Paragraph(row[0], styles["TableHead"]), Paragraph(row[1], styles["TableHead"])])
        else:
            formatted.append([Paragraph(row[0], styles["TableCellBold"]), Paragraph(row[1], styles["Hash"])])
    prov_table = Table(formatted, colWidths=[48 * mm, 114 * mm], repeatRows=1)
    prov_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(INK)),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor(GRID)),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
    ]))
    story += [prov_table, Spacer(1, 6 * mm)]
    source_sha = hashlib.sha256(source_md.read_bytes()).hexdigest()
    try:
        source_label = source_md.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        source_label = source_md.name
    story += [p("Source report", "H2x"), p(source_label, "Smallx"), p(f"Source report SHA-256: {source_sha}", "Hash")]
    story += [Spacer(1, 6 * mm), HRFlowable(width="100%", thickness=0.8, color=colors.HexColor(GRID)), Spacer(1, 4 * mm), p("Prepared from immutable experiment outputs. All numerical state comparisons are analytic BF16 per-sequence architecture accounting.", "Smallx")]

    doc.build(story)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["charts", "pdf"], required=True)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--source-md", type=Path, default=DEFAULT_MD)
    parser.add_argument("--chart-dir", type=Path, default=DEFAULT_CHART_DIR)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    args = parser.parse_args()

    data = load_data(args.data)
    validate_against_markdown(data, args.source_md)
    if args.mode == "charts":
        paths = make_charts(data, args.chart_dir)
        print(json.dumps({key: str(path) for key, path in paths.items()}, indent=2))
    else:
        required = ["fixed_eval_ce.png", "state_scaling.png", "matched_training_ce.png", "benchmark_scores.png"]
        missing = [name for name in required if not (args.chart_dir / name).is_file()]
        if missing:
            raise RuntimeError(f"missing charts: {missing}")
        build_pdf(data, args.chart_dir, args.pdf, args.source_md)
        print(args.pdf)


if __name__ == "__main__":
    main()
