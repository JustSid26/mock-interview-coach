"""Generate downloadable PDF session reports."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from io import BytesIO
from statistics import mean
from typing import Any

from fpdf import FPDF

LIGHT_RED = (254, 226, 226)
LIGHT_RED_HEADER = (252, 165, 165)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BORDER = (220, 38, 38)

FONT = "Times"
MARGIN = 15
CONTENT_W = 180
LABEL_W = 50
VALUE_W = CONTENT_W - LABEL_W
LINE_H = 5


def _strip_markdown(text: str) -> str:
    text = re.sub(r"```[\s\S]*?```", "", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[-*]\s+", "- ", text, flags=re.MULTILINE)
    return text.strip()


def _safe(text: str) -> str:
    return (
        str(text)
        .replace("\u2014", "-")
        .replace("\u2013", "-")
        .replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .encode("latin-1", errors="replace")
        .decode("latin-1")
    )


def _dimension_averages(turn_history: list[dict[str, Any]] | None) -> dict[str, float]:
    if not turn_history:
        return {}
    dims = [
        "clarity",
        "technical_accuracy",
        "communication",
        "confidence",
        "depth",
        "structure",
        "problem_solving",
        "relevance",
    ]
    result: dict[str, float] = {}
    for dim in dims:
        vals = [
            t.get("evaluation", {}).get("scores", {}).get(dim)
            for t in turn_history
            if t.get("evaluation", {}).get("scores", {}).get(dim) is not None
        ]
        if vals:
            result[dim] = round(mean(vals), 1)
    return result


class InterviewReportPDF(FPDF):
    def __init__(self) -> None:
        super().__init__()
        self.set_margins(MARGIN, MARGIN, MARGIN)
        self.set_auto_page_break(auto=True, margin=22)

    def _paint_page(self) -> None:
        self.set_fill_color(*LIGHT_RED)
        self.rect(0, 0, self.w, self.h, style="F")

    def header(self) -> None:
        self._paint_page()
        y0 = 12
        self.set_fill_color(*LIGHT_RED_HEADER)
        self.set_draw_color(*BORDER)
        self.rect(MARGIN, y0, CONTENT_W, 12, style="FD")
        self.set_xy(MARGIN, y0 + 2)
        self.set_font(FONT, "B", 14)
        self.set_text_color(*BLACK)
        self.cell(CONTENT_W, 8, _safe("Mock Interview Coach - Session Report"), align="C")
        self.set_y(y0 + 16)

    def footer(self) -> None:
        self.set_y(-12)
        self.set_font(FONT, "I", 9)
        self.set_text_color(*BLACK)
        self.cell(0, 8, _safe(f"Page {self.page_no()}"), align="C")

    def _need(self, h: float) -> None:
        if self.get_y() + h > self.page_break_trigger:
            self.add_page()

    def section_banner(self, title: str) -> None:
        self._need(12)
        self.set_fill_color(*LIGHT_RED_HEADER)
        self.set_draw_color(*BORDER)
        self.set_font(FONT, "B", 12)
        self.set_text_color(*BLACK)
        self.cell(CONTENT_W, 9, _safe(f"  {title}"), border=1, fill=True)
        self.ln(11)

    def _line_count(self, text: str, width: float, size: int = 10) -> int:
        self.set_font(FONT, "", size)
        lines = self.multi_cell(width, LINE_H, _safe(text), split_only=True)
        return max(1, len(lines))

    def table_row(
        self,
        cells: list[str],
        widths: list[float],
        *,
        header: bool = False,
        aligns: list[str] | None = None,
        stripe_white: bool = True,
    ) -> None:
        aligns = aligns or ["L"] * len(cells)
        size = 10
        style = "B" if header else ""

        heights = []
        for text, w in zip(cells, widths):
            self.set_font(FONT, style, size)
            heights.append(self._line_count(text, w - 2, size) * LINE_H + 2)
        row_h = max(8, *heights)

        self._need(row_h + 1)
        x0, y0 = MARGIN, self.get_y()
        fill = LIGHT_RED_HEADER if header else (WHITE if stripe_white else LIGHT_RED)

        x = x0
        for i, (text, w) in enumerate(zip(cells, widths)):
            self.set_xy(x, y0)
            self.set_fill_color(*fill)
            self.set_draw_color(*BORDER)
            self.set_text_color(*BLACK)
            self.set_font(FONT, style, size)
            self.rect(x, y0, w, row_h, style="FD")
            self.set_xy(x + 1, y0 + 1)
            self.multi_cell(w - 2, LINE_H, _safe(text), align=aligns[i])
            x += w

        self.set_xy(x0, y0 + row_h)

    def label_value_row_bold(self, label: str, value: str, *, stripe_white: bool = True) -> None:
        """Two-column row with bold label cell."""
        self.set_font(FONT, "B", 10)
        label_lines = self._line_count(label, LABEL_W - 2, 10)
        self.set_font(FONT, "", 10)
        value_lines = self._line_count(value, VALUE_W - 2, 10)
        row_h = max(8, label_lines * LINE_H + 2, value_lines * LINE_H + 2)

        self._need(row_h + 1)
        x0, y0 = MARGIN, self.get_y()
        fill = WHITE if stripe_white else LIGHT_RED

        # Label cell
        self.set_fill_color(*fill)
        self.set_draw_color(*BORDER)
        self.rect(x0, y0, LABEL_W, row_h, style="FD")
        self.set_xy(x0 + 1, y0 + 1)
        self.set_font(FONT, "B", 10)
        self.set_text_color(*BLACK)
        self.multi_cell(LABEL_W - 2, LINE_H, _safe(label))

        # Value cell
        self.set_fill_color(*fill)
        self.rect(x0 + LABEL_W, y0, VALUE_W, row_h, style="FD")
        self.set_xy(x0 + LABEL_W + 1, y0 + 1)
        self.set_font(FONT, "", 10)
        self.multi_cell(VALUE_W - 2, LINE_H, _safe(value))

        self.set_xy(x0, y0 + row_h)

    def body_box(self, text: str) -> None:
        self.set_font(FONT, "", 10)
        lines = self._line_count(text, CONTENT_W - 6, 10)
        box_h = lines * LINE_H + 6
        self._need(box_h + 4)

        x0, y0 = MARGIN, self.get_y()
        self.set_fill_color(*WHITE)
        self.set_draw_color(*BORDER)
        self.rect(x0, y0, CONTENT_W, box_h, style="FD")
        self.set_xy(x0 + 3, y0 + 3)
        self.set_text_color(*BLACK)
        self.multi_cell(CONTENT_W - 6, LINE_H, _safe(text))
        self.set_xy(x0, y0 + box_h + 4)


def build_session_pdf(
    *,
    session_id: str,
    role: str,
    focus: str,
    background: str,
    average_score: float,
    coach_feedback: str,
    turn_reviews: list[dict[str, Any]],
    turn_history: list[dict[str, Any]] | None = None,
) -> bytes:
    pdf = InterviewReportPDF()
    pdf.add_page()

    pdf.set_font(FONT, "I", 9)
    pdf.set_text_color(*BLACK)
    pdf.cell(
        CONTENT_W,
        5,
        _safe(f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"),
        align="R",
    )
    pdf.ln(8)

    # Session overview
    pdf.section_banner("Session Overview")
    pdf.table_row(["Field", "Value"], [LABEL_W, VALUE_W], header=True, aligns=["L", "L"])
    overview = [
        ("Session ID", session_id),
        ("Target role", role),
        ("Interview focus", focus),
        ("Background", background or "Not provided"),
        ("Average score", f"{average_score} / 10"),
        ("Total questions", str(len(turn_reviews))),
    ]
    for i, (label, val) in enumerate(overview):
        pdf.label_value_row_bold(label, val, stripe_white=(i % 2 == 0))
    pdf.ln(5)

    # Question scores grid
    if turn_reviews:
        pdf.section_banner("Question Scores")
        qw = [14, 26, 22, CONTENT_W - 62]
        pdf.table_row(
            ["#", "Difficulty", "Score", "Topic / Intent"],
            qw,
            header=True,
            aligns=["C", "C", "C", "L"],
        )
        for i, item in enumerate(turn_reviews):
            score = item.get("score")
            score_txt = f"{score}/10" if score is not None else "-"
            pdf.table_row(
                [
                    str(item.get("turn_number", i + 1)),
                    str(item.get("difficulty", "-")),
                    score_txt,
                    str(item.get("intent", ""))[:100],
                ],
                qw,
                aligns=["C", "C", "C", "L"],
                stripe_white=(i % 2 == 0),
            )
        pdf.ln(5)

    # Dimension averages
    dim_avgs = _dimension_averages(turn_history)
    if dim_avgs:
        pdf.section_banner("Score by Dimension (Session Average)")
        dw = [100, 80]
        pdf.table_row(
            ["Dimension", "Average (/ 10)"],
            dw,
            header=True,
            aligns=["L", "C"],
        )
        for i, (dim, avg) in enumerate(dim_avgs.items()):
            pdf.table_row(
                [dim.replace("_", " ").title(), str(avg)],
                dw,
                aligns=["L", "C"],
                stripe_white=(i % 2 == 0),
            )
        pdf.ln(5)

    # Coach review
    pdf.add_page()
    pdf.section_banner("Session Review (Coach Debrief)")
    feedback = _strip_markdown(coach_feedback)
    for block in feedback.split("\n\n"):
        if block.strip():
            pdf.body_box(block.strip())

    # Q&A section
    pdf.add_page()
    pdf.section_banner("Questions, Your Answers & Model Answers")

    for item in turn_reviews:
        n = item.get("turn_number", "?")
        pdf._need(14)
        pdf.set_fill_color(*LIGHT_RED_HEADER)
        pdf.set_draw_color(*BORDER)
        pdf.set_font(FONT, "B", 11)
        pdf.set_text_color(*BLACK)
        pdf.cell(CONTENT_W, 8, _safe(f"  Question {n}"), border=1, fill=True)
        pdf.ln(10)

        rows = [
            ("Question", item.get("question", "")),
            ("Your answer", item.get("candidate_answer", "")),
            ("Strong answer (exemplar)", item.get("ideal_answer", "")),
        ]
        points = item.get("key_points") or []
        if points:
            rows.append(("Key points to hit", "\n".join(f"- {p}" for p in points)))

        for i, (label, val) in enumerate(rows):
            pdf.label_value_row_bold(label, val, stripe_white=(i % 2 == 0))
        pdf.ln(6)

    out = BytesIO()
    pdf.output(out)
    return out.getvalue()
