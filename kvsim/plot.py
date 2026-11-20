"""Dependency-free SVG charts: line charts for the scaling curves and the roofline.

The files open in any browser and can be dropped into a report. Only what the
project needs is implemented: linear or log10 axes, several labelled series,
dashed roofs and scatter points.
"""

from __future__ import annotations

import math
from html import escape
from pathlib import Path

WIDTH, HEIGHT = 720, 440
LEFT, RIGHT, TOP, BOTTOM = 80, 170, 48, 64
COLORS = ("#2563eb", "#ea580c", "#16a34a", "#dc2626", "#9333ea", "#0891b2", "#ca8a04", "#4b5563")
INK, GRID, MUTED = "#111827", "#e5e7eb", "#6b7280"


class Axis:
    def __init__(self, values, log: bool, lo_px: float, hi_px: float) -> None:
        vals = [v for v in values if v is not None and (v > 0 or not log)]
        if not vals:
            vals = [1.0, 10.0] if log else [0.0, 1.0]
        lo, hi = min(vals), max(vals)
        self.log = log
        if log:
            lo, hi = 10 ** math.floor(math.log10(lo)), 10 ** math.ceil(math.log10(hi))
            if lo == hi:
                hi = lo * 10
        else:
            lo = min(lo, 0.0)
            if hi == lo:
                hi = lo + 1
            step = _nice_step((hi - lo) / 5)
            hi = math.ceil(hi / step) * step
        self.lo, self.hi, self.lo_px, self.hi_px = lo, hi, lo_px, hi_px

    def _t(self, v: float) -> float:
        return math.log10(v) if self.log else v

    def __call__(self, v: float) -> float:
        a, b = self._t(self.lo), self._t(self.hi)
        return self.lo_px + (self._t(v) - a) / (b - a) * (self.hi_px - self.lo_px)

    def ticks(self) -> list[float]:
        if self.log:
            decades = range(round(math.log10(self.lo)), round(math.log10(self.hi)) + 1)
            ticks = [10.0 ** d for d in decades]
            if len(ticks) <= 3:  # few decades: add 2 and 5
                ticks = sorted(t * m for t in ticks for m in (1, 2, 5) if self.lo <= t * m <= self.hi)
            return ticks
        step = _nice_step((self.hi - self.lo) / 5)
        n = int(round((self.hi - self.lo) / step))
        return [self.lo + i * step for i in range(n + 1)]


def _nice_step(raw: float) -> float:
    if raw <= 0:
        return 1.0
    power = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        if raw <= m * power:
            return m * power
    return 10 * power


def _fmt(v: float) -> str:
    if v == 0:
        return "0"
    a = abs(v)
    for limit, suffix in ((1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "k")):
        if a >= limit:
            return f"{v / limit:g}{suffix}"
    return f"{v:g}" if a >= 0.01 else f"{v:.0e}"


def _frame(title: str, xlabel: str, ylabel: str, x: Axis, y: Axis) -> list[str]:
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
        f'viewBox="0 0 {WIDTH} {HEIGHT}" font-family="Segoe UI, Helvetica, Arial, sans-serif">',
        f'<rect width="{WIDTH}" height="{HEIGHT}" fill="#ffffff"/>',
        f'<text x="{LEFT}" y="28" font-size="16" font-weight="600" fill="{INK}">{escape(title)}</text>',
    ]
    for t in x.ticks():
        px = x(t)
        out.append(f'<line x1="{px:.1f}" y1="{TOP}" x2="{px:.1f}" y2="{HEIGHT - BOTTOM}" stroke="{GRID}"/>')
        out.append(f'<text x="{px:.1f}" y="{HEIGHT - BOTTOM + 18}" font-size="11" fill="{MUTED}" '
                   f'text-anchor="middle">{_fmt(t)}</text>')
    for t in y.ticks():
        py = y(t)
        out.append(f'<line x1="{LEFT}" y1="{py:.1f}" x2="{WIDTH - RIGHT}" y2="{py:.1f}" stroke="{GRID}"/>')
        out.append(f'<text x="{LEFT - 8}" y="{py + 4:.1f}" font-size="11" fill="{MUTED}" '
                   f'text-anchor="end">{_fmt(t)}</text>')
    out += [
        f'<rect x="{LEFT}" y="{TOP}" width="{WIDTH - LEFT - RIGHT}" height="{HEIGHT - TOP - BOTTOM}" '
        f'fill="none" stroke="{MUTED}"/>',
        f'<text x="{(LEFT + WIDTH - RIGHT) / 2}" y="{HEIGHT - 20}" font-size="12" fill="{INK}" '
        f'text-anchor="middle">{escape(xlabel)}</text>',
        f'<text transform="translate(20 {(TOP + HEIGHT - BOTTOM) / 2}) rotate(-90)" font-size="12" '
        f'fill="{INK}" text-anchor="middle">{escape(ylabel)}</text>',
    ]
    return out


def _legend(out: list[str], labels: list[tuple[str, str, bool]]) -> None:
    x0 = WIDTH - RIGHT + 16
    for i, (label, color, dashed) in enumerate(labels):
        y0 = TOP + 10 + i * 20
        dash = ' stroke-dasharray="6 4"' if dashed else ""
        out.append(f'<line x1="{x0}" y1="{y0}" x2="{x0 + 22}" y2="{y0}" stroke="{color}" '
                   f'stroke-width="2.5"{dash}/>')
        out.append(f'<text x="{x0 + 28}" y="{y0 + 4}" font-size="11" fill="{INK}">{escape(label)}</text>')


def line_chart(series: list[tuple[str, list[float], list[float]]], title: str, xlabel: str,
               ylabel: str, logx: bool = False, logy: bool = False) -> str:
    """SVG text of a line chart; ``series`` is ``[(label, xs, ys), ...]``."""
    xs = [v for _, sx, _ in series for v in sx]
    ys = [v for _, _, sy in series for v in sy]
    x = Axis(xs, logx, LEFT, WIDTH - RIGHT)
    y = Axis(ys, logy, HEIGHT - BOTTOM, TOP)
    out = _frame(title, xlabel, ylabel, x, y)
    legend = []
    for i, (label, sx, sy) in enumerate(series):
        color = COLORS[i % len(COLORS)]
        pts = [(x(a), y(b)) for a, b in zip(sx, sy)
               if b is not None and (not logx or a > 0) and (not logy or b > 0)]
        if not pts:
            continue
        path = " ".join(f"{px:.1f},{py:.1f}" for px, py in pts)
        out.append(f'<polyline points="{path}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        out += [f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3" fill="{color}"/>' for px, py in pts]
        legend.append((label, color, False))
    _legend(out, legend)
    out.append("</svg>")
    return "\n".join(out)


def roofline_chart(peak_flops: float, bandwidth: float, points: list[tuple[str, float, float]],
                   title: str) -> str:
    """SVG roofline: attainable FLOP/s = min(peak, bandwidth x intensity), plus measured points.

    ``points`` is ``[(label, intensity FLOPs/byte, achieved FLOP/s), ...]``; points with the
    same label share a colour.
    """
    ridge = peak_flops / bandwidth
    intens = [p[1] for p in points if p[1] > 0] + [ridge / 100, ridge * 100]
    x = Axis(intens, True, LEFT, WIDTH - RIGHT)
    lo_perf = min([p[2] for p in points if p[2] > 0] + [bandwidth * x.lo])
    y = Axis([lo_perf, peak_flops * 1.5], True, HEIGHT - BOTTOM, TOP)
    out = _frame(title, "arithmetic intensity (FLOPs / byte)", "FLOP/s", x, y)
    roof = [(x.lo, bandwidth * x.lo), (ridge, peak_flops), (x.hi, peak_flops)]
    roof = [(a, max(b, y.lo)) for a, b in roof]
    path = " ".join(f"{x(a):.1f},{y(b):.1f}" for a, b in roof)
    out.append(f'<polyline points="{path}" fill="none" stroke="{INK}" stroke-width="2.5"/>')
    out.append(f'<line x1="{x(ridge):.1f}" y1="{y(peak_flops):.1f}" x2="{x(ridge):.1f}" '
               f'y2="{HEIGHT - BOTTOM}" stroke="{MUTED}" stroke-dasharray="4 4"/>')
    out.append(f'<text x="{x(ridge) + 6:.1f}" y="{HEIGHT - BOTTOM - 8}" font-size="11" fill="{MUTED}">'
               f'ridge {ridge:.0f}</text>')
    out.append(f'<text x="{x(x.hi) - 6:.1f}" y="{y(peak_flops) - 6:.1f}" font-size="11" fill="{MUTED}" '
               f'text-anchor="end">peak {_fmt(peak_flops)}FLOP/s</text>')
    colors: dict[str, str] = {}
    for label, inten, perf in points:
        if inten <= 0 or perf <= 0:
            continue
        color = colors.setdefault(label, COLORS[len(colors) % len(COLORS)])
        out.append(f'<circle cx="{x(inten):.1f}" cy="{y(perf):.1f}" r="4" fill="{color}" '
                   f'fill-opacity="0.75"/>')
    _legend(out, [("roof", INK, False)] + [(k, v, False) for k, v in colors.items()])
    out.append("</svg>")
    return "\n".join(out)


def save_svg(svg: str, path: str | Path) -> None:
    Path(path).write_text(svg, encoding="utf-8")
