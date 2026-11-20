"""Terminal tables and CSV export shared by all CLI commands."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

UNITS = ["B", "KiB", "MiB", "GiB", "TiB"]


def fmt_bytes(n: float) -> str:
    sign = "-" if n < 0 else ""
    value = float(abs(n))
    for unit in UNITS:
        if value < 1024 or unit == UNITS[-1]:
            return f"{sign}{value:.0f} {unit}" if unit == "B" else f"{sign}{value:.2f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


def _is_bytes_column(name: str) -> bool:
    return name == "bytes" or name.endswith("_bytes") or name.startswith("bytes_")


def _fmt_cell(column: str, value) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)) and _is_bytes_column(column):
        return fmt_bytes(value)
    if column.startswith("share_") and isinstance(value, float):
        return f"{value:.1%}"
    if isinstance(value, float):
        return f"{value:.3g}" if abs(value) < 1000 else f"{value:,.0f}"
    if isinstance(value, int):
        return f"{value:,}"
    return "" if value is None else str(value)


def print_table(rows: list[dict], title: str | None = None, out=None) -> None:
    out = out or sys.stdout
    if title:
        print(title, file=out)
    if not rows:
        print("(no rows)", file=out)
        return
    columns = list(rows[0])
    cells = [[_fmt_cell(c, row.get(c)) for c in columns] for row in rows]
    widths = [max(len(c), *(len(r[i]) for r in cells)) for i, c in enumerate(columns)]

    def line(values):
        return "  ".join(
            v.ljust(w) if i == 0 else v.rjust(w) for i, (v, w) in enumerate(zip(values, widths))
        )

    print(line(columns), file=out)
    print("  ".join("-" * w for w in widths), file=out)
    for r in cells:
        print(line(r), file=out)


def write_csv(rows: list[dict], path: str | Path) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
