#!/usr/bin/env python3
"""Render a real 3GPP TS 38.212 base-graph connectivity pattern as a Tanner graph.

Reads data/bg_tables.json — the same real, spec-extracted base-graph table the
Rust crate compiles into BG1_ENTRIES/BG2_ENTRIES at build time (see build.rs) —
and draws a bipartite graph (variable nodes left, check nodes right) from a
small window of its actual non-zero entries. No connectivity here is invented:
every edge drawn corresponds to a real (row, col) pair present in the table,
i.e. a real non-zero Z x Z circulant sub-block of the 3GPP base graph.

Usage:
    cd <repo_root>
    python bench/dashboard/gen_tanner_graph.py

Requirements:
    pip install matplotlib
"""

import json
from pathlib import Path

DATA = Path("data/bg_tables.json")
EXPORTS = Path("bench/dashboard/exports")

# Same dark-surface palette as gen_charts.py, so this sits visually with the
# rest of the dashboard rather than looking like a different document.
BG = "#1e293b"
GRIDCOLOR = "#2d3748"
TEXT = "#e2e8f0"
DIM = "#94a3b8"
VAR_COLOR = "#ea580c"  # variable nodes — same hue gen_charts.py uses for "encode"
CHECK_COLOR = "#3b82f6"  # check nodes — same hue gen_charts.py uses for "decode"

# A legible window into BG2's real connectivity: the first few check-node
# rows and variable-node columns. BG2 full size is 42 x 52 (2,184 possible
# positions); rendering all of it would be an illegible smear of edges, so
# this crops to a corner small enough to read while keeping every edge real.
BASE_GRAPH = "bg2"
ROWS_SHOWN = 8
COLS_SHOWN = 14


def mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def main():
    data = json.loads(DATA.read_text())
    bg = data[BASE_GRAPH]
    total_rows, total_cols = bg["rows"], bg["cols"]

    # Real edges only: keep entries whose (row, col) both fall inside the
    # displayed window. Nothing here is synthesized — this filters the actual
    # 3GPP-derived entry list built into the crate.
    edges = [
        (e["r"], e["c"])
        for e in bg["entries"]
        if e["r"] < ROWS_SHOWN and e["c"] < COLS_SHOWN
    ]
    if not edges:
        raise SystemExit(f"no entries found in the {ROWS_SHOWN}x{COLS_SHOWN} window — widen it")

    plt = mpl()
    fig, ax = plt.subplots(figsize=(9, 6.5), facecolor=BG)
    ax.set_facecolor(BG)
    ax.axis("off")

    var_x, check_x = 0.15, 0.85
    var_y = {c: 1.0 - (c + 0.5) / COLS_SHOWN for c in range(COLS_SHOWN)}
    check_y = {r: 1.0 - (r + 0.5) / ROWS_SHOWN for r in range(ROWS_SHOWN)}

    for r, c in edges:
        ax.plot(
            [var_x, check_x],
            [var_y[c], check_y[r]],
            color=GRIDCOLOR,
            linewidth=0.8,
            alpha=0.55,
            zorder=1,
        )

    for c in range(COLS_SHOWN):
        ax.scatter([var_x], [var_y[c]], s=220, color=VAR_COLOR, zorder=2, edgecolors=BG, linewidths=1.5)
        ax.text(var_x - 0.05, var_y[c], f"v{c}", color=TEXT, fontsize=8, ha="right", va="center")

    for r in range(ROWS_SHOWN):
        ax.scatter(
            [check_x], [check_y[r]], s=220, color=CHECK_COLOR, marker="s", zorder=2,
            edgecolors=BG, linewidths=1.5,
        )
        ax.text(check_x + 0.05, check_y[r], f"c{r}", color=TEXT, fontsize=8, ha="left", va="center")

    ax.text(var_x, 1.06, "Variable nodes", color=VAR_COLOR, fontsize=10, ha="center", weight="bold")
    ax.text(check_x, 1.06, "Check nodes", color=CHECK_COLOR, fontsize=10, ha="center", weight="bold")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.12)

    ax.set_title(
        f"3GPP TS 38.212 {BASE_GRAPH.upper()} connectivity — real base-graph entries, "
        f"first {ROWS_SHOWN} check rows x {COLS_SHOWN} variable columns\n"
        f"(full {BASE_GRAPH.upper()} is {total_rows} x {total_cols}; {len(edges)} of its "
        f"{len(bg['entries'])} total non-zero blocks fall in this window)",
        color=DIM,
        fontsize=9,
        pad=14,
    )

    EXPORTS.mkdir(parents=True, exist_ok=True)
    out = EXPORTS / "tanner_graph.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=BG)
    print(f"wrote {out} ({len(edges)} real edges from {DATA})")


if __name__ == "__main__":
    main()
