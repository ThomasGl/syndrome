#!/usr/bin/env python3
"""Render the LDPC iterative-decode convergence GIF from real decoder output.

Reads bench/results/ldpc_convergence.json — written by
`cargo run --release --bin ldpc_convergence_export`, which runs one real
802.11 Wi-Fi LDPC encode -> BPSK/AWGN corrupt -> layered offset min-sum
(LOMS) decode cycle and records the hard-decision bits after every
completed iteration — and renders it to an animated GIF.

Nothing in this script invents an error pattern or a convergence curve: every
pixel and every point on the error-count line comes straight from the JSON,
which in turn comes straight from one continuous decoder run. This script
only chooses colors, layout, and frame timing.

Usage:
    cd <repo_root>
    python bench/dashboard/gen_convergence_gif.py

Requirements:
    pip install matplotlib pillow numpy
"""

import io
import json
import sys
from pathlib import Path

RESULTS = Path("bench/results")
EXPORTS = Path("bench/dashboard/exports")
SRC = RESULTS / "ldpc_convergence.json"
OUT = EXPORTS / "ldpc_convergence.gif"

# ─── style — matches bench/dashboard/gen_charts.py ───────────────────────────
BG        = "#1e293b"
GRIDCOLOR = "#2d3748"
TEXT      = "#e2e8f0"
DIM       = "#94a3b8"

# Reused from the validated palette already in gen_charts.py (dark-surface
# lightness/chroma/CVD/contrast checks already passed for this surface).
CORRECT_COLOR = "#22c55e"  # green — bit matches the transmitted codeword
ERROR_COLOR   = "#ef4444"  # red   — bit differs from the transmitted codeword
LINE_COLOR    = "#ea580c"  # orange — bit-error-count curve


def style_ax(ax, xlabel=None, ylabel=None, title=None, grid=True):
    ax.set_facecolor(BG)
    ax.tick_params(colors=DIM, labelcolor=DIM)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRIDCOLOR)
    if grid:
        ax.grid(True, color=GRIDCOLOR, linestyle="--", linewidth=0.6, alpha=0.7)
    if xlabel:
        ax.set_xlabel(xlabel, color=DIM)
    if ylabel:
        ax.set_ylabel(ylabel, color=DIM)
    if title:
        ax.set_title(title, color=TEXT, fontsize=11, pad=10)


def load_data():
    if not SRC.exists():
        print(
            f"  [skip] {SRC} not found "
            "(run: cargo run --release --bin ldpc_convergence_export)",
            file=sys.stderr,
        )
        return None
    with open(SRC) as f:
        return json.load(f)


def render_frame(plt, np, data, frame_idx):
    """Render one frame: bit grid (left) + error-count curve (right)."""
    code = data["code"]
    z, n = code["z"], code["n"]
    channel = data["channel"]
    frames = data["frames"]
    transmitted = np.array(data["transmitted_codeword"], dtype=np.uint8)

    frame = frames[frame_idx]
    hard = np.array(frame["hard_bits"], dtype=np.uint8)
    iteration = frame["iteration"]
    bit_errors = frame["bit_errors"]

    # 24 column-blocks x Z lifted copies — the code's actual QC structure.
    n_blocks = n // z
    mismatch = (hard != transmitted).reshape(n_blocks, z).T  # (z, n_blocks)

    rgb = np.empty((z, n_blocks, 3), dtype=np.float32)
    correct_rgb = np.array(
        [int(CORRECT_COLOR[i : i + 2], 16) / 255 for i in (1, 3, 5)]
    )
    error_rgb = np.array([int(ERROR_COLOR[i : i + 2], 16) / 255 for i in (1, 3, 5)])
    rgb[~mismatch] = correct_rgb
    rgb[mismatch] = error_rgb

    fig, (ax_grid, ax_line) = plt.subplots(
        1, 2, figsize=(8.4, 3.6), facecolor=BG, gridspec_kw={"width_ratios": [1.0, 1.3]}
    )

    ax_grid.imshow(rgb, aspect="equal", interpolation="nearest")
    ax_grid.set_xticks([])
    ax_grid.set_yticks([])
    for spine in ax_grid.spines.values():
        spine.set_edgecolor(GRIDCOLOR)
    status = "receiver input (pre-decode)" if iteration == 0 else f"after iteration {iteration}"
    ax_grid.set_title(
        f"{status}\n{bit_errors} / {n} bits wrong", color=TEXT, fontsize=11, pad=8
    )

    xs = [f["iteration"] for f in frames]
    ys = [f["bit_errors"] for f in frames]
    shown_xs = xs[: frame_idx + 1]
    shown_ys = ys[: frame_idx + 1]
    ax_line.plot(shown_xs, shown_ys, "o-", color=LINE_COLOR, linewidth=2, markersize=5)
    ax_line.scatter([xs[frame_idx]], [ys[frame_idx]], color=LINE_COLOR, s=70, zorder=5)
    ax_line.set_xlim(-0.4, max(xs) + 0.4)
    ax_line.set_ylim(-max(ys) * 0.05 - 1, max(ys) * 1.08 + 1)
    style_ax(
        ax_line,
        xlabel="LOMS iteration",
        ylabel="Bit errors (of 648)",
        title=f"802.11 Wi-Fi LDPC LOMS decode\nZ={z}, rate {code['rate_num']}/{code['rate_den']}, "
        f"Eb/N0={channel['eb_n0_db']:.1f} dB",
    )

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, facecolor=BG)
    plt.close(fig)
    buf.seek(0)
    return buf


def main():
    data = load_data()
    if data is None:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    EXPORTS.mkdir(parents=True, exist_ok=True)

    n_frames = len(data["frames"])
    print(f"Rendering {n_frames} real decoder frames...")
    pil_frames = []
    for i in range(n_frames):
        buf = render_frame(plt, np, data, i)
        pil_frames.append(Image.open(buf).convert("RGB"))

    # Hold the final (converged, zero-error) frame a little longer so the
    # loop reads clearly — this repeats an already-rendered real frame, it
    # does not add or alter any data point.
    hold_frame = pil_frames[-1]
    pil_frames.extend([hold_frame] * 2)

    # Quantize every frame to one shared adaptive palette (built from the
    # first frame, which already contains the full dark background + both
    # bit colors + curve color) to keep the GIF small and flicker-free.
    base_palette = pil_frames[0].quantize(colors=48, method=Image.MEDIANCUT)
    quantized = [f.quantize(palette=base_palette, dither=Image.FLOYDSTEINBERG) for f in pil_frames]

    durations = [450] * n_frames + [1400] * 2
    quantized[0].save(
        OUT,
        save_all=True,
        append_images=quantized[1:],
        duration=durations,
        loop=0,
        optimize=True,
        disposal=2,
    )

    size_kb = OUT.stat().st_size / 1024
    print(f"Wrote {OUT} ({size_kb:.1f} KiB, {len(quantized)} frames)")


if __name__ == "__main__":
    main()
