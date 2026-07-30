#!/usr/bin/env python3
"""Render the Reed-Solomon erasure-recovery GIF from real encoder/decoder output.

Reads bench/results/rs_erasure.json — written by
`cargo run --release --bin rs_erasure_export`, which runs one real RS(10,4)
encode, erases 4 of the 10 data shards (the maximum this code tolerates), and
reconstructs them with the real decoder, asserting byte-for-byte equality to
the original before exporting anything.

Nothing in this script invents pixels: every byte in every frame comes
straight from the JSON, which comes straight from one real encode/erase/decode
cycle. This script only chooses colors, layout, and frame timing.

Usage:
    cd <repo_root>
    python bench/dashboard/gen_rs_erasure_gif.py

Requirements:
    pip install matplotlib pillow numpy
"""

import io
import json
import sys
from pathlib import Path

RESULTS = Path("bench/results")
EXPORTS = Path("bench/dashboard/exports")
SRC = RESULTS / "rs_erasure.json"
OUT = EXPORTS / "rs_erasure.gif"

# ─── style — matches bench/dashboard/gen_charts.py and gen_convergence_gif.py ─
BG        = "#1e293b"
GRIDCOLOR = "#2d3748"
TEXT      = "#e2e8f0"
DIM       = "#94a3b8"

OK_COLOR    = "#22c55e"  # green  — shard present / recovered
LOST_COLOR  = "#ef4444"  # red    — shard erased
PARITY_COLOR = "#3b82f6"  # blue  — parity shard, used to reconstruct


def style_ax(ax, title=None):
    ax.set_facecolor(BG)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRIDCOLOR)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, color=TEXT, fontsize=10.5, pad=8)


def load_data():
    if not SRC.exists():
        print(
            f"  [skip] {SRC} not found "
            "(run: cargo run --release --bin rs_erasure_export)",
            file=sys.stderr,
        )
        return None
    with open(SRC) as f:
        return json.load(f)


def shard_status_bar(ax, np, data_shards, parity_shards, erased, phase):
    """Small horizontal strip of shard indicators: data shards 0..9, then
    parity shards 0..3. `phase` is 'corrupted' or 'recovered', controlling
    whether erased data shards show red (lost) or green (recovered)."""
    total = data_shards + parity_shards
    colors = []
    for i in range(data_shards):
        if i in erased:
            colors.append(LOST_COLOR if phase == "corrupted" else OK_COLOR)
        else:
            colors.append(OK_COLOR)
    colors.extend([PARITY_COLOR] * parity_shards)

    rgb = np.array(
        [[int(c[i : i + 2], 16) / 255 for i in (1, 3, 5)] for c in colors]
    ).reshape(1, total, 3)
    ax.imshow(rgb, aspect="auto", interpolation="nearest")
    ax.set_xticks(range(total))
    ax.set_xticklabels(
        [f"d{i}" for i in range(data_shards)] + [f"p{i}" for i in range(parity_shards)],
        color=DIM, fontsize=7,
    )
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(GRIDCOLOR)


def render_frame(plt, np, data, phase):
    """phase in {'original', 'corrupted', 'recovered'}."""
    side = data["side"]
    data_shards = data["data_shards"]
    parity_shards = data["parity_shards"]
    erased = set(data["erased_shard_indices"])

    image_key = {
        "original": "original_image",
        "corrupted": "corrupted_image",
        "recovered": "recovered_image",
    }[phase]
    img = np.array(data[image_key], dtype=np.uint8).reshape(side, side)

    fig, (ax_img, ax_bar) = plt.subplots(
        2, 1, figsize=(4.6, 5.4), facecolor=BG,
        gridspec_kw={"height_ratios": [4.2, 1.0]},
    )

    ax_img.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    titles = {
        "original": "Original image\n(80x80, RS(10,4) encoded)",
        "corrupted": f"{len(erased)} of {data_shards} data shards erased\n"
                     "(the maximum this code tolerates)",
        "recovered": "Reconstructed by the real decoder\n"
                     "verified byte-for-byte identical",
    }
    style_ax(ax_img, titles[phase])

    bar_phase = "corrupted" if phase in ("original", "corrupted") else "recovered"
    shard_status_bar(ax_bar, np, data_shards, parity_shards,
                      erased if phase != "original" else set(), bar_phase)
    ax_bar.set_title(
        "data shards (d0-d9) + parity shards (p0-p3)"
        if phase != "original" else "data shards (d0-d9) + parity shards (p0-p3), all present",
        color=DIM, fontsize=8, pad=6,
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
    if not data.get("byte_perfect"):
        print("  [abort] exporter did not report byte-perfect recovery", file=sys.stderr)
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    EXPORTS.mkdir(parents=True, exist_ok=True)

    phases = ["original", "corrupted", "recovered"]
    print(f"Rendering {len(phases)} phases from real RS(10,4) encode/erase/decode output...")
    pil_frames = [Image.open(render_frame(plt, np, data, p)).convert("RGB") for p in phases]

    # Hold the corrupted and recovered frames a beat longer than the original,
    # and hold the final recovered frame longest so the loop reads clearly —
    # this repeats already-rendered real frames, it does not alter any pixel.
    hold_recovered = pil_frames[-1]
    sequence = [pil_frames[0], pil_frames[1], pil_frames[1], pil_frames[2], hold_recovered]
    durations = [1200, 900, 900, 1200, 1800]

    base_palette = sequence[0].quantize(colors=48, method=Image.MEDIANCUT)
    quantized = [f.quantize(palette=base_palette, dither=Image.FLOYDSTEINBERG) for f in sequence]

    quantized[0].save(
        OUT,
        save_all=True,
        append_images=quantized[1:],
        duration=durations,
        loop=0,
        optimize=True,
        disposal=2,
    )

    # Report the frame count PIL actually stored, not the pre-save list
    # length: `optimize=True` merges consecutive identical frame objects (the
    # held duplicates above) and combines their durations, so re-opening the
    # written file is the only way to know what was really saved.
    size_kb = OUT.stat().st_size / 1024
    with Image.open(OUT) as written:
        stored_frames = written.n_frames
    print(f"Wrote {OUT} ({size_kb:.1f} KiB, {stored_frames} stored frames after GIF optimization)")


if __name__ == "__main__":
    main()
