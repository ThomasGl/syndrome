#!/usr/bin/env python3
"""Generate PNG chart exports from benchmark result JSON files.

Reads bench/results/*.json and writes PNG images to bench/dashboard/exports/.
These PNGs are embedded in README.md as proof-of-efficiency artifacts.

Usage:
    cd <repo_root>
    python bench/dashboard/gen_charts.py

Requirements:
    pip install matplotlib
"""

import json
import os
import sys
from pathlib import Path

RESULTS = Path("bench/results")
EXPORTS = Path("bench/dashboard/exports")

# ─── style ────────────────────────────────────────────────────────────────────
BG        = "#1e293b"
GRIDCOLOR = "#2d3748"
TEXT      = "#e2e8f0"
DIM       = "#94a3b8"

# Primary two-hue pair validated for the dark surface (#1e293b) with the
# dataviz palette validator: lightness band, chroma floor, CVD separation,
# and 3:1 contrast all pass for #ea580c / #3b82f6.
RUST_COLORS  = ["#ea580c", "#fb923c", "#fdba74"]
CPP_COLOR    = "#3b82f6"
PY_COLOR     = "#a855f7"
PY2_COLOR    = "#22c55e"
BER_COLOR    = "#ea580c"
BLER_COLOR   = "#3b82f6"
SHANNON_COL  = "#ef4444"
ENCODE_COLOR = "#ea580c"
DECODE_COLOR = "#3b82f6"

def mpl():
    """Import matplotlib with a non-interactive backend."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt

def style_ax(ax, xlabel=None, ylabel=None, title=None, grid=True):
    ax.set_facecolor(BG)
    ax.tick_params(colors=DIM, labelcolor=DIM)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRIDCOLOR)
    if grid:
        ax.grid(True, color=GRIDCOLOR, linestyle="--", linewidth=0.6, alpha=0.7)
    if xlabel: ax.set_xlabel(xlabel, color=DIM)
    if ylabel: ax.set_ylabel(ylabel, color=DIM)
    if title:  ax.set_title(title, color=TEXT, fontsize=13, pad=12)


def load_json(path):
    try:
        with open(path) as f:
            data = json.load(f)
        return data if isinstance(data, list) else [data]
    except FileNotFoundError:
        print(f"  [skip] {path} not found", file=sys.stderr)
        return []
    except Exception as e:
        print(f"  [error] {path}: {e}", file=sys.stderr)
        return []


# ─── Chart 1: RS throughput (MiB/s) ──────────────────────────────────────────
def chart_rs_throughput():
    plt = mpl()
    import numpy as np

    rust   = load_json(RESULTS / "rust.json")
    cpp    = load_json(RESULTS / "cpp.json")
    py     = load_json(RESULTS / "python.json")

    all_rec = rust + cpp + py
    if not all_rec:
        print("  [skip] rs_throughput — no data")
        return

    shard_lens = sorted({r["shard_len"] for r in all_rec if r.get("shard_len")})
    if not shard_lens:
        return

    # Collect series keyed by (lang, impl)
    series = {}
    for rec in all_rec:
        if rec.get("mib_per_s") is None:
            continue
        key = (rec.get("lang", "?"), rec.get("impl", "?"))
        series.setdefault(key, {})[rec["shard_len"]] = rec["mib_per_s"]

    COLORS = {
        "rust":              RUST_COLORS[0],
        "cpp":               CPP_COLOR,
        "python_same_algo":  PY_COLOR,
        "python_reedsolo":   PY2_COLOR,
    }

    x = np.arange(len(shard_lens))
    n_series = len(series)
    width = 0.7 / max(n_series, 1)

    fig, ax = plt.subplots(figsize=(10, 5), facecolor=BG)
    for si, ((lang, impl), bylen) in enumerate(series.items()):
        vals = [bylen.get(sl, 0) for sl in shard_lens]
        offset = (si - n_series / 2 + 0.5) * width
        color = COLORS.get(lang, DIM)
        label = f"{lang} {impl}"[:40]
        ax.bar(x + offset, vals, width * 0.9, label=label, color=color, alpha=0.85)

    xlabels = [f"{sl//1024} KiB" if sl >= 1024 else f"{sl} B" for sl in shard_lens]
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, color=DIM)
    style_ax(ax, xlabel="Shard length", ylabel="MiB/s",
             title="Reed-Solomon Encode Throughput")
    ax.legend(facecolor=BG, edgecolor=GRIDCOLOR, labelcolor=TEXT, fontsize=8)
    fig.tight_layout()
    out = EXPORTS / "rs_throughput.png"
    fig.savefig(out, dpi=150, facecolor=BG)
    plt.close(fig)
    print(f"  wrote {out}")


# ─── Chart 2: LDPC Melem/s comparison ────────────────────────────────────────
def chart_ldpc_comparison():
    plt = mpl()

    rust = load_json(RESULTS / "ldpc_rust.json")
    cpp  = load_json(RESULTS / "ldpc_cpp.json")

    all_rec = rust + cpp
    if not all_rec:
        print("  [skip] ldpc_comparison — no data")
        return

    by_impl = {}
    for rec in all_rec:
        key   = f"{rec.get('lang','?')} {rec.get('impl','?')}"
        melem = rec.get("melem_per_s")
        if melem is not None:
            by_impl[key] = melem

    if not by_impl:
        return

    labels = list(by_impl.keys())
    vals   = [by_impl[k] for k in labels]
    colors = [CPP_COLOR if "cpp" in k else RUST_COLORS[0] for k in labels]

    fig, ax = plt.subplots(figsize=(8, 5), facecolor=BG)
    bars = ax.bar(labels, vals, color=colors, width=0.5, alpha=0.85)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1,
                f"{v:.1f}", ha="center", va="bottom", color=TEXT, fontsize=9)

    style_ax(ax, ylabel="Melem/s (variable-node-iteration-updates / s)",
             title="QC-LDPC Decode Throughput — BG1 Z=384, 10 iterations")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, color=DIM, rotation=15, ha="right", fontsize=9)
    fig.tight_layout()
    out = EXPORTS / "ldpc_comparison.png"
    fig.savefig(out, dpi=150, facecolor=BG)
    plt.close(fig)
    print(f"  wrote {out}")


# ─── Chart 3: BER waterfall ───────────────────────────────────────────────────
def chart_ber_waterfall():
    plt = mpl()
    import numpy as np

    ber_data = load_json(RESULTS / "ber_rust.json")
    if not ber_data:
        print("  [skip] ber_waterfall — no data (run: cargo run --release --bin ber_sim)")
        return

    xs   = [r["eb_n0_db"] for r in ber_data]
    bers = [r["ber"]  if r["ber"]  > 0 else None for r in ber_data]
    blers= [r["bler"] if r["bler"] > 0 else None for r in ber_data]

    # BER and BLER are the same unit (error probability), so they share ONE
    # log axis — a dual-axis (twinx) chart here would be misleading.
    fig, ax = plt.subplots(figsize=(10, 5), facecolor=BG)

    ber_y  = [v for v in bers  if v is not None]
    bler_y = [v for v in blers if v is not None]
    ber_x  = [xs[i] for i, v in enumerate(bers)  if v is not None]
    bler_x = [xs[i] for i, v in enumerate(blers) if v is not None]

    ax.semilogy(ber_x,  ber_y,  "o-", color=BER_COLOR,  linewidth=2, markersize=6,
                label="BER (Rust LOMS)")
    ax.semilogy(bler_x, bler_y, "s--", color=BLER_COLOR, linewidth=2, markersize=6,
                label="BLER (Rust LOMS)")

    # Shannon limit for a rate-R code on the real AWGN channel (BPSK is one
    # bit per real dimension): Eb/N0_min = (2^(2R) - 1) / (2R).
    # For R = 22/68 ≈ 0.324 this is ≈ -0.58 dB.
    R = 22 / 68
    shannon_db = 10 * np.log10((2 ** (2 * R) - 1) / (2 * R))
    ax.axvline(shannon_db, color=SHANNON_COL, linestyle="--", linewidth=1.5,
               label=f"Shannon limit R={R:.3f} ≈ {shannon_db:.1f} dB")

    style_ax(ax, xlabel="Eb/N0 (dB)", ylabel="Error rate (log)",
             title="BER / BLER Waterfall — QC-LDPC BG1 Z=384 (BPSK AWGN, β=0.25, 10 iter)")
    ax.legend(facecolor=BG, edgecolor=GRIDCOLOR, labelcolor=TEXT, fontsize=9)

    fig.tight_layout()
    out = EXPORTS / "ber_waterfall.png"
    fig.savefig(out, dpi=150, facecolor=BG)
    plt.close(fig)
    print(f"  wrote {out}")


# ─── Chart 4: latency vs payload size (log-log) ───────────────────────────────
def chart_latency():
    plt = mpl()
    import numpy as np

    rust  = load_json(RESULTS / "rust.json")
    cpp   = load_json(RESULTS / "cpp.json")
    py    = load_json(RESULTS / "python.json")
    ldpc_r = load_json(RESULTS / "ldpc_rust.json")
    ldpc_c = load_json(RESULTS / "ldpc_cpp.json")

    all_rs  = rust + cpp + py
    if not all_rs and not ldpc_r and not ldpc_c:
        print("  [skip] latency — no data")
        return

    IMPL_STYLE = {
        ("rust",             "encode_into"):                ("Rust encode_into",                RUST_COLORS[0], "-", "o"),
        ("rust",             "encode_with_tables_chunked"): ("Rust tables_chunked",              RUST_COLORS[1], "-", "o"),
        ("rust",             "encode_with_avx2"):           ("Rust encode_with_avx2",            "#fde68a",      "-", "o"),
        ("cpp",              "encode_into"):                ("C++ encode_into",                  CPP_COLOR,      "--","s"),
        ("python_same_algo", "encode_into"):                ("Python same_algo",                 PY_COLOR,       ":", "^"),
    }

    by_key = {}
    for rec in all_rs:
        if rec.get("ns_per_iter") is None or rec.get("payload_bytes") is None:
            continue
        key = (rec.get("lang",""), rec.get("impl",""))
        by_key.setdefault(key, []).append((rec["payload_bytes"], rec["ns_per_iter"] / 1000))

    fig, ax = plt.subplots(figsize=(10, 6), facecolor=BG)
    ax.set_yscale("log")
    ax.set_xscale("log")

    for key, pts in by_key.items():
        pts.sort()
        xs, ys = zip(*pts)
        style = IMPL_STYLE.get(key, (str(key), DIM, "-", "o"))
        ax.plot(xs, ys, style[2]+style[3], color=style[1], label=style[0],
                linewidth=2, markersize=7, alpha=0.9)

    # LDPC as single diamond markers.
    for rec, color, label in [
        (ldpc_r, RUST_COLORS[0], "Rust LDPC AVX2"),
        (ldpc_c, CPP_COLOR,      "C++ LDPC scalar"),
    ]:
        if rec and rec[0].get("ns_per_iter") is not None:
            x = rec[0].get("payload_bytes", 26112)
            y = rec[0]["ns_per_iter"] / 1000
            ax.scatter([x], [y], color=color, marker="D", s=120, zorder=5, label=label)
            ax.annotate(f"  {label}\n  {y:,.0f} μs", (x, y),
                        color=color, fontsize=8, va="center")

    style_ax(ax,
             xlabel="Payload bytes",
             ylabel="Latency (μs / call)",
             title="FEC Encode / Decode Latency vs Payload Size (log-log)")
    ax.xaxis.set_major_formatter(
        __import__("matplotlib.ticker", fromlist=["FuncFormatter"]).FuncFormatter(
            lambda v, _: f"{int(v)//1024} KiB" if v >= 1024 else f"{int(v)} B"
        )
    )
    ax.legend(facecolor=BG, edgecolor=GRIDCOLOR, labelcolor=TEXT, fontsize=8)
    fig.tight_layout()
    out = EXPORTS / "latency.png"
    fig.savefig(out, dpi=150, facecolor=BG)
    plt.close(fig)
    print(f"  wrote {out}")


# ─── Chart 5: all-algorithm encode/decode throughput comparison ───────────────
def chart_algo_comparison():
    """Horizontal grouped bars: info-bit throughput of every FEC core.

    Reads bench/results/algos.json (written by `cargo run --release --bin
    algo_bench_export`). One row per algorithm, two bars (encode / decode),
    log x-axis because throughput spans ~4 orders of magnitude between a
    table-lookup Golay decode and an iterative Turbo decode.
    """
    plt = mpl()
    import numpy as np

    recs = load_json(RESULTS / "algos.json")
    if not recs:
        print("  [skip] algo_comparison — no data "
              "(run: cargo run --release --bin algo_bench_export)")
        return

    # {algo_label: {"encode": mbit/s, "decode": mbit/s}}
    by_algo = {}
    for r in recs:
        if r.get("mbit_per_s") is None:
            continue
        label = r.get("label", r.get("algo", "?"))
        by_algo.setdefault(label, {})[r.get("op", "?")] = r["mbit_per_s"]

    if not by_algo:
        return

    # Sort rows by decode throughput so the ranking is readable top-to-bottom.
    labels = sorted(by_algo, key=lambda a: by_algo[a].get("decode", 0.0),
                    reverse=True)
    enc = [by_algo[a].get("encode") for a in labels]
    dec = [by_algo[a].get("decode") for a in labels]

    y = np.arange(len(labels))
    h = 0.36  # bar height; the 0.28 gap between pairs acts as the mark spacer

    fig, ax = plt.subplots(figsize=(11, 0.62 * len(labels) + 1.8), facecolor=BG)
    ax.set_xscale("log")

    def fmt(v):
        if v >= 1000:
            return f"{v/1000:,.1f} Gbit/s"
        if v >= 1:
            return f"{v:,.0f} Mbit/s"
        return f"{v*1000:,.0f} kbit/s"

    for vals, off, color, name in [(enc, -h / 2, ENCODE_COLOR, "encode"),
                                   (dec, +h / 2, DECODE_COLOR, "decode")]:
        ys = [yi + off for yi, v in zip(y, vals) if v is not None]
        vs = [v for v in vals if v is not None]
        bars = ax.barh(ys, vs, h * 0.92, color=color, alpha=0.9, label=name)
        for bar, v in zip(bars, vs):
            ax.text(v * 1.15, bar.get_y() + bar.get_height() / 2, fmt(v),
                    va="center", ha="left", color=TEXT, fontsize=8)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, color=TEXT, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(right=ax.get_xlim()[1] * 8)  # headroom for end labels
    style_ax(ax, xlabel="Information-bit throughput (Mbit/s, log scale)",
             title="FEC Core Throughput — encode vs decode (single thread)")
    ax.legend(facecolor=BG, edgecolor=GRIDCOLOR, labelcolor=TEXT, fontsize=9,
              loc="lower right")
    fig.tight_layout()
    out = EXPORTS / "algo_comparison.png"
    fig.savefig(out, dpi=150, facecolor=BG)
    plt.close(fig)
    print(f"  wrote {out}")


# ─── main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    EXPORTS.mkdir(parents=True, exist_ok=True)
    print("Generating charts…")
    chart_rs_throughput()
    chart_ldpc_comparison()
    chart_ber_waterfall()
    chart_latency()
    chart_algo_comparison()
    print("Done. PNG files are in bench/dashboard/exports/")
    print("Embed in README with: ![Chart](bench/dashboard/exports/<name>.png)")
