#!/usr/bin/env bash
# bench/run_all.sh — build and run all three language drivers, then validate
# that the Rust, C++, and Python-same-algo implementations produce BYTE-IDENTICAL
# parity output for the deterministic seed input.
#
# Usage:  bash bench/run_all.sh
# Must be run from the repo root (syndrome/).
#
# Exit codes:
#   0 — all checksums match, JSON written, dashboard ready
#   1 — build or run failure
#   2 — checksum mismatch (algorithm divergence — fix before plotting numbers)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

RESULTS_DIR="bench/results"
mkdir -p "$RESULTS_DIR"

FAIL=0

# ── colour helpers ──────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
ok()   { echo -e "${GREEN}[OK]${NC}  $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
fail() { echo -e "${RED}[FAIL]${NC} $*"; FAIL=1; }

# ── 1. Write meta.json ───────────────────────────────────────────────────────
echo "── Writing meta.json ──────────────────────────────────────────────────"
UNAME="$(uname -srm 2>/dev/null || echo unknown)"
NPROC="$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo '?')"
RUSTC_VER="$(rustc --version 2>/dev/null || echo 'not found')"
GXX_VER="$(g++ --version 2>/dev/null | head -1 || echo 'not found')"
PYTHON_VER="$(python3 --version 2>/dev/null || echo 'not found')"
RUN_DATE="$(date -u '+%Y-%m-%dT%H:%M:%SZ' 2>/dev/null || date)"

cat > "$RESULTS_DIR/meta.json" <<EOF
{
  "uname":  "$UNAME",
  "nproc":  "$NPROC",
  "rustc":  "$RUSTC_VER",
  "gxx":    "$GXX_VER",
  "python": "$PYTHON_VER",
  "date":   "$RUN_DATE"
}
EOF
ok "meta.json written"

# ── 2. Rust exporter ────────────────────────────────────────────────────────
echo ""
echo "── Rust: cargo run --release --bin bench_export ────────────────────────"
if cargo run --release --bin bench_export 2>&1; then
    ok "rust.json + rust.checksum written"
else
    fail "Rust bench_export failed"
fi

# ── 3. C++ baseline ─────────────────────────────────────────────────────────
echo ""
echo "── C++: build + run ────────────────────────────────────────────────────"
if command -v g++ >/dev/null 2>&1; then
    (cd bench/cpp && make --no-print-directory) && ok "C++ build OK" || { fail "C++ build failed"; }

    if [ "$FAIL" -eq 0 ] || [ -x bench/cpp/rs_encode ]; then
        if ./bench/cpp/rs_encode "$RESULTS_DIR" 2>&1; then
            ok "cpp.json + cpp.checksum written"
        else
            fail "C++ rs_encode run failed"
        fi
    fi
else
    warn "g++ not found — skipping C++ baseline. Install build-essential and re-run."
fi

# ── 4. Python baselines ──────────────────────────────────────────────────────
echo ""
echo "── Python: rs_encode.py ────────────────────────────────────────────────"
if command -v python3 >/dev/null 2>&1; then
    # Install reedsolo if missing (into user site-packages; no root needed).
    if ! python3 -c "import reedsolo" 2>/dev/null; then
        warn "reedsolo not found — attempting: pip install --user reedsolo"
        pip install --user reedsolo 2>&1 || warn "pip install failed; python_reedsolo bars will be absent from charts"
    fi
    if python3 bench/python/rs_encode.py "$RESULTS_DIR" 2>&1; then
        ok "python.json + python_same_algo.checksum written"
    else
        fail "Python rs_encode.py failed"
    fi
else
    warn "python3 not found — skipping Python baselines."
fi

# ── 5. Checksum gate ────────────────────────────────────────────────────────
echo ""
echo "── Checksum gate (algorithm correctness) ───────────────────────────────"

RUST_CS=""
CPP_CS=""
PY_CS=""

[ -f "$RESULTS_DIR/rust.checksum" ] && RUST_CS="$(cat "$RESULTS_DIR/rust.checksum")"
[ -f "$RESULTS_DIR/cpp.checksum"  ] && CPP_CS="$(cat "$RESULTS_DIR/cpp.checksum")"
[ -f "$RESULTS_DIR/python_same_algo.checksum" ] && PY_CS="$(cat "$RESULTS_DIR/python_same_algo.checksum")"

if [ -z "$RUST_CS" ]; then
    fail "rust.checksum missing — Rust exporter did not run successfully"
elif [ -z "$CPP_CS" ] && [ -z "$PY_CS" ]; then
    warn "Only Rust checksum available (C++ and Python may have been skipped)"
else
    GATE_PASS=true

    if [ -n "$CPP_CS" ]; then
        if [ "$RUST_CS" = "$CPP_CS" ]; then
            ok "Rust == C++ parity bytes (byte-identical ✓)"
        else
            fail "CHECKSUM MISMATCH: Rust != C++"
            echo "  Rust:  ${RUST_CS:0:64}…"
            echo "  C++:   ${CPP_CS:0:64}…"
            GATE_PASS=false
        fi
    fi

    if [ -n "$PY_CS" ]; then
        if [ "$RUST_CS" = "$PY_CS" ]; then
            ok "Rust == python_same_algo parity bytes (byte-identical ✓)"
        else
            fail "CHECKSUM MISMATCH: Rust != python_same_algo"
            echo "  Rust:   ${RUST_CS:0:64}…"
            echo "  Python: ${PY_CS:0:64}…"
            GATE_PASS=false
        fi
    fi

    if [ "$GATE_PASS" = "false" ]; then
        echo ""
        echo -e "${RED}STOP: Algorithm divergence detected.${NC}"
        echo "The parity bytes differ between implementations. This means the algorithm"
        echo "specification was not ported exactly. Fix the mismatch before plotting any numbers."
        exit 2
    fi
fi

# ── 6. LDPC benchmarks ───────────────────────────────────────────────────────
echo ""
echo "── LDPC C++ ─────────────────────────────────────────────────────────────"
if command -v g++ >/dev/null 2>&1; then
    if (cd bench/cpp && make --no-print-directory loms_decode); then
        ok "loms_decode built"
        if bench/cpp/loms_decode; then
            ok "ldpc_cpp.json written"
        else
            fail "loms_decode run failed"
        fi
    else
        fail "loms_decode build failed"
    fi
else
    warn "g++ not found — skipping LDPC C++ benchmark."
fi

echo ""
echo "── LDPC Rust ───────────────────────────────────────────────────────────"
if cargo run --release --bin ldpc_bench_export 2>&1; then
    ok "ldpc_rust.json written"
else
    fail "ldpc_bench_export failed"
fi

# ── 7. Summary table ────────────────────────────────────────────────────────
echo ""
echo "── Throughput summary (MiB/s) ──────────────────────────────────────────"
python3 - "$RESULTS_DIR" <<'PYEOF'
import sys, json, os

results_dir = sys.argv[1]
rows = []
for fn in ["rust.json", "cpp.json", "python.json"]:
    p = os.path.join(results_dir, fn)
    if not os.path.exists(p): continue
    data = json.load(open(p))
    if not isinstance(data, list): data = [data]
    rows.extend(data)

shard_lens = [256, 1024, 4096, 16384]
keys_seen = []
by_key = {}
for rec in rows:
    if rec.get("mib_per_s") is None: continue
    key = f"{rec['lang']}/{rec['impl']}"
    if key not in keys_seen: keys_seen.append(key)
    by_key.setdefault(key, {})[rec["shard_len"]] = rec["mib_per_s"]

hdr = f"{'impl':<42} {'256B':>8} {'1KiB':>8} {'4KiB':>8} {'16KiB':>8}"
print(hdr)
print("-" * len(hdr))
for key in keys_seen:
    vals = by_key[key]
    row = f"{key:<42}"
    for sl in shard_lens:
        v = vals.get(sl)
        row += f" {v:>8.1f}" if v is not None else f" {'N/A':>8}"
    print(row)
PYEOF

echo ""
echo "── Dashboard ───────────────────────────────────────────────────────────"
echo "To view the Highcharts dashboard:"
echo "  cd bench/dashboard && python -m http.server"
echo "  then open http://localhost:8000"

if [ "$FAIL" -ne 0 ]; then
    echo ""
    echo -e "${RED}One or more steps failed. See output above.${NC}"
    exit 1
fi

echo ""
ok "All steps completed successfully."
