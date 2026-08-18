#!/usr/bin/env python3
"""
Cross-language Reed-Solomon benchmark driver.

Produces two result sets:
  1. python_same_algo  — pure-Python port of the EXACT algorithm in
     src/reed_solomon.rs (GF(256) 0x11D, Vandermonde matrix, encode_into).
     Byte-identical to Rust and C++ for the same input; used in the
     correctness-gate checksum comparison.
  2. python_reedsolo   — the `reedsolo` library (generator-polynomial RS,
     different algorithm, different output). Included as an ecosystem
     reference only; its parity bytes will NOT match the same-algo group.

Output: bench/results/python.json
        bench/results/python_same_algo.checksum
"""

import os
import sys
import time
import json

# ---------------------------------------------------------------------------
# GF(256) with primitive polynomial 0x11D
# ---------------------------------------------------------------------------

_EXP = [0] * 512
_LOG = [0] * 256

def _build_tables():
    x = 1
    for i in range(255):
        _EXP[i] = x
        _LOG[x] = i
        hi = (x & 0x80) != 0
        x = (x << 1) & 0xFF
        if hi:
            x ^= 0x1D
    for i in range(255, 512):
        _EXP[i] = _EXP[i - 255]

_build_tables()

def gf_mul(a: int, b: int) -> int:
    if a == 0 or b == 0:
        return 0
    return _EXP[_LOG[a] + _LOG[b]]

def pow_alpha(power: int) -> int:
    return _EXP[power % 255]

def gf_inv(a: int) -> int:
    """Multiplicative inverse in GF(256): alpha^(255 - log a)."""
    return _EXP[255 - _LOG[a]]

# ---------------------------------------------------------------------------
# Same-algorithm encoder (mirrors encode_into from reed_solomon.rs)
# ---------------------------------------------------------------------------

DATA_SHARDS   = 10
PARITY_SHARDS = 4
SHARD_LENS    = [256, 1024, 4096, 16384]
CHECKSUM_SHARD_LEN = 1024

ITERS  = 200   # Python is slow; keep timing runs reasonable
WARMUP = 20

def build_coeffs(d: int, p: int) -> list[int]:
    """Cauchy matrix coeffs[i*d+j] = 1 / (i ^ (p + j)).

    The two index sets are disjoint, so no denominator is zero. This mirrors
    MatrixKind::Cauchy in src/reed_solomon.rs; see that type's docs for why
    every square submatrix of it is invertible, which the earlier alpha^(i*j)
    construction did not guarantee.
    """
    coeffs = []
    for i in range(p):
        for j in range(d):
            coeffs.append(gf_inv(i ^ (p + j)))
    return coeffs

COEFFS = build_coeffs(DATA_SHARDS, PARITY_SHARDS)

def encode_into(data: list[bytearray], parity_out: bytearray, shard_len: int) -> None:
    """Pure-Python port of ReedSolomon::encode_into."""
    d, p = DATA_SHARDS, PARITY_SHARDS
    # zero parity
    for b in range(p * shard_len):
        parity_out[b] = 0

    for j in range(d):
        dj = data[j]
        for i in range(p):
            coef = COEFFS[i * d + j]
            if coef == 0:
                continue
            row_start = i * shard_len
            if coef == 1:
                for k in range(shard_len):
                    parity_out[row_start + k] ^= dj[k]
            else:
                for k in range(shard_len):
                    parity_out[row_start + k] ^= gf_mul(coef, dj[k])

def make_seed_data(shard_len: int) -> list[bytearray]:
    """Deterministic seed: shard j filled with (j*3+1) & 0xFF."""
    return [bytearray([(j * 3 + 1) & 0xFF] * shard_len) for j in range(DATA_SHARDS)]

def time_encode_same_algo(shard_len: int) -> float:
    """Return mean ns/iter for encode_into at this shard_len."""
    data = [bytearray([j] * shard_len) for j in range(DATA_SHARDS)]
    parity = bytearray(PARITY_SHARDS * shard_len)

    for _ in range(WARMUP):
        encode_into(data, parity, shard_len)

    t0 = time.perf_counter_ns()
    for _ in range(ITERS):
        encode_into(data, parity, shard_len)
    t1 = time.perf_counter_ns()
    return (t1 - t0) / ITERS

# ---------------------------------------------------------------------------
# reedsolo baseline
# ---------------------------------------------------------------------------

def time_reedsolo(shard_len: int) -> float | None:
    """Return mean ns/iter for reedsolo encoding of equivalent payload.

    reedsolo works on codewords up to 255 bytes (GF(256)). We use
    RSCodec(nsym=4) — 4 ECC symbols per chunk — and chunk the payload so
    the total number of output parity bytes equals PARITY_SHARDS * shard_len,
    giving a fair comparison of equal-sized output protection.

    Each chunk is at most 251 data bytes (255 - 4 ECC). We tile the payload
    across as many chunks as needed.
    """
    try:
        import reedsolo
    except ImportError:
        return None

    # 4 ECC symbols per chunk; max 251 data bytes per GF(256) codeword.
    nsym = PARITY_SHARDS   # 4
    chunk_data = 255 - nsym  # 251 data bytes per chunk

    payload_bytes = DATA_SHARDS * shard_len
    # pad payload to a multiple of chunk_data
    pad_len = (-payload_bytes) % chunk_data
    data = (bytes(range(256)) * ((payload_bytes + pad_len) // 256 + 1))[:(payload_bytes + pad_len)]

    rs = reedsolo.RSCodec(nsym)

    for _ in range(WARMUP):
        rs.encode(data)

    t0 = time.perf_counter_ns()
    for _ in range(ITERS):
        rs.encode(data)
    t1 = time.perf_counter_ns()
    return (t1 - t0) / ITERS

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def mib_per_s(payload_bytes: int, ns_per_iter: float) -> float:
    return (payload_bytes / ns_per_iter) * 1e9 / (1024 * 1024)

def main():
    out_dir = "bench/results"
    if len(sys.argv) > 1:
        out_dir = sys.argv[1]
    os.makedirs(out_dir, exist_ok=True)

    records = []
    checksum_hex = None

    print("python_same_algo timings:")
    for shard_len in SHARD_LENS:
        payload_bytes = DATA_SHARDS * shard_len
        ns = time_encode_same_algo(shard_len)
        mib = mib_per_s(payload_bytes, ns)
        print(f"  shard_len={shard_len:6d}  ns/iter={ns:10.0f}  MiB/s={mib:.2f}")
        records.append({
            "lang": "python_same_algo",
            "impl": "encode_into",
            "shard_len": shard_len,
            "data_shards": DATA_SHARDS,
            "parity_shards": PARITY_SHARDS,
            "payload_bytes": payload_bytes,
            "ns_per_iter": round(ns, 1),
            "mib_per_s": round(mib, 1),
        })

        if shard_len == CHECKSUM_SHARD_LEN:
            seed = make_seed_data(CHECKSUM_SHARD_LEN)
            parity = bytearray(PARITY_SHARDS * CHECKSUM_SHARD_LEN)
            encode_into(seed, parity, CHECKSUM_SHARD_LEN)
            checksum_hex = parity.hex()

    # reedsolo
    print("\npython_reedsolo timings:")
    for shard_len in SHARD_LENS:
        payload_bytes = DATA_SHARDS * shard_len
        ns = time_reedsolo(shard_len)
        if ns is None:
            print(f"  shard_len={shard_len:6d}  reedsolo not installed")
            records.append({
                "lang": "python_reedsolo",
                "impl": "reedsolo.RSCodec.encode",
                "shard_len": shard_len,
                "data_shards": DATA_SHARDS,
                "parity_shards": PARITY_SHARDS,
                "payload_bytes": payload_bytes,
                "ns_per_iter": None,
                "mib_per_s": None,
                "note": "reedsolo not installed",
            })
        else:
            mib = mib_per_s(payload_bytes, ns)
            print(f"  shard_len={shard_len:6d}  ns/iter={ns:10.0f}  MiB/s={mib:.2f}")
            records.append({
                "lang": "python_reedsolo",
                "impl": "reedsolo.RSCodec.encode",
                "shard_len": shard_len,
                "data_shards": DATA_SHARDS,
                "parity_shards": PARITY_SHARDS,
                "payload_bytes": payload_bytes,
                "ns_per_iter": round(ns, 1),
                "mib_per_s": round(mib, 1),
                "note": "different algorithm (generator-polynomial); parity bytes do NOT match same_algo group",
            })

    json_path = os.path.join(out_dir, "python.json")
    with open(json_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nWrote {json_path}")

    cs_path = os.path.join(out_dir, "python_same_algo.checksum")
    with open(cs_path, "w") as f:
        f.write(checksum_hex or "")
    print(f"Wrote {cs_path}")

if __name__ == "__main__":
    main()
