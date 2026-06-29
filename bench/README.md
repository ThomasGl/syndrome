# glezer-rsv Cross-Language Benchmark Suite

Reproducible Reed-Solomon encode throughput comparison: Rust vs same-algorithm C++ vs Python.

## Quick start

```bash
# From repo root:
bash bench/run_all.sh
```

This single command:

1. Runs the **Rust** exporter (`cargo run --release --bin bench_export`) — writes `bench/results/rust.json` + `rust.checksum`.
2. Builds and runs the **C++** driver (`g++ -O3 -march=native -std=c++17`) — writes `bench/results/cpp.json` + `cpp.checksum`.
3. Runs the **Python** driver (`python3 bench/python/rs_encode.py`) — writes `bench/results/python.json` + `python_same_algo.checksum`.
4. Writes `bench/results/meta.json` with host/compiler info.
5. **Checksum gate**: diffs `rust.checksum` vs `cpp.checksum` vs `python_same_algo.checksum` and **fails loudly** if they differ (algorithm bug).
6. Prints a summary table.

## View the dashboard

```bash
cd bench/dashboard
python -m http.server
# open http://localhost:8000
```

Two charts are displayed:
- Grouped column: throughput (MiB/s) per implementation at each shard size.
- Line chart: throughput vs shard size (log-scale x-axis).

## Algorithm

All four implementations use the **identical** algorithm:

- GF(256), primitive polynomial `0x11D`.
- Encoding matrix: `coeffs[i*d+j] = α^((i*j) mod 255)`.
- `encode_into`: zero parity; for each data shard `j`, each parity row `i`, `parity[i][k] ^= mul(coeffs[i][j], data[j][k])`.
- Bench config: `data_shards=10, parity_shards=4`, `shard_len ∈ {256, 1024, 4096, 16384}`.

`python_reedsolo` is a separate bar — it uses generator-polynomial RS (different algorithm, different output) and is included as an ecosystem reference only.

## Prerequisites

| Tool      | Minimum |
|-----------|---------|
| Rust      | stable (cargo + rustc) |
| g++       | any C++17-capable version |
| Python    | 3.10+ |
| reedsolo  | `pip install reedsolo` (auto-installed by `run_all.sh` if missing) |

## License note

The dashboard (`bench/dashboard/`) uses [Highcharts](https://www.highcharts.com/) via CDN under its
**non-commercial** license. The attribution/credits label is intentionally kept visible. Anyone
forking this project for **commercial** use must obtain a separate Highcharts license or replace
the charting library (e.g., with Apache-licensed ECharts).

## Output files

| File | Contents |
|------|----------|
| `bench/results/rust.json` | Rust timings, one record per (impl, shard_len) |
| `bench/results/cpp.json` | C++ timings |
| `bench/results/python.json` | Python same_algo + reedsolo timings |
| `bench/results/meta.json` | Host/compiler metadata |
| `bench/results/rust.checksum` | Hex parity bytes for correctness gate |
| `bench/results/cpp.checksum` | Same, from C++ |
| `bench/results/python_same_algo.checksum` | Same, from Python |
