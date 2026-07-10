/**
 * syndrome benchmark dashboard.
 *
 * Fetches bench/results/{rust,cpp,python,meta}.json (relative to this file,
 * served by `python -m http.server` from bench/dashboard/).
 *
 * Charts:
 *   1. Grouped column: throughput (MiB/s) per lang at each shard_len.
 *   2. Line chart: throughput vs shard_len per (lang, impl).
 *
 * Highcharts credits label is intentionally left visible.
 */

"use strict";

const RESULT_BASE = "../results";
const SHARD_LENS  = [256, 1024, 4096, 16384];

const IMPL_DISPLAY = {
  // Rust
  "encode_into":                { label: "Rust encode_into",           color: "#f97316" },
  "encode_with_tables_chunked": { label: "Rust encode_with_tables_chunked", color: "#fb923c" },
  // C++
  "cpp::encode_into":           { label: "C++ encode_into",            color: "#3b82f6" },
  // Python same-algo
  "python_same_algo::encode_into": { label: "Python same_algo encode_into", color: "#a855f7" },
  // Python reedsolo
  "python_reedsolo::reedsolo.RSCodec.encode": { label: "Python reedsolo (ref)", color: "#22c55e" },
};

function seriesKey(lang, impl) {
  if (lang === "cpp" || lang.startsWith("python")) return `${lang}::${impl}`;
  return impl;   // rust: key is just the impl name
}

async function fetchJson(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`HTTP ${r.status} for ${path}`);
  return r.json();
}

function showMeta(meta) {
  const box = document.getElementById("meta-box");
  if (!meta) { box.textContent = "meta.json not found."; return; }
  box.innerHTML = `
    <strong>Host:</strong> ${meta.uname || "unknown"} &nbsp;|&nbsp;
    <strong>Cores:</strong> ${meta.nproc || "?"} &nbsp;|&nbsp;
    <strong>rustc:</strong> ${meta.rustc || "?"} &nbsp;|&nbsp;
    <strong>g++:</strong> ${meta.gxx || "?"} &nbsp;|&nbsp;
    <strong>Python:</strong> ${meta.python || "?"}<br>
    <span style="color:#64748b">Run date: ${meta.date || "?"}</span>
  `;
}

function buildSeries(allRecords) {
  // Group records by (lang, impl) → Map<key, Map<shard_len, mib_per_s>>
  const byKey = new Map();

  for (const rec of allRecords) {
    if (rec.mib_per_s == null) continue;  // reedsolo not installed
    const key = seriesKey(rec.lang, rec.impl);
    if (!byKey.has(key)) byKey.set(key, new Map());
    byKey.get(key).set(rec.shard_len, rec.mib_per_s);
  }

  const series = [];
  for (const [key, byLen] of byKey) {
    const display = IMPL_DISPLAY[key] || { label: key, color: "#94a3b8" };
    series.push({
      name:  display.label,
      color: display.color,
      data:  SHARD_LENS.map(sl => byLen.get(sl) ?? null),
    });
  }

  return series;
}

function renderGrouped(series) {
  Highcharts.chart("chart-grouped", {
    chart:  { type: "column", backgroundColor: "#1e2130" },
    title:  { text: "Throughput by shard size", style: { color: "#e2e8f0" } },
    subtitle: { text: "All four shard lengths, grouped by implementation", style: { color: "#94a3b8" } },
    xAxis:  {
      categories: SHARD_LENS.map(sl => sl < 1024 ? `${sl} B` : `${sl/1024} KiB`),
      labels: { style: { color: "#94a3b8" } },
    },
    yAxis:  {
      title: { text: "MiB/s", style: { color: "#94a3b8" } },
      labels: { style: { color: "#94a3b8" } },
      gridLineColor: "#2d3748",
    },
    legend: { itemStyle: { color: "#e2e8f0" } },
    plotOptions: { column: { groupPadding: 0.1, pointPadding: 0.05 } },
    series,
    credits: { style: { color: "#64748b" } },
  });
}

function renderLine(series) {
  Highcharts.chart("chart-line", {
    chart:  { type: "line", backgroundColor: "#1e2130" },
    title:  { text: "Throughput vs shard size", style: { color: "#e2e8f0" } },
    subtitle: { text: "Log-scale x-axis", style: { color: "#94a3b8" } },
    xAxis:  {
      type:       "logarithmic",
      categories: SHARD_LENS.map(sl => sl < 1024 ? `${sl} B` : `${sl/1024} KiB`),
      labels: { style: { color: "#94a3b8" } },
    },
    yAxis:  {
      title: { text: "MiB/s", style: { color: "#94a3b8" } },
      labels: { style: { color: "#94a3b8" } },
      gridLineColor: "#2d3748",
    },
    legend: { itemStyle: { color: "#e2e8f0" } },
    plotOptions: {
      line: { marker: { enabled: true, radius: 4 } },
    },
    series,
    credits: { style: { color: "#64748b" } },
  });
}

// ---------------------------------------------------------------------------
// LDPC chart — Rust vs C++ Melem/s (BG1 Z=384, 10 iterations)
// ---------------------------------------------------------------------------

function renderLdpc(ldpcRecords) {
  const container = document.getElementById("chart-ldpc");
  if (!container) return;

  if (ldpcRecords.length === 0) {
    container.innerHTML = `<p style="color:#94a3b8;padding:16px">
      No LDPC results found. Run <code>bash bench/run_all.sh</code> to generate
      <code>bench/results/ldpc_rust.json</code> and <code>bench/results/ldpc_cpp.json</code>.
    </p>`;
    return;
  }

  const LDPC_COLORS = { rust: "#f97316", cpp: "#3b82f6" };
  const byLang = {};
  for (const rec of ldpcRecords) {
    if (rec.melem_per_s == null) continue;
    byLang[rec.lang] = rec.melem_per_s;
  }

  const series = [{
    name: "Rust loms_scalar",
    color: LDPC_COLORS.rust,
    data: [byLang["rust"] ?? null],
  }, {
    name: "C++ loms_scalar",
    color: LDPC_COLORS.cpp,
    data: [byLang["cpp"] ?? null],
  }];

  Highcharts.chart("chart-ldpc", {
    chart: { type: "column", backgroundColor: "#1e2130" },
    title: {
      text: "LDPC Decode Throughput — Rust vs C++ (BG1 Z=384, 10 iter)",
      style: { color: "#e2e8f0" },
    },
    subtitle: {
      text: "scalar path only; AVX2/NEON not yet wired",
      style: { color: "#94a3b8" },
    },
    xAxis: {
      categories: ["BG1 Z=384"],
      labels: { style: { color: "#94a3b8" } },
    },
    yAxis: {
      title: { text: "Melem/s", style: { color: "#94a3b8" } },
      labels: { style: { color: "#94a3b8" } },
      gridLineColor: "#2d3748",
      min: 0,
    },
    legend: { itemStyle: { color: "#e2e8f0" } },
    plotOptions: { column: { groupPadding: 0.1, pointPadding: 0.05 } },
    series,
    credits: { style: { color: "#64748b" } },
  });
}

// ---------------------------------------------------------------------------
// BER waterfall chart — BG1 Z=384 vs Eb/N0 (BPSK AWGN)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Latency chart — μs/call vs payload bytes (log-log)
// ---------------------------------------------------------------------------

function renderLatency(rsRecords, ldpcRecords) {
  const container = document.getElementById("chart-latency");
  if (!container) return;

  if (rsRecords.length === 0 && ldpcRecords.length === 0) {
    container.innerHTML = `<p style="color:#94a3b8;padding:16px">
      No timing data found. Run <code>bash bench/run_all.sh</code>.
    </p>`;
    return;
  }

  const RS_COLORS = {
    "encode_into":                   { label: "Rust encode_into",               color: "#f97316" },
    "encode_with_tables_chunked":    { label: "Rust encode_with_tables_chunked", color: "#fb923c" },
    "encode_with_avx2":              { label: "Rust encode_with_avx2",           color: "#fde68a" },
    "cpp::encode_into":              { label: "C++ encode_into",                 color: "#3b82f6" },
    "python_same_algo::encode_into": { label: "Python same_algo",               color: "#a855f7" },
  };

  // Build one series per (lang, impl) from RS records.
  // x = payload_bytes, y = ns_per_iter / 1000  (μs/call)
  const byKey = new Map();
  for (const rec of rsRecords) {
    if (rec.ns_per_iter == null || rec.payload_bytes == null) continue;
    const key = seriesKey(rec.lang, rec.impl);
    if (!byKey.has(key)) byKey.set(key, []);
    byKey.get(key).push([rec.payload_bytes, rec.ns_per_iter / 1000]);
  }

  const series = [];
  for (const [key, pts] of byKey) {
    pts.sort((a, b) => a[0] - b[0]);
    const meta = RS_COLORS[key] || { label: key, color: "#94a3b8" };
    series.push({ name: meta.label, color: meta.color, data: pts,
                  marker: { enabled: true, radius: 5 } });
  }

  // Add LDPC as single-point markers (payload = variable-node count in bytes).
  for (const rec of ldpcRecords) {
    if (rec.ns_per_iter == null) continue;
    const isRust  = rec.lang === "rust";
    const payload = rec.payload_bytes || 26112;
    series.push({
      name:      isRust ? "Rust LDPC AVX2" : "C++ LDPC scalar",
      color:     isRust ? "#f97316" : "#3b82f6",
      dashStyle: "Dash",
      data:      [[payload, rec.ns_per_iter / 1000]],
      marker:    { enabled: true, symbol: "diamond", radius: 9 },
    });
  }

  Highcharts.chart("chart-latency", {
    chart:    { type: "line", backgroundColor: "#1e2130" },
    title:    { text: "Encode / Decode Latency vs Payload Size",
                style: { color: "#e2e8f0" } },
    subtitle: { text: "Log-log axes — linear O(n) algorithms produce slope-1 lines; diamonds = LDPC decode",
                style: { color: "#94a3b8" } },
    xAxis: {
      type: "logarithmic",
      title:  { text: "Payload bytes", style: { color: "#94a3b8" } },
      labels: {
        style: { color: "#94a3b8" },
        formatter() {
          const v = this.value;
          return v >= 1048576 ? `${(v/1048576).toFixed(0)} MiB`
               : v >= 1024   ? `${(v/1024).toFixed(0)} KiB`
               :                `${v} B`;
        },
      },
      gridLineColor: "#2d3748",
    },
    yAxis: {
      type:   "logarithmic",
      title:  { text: "Latency (μs / call)", style: { color: "#94a3b8" } },
      labels: { style: { color: "#94a3b8" } },
      gridLineColor: "#2d3748",
    },
    legend: { itemStyle: { color: "#e2e8f0" } },
    plotOptions: { line: { connectNulls: false } },
    tooltip: {
      formatter() {
        const x = this.x;
        const xStr = x >= 1048576 ? `${(x/1048576).toFixed(2)} MiB`
                   : x >= 1024   ? `${(x/1024).toFixed(1)} KiB`
                   :               `${x} B`;
        return `<b>${this.series.name}</b><br>Payload: ${xStr}<br>`
             + `Latency: ${this.y.toFixed(1)} μs`;
      },
    },
    series,
    credits: { style: { color: "#64748b" } },
  });
}

function renderBer(berRecords) {
  const container = document.getElementById("chart-ber");
  if (!container) return;

  if (berRecords.length === 0) {
    container.innerHTML = `<p style="color:#94a3b8;padding:16px">
      No BER results found. Run
      <code>cargo run --release --bin ber_sim</code>
      to generate <code>bench/results/ber_rust.json</code>.
    </p>`;
    return;
  }

  const xData  = berRecords.map(r => r.eb_n0_db);
  const berSer = berRecords.map(r => r.ber  > 0 ? r.ber  : null);
  const blerSer= berRecords.map(r => r.bler > 0 ? r.bler : null);

  // Shannon limit for BPSK AWGN at code rate R = 22/68 ≈ 0.324:
  // C = 0.5*log2(1 + SNR) ≥ R  →  SNR = 2^(2R)−1  →  Eb/N0 = SNR/R
  // Eb/N0_shannon ≈ 10*log10((2^(2*22/68)−1) / (22/68)) ≈ −1.1 dB
  const shannonDb = -1.1;

  Highcharts.chart("chart-ber", {
    chart: { type: "line", backgroundColor: "#1e2130" },
    title: {
      text: "BER Waterfall — BG1 Z=384 (BPSK AWGN, β=0.25, 10 iter)",
      style: { color: "#e2e8f0" },
    },
    subtitle: {
      text: "The sharp drop characterises the waterfall region; coding gain ≈ distance to uncoded BER curve",
      style: { color: "#94a3b8" },
    },
    xAxis: {
      title: { text: "Eb/N0 (dB)", style: { color: "#94a3b8" } },
      labels: { style: { color: "#94a3b8" } },
      categories: xData.map(String),
      plotLines: [{
        value: xData.findIndex(x => x >= shannonDb),
        color: "#ef4444",
        dashStyle: "Dash",
        width: 2,
        label: {
          text: `Shannon limit (R≈1/3) ≈ ${shannonDb} dB`,
          style: { color: "#ef4444", fontSize: "11px" },
          rotation: 0,
          y: 14,
        },
      }],
    },
    yAxis: [{
      title: { text: "BER", style: { color: "#94a3b8" } },
      labels: { style: { color: "#94a3b8" } },
      type: "logarithmic",
      min: 1e-6,
      gridLineColor: "#2d3748",
    }, {
      title: { text: "BLER", style: { color: "#94a3b8" } },
      labels: { style: { color: "#94a3b8" } },
      type: "logarithmic",
      min: 1e-4,
      opposite: true,
      gridLineColor: "#2d3748",
    }],
    legend: { itemStyle: { color: "#e2e8f0" } },
    plotOptions: {
      line: { marker: { enabled: true, radius: 5 }, connectNulls: false },
    },
    series: [{
      name: "BER (Rust LOMS)",
      color: "#f97316",
      yAxis: 0,
      data: berSer,
    }, {
      name: "BLER (Rust LOMS)",
      color: "#3b82f6",
      yAxis: 1,
      data: blerSer,
    }],
    credits: { style: { color: "#64748b" } },
  });
}

async function main() {
  let meta = null;
  try {
    meta = await fetchJson(`${RESULT_BASE}/meta.json`);
  } catch (_) { /* optional */ }
  showMeta(meta);

  let allRecords = [];
  const sources = [
    `${RESULT_BASE}/rust.json`,
    `${RESULT_BASE}/cpp.json`,
    `${RESULT_BASE}/python.json`,
  ];

  for (const src of sources) {
    try {
      const data = await fetchJson(src);
      allRecords = allRecords.concat(Array.isArray(data) ? data : [data]);
    } catch (e) {
      console.warn(`Could not load ${src}:`, e.message);
    }
  }

  if (allRecords.length === 0) {
    document.getElementById("chart-grouped").innerHTML =
      `<p class="error">No result data found. Run <code>bash bench/run_all.sh</code> first.</p>`;
  } else {
    const series = buildSeries(allRecords);
    renderGrouped(series);
    renderLine(series);
  }

  // Load LDPC results (independent of RS results)
  let ldpcRecords = [];
  const ldpcSources = [
    `${RESULT_BASE}/ldpc_rust.json`,
    `${RESULT_BASE}/ldpc_cpp.json`,
  ];
  for (const src of ldpcSources) {
    try {
      const data = await fetchJson(src);
      ldpcRecords = ldpcRecords.concat(Array.isArray(data) ? data : [data]);
    } catch (e) {
      console.warn(`Could not load ${src}:`, e.message);
    }
  }
  renderLdpc(ldpcRecords);

  // Latency chart uses the same RS + LDPC records already loaded above.
  renderLatency(allRecords, ldpcRecords);

  // Load BER waterfall results (independent of RS/LDPC throughput results).
  let berRecords = [];
  try {
    const data = await fetchJson(`${RESULT_BASE}/ber_rust.json`);
    berRecords = Array.isArray(data) ? data : [data];
  } catch (e) {
    console.warn("Could not load ber_rust.json:", e.message);
  }
  renderBer(berRecords);
}

main().catch(err => {
  console.error(err);
  document.getElementById("meta-box").innerHTML =
    `<span class="error">Error loading data: ${err.message}</span>`;
});
