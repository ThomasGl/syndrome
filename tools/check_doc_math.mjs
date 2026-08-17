#!/usr/bin/env node
//
// Verify that every LaTeX span in the crate's rustdoc comments still parses
// after CommonMark has had its way with it.
//
// WHY THIS EXISTS
//
// rustdoc renders a doc comment as CommonMark and only then hands the result
// to KaTeX, and CommonMark drops a backslash that precedes ASCII punctuation.
// So `\,` reaches KaTeX as a literal comma, `\{` as a brace that vanishes into
// grouping, `\\` as a single backslash that no longer breaks a row, and `\&`
// as a bare `&` that is a hard parse error. `katex-header.html` sets
// `throwOnError: false`, which means a broken expression does not fail
// loudly — it renders as red raw source on docs.rs. Eyeballing the generated
// pages does not catch the quiet half of that failure mode, where the
// expression renders cleanly and *wrongly*.
//
// This script reproduces the pipeline: extract each math span, apply
// CommonMark's escape rule to it, then parse the result with KaTeX in
// throw-on-error mode. `katex-header.html` documents the source conventions
// that survive the pass; this is the check that they were followed.
//
// A parse check alone is not enough, because the worst case of this bug
// parses: see `shortRowBreaks` below, which catches it on the source instead.
//
// SCOPE
//
// Rust doc comments only (`///`, `//!`) under the directories given on the
// command line, defaulting to `src`. README.md, CHANGELOG.md and
// system_architecture.md are rendered by GitHub, whose math extension keeps
// backslashes verbatim, so they use the opposite convention and are
// deliberately not checked here.
//
// USAGE
//
//   npm install katex@0.18.1
//   node tools/check_doc_math.mjs [dir ...]
//
// Exits non-zero and prints `file:line` for every span that fails to parse.

import { readdirSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";
import katex from "katex";

/** Every ASCII punctuation character CommonMark allows a backslash to escape. */
const ASCII_PUNCT = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";

/**
 * Apply CommonMark's backslash-escape rule: a backslash before ASCII
 * punctuation is removed and the punctuation kept literally. Everything else,
 * including a backslash before a letter, passes through untouched.
 */
function applyCommonMarkEscapes(text) {
  let out = "";
  for (let i = 0; i < text.length; i++) {
    if (text[i] === "\\" && i + 1 < text.length && ASCII_PUNCT.includes(text[i + 1])) {
      out += text[i + 1];
      i++;
    } else {
      out += text[i];
    }
  }
  return out;
}

/**
 * Report row breaks written with two backslashes where four are needed.
 *
 * This is the one failure the parse check cannot see. A row break written
 * `\\` arrives at KaTeX as a single `\`, which KaTeX reads as a control space
 * rather than rejecting — so a `cases` or `bmatrix` block silently collapses
 * to one row and renders cleanly and wrongly.
 *
 * The test is on the *source*, where it is unambiguous. Looking at maximal
 * runs of backslashes that are followed by whitespace or end the span:
 *
 * - a run of four is a correct row break (it reaches KaTeX as `\\`);
 * - a run of one is a LaTeX control space `\ `, which CommonMark leaves alone
 *   because a space is not ASCII punctuation, and is therefore fine;
 * - a run of two or three is a row break that will not survive.
 *
 * Runs followed by something other than whitespace are left alone: `\\&` and
 * `\\_` are exactly the doubled forms `katex-header.html` prescribes.
 *
 * Returns a list of human-readable descriptions, empty when the span is clean.
 */
function shortRowBreaks(source) {
  const found = [];
  const runs = /\\+/g;
  let m;
  while ((m = runs.exec(source)) !== null) {
    const len = m[0].length;
    const next = source[m.index + len];
    if (next !== undefined && !/\s/.test(next)) continue;
    if (len === 2 || len === 3) {
      const around = source
        .slice(Math.max(0, m.index - 24), m.index + len + 12)
        .replace(/\s+/g, " ");
      found.push(
        `row break written with ${len} backslashes near "...${around}..." — CommonMark ` +
          "eats one, so it needs four to reach KaTeX as a row break",
      );
    }
  }
  return found;
}

/** Recursively collect `.rs` files under `dir`. */
function rustFiles(dir) {
  const out = [];
  for (const entry of readdirSync(dir)) {
    const p = join(dir, entry);
    if (statSync(p).isDirectory()) out.push(...rustFiles(p));
    else if (entry.endsWith(".rs")) out.push(p);
  }
  return out;
}

/**
 * Extract the doc-comment text of a file as a list of `{ line, text }`,
 * with the `///` / `//!` markers stripped. Ordinary `//` comments and code
 * are skipped: only what rustdoc renders is checked.
 */
function docLines(source) {
  const out = [];
  source.split("\n").forEach((line, i) => {
    const m = line.match(/^\s*\/\/(?:\/|!)(.*)$/);
    if (m) out.push({ line: i + 1, text: m[1] });
  });
  return out;
}

/**
 * Find math spans in the joined doc text.
 *
 * Display spans (`$$...$$`) are matched first and may run across lines, which
 * is how every block equation in this crate is written; inline spans (`$...$`)
 * are matched in what remains. A `$` that is not part of a pair — a dollar
 * sign in prose — is left alone.
 */
function mathSpans(docs) {
  // One string per file, remembering which line each character came from, so
  // a failure can be reported at the line the span starts on.
  let text = "";
  const lineOf = [];
  for (const { line, text: t } of docs) {
    for (const ch of t + "\n") {
      text += ch;
      lineOf.push(line);
    }
  }

  const spans = [];
  const taken = new Array(text.length).fill(false);

  const display = /\$\$([\s\S]*?)\$\$/g;
  let m;
  while ((m = display.exec(text)) !== null) {
    spans.push({ line: lineOf[m.index], body: m[1], display: true });
    for (let i = m.index; i < m.index + m[0].length; i++) taken[i] = true;
  }

  const inline = /\$([^$\n]+?)\$/g;
  while ((m = inline.exec(text)) !== null) {
    if (taken[m.index]) continue;
    spans.push({ line: lineOf[m.index], body: m[1], display: false });
  }

  return spans;
}

const dirs = process.argv.slice(2);
const roots = dirs.length > 0 ? dirs : ["src"];

let checked = 0;
const failures = [];

for (const root of roots) {
  for (const file of rustFiles(root)) {
    const source = readFileSync(file, "utf8");
    for (const span of mathSpans(docLines(source))) {
      const rendered = applyCommonMarkEscapes(span.body);
      checked++;
      try {
        katex.renderToString(rendered, {
          displayMode: span.display,
          throwOnError: true,
        });
      } catch (err) {
        failures.push({
          file,
          line: span.line,
          source: span.body.trim(),
          afterMarkdown: rendered.trim(),
          message: String(err.message ?? err).replace(/\s+/g, " "),
        });
        continue;
      }
      for (const message of shortRowBreaks(span.body)) {
        failures.push({
          file,
          line: span.line,
          source: span.body.trim(),
          afterMarkdown: rendered.trim(),
          message,
        });
      }
    }
  }
}

for (const f of failures) {
  console.error(`${f.file}:${f.line}: math span does not survive markdown's escape pass`);
  console.error(`    source          : ${f.source}`);
  console.error(`    after markdown  : ${f.afterMarkdown}`);
  console.error(`    error           : ${f.message}`);
  console.error("");
}

if (failures.length > 0) {
  console.error(
    `${failures.length} of ${checked} math spans failed. ` +
      "See katex-header.html for the source conventions that survive CommonMark.",
  );
  process.exit(1);
}

console.log(`${checked} math spans parse cleanly after markdown's escape pass.`);
