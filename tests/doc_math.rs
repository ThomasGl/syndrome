//! The rustdoc math in this crate must survive CommonMark and reach KaTeX intact.
//!
//! # Why this is a test and not only a CI job
//!
//! rustdoc renders a doc comment as CommonMark *before* KaTeX sees it, and
//! CommonMark deletes a backslash that precedes ASCII punctuation. Every
//! LaTeX command whose next character is punctuation therefore arrives
//! mutilated. `katex-header.html` documents the conventions that survive the
//! pass; this file is what stops the documentation being published without
//! them.
//!
//! Two properties make the failure mode unusually bad, and both argue for a
//! gate rather than review:
//!
//! * `throwOnError: false` in `katex-header.html` means a malformed
//!   expression renders as red raw source on docs.rs instead of failing any
//!   build. Nothing goes wrong until a reader looks at the page.
//! * The worst case does not even do that. A row break written `\\` instead
//!   of `\\\\` reaches KaTeX as a single `\`, which KaTeX accepts as a control
//!   space — so a `cases` or `bmatrix` block silently collapses to one row and
//!   renders *cleanly and wrongly*. There is nothing to notice.
//!
//! # What runs, and where
//!
//! The structural checks below are pure Rust with no dependencies, so they run
//! in every `cargo test` on every machine. They cover each documented failure
//! class by construction — see [`Rule`] for the catalogue.
//!
//! A real KaTeX parse is stronger still, because it rejects expressions no
//! structural rule anticipates. That needs Node and the `katex` package, which
//! cannot be a hard requirement of `cargo test`, so
//! [`katex_parses_every_span`] runs it when `tools/node_modules/katex` is
//! present and reports that it was skipped otherwise. The `doc-math` CI job
//! installs it, so the full check is mandatory on every push and best-effort
//! locally.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

/// Every ASCII punctuation character CommonMark allows a backslash to escape.
const ASCII_PUNCT: &str = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";

/// Apply CommonMark's backslash-escape rule: a backslash before ASCII
/// punctuation is removed and the punctuation kept literally. A backslash
/// before anything else — a letter, a space — passes through untouched.
fn apply_commonmark_escapes(text: &str) -> String {
    let bytes: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == '\\' && i + 1 < bytes.len() && ASCII_PUNCT.contains(bytes[i + 1]) {
            out.push(bytes[i + 1]);
            i += 2;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    out
}

/// Whether `text` contains `needle` in a position where it is *not* escaped by
/// a preceding backslash.
///
/// This is the difference between a bug and the fix for it. After CommonMark
/// has run, `\text{a\_b}` (from a correctly doubled `\\_` in the source) still
/// carries its backslash and is fine; a bare `a_b` is the parse error. The
/// same distinction separates a column-separator `&` from an escaped literal
/// `\&`.
fn contains_unescaped(text: &str, needle: char) -> bool {
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '\\' {
            // Skip the backslash and whatever it escapes.
            i += 2;
            continue;
        }
        if chars[i] == needle {
            return true;
        }
        i += 1;
    }
    false
}

/// One math span found in a doc comment.
struct Span {
    file: PathBuf,
    line: usize,
    /// The source between the delimiters, exactly as written in the `.rs`.
    source: String,
    display: bool,
}

/// Recursively collect `.rs` files under `dir`.
fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            rust_files(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
    out.sort();
}

/// Extract every `$...$` and `$$...$$` span from a file's `///` and `//!`
/// comments.
///
/// Display spans are matched first and may run across lines, which is how
/// every block equation in this crate is written; inline spans are matched in
/// what remains. A lone `$` in prose is left alone.
fn spans_in(file: &Path) -> Vec<Span> {
    let Ok(text) = std::fs::read_to_string(file) else {
        return Vec::new();
    };

    // Join the doc-comment text, remembering which line each character came
    // from so a failure can be reported where the span starts.
    let mut doc = String::new();
    let mut line_of: Vec<usize> = Vec::new();
    for (idx, raw) in text.lines().enumerate() {
        let trimmed = raw.trim_start();
        let body = trimmed
            .strip_prefix("///")
            .or_else(|| trimmed.strip_prefix("//!"));
        if let Some(body) = body {
            for ch in body.chars().chain(std::iter::once('\n')) {
                doc.push(ch);
                line_of.push(idx + 1);
            }
        }
    }

    let chars: Vec<char> = doc.chars().collect();
    let mut spans = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        if chars[i] != '$' {
            i += 1;
            continue;
        }
        let display = i + 1 < chars.len() && chars[i + 1] == '$';
        let delim = if display { 2 } else { 1 };
        let start = i + delim;
        // Find the closing delimiter. Inline spans do not cross a blank line.
        let mut j = start;
        let mut end = None;
        while j < chars.len() {
            if chars[j] == '$' {
                if display {
                    if j + 1 < chars.len() && chars[j + 1] == '$' {
                        end = Some(j);
                        break;
                    }
                } else {
                    end = Some(j);
                    break;
                }
            }
            if !display && chars[j] == '\n' {
                break;
            }
            j += 1;
        }
        match end {
            Some(e) => {
                spans.push(Span {
                    file: file.to_path_buf(),
                    line: line_of[i],
                    source: chars[start..e].iter().collect(),
                    display,
                });
                i = e + delim;
            }
            // Unpaired `$`: prose, not math.
            None => i += 1,
        }
    }
    spans
}

/// The failure classes this file checks for, each named after what it costs a
/// reader.
///
/// Every one corresponds to a specific way CommonMark's escape pass damages
/// LaTeX, documented in `katex-header.html`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Rule {
    /// A row break written with two or three backslashes. CommonMark eats one,
    /// so it reaches KaTeX as `\` — a control space — and the `cases` or
    /// `bmatrix` block collapses into a single row. Renders cleanly and
    /// wrongly, which is why the check is on the source, where the count is
    /// unambiguous: four is a row break, one is a deliberate `\ `, two or
    /// three cannot be either.
    CollapsedRowBreak,
    /// A spacing or grouping macro whose backslash CommonMark will delete:
    /// `\,` `\;` `\!` become literal punctuation, `\{` and `\}` vanish into
    /// grouping. `katex-header.html` prescribes the letter-only forms
    /// (`\thinspace`, `\lbrace`) or a doubled backslash.
    EatenMacro,
    /// A `&` that reaches KaTeX bare. Inside an alignment environment that is
    /// a column separator; outside one it is a parse error rendered as red
    /// raw source.
    BareAmpersand,
    /// An underscore inside `\text{}` or `\mathrm{}`. Text mode has no
    /// subscripts, so this is a hard KaTeX parse error. Outside text mode a
    /// bare `_` is correct and expected.
    UnderscoreInTextMode,
    /// `\begin{x}` without a matching `\end{x}`, or unbalanced braces —
    /// usually the symptom of one of the above rather than a mistake of its
    /// own, but worth naming separately when it is the visible damage.
    Unbalanced,
}

struct Finding {
    file: PathBuf,
    line: usize,
    rule: Rule,
    detail: String,
}

/// Environments in which a bare `&` is a legitimate column separator.
const ALIGNMENT_ENVS: [&str; 8] = [
    "cases", "aligned", "align", "array", "matrix", "bmatrix", "pmatrix", "vmatrix",
];

fn check_span(span: &Span, out: &mut Vec<Finding>) {
    let mut push = |rule: Rule, detail: String| {
        out.push(Finding {
            file: span.file.clone(),
            line: span.line,
            rule,
            detail,
        });
    };

    // ---- Source-level rules --------------------------------------------
    // Runs of backslashes, classified by length and by what follows.
    let src: Vec<char> = span.source.chars().collect();
    let mut i = 0;
    while i < src.len() {
        if src[i] != '\\' {
            i += 1;
            continue;
        }
        let start = i;
        while i < src.len() && src[i] == '\\' {
            i += 1;
        }
        let run = i - start;
        let next = src.get(i).copied();

        match next {
            // A row break: the run is followed by whitespace or ends the span.
            None | Some('\n' | ' ' | '\t') => {
                if run == 2 || run == 3 {
                    push(
                        Rule::CollapsedRowBreak,
                        format!(
                            "{run} backslashes before whitespace; a row break needs four \
                             (CommonMark eats one, KaTeX needs two)"
                        ),
                    );
                }
            }
            // Spacing and grouping macros: an odd run leaves none behind.
            Some(c @ (',' | ';' | '!' | '{' | '}')) if run % 2 == 1 => {
                push(
                    Rule::EatenMacro,
                    format!(
                        "`\\{c}` loses its backslash to CommonMark; use a letter-only macro \
                         (\\thinspace, \\thickspace, \\negthinspace, \\lbrace, \\rbrace) \
                         or double the backslash"
                    ),
                );
            }
            Some('&') if run % 2 == 1 => {
                push(
                    Rule::BareAmpersand,
                    "`\\&` loses its backslash to CommonMark and reaches KaTeX bare; \
                     write `\\\\&`"
                        .to_string(),
                );
            }
            _ => {}
        }
    }

    // ---- Post-markdown rules -------------------------------------------
    let rendered = apply_commonmark_escapes(&span.source);

    // Braces must balance, or grouping has been lost.
    let mut depth = 0i32;
    for ch in rendered.chars() {
        match ch {
            '{' => depth += 1,
            '}' => depth -= 1,
            _ => {}
        }
        if depth < 0 {
            break;
        }
    }
    if depth != 0 {
        push(
            Rule::Unbalanced,
            format!("braces do not balance after markdown's escape pass (depth {depth})"),
        );
    }

    // `\begin{x}` must be matched by `\end{x}`.
    let mut envs: Vec<String> = Vec::new();
    for (kw, opening) in [("\\begin{", true), ("\\end{", false)] {
        let mut from = 0;
        while let Some(pos) = rendered[from..].find(kw) {
            let abs = from + pos + kw.len();
            let name: String = rendered[abs..].chars().take_while(|&c| c != '}').collect();
            if opening {
                envs.push(name.clone());
            } else if let Some(idx) = envs.iter().rposition(|e| *e == name) {
                envs.remove(idx);
            } else {
                envs.push(format!("!unmatched-end:{name}"));
            }
            from = abs;
        }
    }
    if !envs.is_empty() {
        push(
            Rule::Unbalanced,
            format!("unmatched environment(s): {envs:?}"),
        );
    }

    // A bare `&` outside an alignment environment is a KaTeX parse error.
    // An escaped `\&` is a literal ampersand and is fine anywhere.
    if contains_unescaped(&rendered, '&') {
        let in_alignment = ALIGNMENT_ENVS
            .iter()
            .any(|e| rendered.contains(&format!("\\begin{{{e}")));
        if !in_alignment {
            push(
                Rule::BareAmpersand,
                "a bare `&` outside an alignment environment is a KaTeX parse error".to_string(),
            );
        }
    }

    // `_` inside \text{} / \mathrm{} is a hard parse error.
    for kw in ["\\text{", "\\mathrm{"] {
        let mut from = 0;
        while let Some(pos) = rendered[from..].find(kw) {
            let body_start = from + pos + kw.len();
            let body: String = rendered[body_start..]
                .chars()
                .take_while(|&c| c != '}')
                .collect();
            if contains_unescaped(&body, '_') {
                push(
                    Rule::UnderscoreInTextMode,
                    format!(
                        "`_` inside {}{{{}}} — text mode has no subscripts; write `\\\\_`",
                        kw.trim_end_matches('{'),
                        body
                    ),
                );
            }
            from = body_start;
        }
    }

    // A display span that opened an alignment environment but has no row
    // break at all is suspicious only if it also has an `&`; that is already
    // covered above, so nothing more is needed here. Kept as a comment rather
    // than a rule so the omission is deliberate rather than forgotten.
    let _ = span.display;
}

fn collect_findings() -> (usize, Vec<Finding>) {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut files = Vec::new();
    rust_files(&root, &mut files);
    assert!(
        !files.is_empty(),
        "found no .rs files under {} — the doc-math check would pass vacuously",
        root.display()
    );

    let mut findings = Vec::new();
    let mut count = 0usize;
    for file in &files {
        for span in spans_in(file) {
            count += 1;
            check_span(&span, &mut findings);
        }
    }
    (count, findings)
}

/// Every math span in the crate's doc comments must survive CommonMark's
/// escape pass without losing a macro, a row break, a brace or a text-mode
/// escape.
///
/// Pure Rust, no dependencies, so this runs everywhere `cargo test` does.
#[test]
fn doc_comment_math_survives_markdowns_escape_pass() {
    let (checked, findings) = collect_findings();
    assert!(
        checked > 100,
        "only {checked} math spans found; the extractor is probably broken, and a \
         vacuous pass here is worse than no check at all"
    );

    if !findings.is_empty() {
        let mut msg = format!(
            "{} of {checked} math spans will not survive rustdoc's CommonMark pass.\n\
             See katex-header.html for the conventions.\n\n",
            findings.len()
        );
        for f in &findings {
            let _ = writeln!(
                msg,
                "  {}:{}  [{:?}] {}",
                f.file
                    .strip_prefix(env!("CARGO_MANIFEST_DIR"))
                    .unwrap_or(&f.file)
                    .display(),
                f.line,
                f.rule,
                f.detail
            );
        }
        panic!("{msg}");
    }
}

/// The extractor must actually find the crate's math, and the rules must
/// actually fire.
///
/// A checker that silently matches nothing passes forever. This pins both
/// ends: that real spans are being found, and that each rule rejects the
/// construct it exists for.
#[test]
fn the_checker_itself_has_teeth() {
    let (checked, _) = collect_findings();
    assert!(
        checked > 500,
        "expected the crate to contain many math spans, found {checked}"
    );

    let cases: [(&str, Rule); 6] = [
        (
            r"\begin{cases} a & b \\ c & d \end{cases}",
            Rule::CollapsedRowBreak,
        ),
        (r"x \, y", Rule::EatenMacro),
        (r"\{ x \}", Rule::EatenMacro),
        (r"x \& y", Rule::BareAmpersand),
        (r"\text{a\_b}", Rule::UnderscoreInTextMode),
        (r"\begin{cases} a \\\\ b", Rule::Unbalanced),
    ];
    for (source, expect) in cases {
        let span = Span {
            file: PathBuf::from("<self-test>"),
            line: 0,
            source: source.to_string(),
            display: true,
        };
        let mut found = Vec::new();
        check_span(&span, &mut found);
        assert!(
            found.iter().any(|f| f.rule == expect),
            "rule {expect:?} did not fire on {source:?}; it fired {:?}",
            found.iter().map(|f| f.rule).collect::<Vec<_>>()
        );
    }

    // And the correct forms must NOT fire.
    for good in [
        r"\begin{cases} a \\\\ b \end{cases}",
        r"x \thinspace y",
        r"\lbrace x \rbrace",
        r"a \\& b",
        r"\text{a\\_b}",
        r"x\_i",
        r"\frac{1}{2}",
    ] {
        let span = Span {
            file: PathBuf::from("<self-test>"),
            line: 0,
            source: good.to_string(),
            display: true,
        };
        let mut found = Vec::new();
        check_span(&span, &mut found);
        assert!(
            found.is_empty(),
            "{good:?} is a correct form but was flagged: {:?}",
            found
                .iter()
                .map(|f| (f.rule, &f.detail))
                .collect::<Vec<_>>()
        );
    }
}

/// Run the real KaTeX parser over every span, when the tooling is available.
///
/// The structural rules above encode the failure classes that are *known*.
/// KaTeX rejects the ones that are not, so where Node and the `katex` package
/// are installed this is the stronger check and the one that matters. It
/// cannot be a hard requirement of `cargo test` — a Rust crate must not need
/// npm to run its suite — so it degrades to a printed skip, and the `doc-math`
/// CI job installs the dependency so the strong check is mandatory on every
/// push.
///
/// ```text
/// cd tools && npm ci && npm run check
/// ```
#[test]
fn katex_parses_every_span() {
    let tools = Path::new(env!("CARGO_MANIFEST_DIR")).join("tools");
    if !tools.join("node_modules/katex").is_dir() {
        println!(
            "skipping the KaTeX parse: {} not installed. \
             Run `cd tools && npm ci` to enable it; CI always does.",
            tools.join("node_modules/katex").display()
        );
        return;
    }

    let output = std::process::Command::new("node")
        .arg(tools.join("check_doc_math.mjs"))
        .arg(Path::new(env!("CARGO_MANIFEST_DIR")).join("src"))
        .current_dir(&tools)
        .output();

    let Ok(output) = output else {
        println!("skipping the KaTeX parse: `node` is not on PATH.");
        return;
    };

    assert!(
        output.status.success(),
        "KaTeX rejected at least one doc-comment math span:\n{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
}
