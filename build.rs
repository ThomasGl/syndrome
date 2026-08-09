use std::env;
use std::fs;
use std::path::Path;

fn main() {
    // Only the base-graph table feeds this script; without this line Cargo
    // would re-run it on every source change.
    println!("cargo:rerun-if-changed=data/bg_tables.json");

    let out_dir = env::var("OUT_DIR").expect("OUT_DIR not set");
    let dest = Path::new(&out_dir).join("bg_tables.rs");

    // `data/bg_tables.json` is the 3GPP TS 38.212 base-graph table the whole
    // 5G LDPC path compiles against. It ships in both the git tree and the
    // published tarball, so a missing file means a broken checkout — never a
    // situation to paper over. An earlier version of this script silently
    // emitted a 2-entry placeholder here, which compiled cleanly into a crate
    // whose "BG1" was a two-edge toy; that failure mode is exactly why this
    // is now a hard error.
    let data_path = Path::new("data/bg_tables.json");
    assert!(
        data_path.exists(),
        "data/bg_tables.json is missing. This file ships with the crate (git and \
         crates.io tarball alike); a build without it would produce fake 5G NR \
         base graphs. Restore it from the repository, or regenerate it with \
         tools/gen_bg_tables.py from the 3GPP TS 38.212 tables."
    );

    let s = fs::read_to_string(data_path).expect("failed to read data/bg_tables.json");
    let v: serde_json::Value = serde_json::from_str(&s).expect("invalid json");
    let mut out = String::new();

    // Helper: emit BG constants from the entries-based format.
    // Each entry: {"r": u8, "c": u8, "v": [i16; 8]}
    let emit_bg = |out: &mut String, prefix: &str, bg: &serde_json::Value| {
        let rows = bg
            .get("rows")
            .and_then(|r| r.as_u64())
            .unwrap_or_else(|| panic!("{prefix}: missing integer field `rows`"));
        let cols = bg
            .get("cols")
            .and_then(|c| c.as_u64())
            .unwrap_or_else(|| panic!("{prefix}: missing integer field `cols`"));
        let entries = bg
            .get("entries")
            .and_then(|e| e.as_array())
            .expect("entries must be array");

        let n = entries.len();
        out.push_str(&format!(
            "pub const {}_ROWS: usize = {}usize;\n",
            prefix, rows
        ));
        out.push_str(&format!(
            "pub const {}_COLS: usize = {}usize;\n",
            prefix, cols
        ));
        out.push_str(&format!(
            "pub const {}_ENTRY_COUNT: usize = {}usize;\n",
            prefix, n
        ));
        out.push_str(&format!(
            "pub const {}_ENTRIES: [(u8, u8, [i16; 8]); {}] = [\n",
            prefix, n
        ));
        for entry in entries {
            let r = entry.get("r").and_then(|x| x.as_u64()).expect("entry.r") as u8;
            let c = entry.get("c").and_then(|x| x.as_u64()).expect("entry.c") as u8;
            let v = entry.get("v").and_then(|x| x.as_array()).expect("entry.v");
            let vals: Vec<i16> = v
                .iter()
                .map(|x| x.as_i64().expect("shift must be integer") as i16)
                .collect();
            assert_eq!(vals.len(), 8, "each entry must have exactly 8 shift values");
            out.push_str(&format!(
                "    ({}, {}, [{}, {}, {}, {}, {}, {}, {}, {}]),\n",
                r, c, vals[0], vals[1], vals[2], vals[3], vals[4], vals[5], vals[6], vals[7]
            ));
        }
        out.push_str("];\n\n");
    };

    let bg1 = v
        .get("bg1")
        .expect("data/bg_tables.json: missing top-level `bg1` object");
    let bg2 = v
        .get("bg2")
        .expect("data/bg_tables.json: missing top-level `bg2` object");
    emit_bg(&mut out, "BG1", bg1);
    emit_bg(&mut out, "BG2", bg2);

    fs::write(dest, out).expect("failed writing bg_tables.rs");
}
