use std::env;
use std::fs;
use std::path::Path;

fn main() {
    let out_dir = env::var("OUT_DIR").expect("OUT_DIR not set");
    let dest = Path::new(&out_dir).join("bg_tables.rs");

    let data_path = Path::new("data/bg_tables.json");
    if data_path.exists() {
        let s = fs::read_to_string(data_path).expect("failed to read data/bg_tables.json");
        let v: serde_json::Value = serde_json::from_str(&s).expect("invalid json");
        let mut out = String::new();

        // Helper: emit BG constants from the entries-based format.
        // Each entry: {"r": u8, "c": u8, "v": [i16; 8]}
        let emit_bg = |out: &mut String, prefix: &str, bg: &serde_json::Value| {
            let rows = bg.get("rows").and_then(|r| r.as_u64()).unwrap_or(2);
            let cols = bg.get("cols").and_then(|c| c.as_u64()).unwrap_or(3);
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

        if let Some(bg1) = v.get("bg1") {
            emit_bg(&mut out, "BG1", bg1);
        }
        if let Some(bg2) = v.get("bg2") {
            emit_bg(&mut out, "BG2", bg2);
        }

        fs::write(dest, out).expect("failed writing bg_tables.rs");
    } else {
        // Minimal fallback placeholder when data/bg_tables.json is absent.
        // Matches the new entries-based format with 2 placeholder entries each.
        let out = r#"pub const BG1_ROWS: usize = 2usize;
pub const BG1_COLS: usize = 3usize;
pub const BG1_ENTRY_COUNT: usize = 2usize;
pub const BG1_ENTRIES: [(u8, u8, [i16; 8]); 2] = [
    (0, 0, [0, 0, 0, 0, 0, 0, 0, 0]),
    (1, 2, [1, 1, 1, 1, 1, 1, 1, 1]),
];

pub const BG2_ROWS: usize = 2usize;
pub const BG2_COLS: usize = 2usize;
pub const BG2_ENTRY_COUNT: usize = 2usize;
pub const BG2_ENTRIES: [(u8, u8, [i16; 8]); 2] = [
    (0, 0, [0, 0, 0, 0, 0, 0, 0, 0]),
    (1, 1, [1, 1, 1, 1, 1, 1, 1, 1]),
];
"#;
        fs::write(dest, out).expect("failed writing placeholder bg_tables.rs");
    }
}
