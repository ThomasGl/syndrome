#!/usr/bin/env python3
"""Generate data/bg_tables.json from the 3GPP TS 38.212 docx XML.

Reads Table index 11 (BG1) and Table index 12 (BG2) from:
  /tmp/ts38212/docx/word/document.xml

Outputs:
  data/bg_tables.json

Entry format: {"r": row_idx, "c": col_idx, "v": [v0..v7]}
Absent entries (all -1 in spec) are simply omitted from the list.
"""

import json
import os
import sys
import xml.etree.ElementTree as ET

NS = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
XML_PATH = '/tmp/ts38212/docx/word/document.xml'
OUT_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'bg_tables.json')


def cell_text(cell):
    """Return the concatenated text content of a table cell."""
    return ''.join(t.text or '' for t in cell.iter(NS + 't')).strip()


def parse_bg_table(tbl):
    """Parse a BG table element.

    The table layout (confirmed by inspection):
      row0: 4 merged cells (title)
      row1: 6 cells (labels: Rowindex, Columnindex, Set index 0..7, ...)
      row2: 20 cells (set index labels 0-7 repeated twice)
      row3+: data rows with 20 cells each

    Each data row contains two side-by-side entries:
      Left entry:  cells[0..10]  -> [row_idx, col_idx, v0..v7]
      Right entry: cells[10..20] -> [row_idx, col_idx, v0..v7]

    An empty row_idx ('') means inherit the current row_idx from the
    previous non-empty row_idx.
    """
    entries = []
    rows = list(tbl.iter(NS + 'tr'))

    current_row_left = None   # tracks inherited row index for left column
    current_row_right = None  # tracks inherited row index for right column

    for ri, row in enumerate(rows):
        if ri < 3:
            continue  # skip header rows

        cells = list(row.findall('.//' + NS + 'tc'))
        if len(cells) != 20:
            # Some rows may have fewer cells due to XML merging; skip them
            continue

        texts = [cell_text(c) for c in cells]

        # --- Left entry (cells 0..10) ---
        left_row_str = texts[0]
        left_col_str = texts[1]
        left_vals_str = texts[2:10]

        if left_row_str != '':
            try:
                current_row_left = int(left_row_str)
            except ValueError:
                current_row_left = None

        if current_row_left is not None and left_col_str != '':
            try:
                col_idx = int(left_col_str)
                shifts = [int(v) for v in left_vals_str]
                # Only include if at least one shift is non-negative
                # (all -1 would indicate absent, but spec XML doesn't use -1;
                # absence is encoded by missing rows).
                entries.append({'r': current_row_left, 'c': col_idx, 'v': shifts})
            except ValueError:
                pass  # skip malformed cells

        # --- Right entry (cells 10..20) ---
        right_row_str = texts[10]
        right_col_str = texts[11]
        right_vals_str = texts[12:20]

        if right_row_str != '':
            try:
                current_row_right = int(right_row_str)
            except ValueError:
                current_row_right = None

        if current_row_right is not None and right_col_str != '':
            try:
                col_idx = int(right_col_str)
                shifts = [int(v) for v in right_vals_str]
                entries.append({'r': current_row_right, 'c': col_idx, 'v': shifts})
            except ValueError:
                pass

    return entries


def main():
    print(f'Parsing {XML_PATH} ...', file=sys.stderr)
    tree = ET.parse(XML_PATH)
    root = tree.getroot()

    tables = list(root.iter(NS + 'tbl'))
    print(f'Total tables found: {len(tables)}', file=sys.stderr)

    if len(tables) < 13:
        print('ERROR: expected at least 13 tables (indices 0-12)', file=sys.stderr)
        sys.exit(1)

    print('Parsing BG1 (table index 11) ...', file=sys.stderr)
    bg1_entries = parse_bg_table(tables[11])
    print(f'  -> {len(bg1_entries)} entries', file=sys.stderr)

    print('Parsing BG2 (table index 12) ...', file=sys.stderr)
    bg2_entries = parse_bg_table(tables[12])
    print(f'  -> {len(bg2_entries)} entries', file=sys.stderr)

    result = {
        'bg1': {
            'rows': 46,
            'cols': 68,
            'entries': bg1_entries,
        },
        'bg2': {
            'rows': 42,
            'cols': 52,
            'entries': bg2_entries,
        },
    }

    out_path = os.path.normpath(OUT_PATH)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'Written to {out_path}', file=sys.stderr)


if __name__ == '__main__':
    main()
