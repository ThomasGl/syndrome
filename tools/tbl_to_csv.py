#!/usr/bin/env python3
from pathlib import Path
import re
s = Path('/tmp/ts38212/docx/word/document.xml').read_text()
start=0
idx=0
outdir=Path('/home/thomas/projects/rust_learn/glezer_rsv/data')
outdir.mkdir(exist_ok=True)
while True:
    i = s.find('<w:tbl', start)
    if i==-1:
        break
    j = s.find('</w:tbl>', i)
    if j==-1:
        break
    tbl = s[i:j+8]
    # count cols
    cols = tbl.count('<w:gridCol')
    if cols >= 6:
        # extract cell texts in order
        cells = re.findall(r'<w:t[^>]*>(.*?)</w:t>', tbl, flags=re.DOTALL)
        # Word tables may include multiple runs per cell; split into cells by <w:tc>
        tcs = re.split(r'</w:tc>', tbl)
        rows = []
        row = []
        for tc in tcs:
            if '<w:tc' not in tc:
                continue
            texts = re.findall(r'<w:t[^>]*>(.*?)</w:t>', tc, flags=re.DOTALL)
            cell_text = ' '.join(t.strip() for t in texts if t.strip())
            row.append(cell_text)
            # detect end of row by presence of </w:tr> within tc or by counting
            if '</w:tr>' in tc:
                rows.append(row)
                row = []
        if row:
            rows.append(row)
        # write CSV
        out = outdir / f'table_{idx}.csv'
        with out.open('w', encoding='utf8') as f:
            for r in rows:
                f.write(','.join('"%s"' % c.replace('"','""') for c in r))
                f.write('\n')
        print('Wrote', out, 'cols=',cols,'rows=',len(rows))
    start = j+8
    idx += 1
print('done')
