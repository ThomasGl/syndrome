#!/usr/bin/env python3
from pathlib import Path
import re
p = Path('/tmp/ts38212/docx/word/document.xml')
s = p.read_text()
# find all tables and capture preceding paragraph texts (limit lookback 1000 chars)
entries = []
start = 0
idx = 0
while True:
    i = s.find('<w:tbl', start)
    if i == -1:
        break
    j = s.find('</w:tbl>', i)
    if j == -1:
        break
    # look back for last <w:p> before i
    back = s.rfind('<w:p', 0, i)
    caption = ''
    if back != -1:
        endp = s.find('</w:p>', back, i)
        if endp != -1:
            pblock = s[back:endp+4]
            texts = re.findall(r'<w:t[^>]*>(.*?)</w:t>', pblock, flags=re.DOTALL)
            caption = ' '.join(t.strip() for t in texts if t.strip())
    entries.append((idx, caption[:200].replace('\n',' ')))
    start = j+8
    idx += 1
for e in entries:
    print(e[0], '---', e[1])
