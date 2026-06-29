#!/usr/bin/env python3
from pathlib import Path
p = Path('/tmp/ts38212/docx/word/document.xml')
if not p.exists():
    print('document.xml missing')
    raise SystemExit(1)
s = p.read_text()
start = 0
count = 0
while True:
    i = s.find('<w:tbl', start)
    if i == -1 or count >= 10:
        break
    j = s.find('</w:tbl>', i)
    if j == -1:
        break
    tbl = s[i:j+8]
    print('--- TABLE', count)
    # show small slice
    print(tbl[:1000])
    print('...')
    start = j+8
    count += 1
print('found', count, 'tables')
