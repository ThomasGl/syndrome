#!/usr/bin/env python3
import re
from pathlib import Path
p = Path('/tmp/ts38212/docx/word/document.txt')
if not p.exists():
    print('document.txt not found at', p)
    raise SystemExit(1)
text = p.read_text()
# Look for 'Base graph' case-insensitive or 'Base Graph' or 'Base graph 1' or 'Table 5.3.2-1'
patterns = [r'Base\s+graph\s+1', r'Base\s+graph\s+2', r'Table\s+5\.3\.2', r'Base\s+Graph']
found = False
for pat in patterns:
    for m in re.finditer(pat, text, flags=re.IGNORECASE):
        found = True
        i = m.start()
        start = max(0, text.rfind('\n', 0, i-1000))
        end = text.find('\n\n', i+1000)
        if end == -1:
            end = i+2000
        snippet = text[start:end]
        print('--- MATCH', pat, 'at', m.start())
        print(snippet)
        print('----')
if not found:
    print('No base graph patterns found')
