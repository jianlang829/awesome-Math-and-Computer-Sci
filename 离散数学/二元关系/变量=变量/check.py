#!/usr/bin/env python3
import re, sys, pathlib

REFLEX = re.compile(r"\b([A-Za-z_]\w*)\s*=\s*\1\b(?=\s*[;,)}])", re.MULTILINE)

p = pathlib.Path(sys.argv[1])
text = p.read_text(encoding="utf-8", errors="ignore")
print(f"----- 文件共 {len(text)} 字符，{text.count(chr(10))} 行 -----")
for m in REFLEX.finditer(text):
    lineno = text[: m.start()].count("\n") + 1
    print(f'Line {lineno}: 命中 "{m.group(0)}"')
