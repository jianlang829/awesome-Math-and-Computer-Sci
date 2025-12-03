import re, sys

for line in sys.stdin:
    if re.search(r"error", line) and not re.search(r"warning", line):
        print(line, end="")
