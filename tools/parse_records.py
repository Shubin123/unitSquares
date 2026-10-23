"""Regenerate data/records.json from David Ellsworth's best-known table.

    curl -sL https://kingbird.myphotos.cc/packing/squares_in_squares.html -o /tmp/sq.html
    python3 tools/parse_records.py /tmp/sq.html > data/records.json
"""
import json
import re
import sys

text = re.sub(r'<!--.*?-->', '', open(sys.argv[1], encoding='utf-8').read(), flags=re.S)
best = {}
for box in text.split('<div class="box">')[1:]:
    head = re.search(r'<font size="\+3">(.*?)<br>', box, re.S)
    val = re.search(r'\\Nn\{([0-9.]+)\}', box) or re.search(r'\$s = ([0-9.]+)\$', box)
    if head and val:
        for n in map(int, re.findall(r'\d+', head.group(1))):
            best.setdefault(n, float(val.group(1)))

json.dump({
    'source': 'https://kingbird.myphotos.cc/packing/squares_in_squares.html',
    'note': 'Best known side s(n) for n with nontrivial or proven entries, compiled by David Ellsworth '
            '(after Erich Friedman). Unlisted n inherit s from the next listed n (monotonicity).',
    's': {str(k): best[k] for k in sorted(best)},
}, sys.stdout, indent=0)
