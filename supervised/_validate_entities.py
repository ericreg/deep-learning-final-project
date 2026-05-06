import json
import re
from pathlib import Path

import nltk

nltk.download("punkt", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)

HEX_PREFIX_RE = re.compile(r"0x[0-9A-Za-z]+")
LONG_HEX_RE = re.compile(r"^[A-Fa-f0-9]{8,}$")
MEMORY_ADDRISH_RE = re.compile(r"^(?:\d+(?:ULL|LL|UL|L)|[A-Fa-f0-9]{10,})$")
BAD_POS = {"JJ", "JJR", "JJS", "RB", "RBR", "RBS", "IN"}

p = Path("wmdp_entities_template.json")
d = json.loads(p.read_text(encoding="utf-8"))

bad = []
for k in d:
    compact = k.replace(" ", "")
    if HEX_PREFIX_RE.search(k) or LONG_HEX_RE.fullmatch(compact) or MEMORY_ADDRISH_RE.fullmatch(compact):
        bad.append((k, "hex"))
        continue

    tags = [t for _, t in nltk.pos_tag(nltk.word_tokenize(k))]
    if any(t in BAD_POS for t in tags):
        bad.append((k, f"pos:{tags}"))

print("TOTAL", len(d))
print("BAD", len(bad))
print("TOP10", list(d.items())[:10])
print("SAMPLE_BAD", bad[:10])
