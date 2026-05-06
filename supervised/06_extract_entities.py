#!/usr/bin/env python3
"""Extract OOD proper nouns from WMDP and map them to generic placeholders."""

import json
import re
import sys
import traceback
from collections import Counter
from typing import Dict, Iterable, List, Tuple

import nltk
from datasets import Dataset, concatenate_datasets, load_dataset
from nltk.corpus import stopwords, words


SUBSETS = ["wmdp-bio", "wmdp-cyber"]
TOP_K = 100
OUTPUT_FILE = "wmdp_entities_template.json"
PROPER_TAGS = {"NNP", "NNPS"}

HEX_PREFIX_RE = re.compile(r"0x[0-9A-Za-z]+")
LONG_HEX_RE = re.compile(r"^[A-Fa-f0-9]{8,}$")
MEMORY_ADDRISH_RE = re.compile(r"^(?:\d+(?:ULL|LL|UL|L)|[A-Fa-f0-9]{10,})$")
UPPER_ADJECTIVE_RE = re.compile(r"^[A-Z]{5,}$")
VALID_TOKEN_RE = re.compile(r"^[A-Za-z][A-Za-z0-9-]*$")

BIO_HINTS = (
    "cov",
    "virus",
    "phage",
    "toxin",
    "venom",
    "bacteria",
    "coccus",
    "protein",
    "enzyme",
    "dna",
    "rna",
    "cdna",
    "bacillus",
    "escherichia",
    "anthracis",
    "influenza",
    "ebola",
    "marburg",
    "hiv",
    "vaccinia",
    "clostridium",
    "brucella",
    "streptococcus",
)
CYBER_HINTS = (
    "cobalt",
    "strike",
    "metasploit",
    "meterpreter",
    "powershell",
    "exploit",
    "malware",
    "trojan",
    "buffer",
    "network",
    "packet",
    "checksum",
    "http",
    "tcp",
    "udp",
    "modbus",
    "mbtcp",
    "server",
    "windows",
    "linux",
    "android",
    "beacon",
    "covenant",
    "libfuzzer",
    "mimikatz",
    "stuxnet",
)
GENERIC_REJECTS = {
    "if",
    "there",
    "when",
    "upon",
    "use",
    "using",
    "system",
    "network",
    "protein",
    "virus",
}


def bootstrap_nltk() -> None:
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    nltk.download("averaged_perceptron_tagger", quiet=True)
    nltk.download("averaged_perceptron_tagger_eng", quiet=True)
    nltk.download("stopwords", quiet=True)
    nltk.download("words", quiet=True)


def get_preferred_split(ds_dict) -> Dataset:
    if "train" in ds_dict:
        return ds_dict["train"]
    split_name = next(iter(ds_dict.keys()))
    return ds_dict[split_name]


def load_wmdp_dataset() -> Dataset:
    subset_datasets: List[Dataset] = []
    for subset in SUBSETS:
        print(f"Loading dataset subset: {subset}")
        ds_dict = load_dataset("cais/wmdp", subset)
        subset_datasets.append(get_preferred_split(ds_dict))

    merged = concatenate_datasets(subset_datasets)
    print(f"Merged dataset size: {len(merged)} rows")
    return merged


def iter_target_texts(row: Dict) -> Iterable[str]:
    question = row.get("question")
    if question:
        yield str(question)

    choices = row.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            if choice is not None:
                yield str(choice)


def normalize_term(term: str) -> str:
    term = re.sub(r"\s+", " ", term.strip())
    return term.strip(".,;:!?()[]{}\"'`")


def contains_hex_or_memory_noise(text: str) -> bool:
    if HEX_PREFIX_RE.search(text):
        return True

    compact = text.replace(" ", "")
    if LONG_HEX_RE.fullmatch(compact):
        return True
    if MEMORY_ADDRISH_RE.fullmatch(compact):
        return True
    return False


def infer_domain(term: str) -> str:
    low = term.lower()
    if any(k in low for k in BIO_HINTS):
        return "bio"
    if any(k in low for k in CYBER_HINTS):
        return "cyber"
    if re.fullmatch(r"[A-Z][a-z]+\s+[a-z]{2,}", term):
        return "bio"
    return "unknown"


def build_english_vocab() -> set[str]:
    vocab = set(w.lower() for w in words.words())
    # Add common lexical words that can slip through due to capitalization.
    vocab |= set(stopwords.words("english"))
    return vocab


def stitch_technical_tokens(raw_tokens: List[str]) -> List[str]:
    stitched: List[str] = []
    i = 0
    while i < len(raw_tokens):
        tok = raw_tokens[i]
        if tok == "-" and stitched and i + 1 < len(raw_tokens):
            right = raw_tokens[i + 1]
            if re.fullmatch(r"[A-Za-z0-9]+", right) and re.search(r"[A-Za-z0-9]$", stitched[-1]):
                stitched[-1] = stitched[-1] + "-" + right
                i += 2
                continue

        if stitched:
            prev = stitched[-1]
            if re.fullmatch(r"[A-Za-z]+", prev) and re.fullmatch(r"[A-Z][a-z]+", tok):
                if re.fullmatch(r"[A-Z]{2,}$", prev):
                    stitched[-1] = prev + tok
                    i += 1
                    continue

        stitched.append(tok)
        i += 1

    return stitched


def extract_ood_proper_nouns(text: str, english_vocab: set[str]) -> List[str]:
    raw_tokens = nltk.word_tokenize(text)
    tokens = stitch_technical_tokens(raw_tokens)
    tagged = nltk.pos_tag(tokens)

    out: List[str] = []
    for i in range(len(tagged)):
        for n in (1, 2, 3):
            if i + n > len(tagged):
                break

            span = tagged[i : i + n]
            words = [w for w, _ in span]
            tags = [t for _, t in span]

            if not all(t in PROPER_TAGS for t in tags):
                continue
            if any(not VALID_TOKEN_RE.fullmatch(w) for w in words):
                continue

            phrase = normalize_term(" ".join(words))
            if not phrase:
                continue
            if contains_hex_or_memory_noise(phrase):
                continue
            if len(phrase) <= 2:
                continue

            phrase_low = phrase.lower()
            if phrase_low in english_vocab:
                continue
            if phrase_low in GENERIC_REJECTS:
                continue
            if len(words) == 1 and UPPER_ADJECTIVE_RE.fullmatch(words[0]):
                continue

            # OOD gate: remove plain lexical terms when all words are in vocab.
            word_lows = [w.lower() for w in words]
            if all(w in english_vocab for w in word_lows):
                continue

            domain = infer_domain(phrase)
            if domain == "unknown":
                continue

            out.append(phrase)

    return out


def auto_map_term(term: str) -> str:
    low = term.lower()
    domain = infer_domain(term)

    if re.fullmatch(r"[A-Z][a-z]+\s+[a-z]{2,}", term):
        return "a bacterium"
    if re.search(r"\b(?:cov|virus|phage)\b", low) or low.endswith("virus") or low.endswith("viruses"):
        return "a virus"
    if any(k in low for k in ("toxin", "venom")):
        return "a toxic substance"
    if any(k in low for k in ("bacteria", "coccus", "杆菌")):
        return "a bacterium"
    if re.search(r"(protein|enzyme|[a-z]+ase)\b", low):
        return "a protein"
    if re.search(r"\b(?:dna|rna|cdna|rnas)\b", low):
        return "a genetic sequence"

    if domain == "cyber":
        if any(k in low for k in ("malware", "trojan", "worm", "ransom")):
            return "a malware variant"
        return "a software tool"

    if domain == "bio":
        return "a biological compound"

    return ""


def sanitize_mapping(raw_mapping: Dict[str, str], english_vocab: set[str]) -> Dict[str, str]:
    cleaned = dict(raw_mapping)
    changed = True
    while changed:
        changed = False
        for key in list(cleaned.keys()):
            norm = normalize_term(key)
            if norm != key:
                val = cleaned.pop(key)
                if norm:
                    cleaned[norm] = val
                changed = True
                continue

            if len(norm) <= 2:
                cleaned.pop(key, None)
                changed = True
                continue
            if contains_hex_or_memory_noise(norm):
                cleaned.pop(key, None)
                changed = True
                continue

            low = norm.lower()
            if low in english_vocab or low in GENERIC_REJECTS:
                cleaned.pop(key, None)
                changed = True
                continue

            toks = nltk.word_tokenize(norm)
            if not toks:
                cleaned.pop(key, None)
                changed = True
                continue
            if any(not VALID_TOKEN_RE.fullmatch(t) for t in toks):
                cleaned.pop(key, None)
                changed = True
                continue
            if len(toks) == 1 and UPPER_ADJECTIVE_RE.fullmatch(toks[0]):
                cleaned.pop(key, None)
                changed = True
                continue

            tags = [t for _, t in nltk.pos_tag(toks)]
            if not all(t in PROPER_TAGS for t in tags):
                cleaned.pop(key, None)
                changed = True
                continue

            if all(t.lower() in english_vocab for t in toks):
                cleaned.pop(key, None)
                changed = True
                continue

            if infer_domain(norm) == "unknown":
                cleaned.pop(key, None)
                changed = True
                continue

            if not cleaned.get(key):
                cleaned.pop(key, None)
                changed = True

    return cleaned


def run_validation(mapping: Dict[str, str], english_vocab: set[str]) -> tuple[bool, List[str]]:
    issues: List[str] = []

    for key in mapping.keys():
        low = key.lower()
        compact = key.replace(" ", "")

        if low in english_vocab or low in {"if", "network", "protein", "system"}:
            issues.append(f"common-word:{key}")
            continue
        if HEX_PREFIX_RE.search(key) or LONG_HEX_RE.fullmatch(compact) or MEMORY_ADDRISH_RE.fullmatch(compact):
            issues.append(f"hex-or-code:{key}")
            continue
        if len(key.split()) == 1 and UPPER_ADJECTIVE_RE.fullmatch(key):
            issues.append(f"upper-adj:{key}")
            continue

    top20 = list(mapping.keys())[:20]
    specific_hits = 0
    for t in top20:
        low = t.lower()
        if any(k in low for k in (
            "spycep",
            "sars-cov-2",
            "streptococcus",
            "stuxnet",
            "mimikatz",
            "metasploit",
            "meterpreter",
            "cobalt strike",
            "h5n1",
            "hiv-1",
        )):
            specific_hits += 1

    if specific_hits < 3:
        issues.append(f"domain-specificity-too-low:{specific_hits}")

    return (len(issues) == 0, issues)


def main() -> None:
    print("Running OOV proper noun extraction...")
    bootstrap_nltk()

    english_vocab = build_english_vocab()
    ds = load_wmdp_dataset()

    counts: Counter = Counter()
    for row in ds:
        for text in iter_target_texts(row):
            counts.update(extract_ood_proper_nouns(text, english_vocab))

    # Use wider candidate pool before sanitization.
    top_terms = counts.most_common(TOP_K * 8)
    raw_mapping = {term: auto_map_term(term) for term, _ in top_terms}
    sanitized = sanitize_mapping(raw_mapping, english_vocab)

    final_items = sorted(
        ((term, counts[term]) for term in sanitized),
        key=lambda x: x[1],
        reverse=True,
    )[:TOP_K]
    final_mapping = {term: sanitized[term] for term, _ in final_items}

    ok, issues = run_validation(final_mapping, english_vocab)
    if not ok:
        print("Validation issues detected:")
        for issue in issues[:20]:
            print(f"- {issue}")
        raise RuntimeError("Validation checklist failed. Tighten extraction constraints.")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(final_mapping, f, indent=2, ensure_ascii=True)

    print(f"Saved {len(final_mapping)} mappings to {OUTPUT_FILE}")
    print("Top 15 mappings:")
    for term, _ in final_items[:15]:
        print(f"- {term}: {final_mapping[term]}")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"RUNTIME ERROR: {exc}")
        sys.exit(1)
    except Exception as exc:  # pragma: no cover
        print(f"UNEXPECTED ERROR: {exc}")
        traceback.print_exc()
        sys.exit(1)
