#!/usr/bin/env python3
"""Extract text from Harry Potter PDF(s), tokenize, and save as a HuggingFace Dataset."""

import re
import sys
from pathlib import Path

import fitz  # PyMuPDF
import torch
from datasets import Dataset
from transformers import AutoTokenizer

MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
CHUNK_SIZE = 512
OUTPUT_DIR = "./hp_corpus"


def extract_pdf_text(pdf_path: Path) -> str:
    pages = []
    with fitz.open(str(pdf_path)) as doc:
        for page in doc:
            text = page.get_text().strip()
            if text:
                pages.append(text)
    return "\n".join(pages)


def clean_text(raw: str) -> str:
    # Collapse whitespace, strip page headers / lone numbers
    text = re.sub(r'\n{3,}', '\n\n', raw)
    text = re.sub(r'(?m)^\s*\d+\s*$', '', text)   # bare page numbers
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


def chunk_tokens(text: str, tokenizer: AutoTokenizer) -> list[str]:
    """Tokenize the full text then cut into non-overlapping CHUNK_SIZE windows."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    for start in range(0, len(ids), CHUNK_SIZE):
        window = ids[start : start + CHUNK_SIZE]
        if len(window) < 32:   # drop tiny trailing fragments
            continue
        chunks.append(tokenizer.decode(window))
    return chunks


def main() -> None:
    pdfs = sorted(Path(".").glob("hp*.pdf"))
    if not pdfs:
        print("No hp*.pdf files found in the current directory.")
        sys.exit(1)

    print(f"Found PDF(s): {[p.name for p in pdfs]}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_chunks: list[str] = []
    for pdf in pdfs:
        print(f"Extracting: {pdf.name}")
        raw = extract_pdf_text(pdf)
        cleaned = clean_text(raw)
        chunks = chunk_tokens(cleaned, tokenizer)
        print(f"  → {len(chunks)} chunks")
        all_chunks.extend(chunks)

    print(f"Total chunks: {len(all_chunks)}")
    ds = Dataset.from_dict({"text": all_chunks})
    ds.save_to_disk(OUTPUT_DIR)
    print(f"Corpus saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
