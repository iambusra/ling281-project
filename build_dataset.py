#!/usr/bin/env python3
"""
Build a HuggingFace Dataset from all Ekşi Sözlük JSON data.

One row per entry. Uses annotated data (with sentiment/categories) where
available, raw data otherwise.

Output: ./hf_dataset/  (Arrow-backed HF Dataset on disk)
"""

import json
import os
import re
from glob import glob
from datetime import datetime

import pandas as pd
from datasets import Dataset, Features, Value, Sequence

BASE = os.path.dirname(os.path.abspath(__file__))

# ── Data sources ──────────────────────────────────────────────────────
# For suriye/rus/ukrayna, prefer annotated versions (they include all
# raw fields plus sentiment_numeric and categories).
# For the rest, use raw data.
SOURCES = {
    # category_label: (glob_pattern, is_annotated)
    "suriye": (os.path.join(BASE, "annotated_data", "suriye_annotated", "*.json"), True),
    "rus": (os.path.join(BASE, "annotated_data", "rus_annotated", "*.json"), True),
    "ukrayna": (os.path.join(BASE, "annotated_data", "ukrayna_annotated", "*.json"), True),
    "afganistan": (os.path.join(BASE, "afganistan_titles_data", "*.json"), False),
    "pakistan": (os.path.join(BASE, "pakistan_titles_data", "*.json"), False),
    "gocmen": (os.path.join(BASE, "gocmen_titles_data", "gocmen_kw", "*.json"), False),
    "multeci": (os.path.join(BASE, "gocmen_titles_data", "multeci_kw", "*.json"), False),
    "siginmaci": (os.path.join(BASE, "gocmen_titles_data", "siginmaci_kw", "*.json"), False),
}

# Date pattern: "DD.MM.YYYY HH:MM" with optional " ~ HH:MM" edit suffix
DATE_RE = re.compile(r"(\d{2})\.(\d{2})\.(\d{4})\s+(\d{2}):(\d{2})")


def parse_date(raw: str) -> str | None:
    """Parse Ekşi date string into ISO format. Returns None on failure."""
    if not raw:
        return None
    m = DATE_RE.search(raw)
    if not m:
        return None
    day, month, year, hour, minute = m.groups()
    try:
        dt = datetime(int(year), int(month), int(day), int(hour), int(minute))
        return dt.isoformat()
    except ValueError:
        return None


def load_entries(pattern: str, category: str, annotated: bool) -> list[dict]:
    rows = []
    for fpath in sorted(glob(pattern)):
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"  SKIP {fpath}: {e}")
            continue

        title_from_scrape = data.get("scrape_info", {}).get("input", "")

        for entry in data.get("entries", []):
            row = {
                "title": entry.get("title", title_from_scrape),
                "entry_id": str(entry.get("entry_id", "")),
                "author": entry.get("author", ""),
                "date_raw": entry.get("date", ""),
                "date": parse_date(entry.get("date", "")),
                "content": entry.get("content", ""),
                "category": category,
                "has_external_url": bool(entry.get("has_external_url", False)),
                "sentiment": entry.get("sentiment_numeric") if annotated else None,
                "topics": entry.get("categories", []) if annotated else [],
            }
            rows.append(row)
    return rows


def main():
    all_rows = []
    for category, (pattern, annotated) in SOURCES.items():
        print(f"Loading {category} (annotated={annotated}) ...")
        rows = load_entries(pattern, category, annotated)
        print(f"  -> {len(rows):,} entries")
        all_rows.extend(rows)

    print(f"\nTotal entries: {len(all_rows):,}")

    # Build HF Dataset
    features = Features({
        "title": Value("string"),
        "entry_id": Value("string"),
        "author": Value("string"),
        "date_raw": Value("string"),
        "date": Value("string"),
        "content": Value("string"),
        "category": Value("string"),
        "has_external_url": Value("bool"),
        "sentiment": Value("int8"),
        "topics": Sequence(Value("string")),
    })

    # Convert to columnar format for Dataset.from_dict
    columns = {k: [] for k in features}
    for row in all_rows:
        for k in features:
            val = row[k]
            # int8 nulls: use -1 as sentinel for missing sentiment
            if k == "sentiment" and val is None:
                val = -1
            columns[k].append(val)

    ds = Dataset.from_dict(columns, features=features)

    out_path = os.path.join(BASE, "hf_dataset")
    ds.save_to_disk(out_path)
    print(f"\nDataset saved to {out_path}")
    print(ds)
    print("\nColumn summary:")
    print(f"  Categories: {sorted(ds.unique('category'))}")
    print(f"  Entries with sentiment: {sum(1 for s in ds['sentiment'] if s != -1):,}")
    print(f"  Entries without sentiment: {sum(1 for s in ds['sentiment'] if s == -1):,}")


if __name__ == "__main__":
    main()
