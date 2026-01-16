#!/usr/bin/env python3
import json
from pathlib import Path
import argparse
import traceback
import csv


def load_json(path: Path):
    """Load JSON as UTF-8."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path: Path):
    """Save JSON as UTF-8, preserving Turkish characters."""
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def strip_annotation_suffix(stem: str) -> str:
    for suffix in ("_annotations", "_annotated"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def build_annotation_map(annotations_list):
    ann_map = {}
    for ann in annotations_list:
        eid = ann.get("entry_id")
        if eid is None:
            continue
        eid = str(eid)
        ann_map[eid] = ann
    return ann_map


def merge_one_pair(ann_file: Path, data_dir: Path, out_dir: Path, error_writer):
    """
    Merge a single annotation JSON with its data JSON.
    Log any errors to CSV.
    """
    ann_stem = ann_file.stem
    base = strip_annotation_suffix(ann_stem)

    data_file = data_dir / f"{base}.json"
    out_file = out_dir / f"{base}_merged.json"

    # If data file missing → log error
    if not data_file.exists():
        error_writer.writerow([
            ann_file.name,
            f"{base}.json",
            "MissingDataFile",
            f"Data file not found for {ann_file.name}",
        ])
        print(f"[WARN] No data file for {ann_file.name}")
        return

    try:
        annotations_list = load_json(ann_file)
    except Exception as e:
        error_writer.writerow([
            ann_file.name,
            data_file.name,
            "InvalidAnnotationJSON",
            repr(e)
        ])
        print(f"[ERROR] Cannot parse annotation JSON {ann_file.name}")
        return

    try:
        data = load_json(data_file)
    except Exception as e:
        error_writer.writerow([
            ann_file.name,
            data_file.name,
            "InvalidDataJSON",
            repr(e)
        ])
        print(f"[ERROR] Cannot parse data JSON {data_file.name}")
        return

    # Validate annotation list
    if not isinstance(annotations_list, list):
        error_writer.writerow([
            ann_file.name,
            data_file.name,
            "AnnotationsNotList",
            "Annotation file is not a list"
        ])
        print(f"[WARN] Annotation file {ann_file.name} is not a list")
        return

    entries = data.get("entries", [])
    if not isinstance(entries, list):
        error_writer.writerow([
            ann_file.name,
            data_file.name,
            "EntriesNotList",
            "entries field is not a list"
        ])
        print(f"[WARN] entries not list in {data_file.name}")
        return

    ann_map = build_annotation_map(annotations_list)

    merged = 0
    missing = 0

    for entry in entries:
        eid = entry.get("entry_id")
        if eid is None:
            error_writer.writerow([
                ann_file.name,
                data_file.name,
                "MissingEntryID",
                "Entry without entry_id"
            ])
            continue

        eid = str(eid)
        ann = ann_map.get(eid)

        if ann is None:
            missing += 1
            continue

        for k, v in ann.items():
            if k == "entry_id":
                continue
            entry[k] = v
        merged += 1

    save_json(data, out_file)

    print(f"[OK] {ann_file.name} + {data_file.name} -> merged={merged}, missing={missing}")


def main():
    parser = argparse.ArgumentParser(description="Merge Ekşi annotations into JSON files.")
    parser.add_argument("data_dir", help="Path to suriye_titles_data")
    parser.add_argument("ann_dir", help="Path to suriye_annotated")
    parser.add_argument("out_dir", help="Output directory")

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    ann_dir = Path(args.ann_dir)
    out_dir = Path(args.out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    # Prepare CSV error log
    error_csv_path = out_dir / "merge_errors.csv"
    error_csv_file = error_csv_path.open("w", encoding="utf-8", newline="")
    error_writer = csv.writer(error_csv_file)
    error_writer.writerow(["annotation_file", "data_file", "error_type", "error_message"])

    ann_files = sorted(ann_dir.glob("*.json"))

    print(f"Found {len(ann_files)} annotation files.\n")

    for ann_file in ann_files:
        try:
            merge_one_pair(ann_file, data_dir, out_dir, error_writer)
        except Exception as e:
            print(f"[CRASH] Unexpected exception while processing {ann_file.name}")
            error_writer.writerow([
                ann_file.name,
                "",
                "UnexpectedError",
                repr(e) + "\n" + traceback.format_exc()
            ])

    error_csv_file.close()

    print("\nDone. Errors logged to:", error_csv_path)


if __name__ == "__main__":
    main()
