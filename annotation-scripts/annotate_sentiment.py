import os
import json
import glob
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

# ------------------------------
# CONFIG
# ------------------------------

# Use the nano model name your account supports. If this fails, try "gpt-5-nano" or check your dashboard.
MODEL_NAME = "gpt-5-nano-2025-08-07"

# Directories
INPUT_DIR = "input_json"         # folder with your Ekşi JSON files
OUTPUT_DIR = "annotations_json"  # folder where we write annotation-only JSON files

# Batching & parallelism
BATCH_SIZE = 40          # entries per API call (increase if entries are short)
MAX_WORKERS = 4          # number of files to process in parallel

os.makedirs(OUTPUT_DIR, exist_ok=True)

client = OpenAI()

# 12-category label set (adjust names if you want, but be consistent downstream)
CATEGORY_LABELS = [
    "HATE",
    "XENO",
    "CRIME",
    "ECON",
    "CULTURE",
    "SEXUAL",
    "GENDER-MORAL",
    "POLICY",
    "NEUTRAL",
    "HUMAN",
    "POS-STEREO",
    "HUMAN-POS",
]

# We only send codes in the user prompt, not long descriptions, to save tokens.
CATEGORY_CODES_STRING = ", ".join(CATEGORY_LABELS)

# ------------------------------
# PROMPTS
# ------------------------------

SYSTEM_PROMPT = (
    "You annotate Turkish social media posts about immigrants and refugees.\n"
    "For each entry you must output:\n"
    "  - sentiment_numeric: integer 1–5\n"
    "    1 = very negative, 2 = negative, 3 = neutral, 4 = positive, 5 = very positive.\n"
    "  - categories: 1–3 codes from a fixed label set.\n"
    "\n"
    "Allowed category codes:\n"
    f"  {CATEGORY_CODES_STRING}\n"
    "\n"
    "Output format (JSON ONLY):\n"
    "{ \"labels\": [\n"
    "  {\"entry_id\": \"...\", \"sentiment_numeric\": N, \"categories\": [\"CODE1\", \"CODE2\"]},\n"
    "  ...\n"
    "] }\n"
    "\n"
    "Important constraints:\n"
    "  - DO NOT copy, quote, or paraphrase the original entry text.\n"
    "  - DO NOT include author names, dates, titles, or any other metadata.\n"
    "  - DO NOT add explanations or comments.\n"
    "  - Each entry_id in the output must match exactly one entry_id from the input.\n"
    "  - Categories must be 1–3 items, each from the allowed label set.\n"
)


def build_user_prompt(batch_items: List[Dict[str, Any]]) -> str:
    """
    Build a compact user prompt with only:
    - the allowed codes list (once)
    - and the (entry_id, text) pairs.
    """
    parts: List[str] = []
    parts.append("You will now label the following entries.\n")
    parts.append("Allowed category codes:\n")
    parts.append(CATEGORY_CODES_STRING)
    parts.append(
        "\n\nFor each entry, return EXACTLY one object in the JSON array 'labels' with:\n"
        "  entry_id, sentiment_numeric, categories.\n"
        "DO NOT repeat the entry text in the JSON.\n"
    )

    for item in batch_items:
        eid = str(item["entry_id"])
        text = item["content"]
        parts.append(f"\nENTRY_ID: {eid}\nTEXT: {text}")

    return "\n".join(parts)


# ------------------------------
# MODEL CALL
# ------------------------------

def call_model_for_batch_chat(batch_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Call the model in JSON mode for a batch of entries.
    Returns a list of label dicts:
      { "entry_id": str, "sentiment_numeric": int, "categories": [str, ...] }
    """
    user_prompt = build_user_prompt(batch_items)

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
    )

    content = completion.choices[0].message.content

    # Parse JSON
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        # Fallback: try to extract the first {...} block
        first = content.find("{")
        last = content.rfind("}")
        if first == -1 or last == -1 or last <= first:
            raise RuntimeError(f"Model output is not valid JSON and no braces found:\n{content}")
        snippet = content[first:last + 1]
        parsed = json.loads(snippet)

    if "labels" not in parsed or not isinstance(parsed["labels"], list):
        raise RuntimeError(
            "JSON does not contain 'labels' array:\n"
            + json.dumps(parsed, ensure_ascii=False, indent=2)
        )

    result_labels: List[Dict[str, Any]] = []

    for obj in parsed["labels"]:
        # Extract and normalize fields
        entry_id = str(obj.get("entry_id", "")).strip()
        if not entry_id:
            continue

        # sentiment_numeric defaults to 3 (neutral) if missing
        try:
            sentiment = int(obj.get("sentiment_numeric", 3))
        except (ValueError, TypeError):
            sentiment = 3
        if sentiment < 1 or sentiment > 5:
            sentiment = 3

        cats = obj.get("categories", [])
        if not isinstance(cats, list):
            cats = [cats]
        cats = [str(c).strip() for c in cats if str(c).strip()]
        # Keep only valid codes and at most 3
        cats = [c for c in cats if c in CATEGORY_LABELS][:3]
        if not cats:
            cats = ["NEUTRAL"]

        result_labels.append(
            {
                "entry_id": entry_id,
                "sentiment_numeric": sentiment,
                "categories": cats,
            }
        )

    if not result_labels:
        raise RuntimeError(f"Empty labels list from model. Raw content:\n{content}")

    return result_labels


# ------------------------------
# I/O HELPERS
# ------------------------------

def load_json_file(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json_file(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ------------------------------
# ANNOTATION PER FILE
# ------------------------------

def annotate_file(in_path: str, out_path: str) -> None:
    """
    Read one Ekşi-style JSON file:
      { "scrape_info": {...}, "entries": [ {entry_id, content, ...}, ... ] }
    Call the model in batches and write an **annotation-only** file:
      [
        {"entry_id": "...", "sentiment_numeric": 2, "categories": ["XENO"]},
        ...
      ]
    """
    raw = load_json_file(in_path)

    # Unwrap Ekşi structure
    if isinstance(raw, dict) and "entries" in raw:
        entries = raw["entries"]
    elif isinstance(raw, list):
        entries = raw
    else:
        raise RuntimeError(
            f"Unsupported JSON structure in {in_path}. "
            f"Expected dict with 'entries' or a list; got {type(raw)}"
        )

    id_to_text: Dict[str, str] = {}
    for e in entries:
        if not isinstance(e, dict):
            continue
        eid = str(e.get("entry_id") or e.get("id") or "").strip()
        text = e.get("content", "")
        if not eid:
            continue
        id_to_text[eid] = text

    entry_ids = list(id_to_text.keys())
    if not entry_ids:
        print(f"[WARN] No valid entries in {os.path.basename(in_path)}; skipping.")
        save_json_file(out_path, [])
        return

    print(f"Processing file: {os.path.basename(in_path)} "
          f"({len(entry_ids)} entries)")

    all_labels: List[Dict[str, Any]] = []

    for start in range(0, len(entry_ids), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(entry_ids))
        batch_ids = entry_ids[start:end]
        batch_items = [{"entry_id": eid, "content": id_to_text[eid]} for eid in batch_ids]

        print(f"  - Annotating batch {start}–{end - 1} "
              f"({len(batch_items)} entries)")

        labels = call_model_for_batch_chat(batch_items)
        # You could sanity-check that every batch_id appears exactly once
        all_labels.extend(labels)

    save_json_file(out_path, all_labels)
    print(f"  -> Saved annotations to {out_path}\n")


def process_one_file(in_path: str) -> None:
    base = os.path.basename(in_path)
    name, _ = os.path.splitext(base)
    out_path = os.path.join(OUTPUT_DIR, f"{name}_annotations.json")
    annotate_file(in_path, out_path)


# ------------------------------
# MAIN
# ------------------------------

def main() -> None:
    json_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.json")))
    if not json_files:
        print(f"No JSON files found in {INPUT_DIR}")
        return

    print(f"Found {len(json_files)} JSON files in {INPUT_DIR}")

    max_workers = min(MAX_WORKERS, len(json_files))
    if max_workers <= 1:
        for in_path in json_files:
            process_one_file(in_path)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_path = {
                executor.submit(process_one_file, in_path): in_path
                for in_path in json_files
            }
            for fut in as_completed(future_to_path):
                in_path = future_to_path[fut]
                try:
                    fut.result()
                except Exception as e:
                    print(f"[ERROR] While processing {os.path.basename(in_path)}: {e}")


if __name__ == "__main__":
    main()
