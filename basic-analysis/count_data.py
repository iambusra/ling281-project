import os
import json
import re
import pandas as pd

# Automatically use the folder where this script lives
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text):
    if not text:
        return []
    return TOKEN_RE.findall(text)


def process_file(filepath):
    """
    Process one JSON file and return:
    - title string
    - number of entries
    - total tokens in entries
    - total tokens in title
    - title_date (taken from first entry's 'date' field, if present)
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    entries = data.get("entries", [])

    # ----- title -----
    scrape_info = data.get("scrape_info", {})
    title = scrape_info.get("input")

    # fallback to first entry's title if needed
    if not title and entries:
        title = entries[0].get("title", "")
    title = title if title else ""

    # ----- title date -----
    # we define "title date" as the date of the first entry
    if entries:
        title_date = entries[0].get("date", "")
    else:
        title_date = ""

    # ----- token counts -----
    title_tokens = len(tokenize(title))

    entry_token_count = 0
    for e in entries:
        entry_token_count += len(tokenize(e.get("content", "")))

    return title, len(entries), entry_token_count, title_tokens, title_date


def main():
    rows = []  # for per-title dataframe

    # --------- WALK THROUGH CATEGORIES ----------
    for folder in os.listdir(BASE_DIR):
        category_path = os.path.join(BASE_DIR, folder)
        if not os.path.isdir(category_path):
            continue
        if not folder.endswith("_titles_data"):
            continue

        # category names: rus, ukrayna, suriye, gocmen, multeci, siginmaci, ...
        category = folder.replace("_titles_data", "")
        print(f"Processing category: {category}")

        for filename in os.listdir(category_path):
            if not filename.endswith(".json"):
                continue

            filepath = os.path.join(category_path, filename)

            try:
                title, n_entries, entry_tokens, title_tokens, title_date = process_file(
                    filepath
                )
            except Exception as e:
                print(f"Error reading {filepath}: {e}")
                continue

            rows.append(
                {
                    "title": title,
                    "category": category,
                    "date": title_date,
                    "num_entries": n_entries,
                    "title_token_count": title_tokens,
                    "entry_token_count": entry_tokens,
                }
            )

    # --------- BUILD PER-TITLE DATAFRAME ----------
    df = pd.DataFrame(rows)

    summary_csv_path = os.path.join(BASE_DIR, "eksi_summary.csv")
    df.to_csv(summary_csv_path, index=False, encoding="utf-8")
    print("\nSaved per-title summary CSV to:", summary_csv_path)

    if df.empty:
        print("No data found; stats summary will not be created.")
        return

    # --------- COMPUTE SUMMARY STATS ----------

    # Overall totals
    total_titles = len(df)  # one row per title
    total_entries = df["num_entries"].sum()
    total_title_tokens = df["title_token_count"].sum()
    total_entry_tokens = df["entry_token_count"].sum()
    total_tokens = total_title_tokens + total_entry_tokens

    overall_row = {
        "level": "overall",
        "category": "ALL",
        "num_titles": total_titles,
        "num_entries": total_entries,
        "title_token_count": total_title_tokens,
        "entry_token_count": total_entry_tokens,
        "total_tokens": total_tokens,
    }

    # Per-category totals (rus, ukrayna, suriye, gocmen, multeci, siginmaci, ...)
    grouped = (
        df.groupby("category")
        .agg(
            num_titles=("title", "nunique"),
            num_entries=("num_entries", "sum"),
            title_token_count=("title_token_count", "sum"),
            entry_token_count=("entry_token_count", "sum"),
        )
        .reset_index()
    )

    grouped["total_tokens"] = (
        grouped["title_token_count"] + grouped["entry_token_count"]
    )
    grouped["level"] = "category"

    grouped = grouped[
        [
            "level",
            "category",
            "num_titles",
            "num_entries",
            "title_token_count",
            "entry_token_count",
            "total_tokens",
        ]
    ]

    grouped = grouped.sort_values("category")

    summary_df = pd.concat([pd.DataFrame([overall_row]), grouped], ignore_index=True)

    stats_csv_path = os.path.join(BASE_DIR, "eksi_stats_summary.csv")
    summary_df.to_csv(stats_csv_path, index=False, encoding="utf-8")

    print("Saved stats summary CSV to:", stats_csv_path)
    print("\nStats summary preview:")
    print(summary_df)


if __name__ == "__main__":
    main()
