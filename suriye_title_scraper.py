import subprocess
import os
import re
import time
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- CONFIG ---
TITLES_FILE = "suriye-titles.txt"   # one title per line
OUTPUT_DIR = "suriye_titles_data"
FAILED_FILE = os.path.join(OUTPUT_DIR, "failed_titles.txt")

# performance tuning
MAX_WORKERS = 5       # number of concurrent scrapers
SCRAPER_TIMEOUT = 120 # seconds per title
SLEEP_BETWEEN_BATCHES = 1  # small pause between completed threads

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_titles(path):
    """Read all titles from file, skip blanks and duplicates."""
    with open(path, "r", encoding="utf-16") as f:
        lines = [line.strip() for line in f if line.strip()]
    seen, titles = set(), []
    for t in lines:
        if t not in seen:
            seen.add(t)
            titles.append(t)
    return titles


def scrape_title(title, retries=3):
    """Scrape Ekşi Sözlük entries for a title with retries."""
    safe_title = re.sub(r"[^a-zA-Z0-9ğüşöçıİĞÜŞÖÇ]+", "_", title).strip("_")
    output_file = os.path.join(OUTPUT_DIR, f"{safe_title}.json")

    if os.path.exists(output_file):
        return f"⏩ Skipped (exists): {title}"

    for attempt in range(1, retries + 1):
        try:
            result = subprocess.run(
                [
                    "eksisozluk-scraper",
                    title,
                    "--output", output_file,
                    "--delay", "1",
                    "--max-retries", "3",
                    "--retry-delay", "5"
                ],
                capture_output=True,
                text=True,
                timeout=SCRAPER_TIMEOUT
            )

            if result.returncode == 0 and os.path.exists(output_file):
                return f"✅ Success: {title}"

            err = result.stderr.strip()[:200] if result.stderr else "unknown error"
            time.sleep(1)

        except subprocess.TimeoutExpired:
            err = "Timeout"
        except Exception as e:
            err = str(e)

        # brief pause between retries
        time.sleep(3)

    # log failed titles
    with open(FAILED_FILE, "a", encoding="utf-8") as f:
        f.write(title + "\n")

    return f"🚫 Failed after {retries} attempts: {title} ({err})"


def main():
    titles = load_titles(TITLES_FILE)
    print(f"📄 Loaded {len(titles)} titles")
    print(f"📂 Output directory: {OUTPUT_DIR}")
    print(f"⚙️  Running with {MAX_WORKERS} concurrent scrapers\n")

    success = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_title = {executor.submit(scrape_title, t): t for t in titles}

        for i, future in enumerate(as_completed(future_to_title), start=1):
            title = future_to_title[future]
            try:
                msg = future.result()
                print(f"[{i}/{len(titles)}] {msg}")
                if msg.startswith("✅"):
                    success += 1
            except Exception as e:
                print(f"[{i}/{len(titles)}] ❌ Error on {title}: {e}")
            sys.stdout.flush()
            time.sleep(SLEEP_BETWEEN_BATCHES)

    print(f"\n✅ Finished scraping. {success}/{len(titles)} succeeded.")
    print(f"❗ Failed titles (if any) saved to: {FAILED_FILE}")


if __name__ == "__main__":
    main()
