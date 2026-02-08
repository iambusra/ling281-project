import subprocess
import re
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- CONFIG ---
TITLES_FILE = "afganistan-titles.txt"     # one title per line
OUTPUT_DIR = "afganistan_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# speed knobs
MAX_WORKERS = 8          # try 3–8
SCRAPER_DELAY = "0.1"    # seconds between requests inside each worker
MAX_RETRIES = "5"
RETRY_DELAY = "5"

# ---------- helpers ----------

def read_titles_from_file(path: str) -> list[str]:
    """
    Read titles from a text file (one title per line).
    Tries multiple encodings (utf-8, utf-16, latin-1).
    Ignores empty lines and comment lines starting with '#'.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"titles file not found: {path}")

    encodings_to_try = ["utf-8", "utf-16", "latin-1"]

    last_error = None
    for enc in encodings_to_try:
        try:
            titles = []
            with open(path, "r", encoding=enc) as f:
                for line in f:
                    t = line.strip()
                    if not t or t.startswith("#"):
                        continue
                    titles.append(t)
            print(f"📄 Loaded titles.txt using encoding: {enc}")
            return titles
        except UnicodeDecodeError as e:
            last_error = e

    raise UnicodeDecodeError(
        "Unable to decode titles file with utf-8 / utf-16 / latin-1",
        b"",
        0,
        1,
        str(last_error),
    )



def safe_filename(title: str) -> str:
    """
    Create a filesystem-safe filename from the title.
    Also truncates to avoid overly long filenames on macOS.
    """
    safe = re.sub(r"[^a-zA-Z0-9ğüşöçıİĞÜŞÖÇ]+", "_", title).strip("_")
    if not safe:
        safe = "untitled"
    return safe[:180]


def scrape_title(title: str) -> tuple[str, str, int]:
    """
    Scrape a single title using eksisozluk_scraper module.
    Returns (title, status, exit_code).
    Status: ok | skipped | failed
    """
    output_file = os.path.join(OUTPUT_DIR, f"{safe_filename(title)}.json")

    if os.path.exists(output_file):
        return (title, "skipped", 0)

    cmd = [
        sys.executable, "-m", "eksisozluk_scraper",
        title,
        "--output", output_file,
        "--delay", SCRAPER_DELAY,
        "--max-retries", MAX_RETRIES,
        "--retry-delay", RETRY_DELAY,
    ]

    p = subprocess.run(cmd, capture_output=True, text=True)

    if p.returncode == 0:
        return (title, "ok", 0)

    # save debug info
    err_path = output_file + ".error.log"
    with open(err_path, "w", encoding="utf-8") as f:
        f.write("CMD:\n" + " ".join(cmd) + "\n\n")
        f.write("STDOUT:\n" + (p.stdout or "") + "\n\n")
        f.write("STDERR:\n" + (p.stderr or "") + "\n")
    return (title, "failed", p.returncode)


def main():
    # 1) read titles from titles.txt
    titles = read_titles_from_file(TITLES_FILE)

    # 2) dedupe while preserving order (case-insensitive)
    seen = set()
    titles_to_scrape = []
    for t in titles:
        k = t.casefold()
        if k not in seen:
            seen.add(k)
            titles_to_scrape.append(t)

    print(f"\n🧾 Titles loaded from {TITLES_FILE}: {len(titles)} (unique: {len(titles_to_scrape)})")
    if not titles_to_scrape:
        print("Nothing to scrape.")
        return

    print(f"⚙️ Scraping with {MAX_WORKERS} workers | scraper delay={SCRAPER_DELAY}s | retries={MAX_RETRIES}\n")

    ok = skipped = failed = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(scrape_title, t) for t in titles_to_scrape]
        for fut in as_completed(futures):
            title, status, code = fut.result()
            if status == "ok":
                ok += 1
                print(f"✅ {title}")
            elif status == "skipped":
                skipped += 1
                print(f"⏩ {title}")
            else:
                failed += 1
                print(f"❌ {title} (exit {code})")

    print(f"\nDone. ok={ok}, skipped={skipped}, failed={failed}")
    if failed:
        print("Some titles failed. See *.error.log files next to their output JSONs for details.")


if __name__ == "__main__":
    main()
