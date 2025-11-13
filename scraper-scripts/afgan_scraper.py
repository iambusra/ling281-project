import subprocess
import requests
from bs4 import BeautifulSoup
import re
import os
import time

# --- CONFIG ---
KEYWORDS = ["afgan", "afganistan", "afganlı"]
OUTPUT_DIR = "afgan_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BASE_URL = "https://eksisozluk.com/basliklar/ara?SearchForm.Keywords={}&a=search"
HEADERS = {"User-Agent": "Mozilla/5.0"}

# match any morphological variant containing one of the roots
RE_KEYWORD = re.compile(r"(afganistan|afganlı|afgan)", re.IGNORECASE)

def find_titles(keyword):
    """Search Ekşi Sözlük for titles containing the keyword (anywhere)."""
    url = BASE_URL.format(keyword)
    res = requests.get(url, headers=HEADERS)
    res.raise_for_status()
    soup = BeautifulSoup(res.text, "html.parser")
    titles = []
    for a in soup.select("ul.topic-list.partial > li > a"):
        title = a.text.strip()
        href = a.get("href")
        if href and RE_KEYWORD.search(title):  # anywhere in the title
            full_url = "https://eksisozluk.com" + href
            titles.append((title, full_url))
    return titles

def scrape_title(title):
    """Run eksisozluk-scraper CLI for a specific title."""
    safe_title = re.sub(r'[^a-zA-Z0-9ğüşöçıİĞÜŞÖÇ]+', '_', title).strip('_')
    output_file = os.path.join(OUTPUT_DIR, f"{safe_title}.json")

    if os.path.exists(output_file):
        print(f"⏩ Skipping (already scraped): {title}")
        return

    print(f"📘 Scraping: {title}")
    subprocess.run([
        "eksisozluk-scraper",
        title,
        "--output", output_file,
        "--delay", "2",
        "--max-retries", "5",
        "--retry-delay", "10"
    ], check=False)
    time.sleep(2)

def main():
    all_titles = set()
    for kw in KEYWORDS:
        print(f"\n🔍 Searching for titles containing '{kw}'...")
        titles = find_titles(kw)
        for title, _ in titles:
            title_lower = title.lower()
            if title_lower not in all_titles:
                all_titles.add(title_lower)
                scrape_title(title)
    print("\n✅ Finished scraping all titles containing the target stems!")

if __name__ == "__main__":
    main()
