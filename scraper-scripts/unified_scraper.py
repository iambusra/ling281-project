import subprocess
import requests
from bs4 import BeautifulSoup
import re
import os
import time

# ================================================================
# CONFIGURATION
# ================================================================

# dictionary: { folder_name : regex pattern capturing all variants }
STEMS = {
    "suriye":   r"suri[a-zçğıöşü]*",          # suriye, suriyeli, suriyeliler, ...
    "afgan":    r"afgan[a-zçğıöşü]*",         # afgan, afganlı, afganistanlı, ...
    "gocmen":   r"(göçmen|mülteci|sığınmacı)[a-zçğıöşü]*",
    "rus":      r"rus[a-zçğıöşü]*",           # rus, ruslar, rusya, rusyalı, ...
    "ukrayna":  r"ukrayna[a-zçğıöşü]*"         # ukrayna, ukraynalı, ukraynalılar, ...
}

BASE_URL = "https://eksisozluk.com/basliklar/ara?SearchForm.Keywords={}&a=search"
HEADERS = {"User-Agent": "Mozilla/5.0"}

DELAY_BETWEEN_REQUESTS = 2
MAX_RETRIES = 5
RETRY_DELAY = 10
# ================================================================


def find_titles(keyword, regex_pattern):
    """Search Ekşi Sözlük for titles containing the keyword or its variants."""
    url = BASE_URL.format(keyword)
    res = requests.get(url, headers=HEADERS)
    res.raise_for_status()
    soup = BeautifulSoup(res.text, "html.parser")

    RE_KEYWORD = re.compile(regex_pattern, re.IGNORECASE)
    titles = []

    for a in soup.select("ul.topic-list.partial > li > a"):
        title = a.text.strip()
        href = a.get("href")
        if href and RE_KEYWORD.search(title):
            full_url = "https://eksisozluk.com" + href
            titles.append((title, full_url))
    return titles


def scrape_title(title, output_dir):
    """Run eksisozluk-scraper CLI for a specific title."""
    safe_title = re.sub(r'[^a-zA-Z0-9ğüşöçıİĞÜŞÖÇ]+', '_', title).strip('_')
    output_file = os.path.join(output_dir, f"{safe_title}.json")

    if os.path.exists(output_file):
        print(f"⏩ Skipping (already scraped): {title}")
        return

    print(f"📘 Scraping: {title}")
    subprocess.run([
        "eksisozluk-scraper",
        title,
        "--output", output_file,
        "--delay", str(DELAY_BETWEEN_REQUESTS),
        "--max-retries", str(MAX_RETRIES),
        "--retry-delay", str(RETRY_DELAY)
    ], check=False)
    time.sleep(DELAY_BETWEEN_REQUESTS)


def scrape_group(group_name, regex_pattern):
    """Scrape all titles for one group of keywords."""
    keywords_to_query = [group_name]  # you can add plural forms manually if needed
    output_dir = f"{group_name}_data"
    os.makedirs(output_dir, exist_ok=True)

    all_titles = set()
    print(f"\n========== {group_name.upper()} ==========")
    for kw in keywords_to_query:
        print(f"🔍 Searching for titles containing '{kw}'...")
        titles = find_titles(kw, regex_pattern)
        for title, _ in titles:
            t_lower = title.lower()
            if t_lower not in all_titles:
                all_titles.add(t_lower)
                scrape_title(title, output_dir)
    print(f"✅ Finished scraping group: {group_name} | total {len(all_titles)} titles scraped.\n")


def main():
    for group_name, regex_pattern in STEMS.items():
        try:
            scrape_group(group_name, regex_pattern)
        except Exception as e:
            print(f"⚠️ Error in group {group_name}: {e}")
            continue

    print("🎉 All groups completed!")


if __name__ == "__main__":
    main()
