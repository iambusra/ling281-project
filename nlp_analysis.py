#!/usr/bin/env python3
# ============================================================
# Ekşi Sözlük NLP analysis: "suri" vs "suriyeli"
# ============================================================

import os, json, re
from glob import glob
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# 1️⃣ Load and prepare the corpus
# ============================================================

DATA_FOLDERS = [
    "suriye_titles_data",
    "afgan_titles_data",
    "gocmen_titles_data",
    "rus_titles_data",
    "ukrayna_titles_data",
]

def load_json_folder(folder):
    rows = []
    for fpath in glob(os.path.join(folder, "*.json")):
        with open(fpath, "r") as f:
            try:
                data = json.load(f)
            except Exception:
                continue
        title = data.get("title", os.path.basename(fpath))
        entries = data.get("entries", data)
        if not isinstance(entries, list):
            continue
        for e in entries:
            content = e.get("content", "").strip()
            date = e.get("date")
            entry_id = e.get("entry_id", None)
            if not content:
                continue
            rows.append({
                "origin_folder": os.path.basename(folder),
                "title": title.lower(),
                "entry_text": content,
                "date": date,
                "entry_id": entry_id
            })
    return pd.DataFrame(rows)

dfs = [load_json_folder(f) for f in DATA_FOLDERS]
df = pd.concat(dfs, ignore_index=True)
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"])
df["year"] = df["date"].dt.year
print(f"Loaded {len(df):,} entries total")

# ============================================================
# 2️⃣ Identify 'suri' vs 'suriyeli' mentions
# ============================================================

PATTERNS = {
    "suri": re.compile(r"\bsuri[a-zçğıöşü]*", re.IGNORECASE),
    "suriyeli": re.compile(r"\bsuriyel[a-zçğıöşü]*", re.IGNORECASE)
}

def label_group(text):
    if not isinstance(text, str):
        return None
    if PATTERNS["suriyeli"].search(text):
        return "suriyeli"
    elif PATTERNS["suri"].search(text):
        return "suri"
    return None

df["group"] = df["entry_text"].apply(label_group)
df = df.dropna(subset=["group"])
print(df["group"].value_counts())

# ============================================================
# 3️⃣ Mark whether title includes göçmen/mülteci/sığınmacı
# ============================================================

def title_category(t):
    if not isinstance(t, str):
        return None
    if re.search(r"göçmen[a-zçğıöşü]*", t, re.I):
        return "gocmen"
    if re.search(r"mültec[a-zçğıöşü]*", t, re.I):
        return "multeci"
    if re.search(r"sığınmac[a-zçğıöşü]*", t, re.I):
        return "siginmaci"
    return None

df["title_category"] = df["title"].apply(title_category)

# ============================================================
# 4️⃣ Sentiment Analysis (two options)
# ============================================================

# -------- Option A: Lexicon-based (VADER) --------
print("\nRunning VADER sentiment analysis (Option A)...")
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download("vader_lexicon", quiet=True)

vader = SentimentIntensityAnalyzer()
df["vader_score"] = df["entry_text"].apply(lambda x: vader.polarity_scores(x)["compound"])

# classify as Positive/Negative/Neutral
df["vader_sentiment"] = pd.cut(
    df["vader_score"],
    bins=[-1, -0.05, 0.05, 1],
    labels=["Negative", "Neutral", "Positive"]
)

# -------- Option B: Model-based (BERT) --------
print("\nRunning transformer-based sentiment analysis (Option B)...")
from transformers import pipeline
sent_model = pipeline("sentiment-analysis", model="savasy/bert-base-turkish-sentiment-cased")

def model_sentiment(text):
    try:
        out = sent_model(text[:512])[0]
        label = out["label"]
        score = out["score"]
        return label, score
    except Exception:
        return None, None

df[["model_label", "model_score"]] = df["entry_text"].apply(
    lambda t: pd.Series(model_sentiment(t))
)

# ============================================================
# 5️⃣ Summary statistics
# ============================================================

summary = df.groupby("group")[["vader_score","model_score"]].mean().round(3)
print("\nAverage sentiment scores:")
print(summary)

cat_counts = df.groupby(["group","title_category"]).size().unstack(fill_value=0)
print("\nEntries containing each keyword under refugee-related titles:")
print(cat_counts)

# ============================================================
# 6️⃣ Temporal analysis plots
# ============================================================

sns.set(style="whitegrid", rc={"figure.figsize": (12,6)})

# ---- Entry counts per year ----
plt.figure(figsize=(12,6))
counts = df.groupby(["group","year"]).size().reset_index(name="entries")
sns.lineplot(data=counts, x="year", y="entries", hue="group", marker="o")
plt.title("Number of entries per year (Suri vs Suriyeli)")
plt.xlabel("Year")
plt.ylabel("Number of entries")
plt.xticks(range(2010,2026), rotation=45)
plt.tight_layout()
plt.show()

# ---- Average sentiment per year (BERT model) ----
plt.figure(figsize=(12,6))
sent_year = (
    df.groupby(["group","year"])["model_score"]
      .mean().reset_index()
)
sns.lineplot(data=sent_year, x="year", y="model_score", hue="group", marker="o")
plt.title("Average model-based sentiment per year (Suri vs Suriyeli)")
plt.xlabel("Year")
plt.ylabel("Mean sentiment score")
plt.xticks(range(2010,2026), rotation=45)
plt.tight_layout()
plt.show()

# ---- Average sentiment per year (VADER) ----
plt.figure(figsize=(12,6))
vader_year = (
    df.groupby(["group","year"])["vader_score"]
      .mean().reset_index()
)
sns.lineplot(data=vader_year, x="year", y="vader_score", hue="group", marker="o")
plt.title("Average VADER sentiment per year (Suri vs Suriyeli)")
plt.xlabel("Year")
plt.ylabel("Mean sentiment score")
plt.xticks(range(2010,2026), rotation=45)
plt.tight_layout()
plt.show()

# ============================================================
# 7️⃣ Save results for further analysis
# ============================================================

df.to_csv("suri_suriyeli_sentiment_results.csv", index=False)
print("\n✅ Results saved to suri_suriyeli_sentiment_results.csv")

# ============================================================
# END
# ============================================================
