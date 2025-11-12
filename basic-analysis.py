import os, json
import pandas as pd
from glob import glob
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt

# load data
DATA_FOLDERS = [
    "suriye_titles_data",
    "afgan_titles_data",
    "gocmen_titles_data",
    "rus_titles_data",
    "ukrayna_titles_data"
]

def load_json_folder(folder):
    rows = []
    for fpath in glob(os.path.join(folder, "*.json")):
        with open(fpath, "r") as f:
            try:
                data = json.load(f)
            except:
                continue
        title = data.get("title", os.path.basename(fpath))
        entries = data.get("entries", data)
        if not isinstance(entries, list):
            continue
        for e in entries:
            date = e.get("date")
            content = e.get("content", "")
            rows.append({
                "origin_folder": os.path.basename(folder),
                "title": title,
                "date": date,
                "length": len(content)
            })
    return pd.DataFrame(rows)

dfs = [load_json_folder(f) for f in DATA_FOLDERS]
df = pd.concat(dfs, ignore_index=True)
print(f"Loaded {len(df):,} entries total.")

# normalize dates
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"])
df["year"] = df["date"].dt.year

# basic analysis
# entries per category
df.groupby("origin_folder").size()

# unique titles per category
df.groupby("origin_folder")["title"].nunique()

# average entry length per category
df.groupby("origin_folder")["length"].mean().round(1)

# timeline of discourse volume
timeline = df.groupby(["origin_folder","year"]).size().reset_index(name="entries")

plt.figure(figsize=(10,6))
sns.lineplot(data=timeline, x="year", y="entries", hue="origin_folder", marker="o")
plt.title("Number of Ekşi Sözlük entries per year by group")
plt.ylabel("Entries")
plt.xlabel("Year")
plt.tight_layout()
plt.show()

# most active titles per category
top_titles = (
    df.groupby(["origin_folder","title"])
      .size()
      .reset_index(name="entries")
      .sort_values(["origin_folder","entries"], ascending=[True,False])
)

for group in df["origin_folder"].unique():
    print(f"\n🔹 Top titles for {group}:")
    subset = top_titles[top_titles["origin_folder"]==group].head(10)
    print(subset[["title","entries"]].to_string(index=False))

# entry length distributions
sns.boxplot(data=df, x="origin_folder", y="length")
plt.title("Entry length distribution by category")
plt.ylabel("Characters per entry")
plt.show()

