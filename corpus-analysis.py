import pandas as pd
from datetime import datetime
import os, json
from glob import glob
import re
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import itertools


# ============================================================
# 1. LOAD DATA
# ============================================================

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
            except Exception:
                continue
        title = data.get("title", os.path.basename(fpath))
        entries = data.get("entries", data)
        if not isinstance(entries, list):
            continue
        for e in entries:
            rows.append({
                "origin_folder": os.path.basename(folder),
                "title": title.lower(),
                "date": e.get("date")
            })
    return pd.DataFrame(rows)

dfs = [load_json_folder(f) for f in DATA_FOLDERS]
df = pd.concat(dfs, ignore_index=True)

# consistent date parsing -----------------------------------------------------
def parse_date_safe(x):
    if not isinstance(x, str):
        return pd.NaT
    for fmt in ("%d.%m.%Y %H:%M", "%d.%m.%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(x, fmt)
        except ValueError:
            continue
    return pd.NaT

df["date"] = df["date"].apply(parse_date_safe)
df = df.dropna(subset=["date"])
df["year"] = df["date"].dt.year
print(f"✅ Loaded {len(df):,} entries total")

# ============================================================
# 2. SPLIT GÖÇMEN SUBCATEGORIES
# ============================================================

patterns = {
    "gocmen": re.compile(r"göçmen[a-zçğıöşü]*", re.IGNORECASE),
    "multeci": re.compile(r"mültec[a-zçğıöşü]*", re.IGNORECASE),
    "siginmaci": re.compile(r"sığınmac[a-zçğıöşü]*", re.IGNORECASE)
}

def split_gocmen_subcats(df):
    mask = df["origin_folder"] == "gocmen_titles_data"
    gocmen_df = df[mask].copy()
    for cat, regex in patterns.items():
        gocmen_df[cat] = gocmen_df["title"].apply(lambda t: bool(regex.search(t)))
    return gocmen_df

gocmen_df = split_gocmen_subcats(df)

# ============================================================
# 3. COUNT ENTRIES PER YEAR / GROUP
# ============================================================

timeline = (
    df.groupby(["origin_folder", "year"])
      .size()
      .reset_index(name="entries")
)

# subcategories for göçmen
gocmen_timeline = (
    gocmen_df.melt(
        id_vars=["year"],
        value_vars=["gocmen","multeci","siginmaci"],
        var_name="subgroup",
        value_name="present"
    )
    .query("present == True")
    .groupby(["subgroup","year"])
    .size()
    .reset_index(name="entries")
)

# ============================================================
# 4. PLOT MAIN TIMELINE (2010–2025)
# ============================================================

sns.set(style="whitegrid",
        rc={"figure.figsize": (12,6), "axes.titlesize":14, "axes.labelsize":12})

label_map = {
    "suriye_titles_data": "Suriye",
    "afgan_titles_data": "Afgan",
    "gocmen_titles_data": "Göçmen+Mülteci+Sığınmacı",
    "rus_titles_data": "Rus",
    "ukrayna_titles_data": "Ukrayna",
    "gocmen_titles_data": "Göçmen"
}

timeline["Group"] = timeline["origin_folder"].map(label_map)
timeline = timeline[timeline["year"].between(2010, 2025)]

# ensure all years 2010–2025 appear, even with zero
years = list(range(2010, 2026))
timeline_full = (
    timeline.pivot_table(index="year", columns="Group", values="entries", fill_value=0)
            .reindex(years, fill_value=0)
            .reset_index()
            .melt(id_vars="year", var_name="Group", value_name="entries")
)

plt.figure(figsize=(13,6))
sns.lineplot(data=timeline_full, x="year", y="entries", hue="Group", marker="o")

plt.xticks(years, rotation=45)
plt.xlim(2010, 2025)
plt.xlabel("Year")
plt.ylabel("Number of entries")
plt.title("Ekşi Sözlük entries per year by refugee-related keyword group")

# political / social milestones ----------------------------------------------
events = {
    2011: "Syrian Civil War begins",
    2015: "EU–Turkey refugee deal",
    2018: "Economic crisis",
    2020: "COVID-19 & border reopening",
    2022: "Ukraine war",
    2023: "Elections & Earthquakes"
}
for year, label in events.items():
    plt.axvline(x=year, color="gray", linestyle="--", alpha=0.5)
    plt.text(year + 0.1, plt.ylim()[1]*0.95, label,
             rotation=90, va="top", fontsize=9, alpha=0.7)

plt.legend(title="Group")
plt.tight_layout()
plt.show()

# ============================================================
# 5. PLOT SUBCATEGORIES INSIDE GÖÇMEN FOLDER
# ============================================================

plt.figure(figsize=(10,5))
sns.lineplot(data=gocmen_timeline, x="year", y="entries",
             hue="subgroup", marker="o")
plt.xticks(years, rotation=45)
plt.xlim(2010, 2025)
plt.title("Entries per year for Göçmen-related subgroups")
plt.xlabel("Year")
plt.ylabel("Number of entries")
plt.legend(title="Subgroup",
           labels=["Göçmen", "Mülteci", "Sığınmacı"])
plt.tight_layout()
plt.show()

# ============================================================
# 6. COMBINED PLOT: MAIN GROUPS + GÖÇMEN SUBGROUPS
# ============================================================

# merge the göçmen subgroups into the main timeline
# rename their column for consistency
gocmen_timeline["Group"] = gocmen_timeline["subgroup"].map({
    "gocmen": "Göçmen",
    "multeci": "Mülteci",
    "siginmaci": "Sığınmacı"
})
gocmen_timeline = gocmen_timeline.drop(columns=["subgroup"])

# prepare the main timeline (rename for consistency)
timeline_full = (
    timeline.pivot_table(index="year", columns="origin_folder", values="entries", fill_value=0)
            .reset_index()
            .melt(id_vars="year", var_name="origin_folder", value_name="entries")
)
timeline_full["Group"] = timeline_full["origin_folder"].map({
    "suriye_titles_data": "Suriye",
    "afgan_titles_data": "Afgan",
    "gocmen_titles_data": "Göçmen+Mülteci+Sığınmacı (Total)",
    "rus_titles_data": "Rus",
    "ukrayna_titles_data": "Ukrayna"
})

# combine both datasets
combined_timeline = pd.concat([
    timeline_full[["year", "Group", "entries"]],
    gocmen_timeline[["year", "Group", "entries"]]
])

combined_timeline = combined_timeline[combined_timeline["year"].between(2010, 2025)]

# make sure every year is represented
years = list(range(2010, 2026))
combined_timeline = (
    combined_timeline.pivot_table(index="year", columns="Group", values="entries", fill_value=0)
                     .reindex(years, fill_value=0)
                     .reset_index()
                     .melt(id_vars="year", var_name="Group", value_name="entries")
)

# ------------------------------------------------------------
# create the plot
sns.set(style="whitegrid",
        rc={"figure.figsize": (13,6), "axes.titlesize":14, "axes.labelsize":12})

plt.figure(figsize=(13,6))
sns.lineplot(
    data=combined_timeline,
    x="year", y="entries",
    hue="Group", marker="o"
)

plt.xticks(years, rotation=45)
plt.xlim(2010, 2025)
plt.xlabel("Year")
plt.ylabel("Number of entries")
plt.title("Ekşi Sözlük entries per year by refugee-related keyword group and Göçmen subgroups")

# ------------------------------------------------------------
# political / social milestones — now includes all election years
events = {
    2011: "Syrian Civil War begins",
    2014: "Local elections",
    2015: "General elections (June & Nov)",
    2018: "General elections",
    2019: "Local elections",
    2020: "COVID-19 & border reopening",
    2022: "Ukraine war",
    2023: "General & local elections + Earthquakes"
}
for year, label in events.items():
    plt.axvline(x=year, color="gray", linestyle="--", alpha=0.5)
    plt.text(year + 0.1, plt.ylim()[1]*0.95, label,
             rotation=90, va="top", fontsize=9, alpha=0.7)

plt.legend(title="Group / Subgroup", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()


# ============================================================
# 7. SOME SHALLOW ANALYSES
# ============================================================
# Columns we need
# origin_folder | title | date | year
titles_df = df[["origin_folder","title","year"]].drop_duplicates()
titles_df.head()

# title-level framing via regex lexicons
# --- define regex-based framing dictionaries ---
frames = {
    "crime": [
        r"suç", r"tecavüz", r"saldır", r"öldür", r"hırsız", r"güvenlik"
    ],
    "economy": [
        r"ekonomi", r"işsizlik", r"yardım", r"para", r"vergi", r"maliyet"
    ],
    "solidarity": [
        r"yardım", r"insan", r"vicdan", r"empati", r"dayanışma"
    ],
    "culture": [
        r"medeniyet", r"batı", r"avrupa", r"dil", r"kültür", r"modern"
    ],
    "politics": [
        r"hükümet", r"iktidar", r"muhalefet", r"erdoğan", r"seçim"
    ],
    "gender": [
        r"kadın", r"erkek", r"kız", r"evlilik", r"cinsiyet"
    ]
}

def frame_presence(title):
    """Return dict of booleans: which frames appear in this title."""
    return {f: any(re.search(pat, title) for pat in pats) for f, pats in frames.items()}

frame_df = titles_df.copy()
for f_name, patterns in frames.items():
    regex = re.compile("|".join(patterns), re.IGNORECASE)
    frame_df[f_name] = frame_df["title"].apply(lambda t: bool(regex.search(t)))

# count frame frequency per refugee group
frame_counts = (
    frame_df.groupby("origin_folder")[list(frames.keys())]
            .sum()
            .astype(int)
            .reset_index()
)

label_map = {
    "suriye_titles_data": "Suriye",
    "afgan_titles_data": "Afgan",
    "gocmen_titles_data": "Göçmen+Mülteci+Sığınmacı",
    "rus_titles_data": "Rus",
    "ukrayna_titles_data": "Ukrayna"
}
frame_counts["Group"] = frame_counts["origin_folder"].map(label_map)
frame_counts = frame_counts.drop(columns="origin_folder")
frame_counts.set_index("Group", inplace=True)
frame_counts

# visualize as heatmap
plt.figure(figsize=(10,6))
sns.heatmap(frame_counts, annot=True, fmt="d", cmap="YlGnBu")
plt.title("Frame occurrences in Ekşi Sözlük titles by refugee group")
plt.xlabel("Frame category")
plt.ylabel("Group")
plt.tight_layout()
plt.show()


# new heatmap
# ============================================================
# create separate Göçmen / Mülteci / Sığınmacı frame counts
# ============================================================

# start from the title-level frame_df we already built
# and merge in the subgroup labels from gocmen_df

# filter gocmen titles and keep frame columns
gocmen_frames = frame_df[frame_df["origin_folder"] == "gocmen_titles_data"].copy()
subcats = gocmen_df[["title", "gocmen", "multeci", "siginmaci"]]
gocmen_frames = gocmen_frames.merge(subcats, on="title", how="left")

# explode to multiple rows if a title belongs to multiple subgroups
gocmen_frames = gocmen_frames.melt(
    id_vars=["title"] + list(frames.keys()),
    value_vars=["gocmen","multeci","siginmaci"],
    var_name="subgroup",
    value_name="present"
).query("present == True")

# aggregate frame counts per subgroup
gocmen_counts = (
    gocmen_frames.groupby("subgroup")[list(frames.keys())]
                 .sum()
                 .astype(int)
                 .reset_index()
)
gocmen_counts["Group"] = gocmen_counts["subgroup"].map({
    "gocmen": "Göçmen",
    "multeci": "Mülteci",
    "siginmaci": "Sığınmacı"
})
gocmen_counts = gocmen_counts.drop(columns="subgroup")

# base frame counts (Suriye, Afgan, Rus, Ukrayna)
frame_counts_base = (
    frame_df[frame_df["origin_folder"] != "gocmen_titles_data"]
    .groupby("origin_folder")[list(frames.keys())]
    .sum()
    .astype(int)
    .reset_index()
)
label_map = {
    "suriye_titles_data": "Suriye",
    "afgan_titles_data": "Afgan",
    "rus_titles_data": "Rus",
    "ukrayna_titles_data": "Ukrayna"
}
frame_counts_base["Group"] = frame_counts_base["origin_folder"].map(label_map)
frame_counts_base = frame_counts_base.drop(columns="origin_folder")

# combine both sets
frame_counts_sep = pd.concat([frame_counts_base, gocmen_counts])
frame_counts_sep.set_index("Group", inplace=True)

# ============================================================
# plot new heatmap
# ============================================================

plt.figure(figsize=(10,6))
sns.heatmap(frame_counts_sep, annot=True, fmt="d", cmap="YlGnBu")
plt.title("Frame occurrences in Ekşi Sözlük titles by refugee or migrant subgroup")
plt.xlabel("Frame category")
plt.ylabel("Group")
plt.tight_layout()
plt.show()

# ============================================================
# 🟦 Heatmap: Framing across Göçmen subgroups only
# ============================================================

# 1. Extract only göçmen titles and merge their subgroup flags
gocmen_frames = frame_df[frame_df["origin_folder"] == "gocmen_titles_data"].copy()
subcats = gocmen_df[["title", "gocmen", "multeci", "siginmaci"]]
gocmen_frames = gocmen_frames.merge(subcats, on="title", how="left")

# 2. Explode so titles count for each relevant subgroup
gocmen_frames = gocmen_frames.melt(
    id_vars=["title"] + list(frames.keys()),
    value_vars=["gocmen", "multeci", "siginmaci"],
    var_name="subgroup",
    value_name="present"
).query("present == True")

# 3. Aggregate by subgroup × frame type
gocmen_counts = (
    gocmen_frames.groupby("subgroup")[list(frames.keys())]
                 .sum()
                 .astype(int)
                 .reset_index()
)
gocmen_counts["Group"] = gocmen_counts["subgroup"].map({
    "gocmen": "Göçmen",
    "multeci": "Mülteci",
    "siginmaci": "Sığınmacı"
})
gocmen_counts = gocmen_counts.drop(columns="subgroup").set_index("Group")

# 4. Plot subgroup-only heatmap
plt.figure(figsize=(8, 4))
sns.heatmap(gocmen_counts, annot=True, fmt="d", cmap="YlGnBu")
plt.title("Frame occurrences in Göçmen-related titles (by subgroup)")
plt.xlabel("Frame category")
plt.ylabel("Subgroup")
plt.tight_layout()
plt.show()

normed = gocmen_counts.div(gocmen_counts.sum(axis=1), axis=0)
plt.figure(figsize=(8, 4))
sns.heatmap(normed, annot=True, fmt=".2f", cmap="YlGnBu")
plt.title("Relative frame proportions in Göçmen-related titles")
plt.xlabel("Frame category")
plt.ylabel("Subgroup")
plt.tight_layout()
plt.show()

# ============================================================
# 🕒 Temporal evolution of frames within Göçmen subgroups
# ============================================================

# reuse the gocmen_frames from the previous section
if "year" not in gocmen_frames.columns:
    # merge back the year info from df if missing
    year_map = df[["title", "year"]].drop_duplicates()
    gocmen_frames = gocmen_frames.merge(year_map, on="title", how="left")

# 1. reshape so we can group by year + subgroup + frame
# ensure we don't conflict with an existing 'present' column
if "present" in gocmen_frames.columns:
    gocmen_frames = gocmen_frames.rename(columns={"present": "sub_present"})

melted = (
    gocmen_frames.melt(
        id_vars=["year", "subgroup"],
        value_vars=list(frames.keys()),
        var_name="frame",
        value_name="is_frame"
    )
    .query("is_frame == True and year >= 2010 and year <= 2025")
)


# 2. aggregate counts
timeline = (
    melted.groupby(["subgroup", "frame", "year"])
          .size()
          .reset_index(name="count")
)

# 3. normalize per year (optional, clearer for comparison)
totals = (
    gocmen_frames.groupby(["subgroup", "year"])
                 .size()
                 .rename("total")
                 .reset_index()
)
timeline = timeline.merge(totals, on=["subgroup", "year"], how="left")
timeline["ratio"] = timeline["count"] / timeline["total"]

# 4. plot — one panel per subgroup
sns.set(style="whitegrid", rc={"figure.figsize": (14, 4)})
sub_labels = {"gocmen": "Göçmen", "multeci": "Mülteci", "siginmaci": "Sığınmacı"}

for subgroup, label in sub_labels.items():
    sub = timeline[timeline["subgroup"] == subgroup]
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=sub, x="year", y="ratio", hue="frame", marker="o")
    plt.title(f"Framing evolution over time in '{label}' titles")
    plt.xlabel("Year")
    plt.ylabel("Proportion of titles mentioning frame")
    plt.xticks(range(2010, 2026), rotation=45)
    plt.xlim(2010, 2025)
    plt.ylim(0, sub["ratio"].max() * 1.1)
    plt.legend(title="Frame", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()


# shallow word frequency & collocations
def clean_title(t):
    t = re.sub(r"[^a-zçğıöşüİĞÜŞÖÇ\s]", " ", t.lower())
    return t.split()

def top_words(df, n=20):
    words = list(itertools.chain.from_iterable(df["title"].apply(clean_title)))
    common = Counter(words).most_common(n)
    return pd.DataFrame(common, columns=["word","freq"])

for group, subset in titles_df.groupby("origin_folder"):
    label = label_map[group]
    print(f"\n🔹 Top words in {label} titles:")
    print(top_words(subset, 15))

# bigrams
def get_bigrams(words):
    return zip(words[:-1], words[1:])

def top_bigrams(df, n=20):
    bigrams = list(itertools.chain.from_iterable(
        get_bigrams(clean_title(t)) for t in df["title"]
    ))
    common = Counter(bigrams).most_common(n)
    return pd.DataFrame([" ".join(b) for b, _ in common], columns=["bigram"])

for group, subset in titles_df.groupby("origin_folder"):
    label = label_map[group]
    print(f"\n🔹 Common bigrams in {label}:")
    print(top_bigrams(subset, 10))

# semantic framing trends over time
# mark frame presence by year
frame_yearly = frame_df.melt(
    id_vars=["year","origin_folder"],
    value_vars=list(frames.keys()),
    var_name="frame",
    value_name="present"
).query("present == True")

# count frames per year per group
frame_trend = (
    frame_yearly.groupby(["origin_folder","frame","year"])
                .size()
                .reset_index(name="count")
)

# suriye
suriye_trend = frame_trend[frame_trend["origin_folder"]=="suriye_titles_data"]

plt.figure(figsize=(10,5))
sns.lineplot(data=suriye_trend, x="year", y="count", hue="frame", marker="o")
plt.title("Framing evolution over time in Suriye-related titles")
plt.ylabel("Title count mentioning frame keyword")
plt.xlabel("Year")
plt.xticks(range(2010,2026), rotation=45)
plt.tight_layout()
plt.show()

# lets be modular
def plot_frame_trends(sub_df, label, years=range(2010, 2026)):
    """Plot frame frequencies per year for a given subset of titles."""
    melted = (
        sub_df.melt(
            id_vars=["year"],
            value_vars=["crime","economy","solidarity","culture","politics","gender"],
            var_name="frame",
            value_name="present"
        )
        .query("present == True")
        .groupby(["frame","year"])
        .size()
        .reset_index(name="count")
    )
    if melted.empty:
        print(f"⚠️ No data for {label}")
        return

    plt.figure(figsize=(10,5))
    sns.lineplot(data=melted, x="year", y="count", hue="frame", marker="o")
    plt.title(f"Framing evolution over time in '{label}' titles")
    plt.ylabel("Number of titles mentioning frame keyword")
    plt.xlabel("Year")
    plt.xticks(list(years), rotation=45)
    plt.xlim(min(years), max(years))
    plt.tight_layout()
    plt.show()

# Suriye
suriye_df = frame_df[frame_df["origin_folder"]=="suriye_titles_data"]
plot_frame_trends(suriye_df, "Suriye")

# Ukrayna
ukrayna_df = frame_df[frame_df["origin_folder"]=="ukrayna_titles_data"]
plot_frame_trends(ukrayna_df, "Ukrayna")

# Rusya
rusya_df = frame_df[frame_df["origin_folder"]=="rus_titles_data"]
plot_frame_trends(rusya_df, "Rusya")

# göçmen
def get_gocmen_subset(df, subgroup):
    return df[df[subgroup] == True]

plot_frame_trends(get_gocmen_subset(gocmen_df, "gocmen"), "Göçmen")
plot_frame_trends(get_gocmen_subset(gocmen_df, "multeci"), "Mülteci")
plot_frame_trends(get_gocmen_subset(gocmen_df, "siginmaci"), "Sığınmacı")

# normalize by total titles per year
# total titles per year
totals = sub_df.groupby("year").size().rename("total").reset_index()
melted = melted.merge(totals, on="year", how="left")
melted["ratio"] = melted["count"] / melted["total"]

sns.lineplot(data=melted, x="year", y="ratio", hue="frame", marker="o")
plt.ylabel("Proportion of titles mentioning frame")

