#!/usr/bin/env python3
# ============================================================
# Summarize Ekşi Sözlük data by category (titles & entries)
# Outputs: LaTeX-ready booktabs table
# ============================================================

import os, json
from glob import glob
import pandas as pd

# ------------------------------------------------------------
# define your data folders
DATA_FOLDERS = {
    "suriye_titles_data": "Suriye / Suriyeli / Suri (Syrian)",
    "afgan_titles_data": "Afgan / Afganistan / Afganlı (Afghan)",
    "rus_titles_data": "Rus / Rusya / Rusyalı (Russian)",
    "ukrayna_titles_data": "Ukrayna / Ukraynalı (Ukrainian)",
    "gocmen_titles_data": "Göçmen / Mülteci / Sığınmacı (Migrant / Refugee / Asylum seeker)"
}

def summarize_folder(folder_path):
    """Count titles and total entries in a given folder."""
    json_files = glob(os.path.join(folder_path, "*.json"))
    title_count = len(json_files)
    entry_count = 0

    for fpath in json_files:
        try:
            with open(fpath, "r") as f:
                data = json.load(f)
            entries = data.get("entries", data)
            if isinstance(entries, list):
                entry_count += len(entries)
        except Exception:
            continue
    return title_count, entry_count

# ------------------------------------------------------------
# collect stats
summary = []
for folder, label in DATA_FOLDERS.items():
    if not os.path.exists(folder):
        print(f"⚠️ Folder not found: {folder}")
        continue
    titles, entries = summarize_folder(folder)
    summary.append({"Category": label, "Titles": titles, "Entries": entries})

df = pd.DataFrame(summary)
total_titles = df["Titles"].sum()
total_entries = df["Entries"].sum()
df.loc[len(df)] = {"Category": "Total", "Titles": total_titles, "Entries": total_entries}

# ------------------------------------------------------------
# print results nicely
print("\n=== Summary by Category ===")
print(df.to_string(index=False))

# ------------------------------------------------------------
# generate LaTeX booktabs table
latex = [
    "\\begin{table}[h!]",
    "\\centering",
    "\\begin{tabular}{lrr}",
    "\\toprule",
    "\\textbf{Category} & \\textbf{Titles} & \\textbf{Entries} \\\\",
    "\\midrule"
]

for _, row in df.iterrows():
    latex.append(f"{row['Category']} & {row['Titles']:,} & {row['Entries']:,} \\\\")

latex += [
    "\\bottomrule",
    "\\end{tabular}",
    "\\caption{Overview of the Ekşi Sözlük dataset by keyword group.}",
    "\\end{table}"
]

# ------------------------------------------------------------
# save LaTeX output
with open("data_summary_table.tex", "w", encoding="utf-8") as f:
    f.write("\n".join(latex))

print("\n✅ LaTeX table saved to data_summary_table.tex")
