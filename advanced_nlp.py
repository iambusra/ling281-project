#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
advanced_nlp.py

Advanced NLP analysis on Ekşi Sözlük refugee-related data.

- Preprocess entries & titles
- Add lexicon-based frame flags
- Run Turkish sentiment models on entries
- Aggregate sentiment to titles (creation date = first entry date)
- Analyze Suri vs Suriyeli as predictor of negative sentiment
- Export extreme positive/negative examples per group
- Export timelines and tables to exports/

"""

import os
import json
from glob import glob
import re
import logging

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # for headless environments
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import statsmodels.api as sm

# -----------------------------------------------------------
# CONFIG
# -----------------------------------------------------------

RAW_FOLDERS = {
    "suriye_titles_data": "Syrian",
    "rus_titles_data": "Russian",
    "ukrayna_titles_data": "Ukrainian",
    "gocmen_titles_data": "MigrantBlock",  # contains göçmen/mülteci/sığınmacı
}

EXPORT_DIR = "exports"
FIG_DIR = os.path.join(EXPORT_DIR, "figures")
TABLE_DIR = os.path.join(EXPORT_DIR, "tables")
DATA_DIR = os.path.join(EXPORT_DIR, "data")
LOG_DIR = os.path.join(EXPORT_DIR, "logs")

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, "advanced_nlp.log"),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# GPU / batch config
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0
BATCH_SIZE = 32  # change if needed

# Sentiment models (Turkish / multilingual; only working ones)
SENTIMENT_MODELS = {
    "savasy": "savasy/bert-base-turkish-sentiment-cased",
    "nlptown": "nlptown/bert-base-multilingual-uncased-sentiment",
    "electra": "incidelen/electra-base-turkish-sentiment-analysis-cased",
}

# Regex helpers for mention detection
# Option A: make 'suri' and 'suriyeli' disjoint
REGEX_MENTION = {
    # suri* but NOT suriyeli*
    "suri": re.compile(r"\bsur[iı](?!yel)[a-zçğıöşü']*", re.IGNORECASE),
    # suriyeli*
    "suriyeli": re.compile(r"\bsur[iı]yel[iı][a-zçğıöşü']*", re.IGNORECASE),
    "ukrayna": re.compile(r"\bukrayn[a-zçğıöşü']*", re.IGNORECASE),
    "rus": re.compile(r"\brus[a-zçğıöşü']*", re.IGNORECASE),
}

# Göçmen / Mülteci / Sığınmacı (for titles)
REGEX_TITLE_CAT = {
    "gocmen": re.compile(r"göçmen[a-zçğıöşü]*", re.IGNORECASE),
    "multeci": re.compile(r"mültec[a-zçğıöşü]*", re.IGNORECASE),
    "siginmaci": re.compile(r"sığınmac[a-zçğıöşü]*", re.IGNORECASE),
}

# Frame / lexicon lists (manually generated)
LEXICONS = {
    "econ": [
        "enflasyon", "zam", "vergi", "kur", "kriz", "dolar", "lira",
        "işsizlik", "kira", "asgari ücret", "hayat pahalılığı", "maaş",
        "yevmiye", "ücret", "fiyat", "fakirlik", "fakir", "geçinmek",
        "geçim sıkıntısı", "fahiş", "işsiz",
    ],
    "sex_crime": [
        "tecavüz", "taciz", "rahatsızlık", "sik", "cinsel", "saldırı",
        "seks", "azgın",
    ],
    "crime": [
        "suç", "gasp", "yağma", "kavga", "güvenlik", "çatış", "huzursuzluk",
        "huzur", "başıboş", "huzursuz", "tedirgin", "güven", "çete",
    ],
    "solidarity": [
        "insanlık", "dayanışma", "yardım", "empati", "kardeşlik", "vicdan",
        "hoşgörü", "ırkçı", "kafatasçı",
    ],
    "pos_culture": [
        "modern", "medeni", "batılı", "kültürlü", "aydın", "ilerici",
        "laik", "avrupa", "avrupalı", "görgülü", "çalışkan", "eğitimli",
        "okumuş",
    ],
    "neg_culture": [
        "yobaz", "bağnaz", "tutucu", "gerici", "geri kafalı", "vahhabi",
        "çöl", "ortadoğu", "orta doğu", "ork", "uruk-hai", "uruk hai",
        "maymun", "istila", "medeniyetsiz", "rahatsız", "dinci", "islamcı",
        "cihad", "görgüsüz", "cahil", "eğitimsiz",
    ],
    "political": [
        "akp", "chp", "mhp", "erdoğan", "ab", "avrupa birliği", "seçim", "yerel seçim",
        "genel seçim", "belediye seçimi", "iktidar", "muhalefet", "zafer partisi",
        "kılıçdaroğlu", "özgür özel", "imamoğlu", "mansur yavaş", "ümit özdağ",
        "sinan oğan", "devlet bahçeli", "milliyetçi", "ulusalcı", "ittifak",
        "cumhur ittifakı", "millet ittifakı", "oy kullan", "vatandaşlık", "pasaport",
    ],
    "religious": [
        "ensar", "muhacir", "müslüman", "müslüman kardeşliği",
        "ümmet", "ümmetçi",
    ],
}

LEXICON_REGEX = {
    key: re.compile(
        r"(" + "|".join([re.escape(w) for w in words]) + r")",
        re.IGNORECASE
    )
    for key, words in LEXICONS.items()
}


# -----------------------------------------------------------
# Utility functions
# -----------------------------------------------------------

def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    # keep diacritics, just lowercase and strip whitespace
    text = text.strip().lower()
    # remove URLs
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def load_raw_data() -> pd.DataFrame:
    """
    Load all JSON files from RAW_FOLDERS into a single entries DataFrame.
    One row per entry.
    """
    rows = []
    for folder, label in RAW_FOLDERS.items():
        if not os.path.isdir(folder):
            logger.warning(f"Folder not found: {folder}")
            continue
        json_files = glob(os.path.join(folder, "*.json"))
        logger.info(f"Loading {len(json_files)} files from {folder}")
        for fpath in json_files:
            try:
                with open(fpath, "r") as f:
                    data = json.load(f)
            except Exception as e:
                logger.error(f"Error reading {fpath}: {e}")
                continue

            title = data.get("title", os.path.basename(fpath))
            entries = data.get("entries", data)
            if not isinstance(entries, list):
                continue
            for idx, e in enumerate(entries):
                content = e.get("content", "")
                if not content:
                    continue
                date = e.get("date")
                entry_id = e.get("entry_id", None)
                rows.append(
                    {
                        "origin_folder": folder,
                        "origin_label": label,
                        "title": title,
                        "title_file": os.path.basename(fpath),
                        "entry_index": idx,
                        "entry_id": entry_id,
                        "raw_text": content,
                        "date_raw": date,
                    }
                )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No entries loaded. Check your data folders.")
    logger.info(f"Loaded {len(df):,} raw entries.")
    return df


def preprocess_entries(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize text, parse dates, add year/month.
    """
    df = df.copy()
    df["clean_text"] = df["raw_text"].apply(normalize_text)
    df["date"] = pd.to_datetime(df["date_raw"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.to_period("M").astype(str)
    logger.info(f"After date parsing, {len(df):,} entries remain.")
    return df


def build_titles_df(entries: pd.DataFrame) -> pd.DataFrame:
    """
    Build titles table: one row per title_file.
    Use earliest entry date as creation date.
    """
    grp = entries.groupby(["origin_folder", "origin_label", "title_file", "title"])
    titles = grp["date"].min().reset_index(name="first_entry_date")
    titles["year_created"] = titles["first_entry_date"].dt.year
    titles["month_created"] = titles["first_entry_date"].dt.to_period("M").astype(str)
    return titles


def apply_lexicon_flags(df: pd.DataFrame, text_col: str, prefix: str) -> pd.DataFrame:
    """
    Add boolean columns for each lexicon to df based on text_col.
    prefix: e.g., 'entry_' or 'title_'
    """
    df = df.copy()
    for key, pattern in LEXICON_REGEX.items():
        col = f"{prefix}{key}"
        df[col] = df[text_col].apply(lambda t: bool(pattern.search(t)))
    return df


def add_group_flags(entries: pd.DataFrame) -> pd.DataFrame:
    """
    Add booleans: is_suri_entry, is_suriyeli_entry, etc.
    Also group flags (syrian, ukrainian, russian, gocmen/multeci/siginmaci).
    """
    df = entries.copy()
    df["is_suri_entry"] = df["clean_text"].apply(lambda t: bool(REGEX_MENTION["suri"].search(t)))
    df["is_suriyeli_entry"] = df["clean_text"].apply(lambda t: bool(REGEX_MENTION["suriyeli"].search(t)))
    df["mentions_ukrayna"] = df["clean_text"].apply(lambda t: bool(REGEX_MENTION["ukrayna"].search(t)))
    df["mentions_rus"] = df["clean_text"].apply(lambda t: bool(REGEX_MENTION["rus"].search(t)))

    # group flags at entry level (loose, based on folder OR mention)
    df["group_syrian"] = (df["origin_folder"] == "suriye_titles_data") | df["is_suri_entry"] | df["is_suriyeli_entry"]
    df["group_ukrainian"] = (df["origin_folder"] == "ukrayna_titles_data") | df["mentions_ukrayna"]
    df["group_russian"] = (df["origin_folder"] == "rus_titles_data") | df["mentions_rus"]
    df["group_migrant_block"] = (df["origin_folder"] == "gocmen_titles_data")

    return df


def add_title_category_flags(titles: pd.DataFrame) -> pd.DataFrame:
    """
    For gocmen_titles_data, mark titles that are göçmen/mülteci/sığınmacı.
    """
    df = titles.copy()

    def cat_flag(row, cat):
        if row["origin_folder"] != "gocmen_titles_data":
            return False
        return bool(REGEX_TITLE_CAT[cat].search(str(row["title"]).lower()))

    for cat in ["gocmen", "multeci", "siginmaci"]:
        df[f"is_{cat}_title"] = df.apply(lambda r, c=cat: cat_flag(r, c), axis=1)

    return df


def map_label_to_sign(label_str: str) -> int:
    """
    Map model label string to {-1, 0, +1} based on its text.
    Works with:
      - Turkish / English negative/neutral/positive labels
      - nlptown 1–5 star labels
    """
    if not isinstance(label_str, str):
        return 0
    s = label_str.lower().strip()

    # Explicit 1–5 stars (nlptown)
    # common patterns: "1 star", "2 stars", etc.
    if "star" in s:
        if "1" in s or "2" in s:
            return -1
        if "3" in s:
            return 0
        if "4" in s or "5" in s:
            return 1

    # negative-ish
    if any(x in s for x in ["neg", "olumsuz", "kötü", "negative"]):
        return -1
    # positive-ish
    if any(x in s for x in ["pos", "olumlu", "iyi", "positive"]):
        return 1
    # neutral or unknown
    if "neut" in s or "tarafsız" in s or "neutral" in s:
        return 0

    return 0


def run_sentiment_model(texts, model_name, model_path, batch_size=BATCH_SIZE):
    """
    Run one Turkish / multilingual sentiment model on a list of texts.
    Returns:
      labels: list of strings
      signed_scores: list of floats (sign * prob_of_predicted_label)
    """
    logger.info(f"Loading sentiment model {model_name}: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(DEVICE)
    if N_GPUS > 1:
        model = torch.nn.DataParallel(model)

    model.eval()
    labels = []
    signed_scores = []

    # get id2label from config
    id2label = model.module.config.id2label if isinstance(model, torch.nn.DataParallel) else model.config.id2label

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc=f"Sentiment {model_name}"):
            batch_texts = [t if isinstance(t, str) else "" for t in texts[i:i+batch_size]]
            enc = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt"
            )
            enc = {k: v.to(DEVICE) for k, v in enc.items()}
            outputs = model(**enc)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            max_probs, preds = probs.max(dim=-1)

            for idx, p in zip(preds.cpu().numpy(), max_probs.cpu().numpy()):
                lab = id2label[int(idx)]
                labels.append(lab)
                sign = map_label_to_sign(lab)
                signed_scores.append(sign * float(p))

    return labels, signed_scores


def run_all_sentiment(entries: pd.DataFrame, text_col: str = "clean_text") -> pd.DataFrame:
    """
    Run all sentiment models on entries[text_col].
    Adds columns:
      sent_<model>_label, sent_<model>_score,
      sent_ensemble_score, sent_ensemble_sign
    """
    df = entries.copy()
    texts = df[text_col].tolist()

    for short_name, model_path in SENTIMENT_MODELS.items():
        label_col = f"sent_{short_name}_label"
        score_col = f"sent_{short_name}_score"
        if label_col in df.columns and score_col in df.columns:
            logger.info(f"Skipping {short_name} sentiment (columns already exist).")
            continue
        try:
            labels, scores = run_sentiment_model(texts, short_name, model_path)
            df[label_col] = labels
            df[score_col] = scores
        except Exception as e:
            logger.error(f"Error running sentiment model {short_name}: {e}")
            df[label_col] = None
            df[score_col] = np.nan

    # ensemble: average of signed scores across available models
    score_cols = [f"sent_{k}_score" for k in SENTIMENT_MODELS.keys() if f"sent_{k}_score" in df.columns]
    df["sent_ensemble_score"] = df[score_cols].mean(axis=1)
    # sign-based label: -1 / 0 / +1
    df["sent_ensemble_sign"] = df["sent_ensemble_score"].apply(
        lambda x: -1 if x < -0.05 else (1 if x > 0.05 else 0)
    )

    return df


# -----------------------------------------------------------
# Analysis functions
# -----------------------------------------------------------

def analyze_sentiment_timelines(entries: pd.DataFrame):
    """
    RQ1-ish: sentiment per year per group.
    Exports CSV and plot.
    """
    df = entries.copy()

    # define groups we care about at entry level
    def label_group(row):
        if row["group_syrian"]:
            return "Syrian"
        if row["group_ukrainian"]:
            return "Ukrainian"
        if row["group_russian"]:
            return "Russian"
        if row["group_migrant_block"]:
            return "MigrantBlock"
        return "Other"

    df["analysis_group"] = df.apply(label_group, axis=1)
    df = df[df["analysis_group"] != "Other"]

    group_year = (
        df.groupby(["analysis_group", "year"])
        .agg(
            n_entries=("sent_ensemble_sign", "size"),
            mean_sent=("sent_ensemble_score", "mean"),
            prop_neg=("sent_ensemble_sign", lambda x: (x == -1).mean()),
            prop_pos=("sent_ensemble_sign", lambda x: (x == 1).mean()),
        )
        .reset_index()
    )

    out_csv = os.path.join(TABLE_DIR, "sentiment_timeline_by_group_year.csv")
    group_year.to_csv(out_csv, index=False)
    logger.info(f"Saved sentiment timeline stats to {out_csv}")

    # Plot mean sentiment over time
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=group_year,
        x="year",
        y="mean_sent",
        hue="analysis_group",
        marker="o"
    )
    plt.axhline(0, color="gray", linestyle="--", alpha=0.5)
    plt.xlabel("Year")
    plt.ylabel("Mean ensemble sentiment")
    plt.title("Mean sentiment over time by group")
    plt.tight_layout()
    out_fig = os.path.join(FIG_DIR, "sentiment_timeline_mean.png")
    plt.savefig(out_fig, dpi=300)
    plt.close()
    logger.info(f"Saved sentiment timeline figure to {out_fig}")


def analyze_titles_creation_and_sentiment(entries: pd.DataFrame, titles: pd.DataFrame):
    """
    Compute title-level sentiment (mean of entry sentiments) and creation dates.
    Export timelines for new titles and their sentiment.
    """
    df = entries.copy()

    # title-level mean sentiment
    title_sent = (
        df.groupby(["origin_folder", "title_file"])
        .agg(
            mean_sent=("sent_ensemble_score", "mean"),
            n_entries=("sent_ensemble_score", "size"),
        )
        .reset_index()
    )
    titles = titles.merge(title_sent, on=["origin_folder", "title_file"], how="left")

    # save titles
    out_csv = os.path.join(DATA_DIR, "titles_with_sentiment.csv")
    titles.to_csv(out_csv, index=False)
    logger.info(f"Saved titles with sentiment to {out_csv}")

    # timeline: new titles per year with avg sentiment
    summary = (
        titles.groupby(["origin_label", "year_created"])
        .agg(
            n_titles=("title_file", "size"),
            mean_title_sent=("mean_sent", "mean"),
        )
        .reset_index()
    )
    out_csv2 = os.path.join(TABLE_DIR, "title_creation_sentiment_by_year_group.csv")
    summary.to_csv(out_csv2, index=False)
    logger.info(f"Saved title creation summary to {out_csv2}")

    # plot number of new titles per year
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=summary,
        x="year_created",
        y="n_titles",
        hue="origin_label",
        marker="o"
    )
    plt.xlabel("Year")
    plt.ylabel("Number of new titles")
    plt.title("New Ekşi Sözlük titles per year by group (approx. creation date)")
    plt.tight_layout()
    out_fig = os.path.join(FIG_DIR, "titles_new_per_year.png")
    plt.savefig(out_fig, dpi=300)
    plt.close()
    logger.info(f"Saved title creation figure to {out_fig}")


def analyze_suri_vs_suriyeli(entries: pd.DataFrame):
    """
    Analyze whether using 'suri' vs 'suriyeli' predicts negative sentiment.
    - Descriptive stats
    - Logistic regression with year × marker interaction + frame covariates
    - Extra logistic: negative vs neutral (dropping positives)
    """
    df = entries.copy()

    # restrict to entries that mention at least one of them
    df = df[(df["is_suri_entry"]) | (df["is_suriyeli_entry"])].copy()
    if df.empty:
        logger.warning("No entries with suri/suriyeli for analysis.")
        return

    df["marker"] = np.where(df["is_suri_entry"], "suri",
                            np.where(df["is_suriyeli_entry"], "suriyeli", "mixed"))

    # filter to clean contrast: only suri OR only suriyeli, not both
    df_simple = df[df["marker"].isin(["suri", "suriyeli"])].copy()
    if df_simple.empty:
        logger.warning("No clean suri-only / suriyeli-only entries for analysis.")
        return

    # Descriptive stats
    desc = (
        df_simple.groupby("marker")
        .agg(
            n=("sent_ensemble_sign", "size"),
            mean_sent=("sent_ensemble_score", "mean"),
            prop_neg=("sent_ensemble_sign", lambda x: (x == -1).mean()),
            prop_neu=("sent_ensemble_sign", lambda x: (x == 0).mean()),
            prop_pos=("sent_ensemble_sign", lambda x: (x == 1).mean()),
        )
        .reset_index()
    )
    out_csv = os.path.join(TABLE_DIR, "suri_suriyeli_descriptives.csv")
    desc.to_csv(out_csv, index=False)
    logger.info(f"Saved suri vs suriyeli descriptives to {out_csv}")

    # prepare logistic: neg_target ~ is_suri * year_c + frame flags
    df_simple["neg_target"] = (df_simple["sent_ensemble_sign"] == -1).astype(int)
    df_simple["is_suri"] = (df_simple["marker"] == "suri").astype(int)
    # center year to avoid weird intercept scaling
    df_simple["year_c"] = df_simple["year"] - df_simple["year"].mean()
    df_simple["is_suri_year"] = df_simple["is_suri"] * df_simple["year_c"]

    # frame covariates from lexicons
    frame_covs = [
        "econ", "crime", "sex_crime",
        "solidarity", "pos_culture", "neg_culture",
        "political", "religious",
    ]
    for key in frame_covs:
        col = f"entry_{key}"
        if col not in df_simple.columns:
            df_simple[col] = False

    X_cols = ["is_suri", "year_c", "is_suri_year"] + [f"entry_{k}" for k in frame_covs]
    X = df_simple[X_cols].astype(float)
    X = sm.add_constant(X)
    y = df_simple["neg_target"].astype(float)

    # Logistic 1: negative vs (neutral+positive)
    try:
        logit_model = sm.Logit(y, X).fit(disp=False)
        params = logit_model.params
        conf = logit_model.conf_int()
        results = pd.DataFrame(
            {
                "term": params.index,
                "estimate": params.values,
                "conf_low": conf[0].values,
                "conf_high": conf[1].values,
                "p_value": logit_model.pvalues.values,
            }
        )
        out_csv2 = os.path.join(TABLE_DIR, "suri_suriyeli_logit_neg_vs_nonneg.csv")
        results.to_csv(out_csv2, index=False)
        logger.info(f"Saved suri vs suriyeli logistic (neg vs non-neg) results to {out_csv2}")
    except Exception as e:
        logger.error(f"Error fitting suri vs suriyeli logit (neg vs non-neg): {e}")

    # Logistic 2: negative vs neutral (drop positives)
    df_neu_neg = df_simple[df_simple["sent_ensemble_sign"].isin([-1, 0])].copy()
    if len(df_neu_neg) > 0:
        df_neu_neg["neg_vs_neu"] = (df_neu_neg["sent_ensemble_sign"] == -1).astype(int)
        X2 = df_neu_neg[X_cols].astype(float)
        X2 = sm.add_constant(X2)
        y2 = df_neu_neg["neg_vs_neu"].astype(float)
        try:
            logit_model2 = sm.Logit(y2, X2).fit(disp=False)
            params2 = logit_model2.params
            conf2 = logit_model2.conf_int()
            results2 = pd.DataFrame(
                {
                    "term": params2.index,
                    "estimate": params2.values,
                    "conf_low": conf2[0].values,
                    "conf_high": conf2[1].values,
                    "p_value": logit_model2.pvalues.values,
                }
            )
            out_csv2b = os.path.join(TABLE_DIR, "suri_suriyeli_logit_neg_vs_neutral.csv")
            results2.to_csv(out_csv2b, index=False)
            logger.info(f"Saved suri vs suriyeli logistic (neg vs neutral) results to {out_csv2b}")
        except Exception as e:
            logger.error(f"Error fitting suri vs suriyeli logit (neg vs neutral): {e}")
    else:
        logger.warning("No suri/suriyeli entries with only neutral/negative for neg-vs-neutral model.")

    # plot proportion negative over time for suri vs suriyeli
    prop_year = (
        df_simple.groupby(["marker", "year"])
        .agg(
            prop_neg=("neg_target", "mean"),
            n=("neg_target", "size"),
        )
        .reset_index()
    )
    out_csv3 = os.path.join(TABLE_DIR, "suri_suriyeli_propneg_by_year.csv")
    prop_year.to_csv(out_csv3, index=False)
    logger.info(f"Saved suri vs suriyeli yearly prop-neg to {out_csv3}")

    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=prop_year,
        x="year",
        y="prop_neg",
        hue="marker",
        marker="o"
    )
    plt.xlabel("Year")
    plt.ylabel("Proportion negative")
    plt.title("Proportion of negative entries for 'suri' vs 'suriyeli'")
    plt.tight_layout()
    out_fig = os.path.join(FIG_DIR, "suri_vs_suriyeli_propneg_timeline.png")
    plt.savefig(out_fig, dpi=300)
    plt.close()
    logger.info(f"Saved suri vs suriyeli figure to {out_fig}")


def export_extreme_sentiment_examples(entries: pd.DataFrame, top_k: int = 10):
    """
    Export most positive and most negative entries per group:
    - Syrian
    - Ukrainian
    - Russian
    - Göçmen
    - Mülteci
    - Sığınmacı
    """
    df = entries.copy()

    # identify title categories for entries (inherit from titles via precomputed flags)
    # we assume entry has columns: is_gocmen_title, is_multeci_title, is_siginmaci_title
    # if not, set them to False
    for cat in ["gocmen", "multeci", "siginmaci"]:
        col = f"is_{cat}_title"
        if col not in df.columns:
            df[col] = False

    groups = {
        "Syrian": df["group_syrian"],
        "Ukrainian": df["group_ukrainian"],
        "Russian": df["group_russian"],
        "GocmenTitle": df["is_gocmen_title"],
        "MulteciTitle": df["is_multeci_title"],
        "SiginmaciTitle": df["is_siginmaci_title"],
    }

    all_rows = []
    for gname, mask in groups.items():
        sub = df[mask & df["sent_ensemble_score"].notna()].copy()
        if sub.empty:
            continue

        # most negative
        neg_sorted = sub.sort_values("sent_ensemble_score").head(top_k)
        neg_sorted = neg_sorted.assign(group=gname, polarity="negative")
        # most positive
        pos_sorted = sub.sort_values("sent_ensemble_score", ascending=False).head(top_k)
        pos_sorted = pos_sorted.assign(group=gname, polarity="positive")

        all_rows.append(neg_sorted)
        all_rows.append(pos_sorted)

    if not all_rows:
        logger.warning("No entries for extreme sentiment examples.")
        return

    out_df = pd.concat(all_rows, ignore_index=True)
    # keep only useful columns
    sentiment_score_cols = [f"sent_{name}_score" for name in SENTIMENT_MODELS.keys() if f"sent_{name}_score" in out_df.columns]
    cols = [
        "group", "polarity",
        "origin_folder", "origin_label",
        "title", "title_file", "entry_id", "entry_index",
        "date", "year",
        "sent_ensemble_score",
    ] + sentiment_score_cols + [
        "clean_text",
    ]
    cols_existing = [c for c in cols if c in out_df.columns]
    out_df = out_df[cols_existing]

    out_csv = os.path.join(TABLE_DIR, "extreme_sentiment_examples.csv")
    out_df.to_csv(out_csv, index=False)
    logger.info(f"Saved extreme sentiment examples to {out_csv}")


def attach_title_flags_to_entries(entries: pd.DataFrame, titles: pd.DataFrame) -> pd.DataFrame:
    """
    Merge title-level category flags (gocmen/multeci/siginmaci) onto entries.
    """
    title_flags_cols = ["origin_folder", "title_file", "is_gocmen_title", "is_multeci_title", "is_siginmaci_title"]
    for col in ["is_gocmen_title", "is_multeci_title", "is_siginmaci_title"]:
        if col not in titles.columns:
            titles[col] = False

    title_flags = titles[title_flags_cols].copy()
    df = entries.merge(
        title_flags,
        on=["origin_folder", "title_file"],
        how="left",
    )
    for col in ["is_gocmen_title", "is_multeci_title", "is_siginmaci_title"]:
        if col not in df.columns:
            df[col] = False
        df[col] = df[col].fillna(False)
    return df


# -----------------------------------------------------------
# MAIN
# -----------------------------------------------------------

def main():
    logger.info("=== Starting advanced NLP analysis ===")
    # 1. Load / preprocess
    entries_csv = os.path.join(DATA_DIR, "entries_preprocessed.csv")
    titles_csv = os.path.join(DATA_DIR, "titles_preprocessed.csv")

    if os.path.exists(entries_csv) and os.path.exists(titles_csv):
        logger.info("Loading preprocessed entries and titles from disk.")
        entries = pd.read_csv(entries_csv)
        titles = pd.read_csv(titles_csv, parse_dates=["first_entry_date"])
    else:
        logger.info("Loading raw JSON data...")
        raw_entries = load_raw_data()
        logger.info("Preprocessing entries...")
        entries = preprocess_entries(raw_entries)
        logger.info("Building titles table...")
        titles = build_titles_df(entries)

        # normalize title text for later lexicon & category work
        titles["title_normalized"] = titles["title"].apply(normalize_text)

        # add title categories
        titles = add_title_category_flags(titles)

        # save preprocessed
        entries.to_csv(entries_csv, index=False)
        titles.to_csv(titles_csv, index=False)
        logger.info(f"Saved preprocessed entries to {entries_csv}")
        logger.info(f"Saved preprocessed titles to {titles_csv}")

    # Make sure basic fields exist if we loaded old CSVs
    if "clean_text" not in entries.columns and "raw_text" in entries.columns:
        entries["clean_text"] = entries["raw_text"].apply(normalize_text)
    if "date" not in entries.columns and "date_raw" in entries.columns:
        entries["date"] = pd.to_datetime(entries["date_raw"], errors="coerce")

    # ensure datetime for entries.date
    if not np.issubdtype(entries["date"].dtype, np.datetime64):
        entries["date"] = pd.to_datetime(entries["date"], errors="coerce")
    entries = entries.dropna(subset=["date"])
    entries["year"] = entries["date"].dt.year

    # titles sanity: ensure title_normalized and category flags exist
    if "title_normalized" not in titles.columns:
        titles["title_normalized"] = titles["title"].apply(normalize_text)
    if not all(col in titles.columns for col in ["is_gocmen_title", "is_multeci_title", "is_siginmaci_title"]):
        titles = add_title_category_flags(titles)
        titles.to_csv(titles_csv, index=False)

    # 2. Lexicon flags
    logger.info("Applying lexicon flags to entries and titles...")
    entries = apply_lexicon_flags(entries, text_col="clean_text", prefix="entry_")
    titles = apply_lexicon_flags(titles, text_col="title_normalized", prefix="title_")

    # 3. Group flags (Syrian/Ukrainian/Russian, etc.)
    logger.info("Adding group flags to entries...")
    entries = add_group_flags(entries)

    # 4. Sentiment on entries
    logger.info("Running sentiment models on entries...")
    entries = run_all_sentiment(entries, text_col="clean_text")

    # 5. Attach title category flags to entries
    entries = attach_title_flags_to_entries(entries, titles)

    # 6. Save enriched entries
    enriched_entries_csv = os.path.join(DATA_DIR, "entries_with_sentiment_and_flags.csv")
    entries.to_csv(enriched_entries_csv, index=False)
    logger.info(f"Saved enriched entries to {enriched_entries_csv}")

    # 7. Analyses
    logger.info("Analyzing sentiment timelines...")
    analyze_sentiment_timelines(entries)

    logger.info("Analyzing titles creation and sentiment...")
    analyze_titles_creation_and_sentiment(entries, titles)

    logger.info("Analyzing suri vs suriyeli...")
    analyze_suri_vs_suriyeli(entries)

    logger.info("Exporting extreme sentiment examples...")
    export_extreme_sentiment_examples(entries, top_k=10)

    logger.info("=== Finished advanced NLP analysis ===")


if __name__ == "__main__":
    main()
