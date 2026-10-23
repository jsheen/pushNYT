"""Loaders for the MIND news-recommendation dataset."""

import csv

import numpy as np
import pandas as pd

SLOTS = ("morning", "afternoon", "evening")


def time_slot(hours: pd.Series) -> np.ndarray:
    """0 = morning (05-11h), 1 = afternoon (12-16h), 2 = evening (17-04h)."""
    return np.select([(hours >= 5) & (hours < 12), (hours >= 12) & (hours < 17)], [0, 1], 2)


def load_behaviors(path) -> pd.DataFrame:
    cols = ["impression_id", "user_id", "time", "history", "impressions"]
    df = pd.read_csv(path, sep="\t", header=None, names=cols)
    df["time"] = pd.to_datetime(df["time"], format="%m/%d/%Y %I:%M:%S %p")
    df["history"] = df["history"].fillna("").str.split()
    df["slot"] = time_slot(df["time"].dt.hour)
    return df.sort_values("time", kind="stable").reset_index(drop=True)


def load_news(path) -> pd.DataFrame:
    cols = ["news_id", "category", "subcategory", "title", "abstract", "url",
            "title_entities", "abstract_entities"]
    # Titles contain bare quotes, so disable CSV quoting
    return pd.read_csv(path, sep="\t", header=None, names=cols, quoting=csv.QUOTE_NONE)


def load_entity_embeddings(path) -> dict[str, np.ndarray]:
    df = pd.read_csv(path, sep="\t", header=None, index_col=0).dropna(axis=1, how="all")
    return dict(zip(df.index, df.to_numpy(np.float32)))


def explode_impressions(behaviors: pd.DataFrame, news_idx: dict[str, int]) -> pd.DataFrame:
    """One row per (impression, candidate article) with its click label."""
    items = behaviors["impressions"].str.split()
    flat = pd.Series(np.concatenate(items.to_numpy()))
    rows = pd.DataFrame({
        "row": np.repeat(np.arange(len(behaviors)), items.str.len().to_numpy()),
        "news": flat.str[:-2].map(news_idx),
        "label": (flat.str[-1] == "1").astype(np.int8),
    })
    return rows.dropna(subset=["news"]).astype({"news": np.int32})
