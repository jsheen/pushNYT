"""Features for predicting a click on (user, article) at a time slot (morning/afternoon/evening).

Four feature groups:
  - time-slot engagement: how the user has engaged in this slot before
  - similarity: candidate article vs. the user's reading history
  - complementarity: Alibaba-style s_ab = p_ab - q_ab over subcategories
    (https://eugeneyan.com/writing/push/), where p = "clicked b after reading a"
    and q = "clicked b instead of a" (a shown and skipped in the same impression)
  - push fatigue: articles recently shown and skipped in this slot, plus overall volume

Every per-user temporal feature uses only impressions strictly before the current one.
"""

import json

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from pushnyt.data import explode_impressions

FEATURES = [
    # time-slot engagement
    "slot", "slot_prior_impr", "slot_prior_clicks", "slot_engaged_rate", "slot_click_share",
    # similarity
    "sim_text", "sim_entity", "subcat_affinity",
    # complementarity
    "complement_score",
    # push fatigue
    "slot_skipped_72h", "slot_skip_rate_72h", "impr_last_24h",
    "hours_since_last_impr", "hours_since_last_slot_impr",
]


def text_vectors(news: pd.DataFrame, dim: int, seed: int) -> np.ndarray:
    text = news["title"].fillna("") + " " + news["abstract"].fillna("")
    tfidf = TfidfVectorizer(stop_words="english", min_df=2, max_features=50_000,
                            sublinear_tf=True).fit_transform(text)
    return normalize(TruncatedSVD(dim, random_state=seed).fit_transform(tfidf)).astype(np.float32)


def entity_vectors(news: pd.DataFrame, emb: dict[str, np.ndarray]) -> np.ndarray:
    """Mean TransE embedding of the entities in each article's title and abstract."""
    out = np.zeros((len(news), len(next(iter(emb.values())))), np.float32)
    for i, cols in enumerate(zip(news["title_entities"], news["abstract_entities"])):
        vecs = [emb[e["WikidataId"]] for col in cols if isinstance(col, str)
                for e in json.loads(col) if e["WikidataId"] in emb]
        if vecs:
            out[i] = np.mean(vecs, axis=0)
    return normalize(out)


def history_matrix(behaviors: pd.DataFrame, news_idx: dict[str, int]):
    """Sparse users x articles matrix of each user's pre-period click history."""
    users = behaviors.drop_duplicates("user_id")
    user_idx = dict(zip(users["user_id"], range(len(users))))
    rows, cols = [], []
    for u, hist in enumerate(users["history"]):
        ids = [news_idx[n] for n in hist if n in news_idx]
        rows += [u] * len(ids)
        cols += ids
    H = sparse.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)),
                          shape=(len(users), len(news_idx)))
    return H, user_idx


def _onehot(codes: np.ndarray, n: int) -> sparse.csr_matrix:
    return sparse.csr_matrix((np.ones(len(codes), np.float32), (np.arange(len(codes)), codes)),
                             shape=(len(codes), n))


def _pmi(counts: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    n = counts + alpha
    return np.log(n * n.sum() / (n.sum(1, keepdims=True) * n.sum(0, keepdims=True)))


def fit_complements(rows: pd.DataFrame, has_subcat: sparse.csr_matrix, n_sub: int) -> np.ndarray:
    """Subcategory complement matrix s[a, b] = pmi(p)[a, b] - pmi(q)[a, b], diagonal zeroed."""
    clicked = rows[rows["label"] == 1]
    # p: user had read subcategory a, then clicked subcategory b
    p = (has_subcat[clicked["user"].to_numpy()].T @ _onehot(clicked["subcat"].to_numpy(), n_sub))
    # q: a was shown and skipped while b was clicked in the same impression
    imp = rows["row"].factorize()[0]
    by_imp = lambda mask: sparse.csr_matrix(
        (np.ones(mask.sum(), np.float32), (imp[mask], rows["subcat"].to_numpy()[mask])),
        shape=(imp.max() + 1, n_sub))
    q = by_imp(rows["label"].to_numpy() == 0).T @ by_imp(rows["label"].to_numpy() == 1)
    s = _pmi(p.toarray()) - _pmi(q.toarray())
    np.fill_diagonal(s, 0.0)  # same-subcategory overlap is similarity, not complementarity
    return s.astype(np.float32)


def _window_sum(keys: np.ndarray, t: np.ndarray, window: int, weights: np.ndarray) -> np.ndarray:
    """Sum of weights over earlier rows with the same key and time in [t - window, t)."""
    order = np.lexsort((t, keys))
    combined = keys[order].astype(np.int64) * (1 << 40) + t[order]
    lo = np.searchsorted(combined, combined - window, "left")
    hi = np.searchsorted(combined, combined, "left")
    cum = np.concatenate([[0.0], np.cumsum(weights[order])])
    out = np.empty(len(keys))
    out[order] = cum[hi] - cum[lo]
    return out


def temporal_features(beh: pd.DataFrame, fit_end: pd.Timestamp, prior_strength: float) -> pd.DataFrame:
    """Per-impression time-slot engagement and fatigue, using only earlier impressions."""
    # Every MIND impression contains at least one click, so engagement and fatigue
    # are measured per article shown rather than per impression.
    df = beh[["user_id", "time", "slot"]].copy()
    df["n_clicks"] = beh["impressions"].str.count("-1")
    df["n_shown"] = beh["impressions"].str.count(" ") + 1
    df["n_skipped"] = df["n_shown"] - df["n_clicks"]
    fit = df["time"] < fit_end
    base_rate = df.loc[fit, "n_clicks"].sum() / df.loc[fit, "n_shown"].sum()

    by_slot = df.groupby(["user_id", "slot"], sort=False)
    by_user = df.groupby("user_id", sort=False)
    shown_before = by_slot["n_shown"].cumsum() - df["n_shown"]
    user_clicks_before = by_user["n_clicks"].cumsum() - df["n_clicks"]

    out = pd.DataFrame(index=df.index)
    out["slot_prior_impr"] = by_slot.cumcount()
    out["slot_prior_clicks"] = by_slot["n_clicks"].cumsum() - df["n_clicks"]
    out["slot_engaged_rate"] = ((out["slot_prior_clicks"] + base_rate * prior_strength)
                                / (shown_before + prior_strength))
    out["slot_click_share"] = out["slot_prior_clicks"] / user_clicks_before.replace(0, np.nan)

    user = df["user_id"].factorize()[0]
    slot_key = user * 3 + df["slot"].to_numpy()
    secs = ((df["time"] - df["time"].min()).dt.total_seconds()).to_numpy().astype(np.int64)
    window = 72 * 3600
    out["slot_skipped_72h"] = _window_sum(slot_key, secs, window, df["n_skipped"].to_numpy(float))
    shown_72h = _window_sum(slot_key, secs, window, df["n_shown"].to_numpy(float))
    out["slot_skip_rate_72h"] = out["slot_skipped_72h"] / np.where(shown_72h > 0, shown_72h, np.nan)
    out["impr_last_24h"] = _window_sum(user, secs, 24 * 3600, np.ones(len(df)))
    out["hours_since_last_impr"] = by_user["time"].diff().dt.total_seconds() / 3600
    out["hours_since_last_slot_impr"] = by_slot["time"].diff().dt.total_seconds() / 3600
    return out


def build_features(beh: pd.DataFrame, news: pd.DataFrame, entity_emb: dict[str, np.ndarray],
                   fit_end: pd.Timestamp, cfg: dict, seed: int) -> pd.DataFrame:
    """Feature rows for every candidate shown at or after `fit_end`.

    Impressions before `fit_end` are used only to fit the complement matrix
    (and as earlier history for the temporal features).
    """
    news_idx = dict(zip(news["news_id"], range(len(news))))
    subcat, subcat_names = news["subcategory"].factorize()
    n_sub = len(subcat_names)
    text_vec = text_vectors(news, cfg["text_dim"], seed)
    ent_vec = entity_vectors(news, entity_emb)

    H, user_idx = history_matrix(beh, news_idx)
    profile_text = normalize(H @ text_vec)
    profile_ent = normalize(H @ ent_vec)
    subcat_weights = normalize(H @ _onehot(subcat, n_sub), norm="l1").toarray()
    has_history = np.asarray(H.sum(1)).ravel() > 0

    rows = explode_impressions(beh, news_idx)
    rows["user"] = beh["user_id"].map(user_idx).to_numpy()[rows["row"]]
    rows["subcat"] = subcat[rows["news"]]
    is_fit = (beh["time"] < fit_end).to_numpy()[rows["row"]]

    s = fit_complements(rows[is_fit], sparse.csr_matrix(subcat_weights > 0), n_sub)
    complement_by_user = subcat_weights @ s  # weighted mean of s[a, b] over the user's subcategories a

    rows = rows[~is_fit].reset_index(drop=True)
    u, a, b = rows["user"].to_numpy(), rows["news"].to_numpy(), rows["subcat"].to_numpy()
    no_hist = ~has_history[u]

    rows["sim_text"] = np.einsum("ij,ij->i", profile_text[u], text_vec[a])
    rows["sim_entity"] = np.einsum("ij,ij->i", profile_ent[u], ent_vec[a])
    rows["subcat_affinity"] = subcat_weights[u, b]
    other_weight = 1.0 - subcat_weights[u, b]  # exclude the candidate's own subcategory
    rows["complement_score"] = complement_by_user[u, b] / np.where(other_weight > 1e-6, other_weight, np.nan)
    rows.loc[no_hist, ["sim_text", "sim_entity", "subcat_affinity", "complement_score"]] = np.nan
    rows.loc[rows["sim_entity"] == 0, "sim_entity"] = np.nan  # article or history has no entities

    temporal = temporal_features(beh, fit_end, cfg["ctr_prior_strength"])
    rows = rows.join(temporal, on="row")
    rows["slot"] = beh["slot"].to_numpy()[rows["row"]]
    rows["impression_id"] = beh["impression_id"].to_numpy()[rows["row"]]
    rows["time"] = beh["time"].to_numpy()[rows["row"]]
    return rows
