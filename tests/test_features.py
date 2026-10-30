import numpy as np
import pandas as pd
from scipy import sparse

from pushnyt.data import load_behaviors, load_entity_embeddings, load_news
from pushnyt.features import article_first_seen, build_features, subcat_novelty, temporal_features

T0 = pd.Timestamp("2019-11-09")


def test_article_first_seen():
    beh = pd.DataFrame({"time": [T0, T0 + pd.Timedelta(hours=5), T0 + pd.Timedelta(hours=9)]})
    rows = pd.DataFrame({"row": [0, 1, 2, 2], "news": [1, 1, 2, 0]})
    H = sparse.csr_matrix(np.array([[1, 0, 0, 0]]))  # article 0 is in a pre-period history
    first = article_first_seen(beh, rows, H)
    assert first[0] == T0                           # history article: capped at the log start
    assert first[1] == T0                           # shown in the first impression
    assert first[2] == T0 + pd.Timedelta(hours=9)  # first shown later
    assert np.isnat(first[3])                       # never seen


def _novelty(rows: pd.DataFrame) -> np.ndarray:
    subcat = np.array([0, 1, 2])  # article i is in subcategory i
    H = sparse.csr_matrix((1, 3))  # no history
    return subcat_novelty(rows.assign(subcat=subcat[rows["news"]]), H, subcat, 3)


def test_subcat_novelty_uses_only_earlier_impressions():
    rows = pd.DataFrame({"user": 0, "row": [0, 1, 1], "news": [0, 0, 1]})
    # impression 0: no exposure yet -> 1; impression 1: all earlier exposure was subcat 0
    assert np.allclose(_novelty(rows), [1.0, 0.0, 1.0])
    # a later impression must not change earlier values
    later = pd.concat([rows, pd.DataFrame({"user": 0, "row": [2, 2], "news": [1, 2]})], ignore_index=True)
    assert np.allclose(_novelty(later)[:3], [1.0, 0.0, 1.0])


def test_user_prior_impr_counts_earlier_impressions_in_any_slot():
    beh = pd.DataFrame({"user_id": ["a", "b", "a", "a"],
                        "time": T0 + pd.to_timedelta([6, 7, 13, 20], unit="h"),
                        "slot": [0, 0, 1, 2], "impressions": "N1-1 N2-0"})
    out = temporal_features(beh, fit_end=T0 + pd.Timedelta(days=1), prior_strength=5)
    assert out["user_prior_impr"].tolist() == [0, 0, 1, 2]


def test_build_features_on_synthetic_data(mind_dir, train_config):
    beh = load_behaviors(mind_dir / "behaviors.tsv")
    rows = build_features(beh, load_news(mind_dir / "news.tsv"),
                          load_entity_embeddings(mind_dir / "entity_embedding.vec"), None, None,
                          pd.Timestamp(train_config["split"]["fit_end"]), train_config["features"], seed=0)
    assert (rows["time"] >= pd.Timestamp(train_config["split"]["fit_end"])).all()
    assert (rows["article_age_hours"] >= 0).all()
    assert rows["novelty"].between(0, 1).all()
    cold = beh.loc[beh["history"].str.len() == 0, "user_id"].unique()
    assert (rows.loc[rows["user_id"].isin(cold), "user_history_len"] == 0).all()
    assert (rows.loc[~rows["user_id"].isin(cold), "user_history_len"] > 0).all()
