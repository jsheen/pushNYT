"""Train an XGBoost click model on MIND: P(click | user, article, time slot)."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from sklearn.metrics import log_loss, roc_auc_score

from pushnyt.data import SLOTS, load_behaviors, load_entity_embeddings, load_news
from pushnyt.features import FEATURES, build_features


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def group_auc(df: pd.DataFrame, scores: np.ndarray) -> float:
    """Mean AUC within impressions that have both a click and a non-click (MIND's metric)."""
    aucs = [roc_auc_score(g["label"], g["score"])
            for _, g in df.assign(score=scores).groupby("impression_id") if g["label"].nunique() == 2]
    return float(np.mean(aucs))


def evaluate(df: pd.DataFrame, scores: np.ndarray) -> dict:
    metrics = {
        "rows": len(df),
        "ctr": float(df["label"].mean()),
        "auc": float(roc_auc_score(df["label"], scores)),
        "logloss": float(log_loss(df["label"], scores)),
        "group_auc": group_auc(df, scores),
    }
    metrics["auc_by_slot"] = {SLOTS[s]: float(roc_auc_score(g["label"], scores[g.index]))
                              for s, g in df.reset_index(drop=True).groupby("slot")}
    return metrics


def train(config: dict) -> None:
    seed = config["seed"]
    set_seed(seed)
    data_dir = Path(config["data"]["dir"])
    output_dir = Path(config["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print("loading data")
    beh = load_behaviors(data_dir / config["data"]["behaviors"])
    news = load_news(data_dir / config["data"]["news"])
    emb = load_entity_embeddings(data_dir / config["data"]["entity_embeddings"])

    print("building features")
    split = config["split"]
    rows = build_features(beh, news, emb, pd.Timestamp(split["fit_end"]), config["features"], seed)
    train_end = pd.Timestamp(split["train_end"])
    train_df = rows[rows["time"] < train_end]
    test_df = rows[rows["time"] >= train_end].reset_index(drop=True)
    # latest slice of the training period is held out for early stopping
    cut = train_df["time"].quantile(1 - config["model"]["valid_fraction"])
    fit_df, valid_df = train_df[train_df["time"] < cut], train_df[train_df["time"] >= cut]
    print(f"rows: fit={len(fit_df):,} valid={len(valid_df):,} test={len(test_df):,}")

    params = {k: v for k, v in config["model"].items() if k != "valid_fraction"}
    model = xgb.XGBClassifier(**params, tree_method="hist", eval_metric="auc",
                              random_state=seed, n_jobs=-1)
    model.fit(fit_df[FEATURES], fit_df["label"],
              eval_set=[(valid_df[FEATURES], valid_df["label"])], verbose=100)

    scores = model.predict_proba(test_df[FEATURES])[:, 1]
    metrics = evaluate(test_df, scores) | {"best_iteration": int(model.best_iteration)}
    print(json.dumps(metrics, indent=2))

    importance = pd.Series(model.get_booster().get_score(importance_type="gain"))
    importance = importance.reindex(FEATURES).fillna(0).sort_values(ascending=False)
    print(importance.round(1).to_string())

    model.save_model(output_dir / "xgb_click.json")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    importance.rename("gain").to_csv(output_dir / "feature_importance.csv")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yml")
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    train(config)


if __name__ == "__main__":
    main()
