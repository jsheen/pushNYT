"""Push objective: trade predicted engagement off against subcategory novelty.

    score = (1 - friction) * engagement + friction * novelty,    friction in [0, 1]

The two halves are on very different scales: P(click) from the XGBoost model sits in a narrow
band near the base CTR (~0.04), while novelty spans [0, 1]. Adding them raw would let novelty
dominate at almost any friction. So each half is replaced by its percentile rank among the
candidates in the same impression (the set we are choosing one push from): both then mean
"share of the other candidates this one beats", and friction is a true weight between the
two rankings. friction = 0 is the pure click model; friction = 1 pushes the least-seen subcategory.

Runs separately from training (its own image): it reads a run's predictions.parquet, so friction
can be changed without retraining.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from pushnyt.plots import plot_tradeoff


def _rank(values, groups: np.ndarray) -> np.ndarray:
    """Percentile rank within each group, in (0, 1]; ties share their average rank."""
    return pd.Series(np.asarray(values)).groupby(groups).rank(pct=True).to_numpy()


def objective_parts(df: pd.DataFrame, p_click: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Engagement and novelty, each as a percentile rank within its impression."""
    imp = df["impression_id"].to_numpy()
    return _rank(p_click, imp), _rank(df["novelty"], imp)


def push_scores(df: pd.DataFrame, p_click: np.ndarray, friction: float) -> np.ndarray:
    """Objective score for each candidate; push the highest one in each impression."""
    if not 0 <= friction <= 1:
        raise ValueError(f"friction must be in [0, 1], got {friction}")
    engage, novel = objective_parts(df, p_click)
    return (1 - friction) * engage + friction * novel


def tradeoff(df: pd.DataFrame, p_click: np.ndarray, frictions: list[float]) -> pd.DataFrame:
    """What the top-ranked push per impression looks like at each friction.

    ctr_at_1: share of top picks the user actually clicked; novelty_at_1: mean novelty of
    the top pick; new_subcat_at_1: share of top picks from a subcategory never shown before.
    """
    imp = df["impression_id"].to_numpy()
    engage, novel = objective_parts(df, p_click)
    out = []
    for f in frictions:
        score = pd.Series((1 - f) * engage + f * novel)
        top = df.iloc[score.groupby(imp).idxmax().to_numpy()]
        out.append({"friction": f, "ctr_at_1": top["label"].mean(), "novelty_at_1": top["novelty"].mean(),
                    "new_subcat_at_1": (top["novelty"] == 1).mean()})
    return pd.DataFrame(out)


def random_tradeoff(df: pd.DataFrame, frictions: list[float], seed: int, draws: int = 5) -> pd.DataFrame:
    """tradeoff() with engagement replaced by a random ranking, averaged over draws."""
    rng = np.random.default_rng(seed)
    return pd.concat([tradeoff(df, rng.random(len(df)), frictions) for _ in range(draws)]
                     ).groupby("friction", as_index=False).mean()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/objective.yml")
    parser.add_argument("--run-dir", help="overrides run_dir in the config")
    parser.add_argument("--friction", type=float, help="overrides friction in the config")
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    run_dir = Path(args.run_dir or config["run_dir"])
    friction = config["friction"] if args.friction is None else args.friction

    preds = pd.read_parquet(run_dir / "predictions.parquet")
    p_click = preds["p_click"].to_numpy()
    frictions = sorted(set(config["sweep"]) | {friction})
    curve = pd.concat([tradeoff(preds, p_click, frictions).assign(policy="model"),
                       random_tradeoff(preds, frictions, config["seed"]).assign(policy="random")])
    print(curve.round(4).to_string(index=False))

    out = run_dir / "objective"
    out.mkdir(exist_ok=True)
    curve.to_csv(out / "tradeoff.csv", index=False)
    plot_tradeoff(curve, out / "tradeoff.png", friction)
    score = pd.Series(push_scores(preds, p_click, friction))
    pushes = preds.iloc[score.groupby(preds["impression_id"].to_numpy()).idxmax().to_numpy()]
    pushes.assign(score=score[pushes.index]).to_parquet(out / "pushes.parquet", index=False)
    summary = curve[(curve["policy"] == "model") & (curve["friction"] == friction)].iloc[0].to_dict()
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"friction={friction}: {len(pushes):,} pushes written to {out}")


if __name__ == "__main__":
    main()
