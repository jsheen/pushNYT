"""pushNYT serving API: click scores and push decisions for one training run.

Serves the run's precomputed test-set predictions (predictions.parquet), so it covers the
impressions of that run's test day. Scoring arbitrary live requests would need the per-user
features computed online (a feature store); the endpoints are shaped so that can be swapped in.

    GET  /health                       liveness + which run is loaded
    GET  /model                        run name, excluded features, test metrics
    GET  /users/{user_id}/impressions  a user's impressions (to find ids to score)
    POST /score                        P(click) and novelty for an impression's candidates
    POST /push                         the push chosen by the objective at a given friction
"""

import json
import os
from pathlib import Path

import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from pushnyt.data import SLOTS
from pushnyt.objective import objective_parts


class ScoreRequest(BaseModel):
    impression_id: int
    news_ids: list[str] | None = Field(None, description="subset of the impression's candidates; default all")


class PushRequest(ScoreRequest):
    friction: float = Field(0.2, ge=0, le=1, description="0 = pure click model, 1 = least-seen subcategory")


def create_app(run_dir: str | Path) -> FastAPI:
    run_dir = Path(run_dir)
    preds = pd.read_parquet(run_dir / "predictions.parquet")
    metrics = json.loads((run_dir / "metrics.json").read_text())
    config_path = run_dir / "config.yml"
    config = yaml.safe_load(config_path.read_text()) if config_path.exists() else {}
    positions = preds.groupby("impression_id").indices
    impressions = preds.groupby("impression_id").agg(
        user_id=("user_id", "first"), time=("time", "first"), slot=("slot", "first"),
        n_candidates=("news_id", "size")).reset_index()

    app = FastAPI(title="pushNYT", description="Push notification scoring and selection")

    def candidates(req: ScoreRequest) -> pd.DataFrame:
        if req.impression_id not in positions:
            raise HTTPException(404, f"impression {req.impression_id} is not in run {run_dir.name}")
        df = preds.iloc[positions[req.impression_id]]
        if req.news_ids is not None:
            missing = sorted(set(req.news_ids) - set(df["news_id"]))
            if missing:
                raise HTTPException(404, f"not candidates in impression {req.impression_id}: {missing}")
            df = df[df["news_id"].isin(req.news_ids)]
        return df.reset_index(drop=True)

    def header(df: pd.DataFrame) -> dict:
        row = df.iloc[0]
        return {"impression_id": int(row["impression_id"]), "user_id": row["user_id"],
                "time": row["time"].isoformat(), "slot": SLOTS[int(row["slot"])]}

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "run": run_dir.name, "impressions": len(positions)}

    @app.get("/model")
    def model() -> dict:
        return {"run": run_dir.name, "excluded_features": config.get("features", {}).get("exclude", []),
                "metrics": metrics}

    @app.get("/users/{user_id}/impressions")
    def user_impressions(user_id: str) -> list[dict]:
        rows = impressions[impressions["user_id"] == user_id]
        if rows.empty:
            raise HTTPException(404, f"user {user_id} has no impressions in run {run_dir.name}")
        return [{"impression_id": int(r.impression_id), "time": r.time.isoformat(),
                 "slot": SLOTS[int(r.slot)], "n_candidates": int(r.n_candidates)}
                for r in rows.itertuples()]

    @app.post("/score")
    def score(req: ScoreRequest) -> dict:
        df = candidates(req).sort_values("p_click", ascending=False)
        return header(df) | {"candidates": df[["news_id", "p_click", "novelty"]].to_dict("records")}

    @app.post("/push")
    def push(req: PushRequest) -> dict:
        df = candidates(req)
        engage, novel = objective_parts(df, df["p_click"].to_numpy())
        df = df.assign(engagement_rank=engage, novelty_rank=novel,
                       score=(1 - req.friction) * engage + req.friction * novel)
        ranked = df.sort_values("score", ascending=False, kind="stable")[
            ["news_id", "score", "p_click", "novelty", "engagement_rank", "novelty_rank"]]
        top = ranked.head(5).to_dict("records")
        return header(df) | {"friction": req.friction, "push": top[0], "runners_up": top[1:]}

    return app


def main() -> None:
    import uvicorn

    app = create_app(os.environ.get("PUSHNYT_RUN_DIR", "outputs/full"))
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
