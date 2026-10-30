import json

import numpy as np
import pandas as pd
import pytest
import yaml
from fastapi.testclient import TestClient

from pushnyt.api import create_app


@pytest.fixture
def client(tmp_path) -> TestClient:
    """A fake run: impression 1 has three candidates, impression 2 has two."""
    preds = pd.DataFrame({
        "impression_id": [1, 1, 1, 2, 2],
        "user_id": ["U1", "U1", "U1", "U2", "U2"],
        "time": pd.to_datetime(["2019-11-14 08:00"] * 3 + ["2019-11-14 19:00"] * 2),
        "slot": [0, 0, 0, 2, 2],
        "news_id": ["N1", "N2", "N3", "N4", "N5"],
        "label": [1, 0, 0, 0, 1],
        "p_click": [0.30, 0.10, 0.05, 0.2, 0.1],
        "novelty": [0.10, 0.50, 1.00, 0.3, 0.9],
    })
    preds.to_parquet(tmp_path / "predictions.parquet", index=False)
    (tmp_path / "metrics.json").write_text(json.dumps({"auc": 0.66}))
    (tmp_path / "config.yml").write_text(yaml.safe_dump({"features": {"exclude": ["sim_text"]}}))
    return TestClient(create_app(tmp_path))


def test_health_and_model(client):
    assert client.get("/health").json()["impressions"] == 2
    model = client.get("/model").json()
    assert model["metrics"]["auc"] == 0.66 and model["excluded_features"] == ["sim_text"]


def test_user_impressions(client):
    imps = client.get("/users/U1/impressions").json()
    assert imps == [{"impression_id": 1, "time": "2019-11-14T08:00:00", "slot": "morning", "n_candidates": 3}]
    assert client.get("/users/nobody/impressions").status_code == 404


def test_score_sorts_by_click_probability(client):
    body = client.post("/score", json={"impression_id": 1}).json()
    assert [c["news_id"] for c in body["candidates"]] == ["N1", "N2", "N3"]
    assert body["user_id"] == "U1" and body["slot"] == "morning"


@pytest.mark.parametrize("friction, expected", [(0.0, "N1"), (1.0, "N3")])
def test_push_follows_friction(client, friction, expected):
    body = client.post("/push", json={"impression_id": 1, "friction": friction}).json()
    assert body["push"]["news_id"] == expected
    assert len(body["runners_up"]) == 2
    assert np.isclose(body["push"]["score"], (1 - friction) * body["push"]["engagement_rank"]
                      + friction * body["push"]["novelty_rank"])


def test_push_on_a_subset_of_candidates(client):
    body = client.post("/push", json={"impression_id": 1, "friction": 0, "news_ids": ["N2", "N3"]}).json()
    assert body["push"]["news_id"] == "N2"


def test_bad_requests(client):
    assert client.post("/push", json={"impression_id": 1, "friction": 1.5}).status_code == 422
    assert client.post("/score", json={"impression_id": 999}).status_code == 404
    assert client.post("/score", json={"impression_id": 1, "news_ids": ["N9"]}).status_code == 404
