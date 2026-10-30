import numpy as np
import pandas as pd
import pytest

from pushnyt.objective import push_scores, tradeoff


@pytest.fixture
def candidates() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n_imp, per = 50, 6
    return pd.DataFrame({"impression_id": np.repeat(np.arange(n_imp), per),
                         "p_click": rng.random(n_imp * per),
                         "novelty": rng.random(n_imp * per),
                         "label": rng.integers(0, 2, n_imp * per)})


def _top(df: pd.DataFrame, friction: float) -> pd.DataFrame:
    score = pd.Series(push_scores(df, df["p_click"].to_numpy(), friction))
    return df.loc[score.groupby(df["impression_id"]).idxmax()]


@pytest.mark.parametrize("friction", [-0.1, 1.5])
def test_friction_must_be_in_unit_interval(candidates, friction):
    with pytest.raises(ValueError):
        push_scores(candidates, candidates["p_click"].to_numpy(), friction)


def test_friction_zero_is_the_click_model(candidates):
    top = _top(candidates, 0.0)
    best = candidates.groupby("impression_id")["p_click"].max()
    assert np.allclose(top.set_index("impression_id")["p_click"], best)


def test_friction_one_is_pure_novelty(candidates):
    top = _top(candidates, 1.0)
    best = candidates.groupby("impression_id")["novelty"].max()
    assert np.allclose(top.set_index("impression_id")["novelty"], best)


def test_scores_are_scale_free(candidates):
    """Ranks make the objective ignore monotone rescaling of P(click), e.g. recalibration."""
    p = candidates["p_click"].to_numpy()
    assert np.allclose(push_scores(candidates, p, 0.3), push_scores(candidates, p ** 3 / 10, 0.3))


def test_more_friction_means_more_novel_pushes(candidates):
    curve = tradeoff(candidates, candidates["p_click"].to_numpy(), [0.0, 0.5, 1.0])
    assert curve["novelty_at_1"].is_monotonic_increasing
