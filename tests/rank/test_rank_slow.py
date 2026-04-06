import numpy as np
import pytest

from scorio import rank


@pytest.mark.slow
@pytest.mark.parametrize(
    ("name", "call"),
    [
        ("thompson", lambda R: rank.thompson(R, return_scores=True)),
        ("bayesian_mcmc", lambda R: rank.bayesian_mcmc(R, return_scores=True)),
        ("rasch_3pl", lambda R: rank.rasch_3pl(R, return_scores=True)),
        ("rasch_3pl_map", lambda R: rank.rasch_3pl_map(R, return_scores=True)),
        ("rasch_mml", lambda R: rank.rasch_mml(R, return_scores=True)),
        (
            "rasch_mml_credible",
            lambda R: rank.rasch_mml_credible(R, return_scores=True),
        ),
        ("davidson_luce", lambda R: rank.davidson_luce(R, return_scores=True)),
        ("davidson_luce_map", lambda R: rank.davidson_luce_map(R, return_scores=True)),
    ],
)
def test_rank_slow_default_parameter_smoke(
    name: str,
    call,
    ordered_binary_small_R: np.ndarray,
    rank_assertions,
) -> None:
    ranking, scores = rank_assertions.assert_ranking_and_scores(
        call(ordered_binary_small_R)
    )
    assert ranking.shape == (ordered_binary_small_R.shape[0],)
    assert scores.shape == (ordered_binary_small_R.shape[0],)
