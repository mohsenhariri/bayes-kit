import numpy as np
import pytest

from scorio import rank


def test_thompson_seed_determinism(
    ordered_binary_small_R: np.ndarray,
    rank_assertions,
) -> None:
    out1 = rank.thompson(
        ordered_binary_small_R, n_samples=1500, seed=11, return_scores=True
    )
    out2 = rank.thompson(
        ordered_binary_small_R, n_samples=1500, seed=11, return_scores=True
    )

    ranking1, scores1 = rank_assertions.assert_ranking_and_scores(out1)
    ranking2, scores2 = rank_assertions.assert_ranking_and_scores(out2)

    np.testing.assert_allclose(scores1, scores2)
    np.testing.assert_allclose(ranking1, ranking2)


def test_bayesian_mcmc_seed_determinism(
    ordered_binary_small_R: np.ndarray,
    rank_assertions,
) -> None:
    out1 = rank.bayesian_mcmc(
        ordered_binary_small_R,
        n_samples=800,
        burnin=200,
        seed=13,
        return_scores=True,
    )
    out2 = rank.bayesian_mcmc(
        ordered_binary_small_R,
        n_samples=800,
        burnin=200,
        seed=13,
        return_scores=True,
    )

    ranking1, scores1 = rank_assertions.assert_ranking_and_scores(out1)
    ranking2, scores2 = rank_assertions.assert_ranking_and_scores(out2)

    np.testing.assert_allclose(scores1, scores2)
    np.testing.assert_allclose(ranking1, ranking2)


def test_equal_information_behavior(equal_information_R: np.ndarray) -> None:
    ranking_ts, scores_ts = rank.thompson(
        equal_information_R,
        n_samples=3000,
        seed=19,
        return_scores=True,
    )
    assert np.allclose(scores_ts, scores_ts[0])
    assert np.all(ranking_ts == ranking_ts[0])

    ranking_mcmc, scores_mcmc = rank.bayesian_mcmc(
        equal_information_R,
        n_samples=700,
        burnin=100,
        seed=19,
        return_scores=True,
    )
    assert np.allclose(scores_mcmc, scores_mcmc[0])
    assert np.all(ranking_mcmc == ranking_mcmc[0])


def test_bayesian_methods_validation_errors(ordered_binary_small_R: np.ndarray) -> None:
    with pytest.raises(ValueError, match="n_samples must be >= 1"):
        rank.thompson(ordered_binary_small_R, n_samples=0)

    with pytest.raises(ValueError, match="prior_alpha must be > 0 and finite"):
        rank.thompson(ordered_binary_small_R, prior_alpha=0.0)

    with pytest.raises(ValueError, match="prior_beta must be > 0 and finite"):
        rank.thompson(ordered_binary_small_R, prior_beta=0.0)

    with pytest.raises(ValueError, match="n_samples must be >= 1"):
        rank.bayesian_mcmc(ordered_binary_small_R, n_samples=0)

    with pytest.raises(ValueError, match="burnin must be >= 0"):
        rank.bayesian_mcmc(ordered_binary_small_R, burnin=-1)

    with pytest.raises(ValueError, match="prior_var must be > 0 and finite"):
        rank.bayesian_mcmc(ordered_binary_small_R, prior_var=0.0)
