from __future__ import annotations

from fractions import Fraction
from itertools import combinations

import numpy as np
import pytest

from scorio import eval as scorio_eval


def _brute_force_max(R: np.ndarray, k: int, weights: np.ndarray) -> float:
    row_scores = []
    for row in R:
        rewards = weights[row]
        maxima = [
            max(rewards[list(indices)])
            for indices in combinations(range(R.shape[1]), k)
        ]
        row_scores.append(float(np.mean(maxima)))
    return float(np.mean(row_scores))


@pytest.mark.parametrize("k", (1, 2, 5, 7))
def test_max_at_k_matches_bruteforce_with_repeated_reward_levels(k: int) -> None:
    R = np.array(
        [
            [0, 1, 2, 3, 0, 2, 1],
            [3, 3, 1, 0, 2, 2, 0],
            [1, 0, 1, 2, 3, 1, 2],
        ],
        dtype=int,
    )
    weights = np.array([0.5, -1.0, 0.5, 2.0])

    expected = _brute_force_max(R, k, weights)

    assert scorio_eval.max_at_k(R, k, weights) == pytest.approx(expected)


def test_large_binary_max_matches_pass_for_mixed_success_counts() -> None:
    trial_count = 2000
    k = 1000
    R = np.zeros((4, trial_count), dtype=int)
    R[0, 0] = 1
    R[1, : trial_count // 2] = 1
    R[3, :] = 1

    max_score = scorio_eval.max_at_k(R, k)
    pass_score = scorio_eval.pass_at_k(R, k)

    assert np.isfinite(max_score)
    assert max_score == pytest.approx(pass_score, abs=1e-12)


def test_large_sparse_binary_max_preserves_rare_success_probability() -> None:
    trial_count = 2_000_000
    R = np.zeros((1, trial_count), dtype=np.uint8)
    R[0, 0] = 1

    score = scorio_eval.max_at_k(R, 2)

    assert score == pytest.approx(2.0 / trial_count, rel=1e-10)


def test_max_ci_preserves_rare_posterior_uncertainty() -> None:
    trial_count = 1_000_000
    R = np.zeros((1, trial_count), dtype=np.uint8)
    R[0, 0] = 1

    mu, sigma, _, _ = scorio_eval.max_at_k_ci(R, 2)

    alpha = trial_count
    beta = 2

    def beta_power(power: int) -> Fraction:
        moment = Fraction(1)
        for offset in range(power):
            moment *= Fraction(alpha + offset, alpha + beta + offset)
        return moment

    second = beta_power(2)
    fourth = beta_power(4)
    expected_mu = float(1 - second)
    expected_sigma = float(fourth - second * second) ** 0.5
    assert mu == pytest.approx(expected_mu, rel=1e-12)
    assert sigma == pytest.approx(expected_sigma, rel=1e-10)


def test_max_ci_uncertainty_is_translation_invariant() -> None:
    R = np.array([[0, 1]], dtype=int)
    base = scorio_eval.max_at_k_ci(R, 2, w=np.array([0.0, 1.0]))
    shift = 1.0e12
    translated = scorio_eval.max_at_k_ci(
        R,
        2,
        w=np.array([shift, shift + 1.0]),
    )

    assert translated[0] == pytest.approx(base[0] + shift)
    assert translated[1] == pytest.approx(base[1], rel=1e-14)


def test_categorical_metrics_scale_without_variance_overflow() -> None:
    R = np.array([[0, 1]], dtype=int)
    base_weights = np.array([0.0, 1.0])
    scale = 1.0e155
    scaled_weights = scale * base_weights

    pairs = [
        (scorio_eval.bayes(R, base_weights), scorio_eval.bayes(R, scaled_weights)),
        (scorio_eval.avg(R, base_weights), scorio_eval.avg(R, scaled_weights)),
        (
            scorio_eval.max_at_k_ci(R, 2, base_weights)[:2],
            scorio_eval.max_at_k_ci(R, 2, scaled_weights)[:2],
        ),
    ]
    for base, scaled in pairs:
        assert np.all(np.isfinite(scaled))
        np.testing.assert_allclose(scaled, np.asarray(base) * scale, rtol=1e-14)

    assert scorio_eval.max_at_k(R, 2, scaled_weights) == pytest.approx(scale)


def test_bayes_and_max_ci_preserve_flat_prior_reshape() -> None:
    R = np.array([[0, 1, 2], [2, 1, 0]], dtype=int)
    weights = np.array([0.0, 0.25, 1.0])
    flat_prior = np.array([2, 1, 0, 2], dtype=int)
    matrix_prior = flat_prior.reshape(R.shape[0], -1)

    np.testing.assert_allclose(
        scorio_eval.bayes(R, weights, flat_prior),
        scorio_eval.bayes(R, weights, matrix_prior),
    )
    np.testing.assert_allclose(
        scorio_eval.max_at_k_ci(R, 3, weights, flat_prior),
        scorio_eval.max_at_k_ci(R, 3, weights, matrix_prior),
    )


@pytest.mark.parametrize("invalid_k", (True, np.bool_(True), 1.0, np.float64(1.0)))
def test_max_metrics_require_integral_k(invalid_k: object) -> None:
    R = np.array([[0, 1, 1]], dtype=int)

    with pytest.raises(ValueError, match="k must be an integer"):
        scorio_eval.max_at_k(R, invalid_k)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="k must be an integer"):
        scorio_eval.max_at_k_ci(R, invalid_k)  # type: ignore[arg-type]
