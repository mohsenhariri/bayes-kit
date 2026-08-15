"""Focused parity and stability tests for the AUC posterior path."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.special import betaln

from scorio.eval.auc import auc_at_k_ci
from scorio.eval.pass_at_k import pass_at_k_ci


def _auc_coefficients(k: int) -> np.ndarray:
    if k == 1:
        return np.ones(1, dtype=float)
    coefficients = np.full(k, 1.0 / (k - 1), dtype=float)
    coefficients[[0, -1]] = 0.5 / (k - 1)
    return coefficients


def _beta_ratio(alpha: float, beta: float, beta_shift: int) -> float:
    return float(math.exp(betaln(alpha, beta + beta_shift) - betaln(alpha, beta)))


def _auc_reference(
    results: np.ndarray,
    k: int,
    alpha0: float,
    beta0: float,
) -> tuple[float, float]:
    coefficients = _auc_coefficients(k)
    budgets = np.arange(1, k + 1, dtype=int)
    means = np.empty(results.shape[0], dtype=float)
    variances = np.empty(results.shape[0], dtype=float)

    for row_index, row in enumerate(results):
        alpha = alpha0 + float(np.sum(row))
        beta = beta0 + results.shape[1] - float(np.sum(row))
        q_moments = np.array(
            [_beta_ratio(alpha, beta, int(budget)) for budget in budgets]
        )
        mean = 1.0 - float(coefficients @ q_moments)
        second_moment = 1.0 - 2.0 * float(coefficients @ q_moments)
        second_moment += sum(
            coefficients[left_index]
            * coefficients[right_index]
            * _beta_ratio(alpha, beta, int(left_budget + right_budget))
            for left_index, left_budget in enumerate(budgets)
            for right_index, right_budget in enumerate(budgets)
        )
        means[row_index] = mean
        variances[row_index] = second_moment - mean * mean

    return (
        float(np.mean(means)),
        float(math.sqrt(float(np.sum(variances))) / results.shape[0]),
    )


def test_auc_posterior_matches_independent_small_closed_form() -> None:
    results = np.array(
        [
            [0, 0, 0, 0, 0],
            [1, 0, 1, 0, 0],
            [1, 1, 1, 0, 1],
            [1, 1, 1, 1, 1],
        ],
        dtype=int,
    )
    k = 4
    alpha0, beta0 = 1.25, 0.75

    mean, sigma, _, _ = auc_at_k_ci(
        results,
        k,
        alpha0=alpha0,
        beta0=beta0,
    )
    expected_mean, expected_sigma = _auc_reference(
        results,
        k,
        alpha0,
        beta0,
    )

    assert mean == pytest.approx(expected_mean, abs=2e-13)
    assert sigma == pytest.approx(expected_sigma, abs=2e-13)


def test_auc_k1_ci_remains_the_exact_pass_k1_alias() -> None:
    results = np.array([[0, 1, 1], [1, 0, 0]], dtype=int)
    kwargs = {
        "confidence": 0.8,
        "bounds": (0.1, 0.9),
        "alpha0": 0.5,
        "beta0": 2.5,
    }

    assert auc_at_k_ci(results, 1, **kwargs) == pass_at_k_ci(
        results,
        1,
        **kwargs,
    )


def test_auc_posterior_is_finite_at_k517() -> None:
    k = 517
    results = np.vstack(
        (
            np.zeros(k, dtype=int),
            np.ones(k, dtype=int),
        )
    )

    mean, sigma, lower, upper = auc_at_k_ci(results, k)

    assert np.all(np.isfinite([mean, sigma, lower, upper]))
    assert 0.0 <= mean <= 1.0
    assert 0.0 <= sigma <= 0.5
    assert 0.0 <= lower <= mean <= upper <= 1.0
