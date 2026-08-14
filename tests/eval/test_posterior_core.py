"""Tests for the shared Beta-posterior count-score kernel."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import beta as beta_function
from scipy.special import betaln

from scorio.eval._count_score import CountScore
from scorio.eval._inputs import prepare_binary_bank
from scorio.eval._posterior import (
    joint_posterior_moments,
    posterior_moments,
)


def _score_value(score: CountScore, probability: float) -> float:
    complement = 1.0 - probability
    return float(
        sum(
            score.values[successes]
            * math.comb(score.k, successes)
            * probability**successes
            * complement ** (score.k - successes)
            for successes in range(score.k + 1)
        )
    )


def _quadrature_moments(
    score: CountScore,
    alpha: float,
    beta: float,
) -> tuple[float, float]:
    normalization = beta_function(alpha, beta)

    def density(probability: float) -> float:
        return float(
            probability ** (alpha - 1.0)
            * (1.0 - probability) ** (beta - 1.0)
            / normalization
        )

    mean = quad(
        lambda probability: _score_value(score, probability) * density(probability),
        0.0,
        1.0,
        epsabs=1e-13,
        epsrel=1e-13,
    )[0]
    second_moment = quad(
        lambda probability: (
            _score_value(score, probability) ** 2 * density(probability)
        ),
        0.0,
        1.0,
        epsabs=1e-13,
        epsrel=1e-13,
    )[0]
    return float(mean), float(second_moment - mean * mean)


def _quadrature_covariance(
    left: CountScore,
    right: CountScore,
    alpha: float,
    beta: float,
) -> float:
    normalization = beta_function(alpha, beta)

    def density(probability: float) -> float:
        return float(
            probability ** (alpha - 1.0)
            * (1.0 - probability) ** (beta - 1.0)
            / normalization
        )

    left_mean = _quadrature_moments(left, alpha, beta)[0]
    right_mean = _quadrature_moments(right, alpha, beta)[0]
    cross_moment = quad(
        lambda probability: (
            _score_value(left, probability)
            * _score_value(right, probability)
            * density(probability)
        ),
        0.0,
        1.0,
        epsabs=1e-13,
        epsrel=1e-13,
    )[0]
    return float(cross_moment - left_mean * right_mean)


def test_generic_score_matches_independent_quadrature_and_macro_aggregation() -> None:
    results = np.array(
        [
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [1, 1, 1, 0],
            [1, 1, 1, 1],
        ],
        dtype=int,
    )
    bank = prepare_binary_bank(results)
    score = CountScore(3, np.array([0.2, 0.7, 0.1, 0.9]))
    alpha0, beta0 = 1.25, 2.5

    actual = posterior_moments(bank, score, alpha0=alpha0, beta0=beta0)
    expected = np.array(
        [
            _quadrature_moments(
                score,
                alpha0 + successes,
                beta0 + bank.trial_count - successes,
            )
            for successes in bank.unique_successes
        ]
    )
    frequencies = bank.frequencies.astype(float)

    assert actual.state_means == pytest.approx(expected[:, 0], abs=2e-12)
    assert actual.state_variances == pytest.approx(expected[:, 1], abs=2e-12)
    assert actual.mean == pytest.approx(
        frequencies @ expected[:, 0] / bank.question_count,
        abs=2e-12,
    )
    assert actual.variance == pytest.approx(
        frequencies @ expected[:, 1] / bank.question_count**2,
        abs=2e-12,
    )
    assert not actual.state_means.flags.writeable
    assert not actual.state_variances.flags.writeable


def test_pass_and_unanimous_match_closed_form_with_latent_k_above_bank_size() -> None:
    bank = prepare_binary_bank(
        np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [1, 0, 0],
                [1, 1, 1],
            ],
            dtype=int,
        )
    )
    k = 7
    alpha0, beta0 = 0.75, 1.5

    pass_moments = posterior_moments(
        bank,
        CountScore.pass_at_k(k),
        alpha0=alpha0,
        beta0=beta0,
    )
    unanimous_moments = posterior_moments(
        bank,
        CountScore.unanimous_at_k(k),
        alpha0=alpha0,
        beta0=beta0,
    )

    alpha = alpha0 + bank.unique_successes
    beta = beta0 + bank.trial_count - bank.unique_successes
    q_k = np.exp(betaln(alpha, beta + k) - betaln(alpha, beta))
    q_2k = np.exp(betaln(alpha, beta + 2 * k) - betaln(alpha, beta))
    p_k = np.exp(betaln(alpha + k, beta) - betaln(alpha, beta))
    p_2k = np.exp(betaln(alpha + 2 * k, beta) - betaln(alpha, beta))

    assert pass_moments.state_means == pytest.approx(1.0 - q_k, abs=2e-14)
    assert pass_moments.state_variances == pytest.approx(
        q_2k - q_k**2,
        abs=2e-14,
    )
    assert unanimous_moments.state_means == pytest.approx(p_k, abs=2e-14)
    assert unanimous_moments.state_variances == pytest.approx(
        p_2k - p_k**2,
        abs=2e-14,
    )


def test_joint_moments_match_independent_cross_moment_quadrature() -> None:
    bank = prepare_binary_bank(
        np.array(
            [
                [0, 0, 0, 0],
                [1, 0, 0, 0],
                [1, 0, 0, 0],
                [1, 1, 0, 0],
                [1, 1, 1, 1],
            ],
            dtype=int,
        )
    )
    left = CountScore(2, np.array([0.0, 0.3, 1.0]))
    right = CountScore(3, np.array([0.2, 0.1, 0.8, 0.4]))
    alpha0, beta0 = 1.4, 0.8

    actual = joint_posterior_moments(
        bank,
        left,
        right,
        alpha0=alpha0,
        beta0=beta0,
    )
    expected_covariances = np.array(
        [
            _quadrature_covariance(
                left,
                right,
                alpha0 + successes,
                beta0 + bank.trial_count - successes,
            )
            for successes in bank.unique_successes
        ]
    )
    expected_left = posterior_moments(
        bank,
        left,
        alpha0=alpha0,
        beta0=beta0,
    )
    expected_right = posterior_moments(
        bank,
        right,
        alpha0=alpha0,
        beta0=beta0,
    )

    assert actual.left.state_means == pytest.approx(expected_left.state_means)
    assert actual.left.state_variances == pytest.approx(expected_left.state_variances)
    assert actual.right.state_means == pytest.approx(expected_right.state_means)
    assert actual.right.state_variances == pytest.approx(expected_right.state_variances)
    assert actual.state_covariances == pytest.approx(
        expected_covariances,
        abs=3e-12,
    )
    assert actual.covariance == pytest.approx(
        bank.frequencies @ expected_covariances / bank.question_count**2,
        abs=3e-12,
    )
    assert not actual.state_covariances.flags.writeable


def test_k517_pass_equivalent_spectrum_is_finite_and_matches_closed_form() -> None:
    bank = prepare_binary_bank(np.array([[0, 0], [0, 1], [1, 1]], dtype=int))
    k = 517
    pass_score = CountScore.pass_at_k(k)
    weights = np.zeros(k, dtype=float)
    weights[0] = 1.0
    spectrum_score = CountScore.spectrum(weights)

    pass_moments = posterior_moments(bank, pass_score)
    spectrum_moments = posterior_moments(bank, spectrum_score)
    joint = joint_posterior_moments(bank, pass_score, spectrum_score)

    alpha = 1.0 + bank.unique_successes
    beta = 1.0 + bank.trial_count - bank.unique_successes
    q_k = np.exp(betaln(alpha, beta + k) - betaln(alpha, beta))
    q_2k = np.exp(betaln(alpha, beta + 2 * k) - betaln(alpha, beta))
    expected_means = 1.0 - q_k
    expected_variances = q_2k - q_k**2

    assert np.all(np.isfinite(pass_moments.state_means))
    assert np.all(np.isfinite(pass_moments.state_variances))
    assert math.isfinite(pass_moments.mean)
    assert math.isfinite(pass_moments.variance)
    assert pass_moments.state_means == pytest.approx(expected_means, abs=5e-12)
    assert pass_moments.state_variances == pytest.approx(
        expected_variances,
        abs=2e-12,
    )
    assert spectrum_moments.state_means == pytest.approx(
        pass_moments.state_means,
        abs=0.0,
    )
    assert spectrum_moments.state_variances == pytest.approx(
        pass_moments.state_variances,
        abs=0.0,
    )
    assert joint.state_covariances == pytest.approx(
        pass_moments.state_variances,
        abs=0.0,
    )
    assert joint.covariance == pytest.approx(pass_moments.variance, abs=0.0)


def test_k1000_pass_variance_matches_independent_beta_moment_oracle() -> None:
    trial_count = 2000
    k = 1000
    bank = prepare_binary_bank(
        np.vstack(
            (
                np.ones(trial_count, dtype=int),
                np.zeros(trial_count, dtype=int),
            )
        )
    )

    actual = posterior_moments(bank, CountScore.pass_at_k(k))
    failure_posterior_beta = float(trial_count + 1)
    # The all-success q moments are below 1e-1200 and unrepresentable in
    # float64, while Beta(1, b) gives E[q^r] = b / (b + r) exactly.
    q_k = np.array([failure_posterior_beta / (failure_posterior_beta + k), 0.0])
    q_2k = np.array([failure_posterior_beta / (failure_posterior_beta + 2 * k), 0.0])
    expected_means = 1.0 - q_k
    expected_variances = q_2k - q_k**2
    expected_dataset_variance = float(np.sum(expected_variances) / 4.0)

    assert actual.state_means == pytest.approx(expected_means, abs=1e-12)
    assert actual.state_variances == pytest.approx(
        expected_variances,
        abs=3e-12,
    )
    assert actual.mean == pytest.approx(float(np.mean(expected_means)), abs=1e-12)
    assert actual.variance == pytest.approx(expected_dataset_variance, abs=1e-12)
    assert math.sqrt(actual.variance) == pytest.approx(
        0.1178265814598153,
        abs=1e-13,
    )


def test_k1000_pass_accuracy_covariance_matches_independent_beta_oracle() -> None:
    trial_count = 2000
    k = 1000
    bank = prepare_binary_bank(
        np.vstack(
            (
                np.ones(trial_count, dtype=int),
                np.zeros(trial_count, dtype=int),
            )
        )
    )
    pass_score = CountScore.pass_at_k(k)
    accuracy_score = CountScore(1, np.array([0.0, 1.0]))

    actual = joint_posterior_moments(bank, pass_score, accuracy_score)
    failure_posterior_beta = float(trial_count + 1)
    q_k = np.array([failure_posterior_beta / (failure_posterior_beta + k), 0.0])
    mean_p = np.array(
        [1.0 / (trial_count + 2.0), (trial_count + 1.0) / (trial_count + 2.0)]
    )
    p_q_k = np.array(
        [
            failure_posterior_beta
            / ((failure_posterior_beta + k) * (failure_posterior_beta + k + 1.0)),
            0.0,
        ]
    )
    expected_covariances = mean_p * q_k - p_q_k
    expected_dataset_covariance = float(np.sum(expected_covariances) / 4.0)

    assert actual.state_covariances == pytest.approx(
        expected_covariances,
        abs=2e-12,
    )
    assert actual.covariance == pytest.approx(
        expected_dataset_covariance,
        abs=1e-12,
    )


def test_k1000_geo_spectrum_delta_sigma_includes_spectrum_variance() -> None:
    trial_count = 2000
    k = 1000
    bank = prepare_binary_bank(
        np.vstack(
            (
                np.ones(trial_count, dtype=int),
                np.zeros(trial_count, dtype=int),
            )
        )
    )
    successes = np.arange(k + 1, dtype=float)
    spectrum_values = (2.0 / k) * np.maximum(successes - k / 2.0, 0.0)

    actual = joint_posterior_moments(
        bank,
        CountScore.pass_at_k(k),
        CountScore(k, spectrum_values),
    )

    # At the all-success endpoint, the upper-half spectrum is 2p - 1 up to
    # a posterior tail below 1e-280; the all-failure contribution is likewise
    # negligible.  This gives closed-form dataset moments at float64 scale.
    expected_spectrum_mean = 500.0 / 1001.0
    expected_spectrum_variance = 2001.0 / (2002.0**2 * 2003.0)
    assert actual.right.mean == pytest.approx(expected_spectrum_mean, abs=1e-14)
    assert actual.right.variance == pytest.approx(
        expected_spectrum_variance,
        abs=1e-15,
    )
    assert actual.covariance == pytest.approx(0.0, abs=1e-280)

    gradient_pass = 0.5 * math.sqrt(actual.right.mean / actual.left.mean)
    gradient_spectrum = 0.5 * math.sqrt(actual.left.mean / actual.right.mean)
    delta_variance = (
        gradient_pass**2 * actual.left.variance
        + gradient_spectrum**2 * actual.right.variance
        + 2.0 * gradient_pass * gradient_spectrum * actual.covariance
    )
    assert math.sqrt(delta_variance) == pytest.approx(
        0.05099785485421488,
        abs=1e-13,
    )
    assert actual.right.variance > 0.0


def test_nonfinite_moment_is_rejected_and_constant_variance_is_zero() -> None:
    bank = prepare_binary_bank(np.array([[0, 1]], dtype=int))
    score = CountScore(1, np.array([np.finfo(float).max, 0.0]))

    with pytest.raises(FloatingPointError, match="finite"):
        posterior_moments(bank, score)

    constant = posterior_moments(bank, CountScore(517, np.ones(518)))
    assert constant.mean == 1.0
    assert constant.variance == 0.0
    assert np.all(constant.state_variances == 0.0)


@pytest.mark.parametrize("prior", [1e14, 1e308])
def test_finite_extreme_beta_priors_do_not_lose_moment_precision(prior: float) -> None:
    symmetric_bank = prepare_binary_bank(np.array([[0, 1]], dtype=int))
    pass_moments = posterior_moments(
        symmetric_bank,
        CountScore.pass_at_k(2),
        alpha0=prior,
        beta0=prior,
    )

    inverse = 1.0 / prior
    expected_q2 = 0.5 * (1.0 + inverse) / (2.0 + inverse)
    expected_q4 = math.prod(
        (1.0 + offset * inverse) / (2.0 + offset * inverse) for offset in range(4)
    )
    assert pass_moments.mean == pytest.approx(1.0 - expected_q2, abs=2e-14)
    assert pass_moments.variance == pytest.approx(
        expected_q4 - expected_q2**2,
        abs=2e-14,
    )

    all_success_bank = prepare_binary_bank(np.array([[1]], dtype=int))
    tiny_prior_moments = posterior_moments(
        all_success_bank,
        CountScore.unanimous_at_k(1),
        alpha0=1.0,
        beta0=1e-300,
    )
    assert tiny_prior_moments.mean == 1.0
    assert tiny_prior_moments.variance >= 0.0
