"""Stable Beta-posterior moments for count-based binary scores."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp

from ._count_score import CountScore
from ._inputs import BinaryBank, validate_beta_prior

_VARIANCE_ROUNDOFF = 64.0 * np.finfo(float).eps


def _finite_scalar(value: float, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise FloatingPointError(f"{name} must be finite; got {result!r}")
    return result


def _log_beta_power_moment(alpha: float, beta: float, power: int) -> float:
    """Return ``log(E[p**power])`` for a Beta variable by stable ratios."""
    if power == 0:
        return 0.0

    numerators = alpha + np.arange(power, dtype=float)
    ratios = beta / numerators
    terms = np.empty(power, dtype=float)
    finite = np.isfinite(ratios)
    terms[finite] = -np.log1p(ratios[finite])
    terms[~finite] = np.log(numerators[~finite]) - math.log(beta)
    return math.fsum(float(term) for term in terms)


def _log_beta_mixed_moment(
    alpha: float,
    beta: float,
    p_power: int,
    q_power: int,
) -> float:
    """Return ``log(E[p**p_power * (1-p)**q_power])`` stably."""
    return _log_beta_power_moment(
        alpha,
        beta,
        p_power,
    ) + _log_beta_power_moment(
        beta,
        alpha + p_power,
        q_power,
    )


def _positive_difference_from_logs(log_larger: float, log_smaller: float) -> float:
    """Return ``exp(log_larger) - exp(log_smaller)`` without cancellation."""
    difference = log_smaller - log_larger
    tolerance = (
        64.0
        * np.finfo(float).eps
        * max(
            1.0,
            abs(log_larger),
            abs(log_smaller),
        )
    )
    if difference > 0.0:
        if difference <= tolerance:
            difference = 0.0
        else:
            raise FloatingPointError("moment difference is materially negative")
    return math.exp(log_larger) * -math.expm1(difference)


def _power_variance(log_mean: float, log_second_moment: float) -> float:
    """Return ``Var(X)`` from log first and second moments."""
    return _positive_difference_from_logs(log_second_moment, 2.0 * log_mean)


def _endpoint_kind(score: CountScore) -> str | None:
    """Identify exact Pass or unanimity credit vectors."""
    if score.values[0] == 0.0 and np.all(score.values[1:] == 1.0):
        return "pass"
    if score.values[-1] == 1.0 and np.all(score.values[:-1] == 0.0):
        return "unanimous"
    return None


def _normalized_recurrence_weights(
    increments: NDArray[np.float64],
    *,
    name: str,
    precise_sum: bool = False,
) -> NDArray[np.float64]:
    """Normalize adjacent log-weight ratios from their numerical mode."""
    if not np.all(np.isfinite(increments)):
        raise FloatingPointError(f"non-finite {name} recurrence")

    preliminary = np.empty(increments.size + 1, dtype=float)
    preliminary[0] = 0.0
    preliminary[1:] = np.cumsum(increments)
    if not np.all(np.isfinite(preliminary)):
        raise FloatingPointError(f"non-finite {name} log weights")
    anchor = int(np.argmax(preliminary))

    log_weights = np.empty_like(preliminary)
    log_weights[anchor] = 0.0
    if anchor:
        log_weights[:anchor] = -np.cumsum(increments[:anchor][::-1])[::-1]
    if anchor < increments.size:
        log_weights[anchor + 1 :] = np.cumsum(increments[anchor:])

    normalizer = float(logsumexp(log_weights))
    if not math.isfinite(normalizer):
        raise FloatingPointError(f"non-finite {name} normalization")
    weights = np.exp(log_weights - normalizer)
    total = (
        math.fsum(float(weight) for weight in weights)
        if precise_sum
        else float(np.sum(weights))
    )
    if not math.isfinite(total) or total <= 0.0:
        raise FloatingPointError(f"non-finite {name} weights")
    weights /= total
    if not np.all(np.isfinite(weights)):
        raise FloatingPointError(f"non-finite {name} weights")
    return weights


@dataclass(frozen=True, slots=True, eq=False)
class PosteriorMoments:
    """Posterior moments for each unique bank state and their macro-average."""

    state_means: NDArray[np.float64]
    state_variances: NDArray[np.float64]
    mean: float
    variance: float


@dataclass(frozen=True, slots=True, eq=False)
class JointPosteriorMoments:
    """Marginal moments and covariance for two scores on the same bank."""

    left: PosteriorMoments
    right: PosteriorMoments
    state_covariances: NDArray[np.float64]
    covariance: float


def _bernstein_product_values(
    left: NDArray[np.float64],
    right: NDArray[np.float64],
) -> NDArray[np.float64]:
    r"""Return Bernstein coefficients for the product of two scores.

    For degrees ``k`` and ``m``, the degree-``k + m`` coefficient at total
    success count ``t`` is

    .. math::

        h_t = \sum_j
            \frac{\binom{k}{j}\binom{m}{t-j}}{\binom{k+m}{t}}
            x_j y_{t-j}.

    The fraction is a conditional hypergeometric distribution.  Computing
    and normalizing its log weights separately for each ``t`` avoids the
    single global scale that loses endpoint coefficients at high degree.
    """
    left_degree = left.size - 1
    right_degree = right.size - 1
    product_degree = left_degree + right_degree
    product = np.empty(product_degree + 1, dtype=float)

    for total in range(product_degree + 1):
        lower = max(0, total - right_degree)
        upper = min(left_degree, total)
        left_indices = np.arange(lower, upper + 1, dtype=int)
        right_indices = total - left_indices
        if left_indices.size > 1:
            indices = left_indices[:-1].astype(float)
            increments = (
                np.log(left_degree - indices)
                - np.log(indices + 1.0)
                + np.log(total - indices)
                - np.log(right_degree - total + indices + 1.0)
            )
            weights = _normalized_recurrence_weights(
                increments,
                name="Bernstein product",
            )
        else:
            weights = np.ones(1, dtype=float)
        with np.errstate(over="ignore", invalid="ignore"):
            terms = weights * left[left_indices] * right[right_indices]
        if not np.all(np.isfinite(terms)):
            raise FloatingPointError("non-finite Bernstein product terms")

        value = float(np.sum(terms))
        magnitude = float(np.sum(np.abs(terms)))
        if magnitude and abs(value) <= math.sqrt(np.finfo(float).eps) * magnitude:
            value = math.fsum(float(term) for term in terms)
        product[total] = _finite_scalar(
            value,
            name="Bernstein product coefficient",
        )
    return product


def _beta_binomial_probabilities(
    degree: int,
    alpha: float,
    beta: float,
) -> NDArray[np.float64]:
    """Return normalized Beta-binomial probabilities by a log recurrence."""
    if not degree:
        return np.ones(1, dtype=float)

    indices = np.arange(degree, dtype=float)
    increments = (
        np.log(degree - indices)
        - np.log(indices + 1.0)
        + np.log(alpha + indices)
        - np.log(beta + (degree - indices - 1.0))
    )
    return _normalized_recurrence_weights(
        increments,
        name="Beta-binomial",
        precise_sum=True,
    )


def _beta_bernstein_moment(
    values: NDArray[np.float64],
    alpha: float,
    beta: float,
) -> float:
    """Integrate Bernstein credit values against ``Beta(alpha, beta)``."""
    probabilities = _beta_binomial_probabilities(values.size - 1, alpha, beta)
    terms = values * probabilities
    if not np.all(np.isfinite(terms)):
        raise FloatingPointError("non-finite Beta-posterior moment terms")

    try:
        value = math.fsum(float(term) for term in terms)
    except OverflowError as exc:
        raise FloatingPointError("non-finite Beta-posterior moment") from exc
    return _finite_scalar(value, name="Beta-posterior moment")


def _posterior_parameters(
    bank: BinaryBank,
    alpha0: float,
    beta0: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    alpha0, beta0 = validate_beta_prior(alpha0, beta0)
    successes = bank.unique_successes.astype(float, copy=False)
    alpha = alpha0 + successes
    beta = beta0 + (bank.trial_count - successes)
    if not np.all(np.isfinite(alpha)) or not np.all(np.isfinite(beta)):
        raise FloatingPointError("non-finite Beta-posterior parameters")
    if np.any(alpha <= 0.0) or np.any(beta <= 0.0):
        raise FloatingPointError("Beta-posterior parameters must be positive")
    return alpha, beta


def _state_moments(
    score: CountScore,
    alpha: NDArray[np.float64],
    beta: NDArray[np.float64],
    endpoint: str | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if np.all(score.values == score.values[0]):
        return (
            np.full(alpha.size, score.values[0], dtype=float),
            np.zeros(alpha.size, dtype=float),
        )

    if endpoint is not None:
        means = np.empty(alpha.size, dtype=float)
        variances = np.empty(alpha.size, dtype=float)
        for index, (state_alpha, state_beta) in enumerate(
            zip(alpha, beta, strict=True)
        ):
            if endpoint == "pass":
                log_mean_power = _log_beta_power_moment(
                    float(state_beta),
                    float(state_alpha),
                    score.k,
                )
                log_second_power = _log_beta_power_moment(
                    float(state_beta),
                    float(state_alpha),
                    2 * score.k,
                )
                means[index] = -math.expm1(log_mean_power)
            else:
                log_mean_power = _log_beta_power_moment(
                    float(state_alpha),
                    float(state_beta),
                    score.k,
                )
                log_second_power = _log_beta_power_moment(
                    float(state_alpha),
                    float(state_beta),
                    2 * score.k,
                )
                means[index] = math.exp(log_mean_power)
            variances[index] = _power_variance(
                log_mean_power,
                log_second_power,
            )
        return means, variances

    squared = _bernstein_product_values(score.values, score.values)
    means = np.empty(alpha.size, dtype=float)
    variances = np.empty(alpha.size, dtype=float)
    for index, (state_alpha, state_beta) in enumerate(zip(alpha, beta, strict=True)):
        mean = _beta_bernstein_moment(
            score.values,
            float(state_alpha),
            float(state_beta),
        )
        second_moment = _beta_bernstein_moment(
            squared,
            float(state_alpha),
            float(state_beta),
        )
        means[index] = mean
        raw_variance = _finite_scalar(
            second_moment - mean * mean,
            name="posterior variance",
        )
        scale = max(1.0, abs(second_moment), abs(mean * mean))
        cancellation_threshold = (score.k + 1) * _VARIANCE_ROUNDOFF * scale
        if raw_variance > cancellation_threshold:
            variances[index] = raw_variance
            continue

        centered = score.values - mean
        centered_squared = _bernstein_product_values(
            centered,
            centered,
        )
        centered_variance = _beta_bernstein_moment(
            centered_squared,
            float(state_alpha),
            float(state_beta),
        )
        if centered_variance >= 0.0:
            variances[index] = centered_variance
        elif centered_variance >= -cancellation_threshold:
            variances[index] = 0.0
        else:
            raise FloatingPointError(
                f"posterior variance is materially negative ({centered_variance!r})"
            )
    return means, variances


def _aggregate_moments(
    bank: BinaryBank,
    state_means: NDArray[np.float64],
    state_variances: NDArray[np.float64],
) -> PosteriorMoments:
    if not np.all(np.isfinite(state_means)):
        raise FloatingPointError("state_means must be finite")
    if not np.all(np.isfinite(state_variances)):
        raise FloatingPointError("state_variances must be finite")
    if np.any(state_variances < 0.0):
        raise FloatingPointError("state_variances must be non-negative")

    frequencies = bank.frequencies.astype(float, copy=False)
    mean = _finite_scalar(
        float(np.dot(frequencies, state_means) / bank.question_count),
        name="dataset posterior mean",
    )
    variance = _finite_scalar(
        float(
            np.dot(frequencies, state_variances)
            / (bank.question_count * bank.question_count)
        ),
        name="dataset posterior variance",
    )
    if variance < 0.0:
        raise FloatingPointError("dataset posterior variance must be non-negative")
    state_means.setflags(write=False)
    state_variances.setflags(write=False)
    return PosteriorMoments(state_means, state_variances, mean, variance)


def posterior_moments(
    bank: BinaryBank,
    score: CountScore,
    alpha0: float = 1.0,
    beta0: float = 1.0,
) -> PosteriorMoments:
    """Return Beta-posterior moments for a latent count score.

    The latent budget ``score.k`` is independent of the observed trial count,
    so this function deliberately supports ``score.k > bank.trial_count``.
    """
    alpha, beta = _posterior_parameters(bank, alpha0, beta0)
    state_means, state_variances = _state_moments(
        score,
        alpha,
        beta,
        _endpoint_kind(score),
    )
    return _aggregate_moments(bank, state_means, state_variances)


def joint_posterior_moments(
    bank: BinaryBank,
    left: CountScore,
    right: CountScore,
    alpha0: float = 1.0,
    beta0: float = 1.0,
) -> JointPosteriorMoments:
    """Return marginal moments and covariance for two latent count scores."""
    alpha, beta = _posterior_parameters(bank, alpha0, beta0)

    same_score = left is right or np.array_equal(left.values, right.values)
    left_endpoint = _endpoint_kind(left)
    right_endpoint = left_endpoint if same_score else _endpoint_kind(right)

    left_means, left_variances = _state_moments(
        left,
        alpha,
        beta,
        left_endpoint,
    )
    left_moments = _aggregate_moments(bank, left_means, left_variances)
    if same_score:
        right_means = left_means
        right_variances = left_variances
        right_moments = left_moments
    else:
        right_means, right_variances = _state_moments(
            right,
            alpha,
            beta,
            right_endpoint,
        )
        right_moments = _aggregate_moments(bank, right_means, right_variances)

    if same_score:
        state_covariances = left_variances
    elif np.all(left.values == left.values[0]) or np.all(
        right.values == right.values[0]
    ):
        state_covariances = np.empty(alpha.size, dtype=float)
        state_covariances.fill(0.0)
    elif {left_endpoint, right_endpoint} == {"pass", "unanimous"}:
        state_covariances = np.empty(alpha.size, dtype=float)
        pass_score = left if left_endpoint == "pass" else right
        unanimous_score = right if pass_score is left else left
        for index, (state_alpha, state_beta) in enumerate(
            zip(alpha, beta, strict=True)
        ):
            log_q = _log_beta_power_moment(
                float(state_beta),
                float(state_alpha),
                pass_score.k,
            )
            log_p = _log_beta_power_moment(
                float(state_alpha),
                float(state_beta),
                unanimous_score.k,
            )
            log_mixed = _log_beta_mixed_moment(
                float(state_alpha),
                float(state_beta),
                unanimous_score.k,
                pass_score.k,
            )
            state_covariances[index] = _positive_difference_from_logs(
                log_p + log_q,
                log_mixed,
            )
    else:
        state_covariances = np.empty(alpha.size, dtype=float)
        product = _bernstein_product_values(left.values, right.values)
        for index, (state_alpha, state_beta) in enumerate(
            zip(alpha, beta, strict=True)
        ):
            cross_moment = _beta_bernstein_moment(
                product,
                float(state_alpha),
                float(state_beta),
            )
            raw_covariance = _finite_scalar(
                cross_moment - left_means[index] * right_means[index],
                name="posterior covariance",
            )
            scale = max(
                1.0,
                abs(cross_moment),
                abs(left_means[index] * right_means[index]),
            )
            cancellation_threshold = (left.k + right.k + 1) * _VARIANCE_ROUNDOFF * scale
            if abs(raw_covariance) > cancellation_threshold:
                state_covariances[index] = raw_covariance
                continue

            centered_left = left.values - left_means[index]
            centered_right = right.values - right_means[index]
            centered_product = _bernstein_product_values(
                centered_left,
                centered_right,
            )
            state_covariances[index] = _beta_bernstein_moment(
                centered_product,
                float(state_alpha),
                float(state_beta),
            )

    frequencies = bank.frequencies.astype(float, copy=False)
    if not np.all(np.isfinite(state_covariances)):
        raise FloatingPointError("state_covariances must be finite")
    covariance = _finite_scalar(
        float(
            np.dot(frequencies, state_covariances)
            / (bank.question_count * bank.question_count)
        ),
        name="dataset posterior covariance",
    )
    state_covariances.setflags(write=False)
    return JointPosteriorMoments(
        left_moments,
        right_moments,
        state_covariances,
        covariance,
    )


__all__ = [
    "JointPosteriorMoments",
    "PosteriorMoments",
    "joint_posterior_moments",
    "posterior_moments",
]
