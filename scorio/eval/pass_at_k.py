"""Pass family metrics and uncertainty estimators for binary outcomes.

Quantify performance under test-time sampling by evaluating what happens when
``k`` responses are selected per question. This module provides the classic
Pass@k point estimators for finite observed trials and Bayesian uncertainty
estimators under a Beta posterior model for per-question success
probabilities.

Methods
-------
- ``pass_at_k``: probability that at least one selected trial is successful.
- ``pass_hat_k``: probability that all selected trials are successful.

Each metric has a companion ``*_ci`` function that returns
``(mu, sigma, lo, hi)`` under the Bayesian uncertainty model used in this
module. Generalized G-Pass variants live in :mod:`scorio.eval.gpass`.
"""

import math

import numpy as np

from ._count_score import CountScore
from ._inputs import prepare_binary_bank, validate_finite_k
from ._posterior import posterior_moments
from .utils import normal_credible_interval


def pass_at_k(R: np.ndarray, k: int) -> float:
    r"""
    Unbiased Pass@k estimator.

    Computes the probability that at least one of *k* randomly selected
    samples is correct, averaged over all *M* questions.

    References:
        Chen, M., Tworek, J., Jun, H., et al. (2021).
        Evaluating Large Language Models Trained on Code.
        *arXiv preprint arXiv:2107.03374*.
        https://arxiv.org/abs/2107.03374

    Args:
        R: :math:`M \times N` binary matrix with entries in :math:`\{0, 1\}`.
           :math:`R_{\alpha i} = 1` if trial :math:`i` for question
           :math:`\alpha` passed, 0 otherwise.
        k: Number of samples to select (:math:`1 \le k \le N`).

    Returns:
        float: The average Pass@k score across all *M* questions.

    Notation:
        For each row :math:`\alpha`:

        .. math::

            \nu_\alpha = \sum_{i=1}^{N} R_{\alpha i} \quad \text{(number of correct samples)}

        :math:`C(a, b)` denotes the binomial coefficient :math:`\binom{a}{b}`.

    Formula:
        .. math::

            \text{Pass@k}_\alpha = 1 - \frac{C(N - \nu_\alpha, k)}{C(N, k)}

        .. math::

            \text{Pass@k} = \frac{1}{M} \sum_{\alpha=1}^{M} \text{Pass@k}_\alpha

    Examples:
        >>> import numpy as np
        >>> R = np.array([[0, 1, 1, 0, 1],
        ...               [1, 1, 0, 1, 1]])
        >>> round(pass_at_k(R, 1), 6)
        0.7
        >>> round(pass_at_k(R, 2), 6)
        0.95
    """
    bank = prepare_binary_bank(R)
    k = validate_finite_k(bank.trial_count, k)
    return CountScore.pass_at_k(k).mean(bank)


def pass_hat_k(R: np.ndarray, k: int) -> float:
    r"""
    Pass^k (Pass-hat@k): probability that all *k* selected trials
    are correct.

    Computes the probability that *k* randomly selected samples are ALL
    correct, averaged over all *M* questions. Also known as G-Pass@k.

    References:
        Yao, S., Shinn, N., Razavi, P., & Narasimhan, K. (2024).
        :math:`\tau`-bench: A Benchmark for Tool-Agent-User Interaction
        in Real-World Domains.
        *arXiv preprint arXiv:2406.12045*.
        https://arxiv.org/abs/2406.12045

    Args:
        R: :math:`M \times N` binary matrix with entries in :math:`\{0, 1\}`.
           :math:`R_{\alpha i} = 1` if trial :math:`i` for question
           :math:`\alpha` passed, 0 otherwise.
        k: Number of samples to select (:math:`1 \le k \le N`).

    Returns:
        float: The average Pass^k score across all *M* questions.

    Notation:
        For each row :math:`\alpha`:

        .. math::

            \nu_\alpha = \sum_{i=1}^{N} R_{\alpha i} \quad \text{(number of correct samples)}

        :math:`C(a, b)` denotes the binomial coefficient :math:`\binom{a}{b}`.

    Formula:
        .. math::

            \hat{\text{Pass@k}}_\alpha = \frac{C(\nu_\alpha, k)}{C(N, k)}

        .. math::

            \hat{\text{Pass@k}} = \frac{1}{M} \sum_{\alpha=1}^{M} \hat{\text{Pass@k}}_\alpha

    Examples:
        >>> import numpy as np
        >>> R = np.array([[0, 1, 1, 0, 1],
        ...               [1, 1, 0, 1, 1]])
        >>> round(pass_hat_k(R, 1), 6)
        0.7
        >>> round(pass_hat_k(R, 2), 6)
        0.45
    """
    bank = prepare_binary_bank(R)
    k = validate_finite_k(bank.trial_count, k)
    return CountScore.unanimous_at_k(k).mean(bank)


def _pass_at_k_bayes(
    R: np.ndarray, k: int, alpha0: float = 1.0, beta0: float = 1.0
) -> tuple[float, float]:
    """Posterior mean/std for the i.i.d. Pass@k quantity: 1 - (1 - p)^k."""
    bank = prepare_binary_bank(R)
    k = validate_finite_k(bank.trial_count, k)
    moments = posterior_moments(
        bank,
        CountScore.pass_at_k(k),
        alpha0=alpha0,
        beta0=beta0,
    )
    return moments.mean, float(math.sqrt(moments.variance))


def _pass_hat_k_bayes(
    R: np.ndarray, k: int, alpha0: float = 1.0, beta0: float = 1.0
) -> tuple[float, float]:
    """Posterior mean/std for the i.i.d. Pass^k quantity: p^k."""
    bank = prepare_binary_bank(R)
    k = validate_finite_k(bank.trial_count, k)
    moments = posterior_moments(
        bank,
        CountScore.unanimous_at_k(k),
        alpha0=alpha0,
        beta0=beta0,
    )
    return moments.mean, float(math.sqrt(moments.variance))


def pass_at_k_ci(
    R: np.ndarray,
    k: int,
    confidence: float = 0.95,
    bounds: tuple[float, float] = (0.0, 1.0),
    alpha0: float = 1.0,
    beta0: float = 1.0,
) -> tuple[float, float, float, float]:
    r"""
    Bayesian :math:`\mu`, :math:`\sigma`, and credible interval for
    i.i.d. Pass@k.

    Treats each question's underlying correctness probability :math:`p` as
    latent with a :math:`\text{Beta}(\alpha_0, \beta_0)` posterior and
    propagates uncertainty to the dataset-level metric.

    Args:
        R: :math:`M \times N` binary matrix with entries in :math:`\{0, 1\}`.
           :math:`R_{\alpha i} = 1` if trial :math:`i` for question
           :math:`\alpha` passed, 0 otherwise.
        k: Number of samples to select (:math:`1 \le k \le N`).
        confidence: credibility level of the interval (default 0.95).
        bounds: ``(lo, hi)`` clipping bounds for the interval
                (default ``(0, 1)``).
        alpha0: Beta prior parameter :math:`\alpha_0` (default 1).
        beta0: Beta prior parameter :math:`\beta_0` (default 1).

    Returns:
        tuple[float, float, float, float]:
            :math:`(\mu,\; \sigma,\; \text{lo},\; \text{hi})`

    Notation:
        Per-question posterior:
        :math:`p_\alpha \mid R \sim \text{Beta}(\alpha_0 + c_\alpha,\;
        \beta_0 + N - c_\alpha)` where
        :math:`c_\alpha = \sum_i R_{\alpha i}`.

    Formula:
        The per-question i.i.d. quantity is:

        .. math::

            g(p) = 1 - (1 - p)^k

        Its posterior mean and variance are:

        .. math::

            \mathbb{E}[g(p_\alpha)] &= 1 - \frac{B(\alpha_\alpha,\;
                \beta_\alpha + k)}{B(\alpha_\alpha, \beta_\alpha)}

            \text{Var}[g(p_\alpha)] &= \mathbb{E}[g(p_\alpha)^2]
                - \mathbb{E}[g(p_\alpha)]^2

        Dataset-level aggregation:

        .. math::

            \mu &= \frac{1}{M} \sum_{\alpha} \mathbb{E}[g(p_\alpha)]

            \sigma &= \frac{1}{M} \sqrt{\sum_{\alpha}
                \text{Var}[g(p_\alpha)]}

    Examples:
        >>> import numpy as np
        >>> R = np.array([[0, 1, 1, 0, 1],
        ...               [1, 1, 0, 1, 1]])
        >>> mu, sigma, lo, hi = pass_at_k_ci(R, 1)
        >>> round(mu, 6), round(sigma, 6), round(lo, 4), round(hi, 4)
        (0.642857, 0.118451, 0.4107, 0.875)
        >>> mu, sigma, lo, hi = pass_at_k_ci(R, 2)
        >>> round(mu, 6), round(sigma, 6), round(lo, 4), round(hi, 4)
        (0.839286, 0.097263, 0.6487, 1.0)
    """
    mu, sigma = _pass_at_k_bayes(R, k, alpha0=alpha0, beta0=beta0)
    lo, hi = normal_credible_interval(
        mu, sigma, credibility=confidence, two_sided=True, bounds=bounds
    )
    return float(mu), float(sigma), float(lo), float(hi)


def pass_hat_k_ci(
    R: np.ndarray,
    k: int,
    confidence: float = 0.95,
    bounds: tuple[float, float] = (0.0, 1.0),
    alpha0: float = 1.0,
    beta0: float = 1.0,
) -> tuple[float, float, float, float]:
    r"""
    Bayesian :math:`\mu`, :math:`\sigma`, and credible interval for
    i.i.d. Pass^k.

    Treats each question's underlying correctness probability :math:`p` as
    latent with a :math:`\text{Beta}(\alpha_0, \beta_0)` posterior and
    propagates uncertainty to the dataset-level metric.

    Args:
        R: :math:`M \times N` binary matrix with entries in :math:`\{0, 1\}`.
           :math:`R_{\alpha i} = 1` if trial :math:`i` for question
           :math:`\alpha` passed, 0 otherwise.
        k: Number of samples to select (:math:`1 \le k \le N`).
        confidence: credibility level of the interval (default 0.95).
        bounds: ``(lo, hi)`` clipping bounds for the interval
                (default ``(0, 1)``).
        alpha0: Beta prior parameter :math:`\alpha_0` (default 1).
        beta0: Beta prior parameter :math:`\beta_0` (default 1).

    Returns:
        tuple[float, float, float, float]:
            :math:`(\mu,\; \sigma,\; \text{lo},\; \text{hi})`

    Formula:
        The per-question i.i.d. quantity is:

        .. math::

            g(p) = p^k

        Its posterior mean and variance are:

        .. math::

            \mathbb{E}[g(p_\alpha)] &= \frac{B(\alpha_\alpha + k,\;
                \beta_\alpha)}{B(\alpha_\alpha, \beta_\alpha)}

            \text{Var}[g(p_\alpha)] &= \mathbb{E}[g(p_\alpha)^2]
                - \mathbb{E}[g(p_\alpha)]^2

    Examples:
        >>> import numpy as np
        >>> R = np.array([[0, 1, 1, 0, 1],
        ...               [1, 1, 0, 1, 1]])
        >>> mu, sigma, lo, hi = pass_hat_k_ci(R, 1)
        >>> round(mu, 6), round(sigma, 6), round(lo, 4), round(hi, 4)
        (0.642857, 0.118451, 0.4107, 0.875)
        >>> mu, sigma, lo, hi = pass_hat_k_ci(R, 2)
        >>> round(mu, 6), round(sigma, 6), round(lo, 4), round(hi, 4)
        (0.446429, 0.146167, 0.1599, 0.7329)
    """
    mu, sigma = _pass_hat_k_bayes(R, k, alpha0=alpha0, beta0=beta0)
    lo, hi = normal_credible_interval(
        mu, sigma, credibility=confidence, two_sided=True, bounds=bounds
    )
    return float(mu), float(sigma), float(lo), float(hi)


__all__ = [
    "pass_at_k",
    "pass_hat_k",
    "pass_at_k_ci",
    "pass_hat_k_ci",
]
