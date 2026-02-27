"""Pass family metrics and uncertainty estimators for binary outcomes.

Quantify performance under test-time sampling by evaluating what happens when
``k`` responses are selected per question. The module provides point estimators
for finite observed trials and Bayesian uncertainty estimators under a Beta
posterior model for per-question success probabilities.


Methods
----------
- ``pass_at_k``: probability that at least one selected trial is successful.
- ``pass_hat_k``: probability that all selected trials are successful.
- ``g_pass_at_k``: alias of ``pass_hat_k``.
- ``g_pass_at_k_tau``: probability that at least
  :math:`\\lceil \\tau k \\rceil` selected trials are successful.
- ``mg_pass_at_k``: mean generalized pass metric over
  :math:`\\tau \\in [0.5, 1.0]`.

Each metric has a companion ``*_ci`` function that returns
``(mu, sigma, lo, hi)`` under the Bayesian uncertainty model used in this
module.
"""

import math

import numpy as np
from scipy.special import betaln, comb

from .utils import _as_2d_int_matrix, _validate_binary, normal_credible_interval


def pass_at_k(R: np.ndarray, k: int) -> float:
    """
    Unbiased Pass\u0040k estimator.

    Computes the probability that at least one of *k* randomly selected
    samples is correct, averaged over all *M* questions.

    References:
        Chen, M., Tworek, J., Jun, H., et al. (2021).
        Evaluating Large Language Models Trained on Code.
        *arXiv preprint arXiv:2107.03374*.
        https://arxiv.org/abs/2107.03374

    Args:
        R: :math:`M \\times N` binary matrix with entries in :math:`\\{0, 1\\}`.
           :math:`R_{\\alpha i} = 1` if trial :math:`i` for question
           :math:`\\alpha` passed, 0 otherwise.
        k: Number of samples to select (:math:`1 \\le k \\le N`).

    Returns:
        float: The average Pass\u0040k score across all *M* questions.

    Notation:
        For each row :math:`\\alpha`:

        .. math::

            \\nu_\\alpha = \\sum_{i=1}^{N} R_{\\alpha i} \\quad \\text{(number of correct samples)}

        :math:`C(a, b)` denotes the binomial coefficient :math:`\\binom{a}{b}`.

    Formula:
        .. math::

            \\text{Pass@k}_\\alpha = 1 - \\frac{C(N - \\nu_\\alpha, k)}{C(N, k)}

        .. math::

            \\text{Pass@k} = \\frac{1}{M} \\sum_{\\alpha=1}^{M} \\text{Pass@k}_\\alpha

    Examples:
        >>> import numpy as np
        >>> R = np.array([[0, 1, 1, 0, 1],
        ...               [1, 1, 0, 1, 1]])
        >>> round(pass_at_k(R, 1), 6)
        0.7
        >>> round(pass_at_k(R, 2), 6)
        0.95
    """
    R = _as_2d_int_matrix(R)
    _validate_binary(R)
    _, N = R.shape
    if not (1 <= k <= N):
        raise ValueError(f"k must satisfy 1 <= k <= N (N={N}); got k={k}")
    nu = np.sum(R, axis=1)
    denom = comb(N, k)
    vals = 1 - comb(N - nu, k) / denom  # (M,)
    return float(np.mean(vals))


def pass_hat_k(R: np.ndarray, k: int) -> float:
    """
    Pass\u005ek (Pass-hat\u0040k): probability that all *k* selected trials
    are correct.

    Computes the probability that *k* randomly selected samples are ALL
    correct, averaged over all *M* questions.  Also known as G-Pass\u0040k.

    References:
        Yao, S., Shinn, N., Razavi, P., & Narasimhan, K. (2024).
        :math:`\\tau`-bench: A Benchmark for Tool-Agent-User Interaction
        in Real-World Domains.
        *arXiv preprint arXiv:2406.12045*.
        https://arxiv.org/abs/2406.12045

    Args:
        R: :math:`M \\times N` binary matrix with entries in :math:`\\{0, 1\\}`.
           :math:`R_{\\alpha i} = 1` if trial :math:`i` for question
           :math:`\\alpha` passed, 0 otherwise.
        k: Number of samples to select (:math:`1 \\le k \\le N`).

    Returns:
        float: The average Pass\u005ek score across all *M* questions.

    Notation:
        For each row :math:`\\alpha`:

        .. math::

            \\nu_\\alpha = \\sum_{i=1}^{N} R_{\\alpha i} \\quad \\text{(number of correct samples)}

        :math:`C(a, b)` denotes the binomial coefficient :math:`\\binom{a}{b}`.

    Formula:
        .. math::

            \\hat{\\text{Pass@k}}_\\alpha = \\frac{C(\\nu_\\alpha, k)}{C(N, k)}

        .. math::

            \\hat{\\text{Pass@k}} = \\frac{1}{M} \\sum_{\\alpha=1}^{M} \\hat{\\text{Pass@k}}_\\alpha

    Examples:
        >>> import numpy as np
        >>> R = np.array([[0, 1, 1, 0, 1],
        ...               [1, 1, 0, 1, 1]])
        >>> round(pass_hat_k(R, 1), 6)
        0.7
        >>> round(pass_hat_k(R, 2), 6)
        0.45
    """
    R = _as_2d_int_matrix(R)
    _validate_binary(R)
    _, N = R.shape
    if not (1 <= k <= N):
        raise ValueError(f"k must satisfy 1 <= k <= N (N={N}); got k={k}")
    nu = np.sum(R, axis=1)
    denom = comb(N, k)
    vals = comb(nu, k) / denom  # (M,)
    return float(np.mean(vals))


def g_pass_at_k(R: np.ndarray, k: int) -> float:
    """
    Alias for :func:`pass_hat_k`.

    Provided for compatibility with literature that uses the
    G-Pass\u0040k naming convention.
    """
    return pass_hat_k(R, k)


def g_pass_at_k_tau(R: np.ndarray, k: int, tau: float) -> float:
    """
    G-Pass\u0040k\\ :sub:`τ`: Generalized Pass\u0040k with threshold
    :math:`\\tau`.

    Computes the probability that at least
    :math:`\\lceil \\tau \\cdot k \\rceil` of *k* randomly selected samples
    are correct, averaged over all *M* questions.

    References:
        Liu, J., Liu, H., Xiao, L., et al. (2024).
        Are Your LLMs Capable of Stable Reasoning?
        *arXiv preprint arXiv:2412.13147*.
        https://arxiv.org/abs/2412.13147

    Args:
        R: :math:`M \\times N` binary matrix with entries in :math:`\\{0, 1\\}`.
           :math:`R_{\\alpha i} = 1` if trial :math:`i` for question
           :math:`\\alpha` passed, 0 otherwise.
        k: Number of samples to select (:math:`1 \\le k \\le N`).
        tau: Threshold parameter :math:`\\tau \\in [0, 1]`.  Requires at
             least :math:`\\lceil \\tau \\cdot k \\rceil` successes.
             When :math:`\\tau = 0`, equivalent to Pass\u0040k.
             When :math:`\\tau = 1`, equivalent to Pass\u005ek.

    Returns:
        float: The average G-Pass\u0040k\\ :sub:`τ` score across all
        *M* questions.

    Notation:
        For each row :math:`\\alpha`:

        .. math::

            \\nu_\\alpha = \\sum_{i=1}^{N} R_{\\alpha i} \\quad \\text{(number of correct samples)}

        :math:`C(a, b)` denotes the binomial coefficient :math:`\\binom{a}{b}`.

        :math:`j_0 = \\lceil \\tau \\cdot k \\rceil` is the minimum number of
        successes required.

    Formula:
        .. math::

            \\text{G-Pass@k}_{\\tau, \\alpha} = \\sum_{j=j_0}^{k}
                \\frac{C(\\nu_\\alpha, j) \\cdot C(N - \\nu_\\alpha, k - j)}{C(N, k)}

        .. math::

            \\text{G-Pass@k}_\\tau = \\frac{1}{M} \\sum_{\\alpha=1}^{M}
                \\text{G-Pass@k}_{\\tau, \\alpha}

    Examples:
        >>> import numpy as np
        >>> R = np.array([[0, 1, 1, 0, 1],
        ...               [1, 1, 0, 1, 1]])
        >>> round(g_pass_at_k_tau(R, 2, 0.5), 6)
        0.95
        >>> round(g_pass_at_k_tau(R, 2, 1.0), 6)
        0.45
    """
    R = _as_2d_int_matrix(R)
    _validate_binary(R)
    M, N = R.shape

    if not (0.0 <= tau <= 1.0):
        raise ValueError(f"tau must be in [0, 1]; got {tau}")
    if not (1 <= k <= N):
        raise ValueError(f"k must satisfy 1 <= k <= N (N={N}); got k={k}")

    # Edge case: if tau -> 0, return pass_at_k(R, k)
    if tau <= 0.0:
        return pass_at_k(R, k)

    nu = np.sum(R, axis=1)
    denom = comb(N, k)

    j0 = int(math.ceil(tau * k))
    if j0 > k:
        return 0.0

    vals = np.zeros(M, dtype=float)
    for j in range(j0, k + 1):
        vals += comb(nu, j) * comb(N - nu, k - j) / denom
    return float(np.mean(vals))


def mg_pass_at_k(R: np.ndarray, k: int) -> float:
    """
    mG-Pass\u0040k: mean Generalized Pass\u0040k.

    Computes the mean of G-Pass\u0040k\\ :sub:`τ` over the range
    :math:`\\tau \\in [0.5, 1.0]`, inspired by the mean Average Precision
    (mAP) metric.  This provides a comprehensive metric that integrates
    performance potential and stability across multiple thresholds.

    References:
        Liu, J., Liu, H., Xiao, L., et al. (2024).
        Are Your LLMs Capable of Stable Reasoning?
        *arXiv preprint arXiv:2412.13147*.
        https://arxiv.org/abs/2412.13147

    Args:
        R: :math:`M \\times N` binary matrix with entries in :math:`\\{0, 1\\}`.
           :math:`R_{\\alpha i} = 1` if trial :math:`i` for question
           :math:`\\alpha` passed, 0 otherwise.
        k: Number of samples to select (:math:`1 \\le k \\le N`).

    Returns:
        float: The average mG-Pass\u0040k score across all *M* questions.

    Notation:
        For each row :math:`\\alpha`:

        .. math::

            \\nu_\\alpha = \\sum_{i=1}^{N} R_{\\alpha i} \\quad \\text{(number of correct samples)}

        :math:`m = \\lceil k/2 \\rceil` is the majority threshold (the
        integration starts at :math:`\\tau = 0.5`).

    Formula:
        .. math::

            \\text{mG-Pass@k} = 2 \\int_{0.5}^{1.0} \\text{G-Pass@k}_\\tau
                \\, d\\tau

        The discrete approximation used in computation:

        .. math::

            \\text{mG-Pass@k}_\\alpha = \\frac{2}{k} \\sum_{j=m+1}^{k}
                (j - m) \\cdot P(X = j)

        where :math:`X \\sim \\text{Hypergeometric}(N, \\nu_\\alpha, k)` and
        the probability mass function is:

        .. math::

            P(X = j) = \\frac{C(\\nu_\\alpha, j) \\cdot C(N - \\nu_\\alpha, k - j)}{C(N, k)}

        The final metric is averaged over all questions:

        .. math::

            \\text{mG-Pass@k} = \\frac{1}{M} \\sum_{\\alpha=1}^{M}
                \\text{mG-Pass@k}_\\alpha

    Examples:
        >>> import numpy as np
        >>> R = np.array([[0, 1, 1, 0, 1],
        ...               [1, 1, 0, 1, 1]])
        >>> round(mg_pass_at_k(R, 2), 6)
        0.45
        >>> round(mg_pass_at_k(R, 3), 6)
        0.166667
    """
    R = _as_2d_int_matrix(R)
    _validate_binary(R)
    M, N = R.shape

    if not (1 <= k <= N):
        raise ValueError(f"k must satisfy 1 <= k <= N (N={N}); got k={k}")

    nu = np.sum(R, axis=1)
    denom = comb(N, k)

    majority = int(math.ceil(0.5 * k))
    if majority >= k:
        return 0.0

    vals = np.zeros(M, dtype=float)
    # mG per-question = (2/k) * E[(X - majority)_+], X ~ Hypergeom(N, nu, k)
    for j in range(majority + 1, k + 1):
        pmf = comb(nu, j) * comb(N - nu, k - j) / denom
        vals += (j - majority) * pmf

    vals *= 2.0 / k
    return float(np.mean(vals))


def _beta_ratio(alpha: float, beta: float, a: int, b: int) -> float:
    """Compute Beta(alpha+a, beta+b) / Beta(alpha, beta) stably."""
    return float(math.exp(betaln(alpha + a, beta + b) - betaln(alpha, beta)))


def _binary_beta_posterior_params(
    R: np.ndarray, alpha0: float = 1.0, beta0: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Per-row Beta posterior parameters for binary outcomes with Beta(alpha0,beta0) prior."""
    Rm = _as_2d_int_matrix(R)
    _validate_binary(Rm)
    _, N = Rm.shape
    c = np.sum(Rm, axis=1).astype(float)
    alpha = alpha0 + c
    beta = beta0 + (N - c)
    return alpha, beta


def _pass_at_k_bayes(
    R: np.ndarray, k: int, alpha0: float = 1.0, beta0: float = 1.0
) -> tuple[float, float]:
    """Posterior mean/std for the i.i.d. Pass@k quantity: 1 - (1 - p)^k."""
    Rm = _as_2d_int_matrix(R)
    _validate_binary(Rm)
    M, N = Rm.shape
    if not (1 <= k <= N):
        raise ValueError(f"k must satisfy 1 <= k <= N (N={N}); got k={k}")

    alpha, beta = _binary_beta_posterior_params(Rm, alpha0=alpha0, beta0=beta0)

    means = np.empty(M, dtype=float)
    vars_ = np.empty(M, dtype=float)

    # g(p) = 1 - (1-p)^k
    for i in range(M):
        a_i = float(alpha[i])
        b_i = float(beta[i])
        e_qk = _beta_ratio(a_i, b_i, 0, k)  # E[(1-p)^k]
        e_q2k = _beta_ratio(a_i, b_i, 0, 2 * k)  # E[(1-p)^(2k)]
        m = 1.0 - e_qk
        e2 = 1.0 - 2.0 * e_qk + e_q2k
        v = max(0.0, e2 - m * m)
        means[i] = m
        vars_[i] = v

    mu = float(np.mean(means))
    sigma = float(math.sqrt(float(np.sum(vars_))) / M)
    return mu, sigma


def _pass_hat_k_bayes(
    R: np.ndarray, k: int, alpha0: float = 1.0, beta0: float = 1.0
) -> tuple[float, float]:
    """Posterior mean/std for the i.i.d. Pass^k quantity: p^k."""
    Rm = _as_2d_int_matrix(R)
    _validate_binary(Rm)
    M, N = Rm.shape
    if not (1 <= k <= N):
        raise ValueError(f"k must satisfy 1 <= k <= N (N={N}); got k={k}")

    alpha, beta = _binary_beta_posterior_params(Rm, alpha0=alpha0, beta0=beta0)

    means = np.empty(M, dtype=float)
    vars_ = np.empty(M, dtype=float)

    for i in range(M):
        a_i = float(alpha[i])
        b_i = float(beta[i])
        e_pk = _beta_ratio(a_i, b_i, k, 0)  # E[p^k]
        e_p2k = _beta_ratio(a_i, b_i, 2 * k, 0)  # E[p^(2k)]
        m = e_pk
        v = max(0.0, e_p2k - m * m)
        means[i] = m
        vars_[i] = v

    mu = float(np.mean(means))
    sigma = float(math.sqrt(float(np.sum(vars_))) / M)
    return mu, sigma


def _g_pass_at_k_tau_bayes(
    R: np.ndarray, k: int, tau: float, alpha0: float = 1.0, beta0: float = 1.0
) -> tuple[float, float]:
    """Posterior mean/std for the i.i.d. G-Pass@k_τ quantity."""
    Rm = _as_2d_int_matrix(R)
    _validate_binary(Rm)
    M, N = Rm.shape

    if not (0.0 <= tau <= 1.0):
        raise ValueError(f"tau must be in [0, 1]; got {tau}")
    if not (1 <= k <= N):
        raise ValueError(f"k must satisfy 1 <= k <= N (N={N}); got k={k}")

    if tau <= 0.0:
        return _pass_at_k_bayes(Rm, k, alpha0=alpha0, beta0=beta0)
    if tau >= 1.0:
        return _pass_hat_k_bayes(Rm, k, alpha0=alpha0, beta0=beta0)

    j0 = int(math.ceil(tau * k))
    alpha, beta = _binary_beta_posterior_params(Rm, alpha0=alpha0, beta0=beta0)

    means = np.empty(M, dtype=float)
    vars_ = np.empty(M, dtype=float)

    # g(p) = Σ_{j=j0..k} C(k,j) p^j (1-p)^{k-j}
    js = list(range(j0, k + 1))
    coeff = [float(comb(k, j)) for j in js]

    for i in range(M):
        a_i = float(alpha[i])
        b_i = float(beta[i])

        m = 0.0
        for c_j, j in zip(coeff, js, strict=True):
            m += c_j * _beta_ratio(a_i, b_i, j, k - j)

        e2 = 0.0
        for idx_j, j in enumerate(js):
            c_j = coeff[idx_j]
            for idx_l, l in enumerate(js):
                c_l = coeff[idx_l]
                e2 += c_j * c_l * _beta_ratio(a_i, b_i, j + l, 2 * k - (j + l))

        v = max(0.0, e2 - m * m)
        means[i] = m
        vars_[i] = v

    mu = float(np.mean(means))
    sigma = float(math.sqrt(float(np.sum(vars_))) / M)
    return mu, sigma


def _mg_pass_at_k_bayes(
    R: np.ndarray, k: int, alpha0: float = 1.0, beta0: float = 1.0
) -> tuple[float, float]:
    """Posterior mean/std for the i.i.d. mG-Pass@k quantity."""
    Rm = _as_2d_int_matrix(R)
    _validate_binary(Rm)
    M, N = Rm.shape
    if not (1 <= k <= N):
        raise ValueError(f"k must satisfy 1 <= k <= N (N={N}); got k={k}")

    alpha, beta = _binary_beta_posterior_params(Rm, alpha0=alpha0, beta0=beta0)

    majority = int(math.ceil(0.5 * k))
    if majority >= k:
        return 0.0, 0.0

    js = list(range(majority + 1, k + 1))
    coeff = [float((2.0 / k) * (j - majority) * comb(k, j)) for j in js]

    means = np.empty(M, dtype=float)
    vars_ = np.empty(M, dtype=float)

    for i in range(M):
        a_i = float(alpha[i])
        b_i = float(beta[i])

        m = 0.0
        for c_j, j in zip(coeff, js, strict=True):
            m += c_j * _beta_ratio(a_i, b_i, j, k - j)

        e2 = 0.0
        for idx_j, j in enumerate(js):
            c_j = coeff[idx_j]
            for idx_l, l in enumerate(js):
                c_l = coeff[idx_l]
                e2 += c_j * c_l * _beta_ratio(a_i, b_i, j + l, 2 * k - (j + l))

        v = max(0.0, e2 - m * m)
        means[i] = m
        vars_[i] = v

    mu = float(np.mean(means))
    sigma = float(math.sqrt(float(np.sum(vars_))) / M)
    return mu, sigma


def pass_at_k_ci(
    R: np.ndarray,
    k: int,
    confidence: float = 0.95,
    bounds: tuple[float, float] = (0.0, 1.0),
    alpha0: float = 1.0,
    beta0: float = 1.0,
) -> tuple[float, float, float, float]:
    """
    Bayesian :math:`\\mu`, :math:`\\sigma`, and credible interval for
    i.i.d. Pass\u0040k.

    Treats each question's underlying correctness probability :math:`p` as
    latent with a :math:`\\text{Beta}(\\alpha_0, \\beta_0)` posterior and
    propagates uncertainty to the dataset-level metric.

    Args:
        R: :math:`M \\times N` binary matrix with entries in :math:`\\{0, 1\\}`.
           :math:`R_{\\alpha i} = 1` if trial :math:`i` for question
           :math:`\\alpha` passed, 0 otherwise.
        k: Number of samples to select (:math:`1 \\le k \\le N`).
        confidence: credibility level of the interval (default 0.95).
        bounds: ``(lo, hi)`` clipping bounds for the interval
                (default ``(0, 1)``).
        alpha0: Beta prior parameter :math:`\\alpha_0` (default 1).
        beta0: Beta prior parameter :math:`\\beta_0` (default 1).

    Returns:
        tuple[float, float, float, float]:
            :math:`(\\mu,\\; \\sigma,\\; \\text{lo},\\; \\text{hi})`

    Notation:
        Per-question posterior:
        :math:`p_\\alpha \\mid R \\sim \\text{Beta}(\\alpha_0 + c_\\alpha,\\;
        \\beta_0 + N - c_\\alpha)` where
        :math:`c_\\alpha = \\sum_i R_{\\alpha i}`.

    Formula:
        The per-question i.i.d. quantity is:

        .. math::

            g(p) = 1 - (1 - p)^k

        Its posterior mean and variance are:

        .. math::

            \\mathbb{E}[g(p_\\alpha)] &= 1 - \\frac{B(\\alpha_\\alpha,\\;
                \\beta_\\alpha + k)}{B(\\alpha_\\alpha, \\beta_\\alpha)}

            \\text{Var}[g(p_\\alpha)] &= \\mathbb{E}[g(p_\\alpha)^2]
                - \\mathbb{E}[g(p_\\alpha)]^2

        Dataset-level aggregation:

        .. math::

            \\mu &= \\frac{1}{M} \\sum_{\\alpha} \\mathbb{E}[g(p_\\alpha)]

            \\sigma &= \\frac{1}{M} \\sqrt{\\sum_{\\alpha}
                \\text{Var}[g(p_\\alpha)]}

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
    """
    Bayesian :math:`\\mu`, :math:`\\sigma`, and credible interval for
    i.i.d. Pass\u005ek.

    Treats each question's underlying correctness probability :math:`p` as
    latent with a :math:`\\text{Beta}(\\alpha_0, \\beta_0)` posterior and
    propagates uncertainty to the dataset-level metric.

    Args:
        R: :math:`M \\times N` binary matrix with entries in :math:`\\{0, 1\\}`.
           :math:`R_{\\alpha i} = 1` if trial :math:`i` for question
           :math:`\\alpha` passed, 0 otherwise.
        k: Number of samples to select (:math:`1 \\le k \\le N`).
        confidence: credibility level of the interval (default 0.95).
        bounds: ``(lo, hi)`` clipping bounds for the interval
                (default ``(0, 1)``).
        alpha0: Beta prior parameter :math:`\\alpha_0` (default 1).
        beta0: Beta prior parameter :math:`\\beta_0` (default 1).

    Returns:
        tuple[float, float, float, float]:
            :math:`(\\mu,\\; \\sigma,\\; \\text{lo},\\; \\text{hi})`

    Formula:
        The per-question i.i.d. quantity is:

        .. math::

            g(p) = p^k

        Its posterior mean and variance are:

        .. math::

            \\mathbb{E}[g(p_\\alpha)] &= \\frac{B(\\alpha_\\alpha + k,\\;
                \\beta_\\alpha)}{B(\\alpha_\\alpha, \\beta_\\alpha)}

            \\text{Var}[g(p_\\alpha)] &= \\mathbb{E}[g(p_\\alpha)^2]
                - \\mathbb{E}[g(p_\\alpha)]^2

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


def g_pass_at_k_ci(
    R: np.ndarray,
    k: int,
    confidence: float = 0.95,
    bounds: tuple[float, float] = (0.0, 1.0),
    alpha0: float = 1.0,
    beta0: float = 1.0,
) -> tuple[float, float, float, float]:
    """
    Alias for :func:`pass_hat_k_ci`.

    Provided for compatibility with literature that uses the
    G-Pass\u0040k naming convention.
    """
    return pass_hat_k_ci(
        R, k, confidence=confidence, bounds=bounds, alpha0=alpha0, beta0=beta0
    )


def g_pass_at_k_tau_ci(
    R: np.ndarray,
    k: int,
    tau: float,
    confidence: float = 0.95,
    bounds: tuple[float, float] = (0.0, 1.0),
    alpha0: float = 1.0,
    beta0: float = 1.0,
) -> tuple[float, float, float, float]:
    """
    Bayesian :math:`\\mu`, :math:`\\sigma`, and credible interval for
    i.i.d. G-Pass\u0040k\\ :sub:`τ`.

    Under i.i.d. sampling with per-question success probability :math:`p`,
    the quantity of interest is
    :math:`g(p) = P[\\text{Binomial}(k, p) \\ge \\lceil \\tau k \\rceil]`.

    Args:
        R: :math:`M \\times N` binary matrix with entries in :math:`\\{0, 1\\}`.
           :math:`R_{\\alpha i} = 1` if trial :math:`i` for question
           :math:`\\alpha` passed, 0 otherwise.
        k: Number of samples to select (:math:`1 \\le k \\le N`).
        tau: Threshold parameter :math:`\\tau \\in [0, 1]`.
        confidence: credibility level of the interval (default 0.95).
        bounds: ``(lo, hi)`` clipping bounds for the interval
                (default ``(0, 1)``).
        alpha0: Beta prior parameter :math:`\\alpha_0` (default 1).
        beta0: Beta prior parameter :math:`\\beta_0` (default 1).

    Returns:
        tuple[float, float, float, float]:
            :math:`(\\mu,\\; \\sigma,\\; \\text{lo},\\; \\text{hi})`

    Formula:
        The per-question i.i.d. quantity is:

        .. math::

            g(p) = \\sum_{j=j_0}^{k} \\binom{k}{j}\\, p^j (1-p)^{k-j}

        where :math:`j_0 = \\lceil \\tau \\cdot k \\rceil`.  Posterior moments
        are computed analytically via Beta-function ratios (see
        :func:`pass_at_k_ci` for the aggregation pattern).

    Examples:
        >>> import numpy as np
        >>> R = np.array([[0, 1, 1, 0, 1],
        ...               [1, 1, 0, 1, 1]])
        >>> mu, sigma, lo, hi = g_pass_at_k_tau_ci(R, 2, 0.5)
        >>> round(mu, 6), round(sigma, 6), round(lo, 4), round(hi, 4)
        (0.839286, 0.097263, 0.6487, 1.0)
        >>> mu, sigma, lo, hi = g_pass_at_k_tau_ci(R, 2, 1.0)
        >>> round(mu, 6), round(sigma, 6), round(lo, 4), round(hi, 4)
        (0.446429, 0.146167, 0.1599, 0.7329)
    """
    mu, sigma = _g_pass_at_k_tau_bayes(R, k, tau, alpha0=alpha0, beta0=beta0)
    lo, hi = normal_credible_interval(
        mu, sigma, credibility=confidence, two_sided=True, bounds=bounds
    )
    return float(mu), float(sigma), float(lo), float(hi)


def mg_pass_at_k_ci(
    R: np.ndarray,
    k: int,
    confidence: float = 0.95,
    bounds: tuple[float, float] = (0.0, 1.0),
    alpha0: float = 1.0,
    beta0: float = 1.0,
) -> tuple[float, float, float, float]:
    """
    Bayesian :math:`\\mu`, :math:`\\sigma`, and credible interval for
    i.i.d. mG-Pass\u0040k.

    Mirrors the computation of :func:`mg_pass_at_k`, but replaces the
    per-question hypergeometric distribution with
    :math:`\\text{Binomial}(k, p)` under a Beta posterior over :math:`p`.

    Args:
        R: :math:`M \\times N` binary matrix with entries in :math:`\\{0, 1\\}`.
           :math:`R_{\\alpha i} = 1` if trial :math:`i` for question
           :math:`\\alpha` passed, 0 otherwise.
        k: Number of samples to select (:math:`1 \\le k \\le N`).
        confidence: credibility level of the interval (default 0.95).
        bounds: ``(lo, hi)`` clipping bounds for the interval
                (default ``(0, 1)``).
        alpha0: Beta prior parameter :math:`\\alpha_0` (default 1).
        beta0: Beta prior parameter :math:`\\beta_0` (default 1).

    Returns:
        tuple[float, float, float, float]:
            :math:`(\\mu,\\; \\sigma,\\; \\text{lo},\\; \\text{hi})`

    Formula:
        The per-question i.i.d. quantity is:

        .. math::

            g(p) = \\frac{2}{k} \\sum_{j=m+1}^{k} (j - m)\\,
                \\binom{k}{j}\\, p^j (1-p)^{k-j}

        where :math:`m = \\lceil k/2 \\rceil`.  Posterior moments are
        computed analytically via Beta-function ratios.

    Examples:
        >>> import numpy as np
        >>> R = np.array([[0, 1, 1, 0, 1],
        ...               [1, 1, 0, 1, 1]])
        >>> mu, sigma, lo, hi = mg_pass_at_k_ci(R, 2)
        >>> round(mu, 6), round(sigma, 6), round(lo, 4), round(hi, 4)
        (0.446429, 0.146167, 0.1599, 0.7329)
        >>> mu, sigma, lo, hi = mg_pass_at_k_ci(R, 3)
        >>> round(mu, 6), round(sigma, 6), round(lo, 4), round(hi, 4)
        (0.218254, 0.098816, 0.0246, 0.4119)
    """
    mu, sigma = _mg_pass_at_k_bayes(R, k, alpha0=alpha0, beta0=beta0)
    lo, hi = normal_credible_interval(
        mu, sigma, credibility=confidence, two_sided=True, bounds=bounds
    )
    return float(mu), float(sigma), float(lo), float(hi)


__all__ = [
    # Point estimators
    "pass_at_k",
    "pass_hat_k",
    "g_pass_at_k",
    "g_pass_at_k_tau",
    "mg_pass_at_k",
    # Bayesian CI
    "pass_at_k_ci",
    "pass_hat_k_ci",
    "g_pass_at_k_ci",
    "g_pass_at_k_tau_ci",
    "mg_pass_at_k_ci",
]
