"""
Bayesian ranking methods.

These methods rank models using posterior summaries rather than point
estimates alone.

Notation
--------

Let :math:`R \\in \\{0,1\\}^{L \\times M \\times N}`.
Each method introduces latent model parameters :math:`\\theta_l`, computes a
posterior :math:`p(\\theta \\mid R)`, and ranks with posterior scores
:math:`s_l`.

The generic form is

.. math::
    s_l = \\mathbb{E}\\!\\left[g(\\theta_l) \\mid R\\right]
    \\quad\\text{or}\\quad
    s_l = \\mathbb{Q}_q\\!\\left(g(\\theta_l) \\mid R\\right),

where :math:`\\mathbb{Q}_q` denotes a posterior quantile.
"""

import numpy as np

from scorio.utils import rank_scores

from ._base import (
    average_equivalent_scores,
    build_pairwise_wins,
    validate_input,
)
from ._types import RankMethod, RankResult


def thompson(
    R: np.ndarray,
    n_samples: int = 10000,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
    seed: int = 42,
    method: RankMethod = "competition",
    return_scores: bool = False,
) -> RankResult:
    """
    Rank models by posterior expected rank via Beta posterior sampling.

    Method context:
        Inspired by Thompson sampling, this offline estimator assumes each
        model has one latent Bernoulli success
        probability over all ``M*N`` outcomes and uses the conjugate
        Beta-Binomial posterior. Ranking score is the negative Monte Carlo
        estimate of posterior expected rank.

    Args:
        R: Binary outcome tensor with shape ``(L, M, N)`` or matrix
            ``(L, M)`` (treated as ``N=1``).
        n_samples: Positive number of posterior Monte Carlo samples ``T``.
        prior_alpha: Positive Beta prior alpha.
        prior_beta: Positive Beta prior beta.
        seed: Random seed for reproducibility.
        method: Tie-handling rule passed to :func:`scorio.utils.rank_scores`.
        return_scores: If ``True``, return ``(ranking, scores)``.

    Returns:
        Ranking array of shape ``(L,)``.
        If ``return_scores=True``, also returns
        ``scores = - E[r_l | R]`` approximations (shape ``(L,)``),
        where higher is better.

    Notation:
        ``S_l = sum_{m,n} R[l,m,n]`` and ``T_tot = M*N``.
        ``r_l^(t)`` is model ``l`` rank in posterior draw ``t``.

    Formula:
        .. math::
            p_l \\mid R \\sim
            \\mathrm{Beta}(\\alpha + S_l,\\ \\beta + T_{\\mathrm{tot}} - S_l)

        .. math::
            s_l^{\\mathrm{TS}} =
            -\\frac{1}{T}\\sum_{t=1}^{T} r_l^{(t)}

    References:
        Thompson, W. R. (1933). On the Likelihood that One Unknown Probability
        Exceeds Another in View of the Evidence of Two Samples. Biometrika.
        https://doi.org/10.1093/biomet/25.3-4.285

        Russo, D. J., et al. (2018). A Tutorial on Thompson Sampling.
        Foundations and Trends in Machine Learning.
        https://doi.org/10.1561/2200000070

        Gelman, A., et al. (2013). Bayesian Data Analysis (3rd ed.).
        https://doi.org/10.1201/b16018

    Examples:
        >>> import numpy as np
        >>> from scorio import rank
        >>> R = np.array([
        ...     [[1, 1], [1, 1]],
        ...     [[0, 0], [0, 0]],
        ... ])
        >>> ranks, scores = rank.thompson(R, n_samples=2000, return_scores=True)
        >>> ranks.tolist()
        [1, 2]

    Notes:
        If all models have identical posterior Beta parameters, the exact
        expected ranks are equal and the implementation returns equal scores
        deterministically (instead of Monte Carlo tie-breaking noise).
    """
    R = validate_input(R)
    if isinstance(n_samples, bool) or not isinstance(n_samples, (int, np.integer)):
        raise TypeError(f"n_samples must be an integer, got {type(n_samples).__name__}")
    n_samples = int(n_samples)
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")

    prior_alpha = float(prior_alpha)
    if not np.isfinite(prior_alpha) or prior_alpha <= 0.0:
        raise ValueError("prior_alpha must be > 0 and finite.")
    prior_beta = float(prior_beta)
    if not np.isfinite(prior_beta) or prior_beta <= 0.0:
        raise ValueError("prior_beta must be > 0 and finite.")

    L, M, N = R.shape
    rng = np.random.default_rng(seed)

    # Compute posterior parameters for each model.
    successes = R.reshape(L, -1).sum(axis=1).astype(float)
    total = float(M * N)
    post_alphas = prior_alpha + successes
    post_betas = prior_beta + (total - successes)

    # If posterior marginals are identical, expected ranks are exactly equal.
    if np.allclose(post_alphas, post_alphas[0]) and np.allclose(
        post_betas, post_betas[0]
    ):
        scores = np.full(L, -(L + 1) / 2.0, dtype=float)
        ranking = rank_scores(scores)[method]
        return (ranking, scores) if return_scores else ranking

    # Monte Carlo posterior expected rank.
    rank_sums = np.zeros(L)
    for _ in range(n_samples):
        samples = rng.beta(post_alphas, post_betas)
        # Rank sampled probabilities: rank 1 is best.
        ranks = np.argsort(np.argsort(-samples)) + 1
        rank_sums += ranks

    avg_ranks = rank_sums / n_samples
    # Larger score is better: negative average rank.
    scores = -avg_ranks
    scores = average_equivalent_scores(
        scores, np.column_stack((post_alphas, post_betas))
    )
    ranking = rank_scores(scores)[method]
    return (ranking, scores) if return_scores else ranking


def bayesian_mcmc(
    R: np.ndarray,
    n_samples: int = 5000,
    burnin: int = 1000,
    prior_var: float = 1.0,
    seed: int = 42,
    method: RankMethod = "competition",
    return_scores: bool = False,
) -> RankResult:
    """
    Rank models via Bayesian Bradley-Terry posterior means from MCMC.

    Method context:
        This method uses decisive pairwise counts with Bradley-Terry likelihood
        and independent Gaussian priors on log-strengths. Posterior inference is
        approximated by random-walk Metropolis-Hastings.

    Args:
        R: Binary outcome tensor with shape ``(L, M, N)`` or matrix
            ``(L, M)`` (treated as ``N=1``).
        n_samples: Positive number of retained MCMC samples.
        burnin: Nonnegative number of warmup iterations.
        prior_var: Positive Gaussian prior variance on ``theta``.
        seed: Random seed for reproducibility.
        method: Tie-handling rule passed to :func:`scorio.utils.rank_scores`.
        return_scores: If ``True``, return ``(ranking, scores)``.

    Returns:
        Ranking array of shape ``(L,)``.
        If ``return_scores=True``, also returns posterior mean
        ``E[theta|W]`` estimates (shape ``(L,)``).

    Notation:
        ``W_ij`` is decisive wins where model ``i`` beats ``j``.
        ``theta_i`` are log-strengths with ``pi_i = exp(theta_i)``.

    Formula:
        .. math::
            \\Pr(i\\succ j\\mid\\theta)=
            \\frac{\\exp(\\theta_i)}{\\exp(\\theta_i)+\\exp(\\theta_j)},\\quad
            \\theta_i\\sim\\mathcal{N}(0,\\sigma^2)

        .. math::
            s_i^{\\mathrm{MCMC}} = \\mathbb{E}[\\theta_i\\mid W]

    References:
        Bradley, R. A., & Terry, M. E. (1952). Rank Analysis of Incomplete
        Block Designs: I. The Method of Paired Comparisons. Biometrika.
        https://doi.org/10.1093/biomet/39.3-4.324

        Metropolis, N., et al. (1953). Equation of State Calculations by Fast
        Computing Machines. The Journal of Chemical Physics.
        https://doi.org/10.1063/1.1699114

        Hastings, W. K. (1970). Monte Carlo Sampling Methods Using Markov
        Chains and Their Applications. Biometrika.
        https://doi.org/10.1093/biomet/57.1.97

        Caron, F., & Doucet, A. (2012). Efficient Bayesian inference for
        generalized Bradley-Terry models.
        https://doi.org/10.1080/10618600.2012.638220

    Examples:
        >>> import numpy as np
        >>> from scorio import rank
        >>> R = np.array([
        ...     [[1, 1], [1, 1]],
        ...     [[0, 0], [0, 0]],
        ... ])
        >>> ranks, scores = rank.bayesian_mcmc(
        ...     R, n_samples=2000, burnin=500, return_scores=True
        ... )
        >>> ranks.tolist()
        [1, 2]

    Notes:
        If there are no decisive outcomes, posterior means are exactly equal
        under the symmetric Gaussian prior, and zeros are returned directly.
    """
    R = validate_input(R)
    if isinstance(n_samples, bool) or not isinstance(n_samples, (int, np.integer)):
        raise TypeError(f"n_samples must be an integer, got {type(n_samples).__name__}")
    n_samples = int(n_samples)
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")

    if isinstance(burnin, bool) or not isinstance(burnin, (int, np.integer)):
        raise TypeError(f"burnin must be an integer, got {type(burnin).__name__}")
    burnin = int(burnin)
    if burnin < 0:
        raise ValueError(f"burnin must be >= 0, got {burnin}")

    prior_var = float(prior_var)
    if not np.isfinite(prior_var) or prior_var <= 0.0:
        raise ValueError("prior_var must be > 0 and finite.")

    L = R.shape[0]
    rng = np.random.default_rng(seed)

    wins = build_pairwise_wins(R)

    if float(np.sum(wins)) <= 0.0:
        scores = np.zeros(L, dtype=float)
        ranking = rank_scores(scores)[method]
        return (ranking, scores) if return_scores else ranking

    def log_likelihood(theta):
        ll = 0.0
        for i in range(L):
            for j in range(L):
                if i == j or wins[i, j] == 0:
                    continue
                diff = theta[j] - theta[i]
                # log P(i beats j) = -log(1 + exp(θ_j - θ_i))
                if diff > 20:
                    log_p = -diff
                elif diff < -20:
                    log_p = 0.0
                else:
                    log_p = -np.log(1 + np.exp(diff))
                ll += wins[i, j] * log_p
        return ll

    def log_prior(theta):
        return -0.5 * np.sum(theta**2) / prior_var

    def log_posterior(theta):
        return log_likelihood(theta) + log_prior(theta)

    # Initialize at prior mean
    theta_current = np.zeros(L)
    log_post_current = log_posterior(theta_current)

    samples = []
    proposal_std = 0.1
    accepted = 0

    # MCMC sampling with adaptive proposal
    for iteration in range(n_samples + burnin):
        # Propose new theta
        theta_proposed = theta_current + rng.normal(0, proposal_std, L)

        log_post_proposed = log_posterior(theta_proposed)
        log_accept_prob = log_post_proposed - log_post_current

        if np.log(rng.random()) < min(log_accept_prob, 0.0):
            theta_current = theta_proposed
            log_post_current = log_post_proposed
            accepted += 1

        if iteration >= burnin:
            samples.append(theta_current.copy())

        # Adaptive proposal tuning
        if iteration > 0 and iteration % 500 == 0 and iteration < burnin:
            accept_rate = accepted / iteration
            if accept_rate < 0.2:
                proposal_std *= 0.8
            elif accept_rate > 0.5:
                proposal_std *= 1.2

    # Posterior mean estimate.
    scores = np.mean(samples, axis=0)
    scores = average_equivalent_scores(scores, R)

    ranking = rank_scores(scores)[method]
    return (ranking, scores) if return_scores else ranking


__all__ = ["thompson", "bayesian_mcmc"]
