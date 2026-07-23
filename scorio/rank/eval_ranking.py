"""
Evaluation-metric ranking methods.

These methods map each model's responses to a scalar score and then convert
scores to ranks with :func:`scorio.utils.rank_scores`.

Notation
--------

Let :math:`R \\in \\{0,1,\\ldots,C\\}^{L \\times M \\times N}` denote model
outcomes, and define per-question correct-count summaries
:math:`k_{lm}=\\sum_{n=1}^{N} R_{lmn}` when outcomes are binary.

The module follows the score template

.. math::
    s_l = \\frac{1}{M}\\sum_{m=1}^{M} g_m(k_{lm}, N; \\psi),

where :math:`g_m` depends on the selected evaluation metric.
"""

import numpy as np
from scipy.stats import norm

from scorio import eval
from scorio.utils import rank_scores

from ._base import validate_input
from ._types import RankMethod, RankResult


def avg(
    R: np.ndarray,
    method: RankMethod = "competition",
    return_scores: bool = False,
) -> RankResult:
    """
    Rank models by mean accuracy over all questions and trials.

    Method context:
        This is the simplest pointwise ranking baseline: each model receives
        one score equal to its empirical success rate across all ``M * N``
        outcomes.

    Args:
        R: Binary outcome tensor with shape ``(L, M, N)`` or matrix
            ``(L, M)`` (treated as ``N=1``).
        method: Tie-handling rule passed to ``rank_scores``.
            One of ``"competition"``, ``"dense"``, ``"avg"``,
            ``"competition_max"``.
        return_scores: If ``True``, return ``(ranking, scores)``.

    Returns:
        Ranking array of shape ``(L,)``.
        If ``return_scores=True``, also returns ``scores`` of shape ``(L,)``.

    Notation:
        ``R[l, m, n]`` is the binary outcome for model ``l``, question ``m``,
        trial ``n``.

    Formula:
        .. math::
            s_l^{\\mathrm{avg}} = \\frac{1}{MN}
            \\sum_{m=1}^{M}\\sum_{n=1}^{N} R_{lmn}

    Examples:
        >>> import numpy as np
        >>> from scorio import rank
        >>> R = np.array([
        ...     [[1, 1], [0, 1]],
        ...     [[1, 0], [0, 0]],
        ... ])
        >>> ranks, scores = rank.avg(R, return_scores=True)
        >>> scores.round(3).tolist()
        [0.75, 0.25]
        >>> ranks.tolist()
        [1, 2]


    """
    R = validate_input(R)
    L, _, _ = R.shape
    scores = np.array([eval.avg(R[model, :, :])[0] for model in range(L)])
    ranking = rank_scores(scores)[method]
    return (ranking, scores) if return_scores else ranking


def bayes(
    R: np.ndarray,
    w: np.ndarray | None = None,
    R0: np.ndarray | None = None,
    quantile: float | None = None,
    method: RankMethod = "competition",
    return_scores: bool = False,
) -> RankResult:
    """
    Rank models with Bayes@N posterior statistics.

    Method context:
        For each model, this method computes Bayes@N posterior summary
        statistics ``(mu_l, sigma_l)`` from categorical outcomes. Ranking can
        be based on posterior mean (default) or a Normal-quantile conservative
        score.

    References:
        Hariri, M., Samandar, A., Hinczewski, M., & Chaudhary, V. (2026).
        Don't Pass@k: A Bayesian Framework for Large Language Model Evaluation.
        *ICLR 2026*.
        https://openreview.net/forum?id=PTXi3Ef4sT

    Args:
        R: Categorical outcome tensor with shape ``(L, M, N)`` or matrix
            ``(L, M)`` (treated as ``N=1``). Entries must be integers in
            ``{0, ..., C}``.
        w: Weight vector of shape ``(C+1,)`` mapping categories to scores.
            If not provided and R is binary (contains only 0 and 1), defaults
            to ``[0, 1]``. For non-binary R, w is required.
        R0: Optional prior outcomes. Supported shapes:
            - ``(M, D)``: one shared prior matrix reused for all models.
            - ``(L, M, D)``: model-specific prior outcomes.
        quantile: Optional quantile ``q`` in ``(0, 1)``. If ``None``, rank by
            posterior mean. Otherwise rank by ``mu_l + Phi^{-1}(q) sigma_l``.
        method: Tie-handling rule for score-to-rank conversion.
        return_scores: If ``True``, return ``(ranking, scores)``.

    Returns:
        Ranking array of shape ``(L,)``.
        If ``return_scores=True``, also returns per-model scores used for
        ranking (posterior means or quantile scores), shape ``(L,)``.

    Notation:
        ``mu_l, sigma_l`` are Bayes@N posterior mean and uncertainty for model
        ``l`` computed by :func:`scorio.eval.bayes`.

    Formula:
        .. math::
            s_l =
            \\begin{cases}
            \\mu_l, & \\text{if } q\\text{ is None} \\\\
            \\mu_l + \\Phi^{-1}(q)\\,\\sigma_l, & \\text{otherwise}
            \\end{cases}

    Examples:
        >>> import numpy as np
        >>> from scorio import rank
        >>> R = np.array([
        ...     [[1, 0], [1, 1], [0, 0]],
        ...     [[0, 0], [1, 0], [1, 1]],
        ... ])
        >>> w = np.array([0.0, 1.0])
        >>> R0 = np.array([[1, 1], [0, 1], [0, 0]])  # shared prior
        >>> ranks, scores = rank.bayes(R, w=w, R0=R0, return_scores=True)
        >>> ranks.shape, scores.shape
        ((2,), (2,))

    Notes:
        Lower quantiles (for example ``q=0.05``) implement conservative ranking
        by penalizing posterior uncertainty.
    """
    R = validate_input(R, binary_only=False)
    L, M, N = R.shape

    if quantile is not None:
        try:
            quantile = float(quantile)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"quantile must be in (0, 1); got {quantile}") from exc
        if not np.isfinite(quantile) or not (0.0 < quantile < 1.0):
            raise ValueError(f"quantile must be in (0, 1); got {quantile}")

    R0_shared: np.ndarray | None = None
    R0_per_model: np.ndarray | None = None

    # Validate and normalize R0
    if R0 is not None:
        R0_array = np.asarray(R0)
        if np.issubdtype(R0_array.dtype, np.bool_):
            R0 = R0_array.astype(int, copy=False)
        else:
            if not np.issubdtype(R0_array.dtype, np.number) or np.issubdtype(
                R0_array.dtype, np.complexfloating
            ):
                raise ValueError("R0 must contain real, finite integer-valued outcomes")
            if not np.isfinite(R0_array).all():
                raise ValueError("R0 must contain real, finite integer-valued outcomes")
            if not np.equal(R0_array, np.floor(R0_array)).all():
                raise ValueError("R0 must contain real, finite integer-valued outcomes")
            R0 = R0_array.astype(int, copy=False)

        if R0.ndim == 2:
            if R0.shape[0] != M:
                raise ValueError(
                    f"Shared R0 must have shape (M={M}, D), got {R0.shape}"
                )
            R0_shared = R0
        elif R0.ndim == 3:
            if R0.shape[0] != L or R0.shape[1] != M:
                raise ValueError(
                    f"Model-specific R0 must have shape (L={L}, M={M}, D), got {R0.shape}"
                )
            R0_per_model = R0
        else:
            raise ValueError(
                "R0 must be shape (M, D) or (L, M, D); "
                f"got ndim={R0.ndim} with shape {R0.shape}"
            )

    scores = np.zeros(L)
    z = norm.ppf(quantile) if quantile is not None else None
    for model in range(L):
        model_R0 = R0_shared if R0_shared is not None else None
        if R0_per_model is not None:
            model_R0 = R0_per_model[model]

        mu, sigma = eval.bayes(R[model], w, R0=model_R0)

        if z is not None:
            scores[model] = mu + z * sigma
        else:
            scores[model] = mu

    ranking = rank_scores(scores)[method]
    return (ranking, scores) if return_scores else ranking


def pass_at_k(
    R: np.ndarray,
    k: int,
    method: RankMethod = "competition",
    return_scores: bool = False,
) -> RankResult:
    """
    Rank models by the Pass@k metric.

    Method context:
        Pass@k measures the probability that at least one of ``k`` draws
        without replacement is correct for a question. Scores are averaged
        across questions per model.

    References:
        Chen, M., Tworek, J., Jun, H., et al. (2021).
        Evaluating Large Language Models Trained on Code.
        *arXiv:2107.03374*.
        https://arxiv.org/abs/2107.03374

    Args:
        R: Binary outcome tensor of shape ``(L, M, N)`` or matrix ``(L, M)``.
        k: Number of selected samples, with ``1 <= k <= N``.
        method: Tie-handling rule for ``rank_scores``.
        return_scores: If ``True``, return ``(ranking, scores)``.

    Returns:
        Ranking array of shape ``(L,)``.
        If ``return_scores=True``, also returns per-model Pass@k scores.

    Notation:
        ``nu_lm = sum_{n=1}^N R_lmn`` is the number of successes for model
        ``l`` on question ``m``.

    Formula:
        .. math::
            s_l^{\\mathrm{Pass@}k}
            = \\frac{1}{M} \\sum_{m=1}^{M}
            \\left(1 - \\frac{{N-\\nu_{lm} \\choose k}}{{N \\choose k}}\\right)

    Examples:
        >>> import numpy as np
        >>> from scorio import rank
        >>> R = np.array([
        ...     [[1, 1, 0], [0, 1, 0]],
        ...     [[1, 0, 0], [0, 0, 0]],
        ... ])
        >>> ranks, scores = rank.pass_at_k(R, k=2, return_scores=True)
        >>> ranks.tolist()
        [1, 2]


    """
    R = validate_input(R)
    L, _, _ = R.shape
    scores = np.array([eval.pass_at_k(R[model, :, :], k) for model in range(L)])
    ranking = rank_scores(scores)[method]
    return (ranking, scores) if return_scores else ranking


def pass_hat_k(
    R: np.ndarray,
    k: int,
    method: RankMethod = "competition",
    return_scores: bool = False,
) -> RankResult:
    """
    Rank models by Pass-hat@k (G-Pass@k).

    Method context:
        Pass-hat@k is the probability that all ``k`` selected samples are
        correct for a question, then averaged across questions.

    References:
        Yao, S., Shinn, N., Razavi, P., & Narasimhan, K. (2024).
        tau-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains.
        *arXiv:2406.12045*.
        https://arxiv.org/abs/2406.12045

    Args:
        R: Binary outcome tensor of shape ``(L, M, N)`` or matrix ``(L, M)``.
        k: Number of selected samples, with ``1 <= k <= N``.
        method: Tie-handling rule for ``rank_scores``.
        return_scores: If ``True``, return ``(ranking, scores)``.

    Returns:
        Ranking array of shape ``(L,)``.
        If ``return_scores=True``, also returns per-model Pass-hat@k scores.

    Notation:
        ``nu_lm = sum_{n=1}^N R_lmn``.

    Formula:
        .. math::
            s_l^{\\widehat{\\mathrm{Pass@}k}}
            = \\frac{1}{M} \\sum_{m=1}^{M}
            \\frac{{\\nu_{lm} \\choose k}}{{N \\choose k}}

    Examples:
        >>> import numpy as np
        >>> from scorio import rank
        >>> R = np.array([
        ...     [[1, 1, 0], [0, 1, 0]],
        ...     [[1, 0, 0], [0, 0, 0]],
        ... ])
        >>> rank.pass_hat_k(R, k=1).tolist()
        [1, 2]

    """
    R = validate_input(R)
    L, _, _ = R.shape
    scores = np.array([eval.pass_hat_k(R[model, :, :], k) for model in range(L)])
    ranking = rank_scores(scores)[method]
    return (ranking, scores) if return_scores else ranking


def g_pass_at_k_tau(
    R: np.ndarray,
    k: int,
    tau: float,
    method: RankMethod = "competition",
    return_scores: bool = False,
) -> RankResult:
    """
    Rank models by generalized G-Pass@k_tau.

    Method context:
        G-Pass@k_tau measures the probability of obtaining at least
        ``max(1, ceil(tau * k))`` successes in ``k`` draws without replacement.
        It interpolates between Pass@k (small tau) and Pass-hat@k (tau=1).

    References:
        Liu, J., Liu, H., Xiao, L., et al. (2025).
        Are Your LLMs Capable of Stable Reasoning?
        *Findings of ACL 2025*, 17594--17632.
        https://doi.org/10.18653/v1/2025.findings-acl.905

    Args:
        R: Binary outcome tensor of shape ``(L, M, N)`` or matrix ``(L, M)``.
        k: Number of selected samples, with ``1 <= k <= N``.
        tau: Threshold parameter in ``[0, 1]``.
        method: Tie-handling rule for ``rank_scores``.
        return_scores: If ``True``, return ``(ranking, scores)``.

    Returns:
        Ranking array of shape ``(L,)``.
        If ``return_scores=True``, also returns per-model G-Pass@k_tau scores.

    Notation:
        ``X_lm ~ Hypergeom(N, nu_lm, k)`` where ``nu_lm`` is the success count
        for model ``l`` and question ``m``.

    Formula:
        .. math::
            s_l^{\\mathrm{G\\text{-}Pass@}k_\\tau}
            = \\frac{1}{M} \\sum_{m=1}^{M}
            \\Pr\\left(X_{lm} \\ge \\max(1,\\lceil \\tau k \\rceil)\\right)

        .. math::
            \\Pr\\left(X_{lm} \\ge \\max(1,\\lceil \\tau k \\rceil)\\right)
            =
            \\sum_{j=\\max(1,\\lceil \\tau k \\rceil)}^{k}
            \\frac{{\\nu_{lm} \\choose j}{N-\\nu_{lm} \\choose k-j}}
                 {{N \\choose k}}

    Examples:
        >>> import numpy as np
        >>> from scorio import rank
        >>> R = np.array([
        ...     [[1, 1, 0], [0, 1, 0]],
        ...     [[1, 0, 0], [0, 0, 0]],
        ... ])
        >>> rank.g_pass_at_k_tau(R, k=2, tau=1.0).tolist() == rank.pass_hat_k(R, 2).tolist()
        True

    """
    R = validate_input(R)
    L, _, _ = R.shape
    scores = np.array(
        [eval.g_pass_at_k_tau(R[model, :, :], k, tau) for model in range(L)]
    )
    ranking = rank_scores(scores)[method]
    return (ranking, scores) if return_scores else ranking


def mg_pass_at_k(
    R: np.ndarray,
    k: int,
    method: RankMethod = "competition",
    return_scores: bool = False,
) -> RankResult:
    """
    Rank models by mG-Pass@k (mean generalized pass metric).

    Method context:
        mG-Pass@k aggregates G-Pass@k_tau for ``tau in [0.5, 1]`` via the
        discrete summation proposed in the G-Pass literature, producing a
        stability-focused score. This summation approximates the continuous
        area over that interval; for odd ``k`` the two are not exactly equal.

    References:
        Liu, J., Liu, H., Xiao, L., et al. (2025).
        Are Your LLMs Capable of Stable Reasoning?
        *Findings of ACL 2025*, 17594--17632.
        https://doi.org/10.18653/v1/2025.findings-acl.905

    Args:
        R: Binary outcome tensor of shape ``(L, M, N)`` or matrix ``(L, M)``.
        k: Number of selected samples, with ``1 <= k <= N``.
        method: Tie-handling rule for ``rank_scores``.
        return_scores: If ``True``, return ``(ranking, scores)``.

    Returns:
        Ranking array of shape ``(L,)``.
        If ``return_scores=True``, also returns per-model mG-Pass@k scores.

    Notation:
        ``X_lm ~ Hypergeom(N, nu_lm, k)``, and
        ``m0 = ceil(k/2)``.

    Formula:
        .. math::
            s_l^{\\mathrm{mG\\text{-}Pass@}k}
            = \\frac{1}{M} \\sum_{m=1}^{M}
            \\frac{2}{k} \\sum_{i=m_0+1}^{k}
            \\Pr(X_{lm} \\ge i)

        .. math::
            \\frac{2}{k} \\sum_{i=m_0+1}^{k}
            \\Pr(X_{lm} \\ge i)
            =
            \\frac{2}{k} \\, \\mathbb{E}\\left[(X_{lm}-m_0)_+\\right]

    Examples:
        >>> import numpy as np
        >>> from scorio import rank
        >>> R = np.array([
        ...     [[1, 1, 0], [0, 1, 0]],
        ...     [[1, 0, 0], [0, 0, 0]],
        ... ])
        >>> ranks, scores = rank.mg_pass_at_k(R, k=2, return_scores=True)
        >>> ranks.tolist()
        [1, 2]

    """
    R = validate_input(R)
    L, _, _ = R.shape
    scores = np.array([eval.mg_pass_at_k(R[model, :, :], k) for model in range(L)])
    ranking = rank_scores(scores)[method]
    return (ranking, scores) if return_scores else ranking


__all__ = [
    "avg",
    "bayes",
    "pass_at_k",
    "pass_hat_k",
    "g_pass_at_k_tau",
    "mg_pass_at_k",
]
