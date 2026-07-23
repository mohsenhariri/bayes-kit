r"""Per-trace confidence signals for test-time scaling.

The voting and reward-selection rules in :mod:`scorio.aggregate` collapse a pool
of candidate traces into one answer *given a per-candidate score*. This module
produces those scores from a trace's **own token-level statistics** -- the
sequence of chosen-token log-probabilities and the per-position top-:math:`k`
log-probabilities -- with no external verifier or reward model. Feeding the
resulting scalar into :func:`~scorio.aggregate.weighted_majority_vote`,
:func:`~scorio.aggregate.filtered_vote`, or :func:`~scorio.aggregate.best_of_n`
reproduces a large family of literature methods (confidence-weighted
self-consistency, DeepConf offline voting, self-certainty Best-of-N,
perplexity/likelihood reranking).

Inputs (one trace)
------------------
Each estimator consumes one of two token-level arrays for a single trace:

- ``logprobs`` -- ``(T,)`` chosen-token log-probabilities
  :math:`\log p_\theta(y_t \mid y_{<t}, x)` (``completion_logprob_list``).
- ``topk_logprobs`` -- ``(T, k)`` (or a ragged length-``T`` list of rows) of the
  top-:math:`k` candidate log-probabilities at each position
  (``completion_topk_logprobs_list``). With only the top-:math:`k` exposed (here
  :math:`k = 20`), full-vocabulary quantities (entropy, self-certainty) are
  computed over the observed top-:math:`k` support -- a biased but empirically
  monotone proxy for the exact value, documented per function.

Orientation
-----------
Most estimators are **confidences**: *higher = more confident*, so they drop
straight into the score-consuming rules. The two exceptions are *uncertainties*
-- :func:`perplexity`, :func:`token_entropy`, and :func:`varentropy` -- for which
*lower = more confident*; negate them (or pass a decreasing transform) before
using them as a selection score.

Methods
-------
- ``mean_logprob`` / ``sequence_logprob`` / ``perplexity``: sequence-likelihood
  confidences from the chosen-token log-probabilities.
- ``self_certainty``: average KL-from-uniform of the next-token distribution
  (Kang et al., 2025).
- ``token_confidence`` / ``deepconf_confidence``: DeepConf negative-mean-top-k
  confidence and its trace-level reductions -- average, tail, bottom-percentile
  group, and lowest group (Fu et al., 2025).
- ``token_entropy`` / ``varentropy``: Shannon entropy and its variance
  (varentropy) of the top-k distribution (Malinin & Gales, 2021; entropix, 2024).
- ``max_softmax_probability`` / ``logprob_margin``: top-1 probability and
  top1-top2 margin (Hendrycks & Gimpel, 2017; Scheffer et al., 2001).
- ``picsar``: reasoning + answer log-likelihood Best-of-N selector
  (Leang et al., 2026).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

__all__ = [
    "mean_logprob",
    "sequence_logprob",
    "perplexity",
    "self_certainty",
    "token_confidence",
    "deepconf_confidence",
    "token_entropy",
    "varentropy",
    "max_softmax_probability",
    "logprob_margin",
    "picsar",
]

_REDUCERS = ("mean", "min", "max")


def _as_logprobs(logprobs: Any) -> np.ndarray:
    """Coerce chosen-token log-probabilities to a finite 1-D ``(T,)`` array."""
    lp = np.asarray(logprobs, dtype=float).reshape(-1)
    if lp.size == 0:
        raise ValueError("need at least one token (T >= 1).")
    if not np.all(np.isfinite(lp)):
        raise ValueError("logprobs must all be finite.")
    return lp


def _as_topk(topk_logprobs: Any) -> tuple[np.ndarray | None, list[np.ndarray]]:
    """Coerce top-k log-probabilities to ``(mat, rows)``.

    Returns ``mat`` -- a ``(T, k)`` float array when every position has the same
    number of candidates (the common, vectorizable case), else ``None`` -- and
    ``rows``, always the length-``T`` list of per-position 1-D log-prob arrays.
    """
    try:
        mat = np.asarray(topk_logprobs, dtype=float)
    except (ValueError, TypeError):
        mat = np.asarray(topk_logprobs, dtype=object)
    if mat.dtype != object and mat.ndim == 2:
        if mat.shape[0] == 0 or mat.shape[1] == 0:
            raise ValueError("need at least one token and one top-k candidate.")
        if not np.all(np.isfinite(mat)):
            raise ValueError("topk_logprobs must all be finite.")
        rows = [mat[i] for i in range(mat.shape[0])]
        return mat, rows
    # Ragged (or 1-D of one row): fall back to a per-position list.
    if mat.ndim == 1 and mat.dtype != object:
        # A single position given as a bare (k,) row.
        rows = [np.asarray(mat, dtype=float)]
    else:
        rows = [np.asarray(r, dtype=float).reshape(-1) for r in topk_logprobs]
    if not rows:
        raise ValueError("need at least one token (T >= 1).")
    for r in rows:
        if r.size == 0:
            raise ValueError("every position needs at least one top-k candidate.")
        if not np.all(np.isfinite(r)):
            raise ValueError("topk_logprobs must all be finite.")
    return None, rows


def _reduce(x: np.ndarray, how: str) -> float:
    """Reduce a per-token ``(T,)`` statistic to one scalar."""
    if how == "mean":
        return float(np.mean(x))
    if how == "min":
        return float(np.min(x))
    if how == "max":
        return float(np.max(x))
    raise ValueError(f"aggregate must be one of {_REDUCERS}; got {how!r}.")


def _normalized_logprobs(row: np.ndarray) -> np.ndarray:
    """Normalize finite log-probabilities without taking ``log(exp(x))``.

    Keeping the normalized values in log space prevents a finite, very unlikely
    top-k entry (for example ``-1000``) from becoming ``log(0)`` after its
    probability underflows to zero.
    """
    maximum = float(np.max(row))
    shifted = row - maximum
    return shifted - math.log(float(np.sum(np.exp(shifted))))


def _per_token_entropy(mat: np.ndarray | None, rows: list[np.ndarray]) -> np.ndarray:
    """Per-token Shannon entropy of the renormalized top-k distribution (nats)."""
    if mat is not None:
        shifted = mat - mat.max(axis=1, keepdims=True)
        log_p = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
        p = np.exp(log_p)
        return -np.sum(p * log_p, axis=1)
    out = np.empty(len(rows))
    for i, r in enumerate(rows):
        log_p = _normalized_logprobs(r)
        out[i] = -float(np.sum(np.exp(log_p) * log_p))
    return out


def _per_token_varentropy(mat: np.ndarray | None, rows: list[np.ndarray]) -> np.ndarray:
    r"""Per-token varentropy :math:`\sum_j p_j(\log p_j)^2 - H^2` of the top-k."""
    if mat is not None:
        shifted = mat - mat.max(axis=1, keepdims=True)
        log_p = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
        p = np.exp(log_p)
        surprisal = -log_p
        entropy = np.sum(p * surprisal, axis=1, keepdims=True)
        return np.sum(p * (surprisal - entropy) ** 2, axis=1)
    out = np.empty(len(rows))
    for i, r in enumerate(rows):
        log_p = _normalized_logprobs(r)
        p = np.exp(log_p)
        surprisal = -log_p
        entropy = float(np.sum(p * surprisal))
        out[i] = float(np.sum(p * (surprisal - entropy) ** 2))
    return out


def _per_token_self_certainty(
    mat: np.ndarray | None, rows: list[np.ndarray]
) -> np.ndarray:
    r"""Per-token KL-from-uniform over the renormalized top-k support.

    :math:`\mathrm{KL}(U \| p) = -\log k - \tfrac1k \sum_{j} \log p_j` with the
    uniform reference over the :math:`k` observed candidates.
    """
    if mat is not None:
        shifted = mat - mat.max(axis=1, keepdims=True)
        log_p = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
        k = mat.shape[1]
        return -math.log(k) - np.mean(log_p, axis=1)
    out = np.empty(len(rows))
    for i, r in enumerate(rows):
        log_p = _normalized_logprobs(r)
        out[i] = -math.log(len(log_p)) - float(np.mean(log_p))
    return out


def _per_token_confidence(mat: np.ndarray | None, rows: list[np.ndarray]) -> np.ndarray:
    r"""DeepConf per-token confidence :math:`C_i = -\tfrac1k \sum_j \log P_i(j)`.

    Uses the *raw* top-k log-probabilities (not renormalized), matching the
    DeepConf definition (Fu et al., 2025, Eq. 2).
    """
    if mat is not None:
        return -mat.mean(axis=1)
    return np.array([-float(np.mean(r)) for r in rows])


def _per_token_max_prob(mat: np.ndarray | None, rows: list[np.ndarray]) -> np.ndarray:
    """Per-token top-1 (maximum) probability, from the raw top-k log-probs."""
    if mat is not None:
        return np.exp(mat.max(axis=1))
    return np.array([math.exp(float(np.max(r))) for r in rows])


def _per_token_margin(
    mat: np.ndarray | None, rows: list[np.ndarray], use_prob: bool
) -> np.ndarray:
    """Per-token top1-top2 margin in log-prob (default) or probability space.

    A position with a single candidate contributes a margin of ``0``.
    """

    def one(r: np.ndarray) -> float:
        if r.size < 2:
            return 0.0
        two = np.partition(r, -2)[-2:]
        top1, top2 = float(two[1]), float(two[0])
        if top1 < top2:  # np.partition leaves the largest last; guard anyway
            top1, top2 = top2, top1
        if use_prob:
            return math.exp(top1) - math.exp(top2)
        return top1 - top2

    if mat is not None and mat.shape[1] >= 2:
        part = np.partition(mat, -2, axis=1)
        top1 = part[:, -1]
        top2 = part[:, -2]
        if use_prob:
            return np.exp(top1) - np.exp(top2)
        return top1 - top2
    return np.array([one(r) for r in rows])


def mean_logprob(logprobs: Any) -> float:
    r"""
    Mean chosen-token log-probability (length-normalized sequence log-likelihood).

    Averages the log-probability the model assigned to each token it actually
    generated. Higher (closer to zero) means the trace was, on average, more
    predictable to the model and is used as a length-fair confidence for
    reward-free selection. As a Best-of-N score this is the *sample-and-rank*
    rule of Meena. Exponentiating it gives the length-normalized likelihood
    weight used by normalized weighted self-consistency,
    :math:`\exp(\text{mean\_logprob}) = 1 / \mathrm{PPL}`. Do not use the raw
    negative value as an additive vote weight.

    References:
        Adiwardana, D., Luong, M.-T., So, D. R., Hall, J., Fiedel, N., Thoppilan,
        R., Yang, Z., Kulshreshtha, A., Nemade, G., Lu, Y., & Le, Q. V. (2020).
        Towards a Human-like Open-Domain Chatbot (Meena; sample-and-rank).
        *arXiv:2001.09977*. https://arxiv.org/abs/2001.09977

        Wang, X., Wei, J., Schuurmans, D., Le, Q., Chi, E., Narang, S.,
        Chowdhery, A., & Zhou, D. (2023). Self-Consistency Improves Chain of
        Thought Reasoning in Language Models (Sec. 2, normalized weighted sum).
        *ICLR 2023*, *arXiv:2203.11171*. https://arxiv.org/abs/2203.11171

    Args:
        logprobs: ``(T,)`` array-like of chosen-token log-probabilities for one
            trace.

    Returns:
        The mean log-probability (``<= 0``); higher is more confident.

    Formula:
        .. math::

            s(y) = \frac{1}{T} \sum_{t=1}^{T} \log p_\theta(y_t \mid y_{<t}, x).

    Examples:
        >>> round(mean_logprob([-0.1, -0.2, -0.3]), 4)
        -0.2
    """
    return float(np.mean(_as_logprobs(logprobs)))


def sequence_logprob(logprobs: Any) -> float:
    r"""
    Total (un-normalized) sequence log-likelihood, :math:`\sum_t \log p_\theta`.

    The sum of the chosen-token log-probabilities -- the log-probability of the
    whole generation. Unlike :func:`mean_logprob` it is **not** length-normalized,
    so it penalizes longer traces. Exponentiating it gives the path probability
    used by the paper's *unnormalized weighted sum*, where paths vote with weight
    :math:`\exp(\text{sequence\_logprob})`.

    References:
        Wang, X., Wei, J., Schuurmans, D., Le, Q., Chi, E., Narang, S.,
        Chowdhery, A., & Zhou, D. (2023). Self-Consistency Improves Chain of
        Thought Reasoning in Language Models (Sec. 2, unnormalized weighted sum).
        *ICLR 2023*, *arXiv:2203.11171*. https://arxiv.org/abs/2203.11171

    Args:
        logprobs: ``(T,)`` array-like of chosen-token log-probabilities.

    Returns:
        The summed log-probability (``<= 0``); higher is more confident.

    Formula:
        .. math::

            \log p_\theta(y \mid x) = \sum_{t=1}^{T}
            \log p_\theta(y_t \mid y_{<t}, x).

    Examples:
        >>> round(sequence_logprob([-0.1, -0.2, -0.3]), 4)
        -0.6
    """
    return float(np.sum(_as_logprobs(logprobs)))


def perplexity(logprobs: Any) -> float:
    r"""
    Sequence perplexity, :math:`\exp(-\text{mean\_logprob})`.

    The exponentiated mean negative log-probability. **Lower is more confident**
    (perplexity ``1`` is a perfectly predicted trace), so this is an
    *uncertainty*. For an order-based selector, negate it or use
    :func:`mean_logprob`; for magnitude-weighted voting, use reciprocal
    perplexity so the vote weight remains non-negative.

    Args:
        logprobs: ``(T,)`` array-like of chosen-token log-probabilities.

    Returns:
        The perplexity (``>= 1``); **lower** is more confident.

    Formula:
        .. math::

            \mathrm{PPL}(y) = \exp\!\Big(-\tfrac{1}{T} \sum_{t=1}^{T}
            \log p_\theta(y_t \mid y_{<t}, x)\Big).

    Examples:
        >>> round(perplexity([0.0, 0.0, 0.0]), 4)
        1.0
    """
    return float(math.exp(-mean_logprob(logprobs)))


def picsar(
    logprobs: Any,
    *,
    answer_start: int | None = None,
    normalize_reasoning: bool = False,
) -> float:
    r"""
    PiCSAR: reasoning + answer log-likelihood Best-of-N selector.

    Scores a trace by the sum of its reasoning log-likelihood and its answer
    log-likelihood -- the joint confidence in *how* it reasoned and *what* it
    concluded. With ``answer_start`` splitting the token stream into a reasoning
    span ``y_{<answer_start}`` and an answer span ``y_{>=answer_start}``, the
    score is the reasoning log-likelihood (optionally length-normalized, the
    ``PiCSAR-N`` variant) plus the answer log-likelihood. Without a split it
    reduces to :func:`sequence_logprob`, the whole-trace log-likelihood.

    References:
        Leang, J. O. J., Zhao, Z., Gema, A. P., Yang, S., Kwan, W.-C., He, X.,
        Li, W., Minervini, P., Giunchiglia, E., & Cohen, S. B. (2026).
        Probabilistic Confidence via Sum of Answer and Reasoning Log-Likelihoods
        (PiCSAR). *Findings of ACL 2026*, *arXiv:2508.21787*.
        https://arxiv.org/abs/2508.21787

    Args:
        logprobs: ``(T,)`` array-like of chosen-token log-probabilities.
        answer_start: Index of the first answer-span token. Tokens before it are
            the reasoning span; from it onward is the answer span. ``None``
            (default) treats the whole trace as one span (equals
            :func:`sequence_logprob`).
        normalize_reasoning: If ``True`` (``PiCSAR-N``), divide the reasoning
            log-likelihood by its token count before adding the (un-normalized)
            answer log-likelihood. Ignored when ``answer_start`` is ``None``.

    Returns:
        The PiCSAR score; higher is more confident.

    Formula:
        .. math::

            \mathrm{PiCSAR}(r, y) = \log p(r \mid x)
            + \log p(y \mid \langle a\rangle, r, x),

        with ``PiCSAR-N`` replacing :math:`\log p(r\mid x)` by
        :math:`\tfrac{1}{|r|}\log p(r\mid x)`.

    Examples:
        >>> lp = [-0.1, -0.2, -0.3, -0.4]
        >>> round(picsar(lp), 4)                       # whole-trace log-likelihood
        -1.0
        >>> round(picsar(lp, answer_start=3), 4)       # reasoning(-0.6) + answer(-0.4)
        -1.0
    """
    lp = _as_logprobs(logprobs)
    if answer_start is None:
        return float(np.sum(lp))
    if not 0 <= answer_start <= lp.size:
        raise ValueError(f"answer_start must be in [0, {lp.size}]; got {answer_start}.")
    reasoning = lp[:answer_start]
    answer = lp[answer_start:]
    r_ll = float(np.sum(reasoning))
    if normalize_reasoning and reasoning.size:
        r_ll /= reasoning.size
    return r_ll + float(np.sum(answer))


def self_certainty(topk_logprobs: Any, *, aggregate: str = "mean") -> float:
    r"""
    Self-certainty: average KL divergence from uniform to the next-token law.

    Measures how far each step's next-token distribution sits from maximum
    uncertainty (the uniform distribution): a peaked distribution is far from
    uniform and scores high. Averaged over the trace it is a reward-free
    confidence that ranks Best-of-N candidates as well as a trained verifier, and
    its rank-weighted vote is :func:`~scorio.aggregate.rank_weighted_vote`.

    Only the top-:math:`k` log-probabilities are exposed here, so the per-token
    KL is computed over the renormalized top-:math:`k` support (uniform reference
    over the :math:`k` observed candidates). This is the top-:math:`k`
    approximation of the paper's full-vocabulary Eq. (9); because the near-zero
    tail tokens the paper sums over are unavailable, the value is a biased but
    empirically monotone proxy, not the exact quantity.

    References:
        Kang, Z., Zhao, X., & Song, D. (2025). Scalable Best-of-N Selection for
        Large Language Models via Self-Certainty. *NeurIPS 2025*,
        *arXiv:2502.18581*. https://arxiv.org/abs/2502.18581

    Args:
        topk_logprobs: ``(T, k)`` array-like (or a ragged length-``T`` list of
            rows) of per-position top-:math:`k` log-probabilities.
        aggregate: How to reduce the per-token KL over the trace: ``"mean"``
            (default), ``"min"`` (least-certain token), or ``"max"``.

    Returns:
        The (top-:math:`k`) self-certainty; higher is more confident.

    Formula:
        .. math::

            C_i^{\mathrm{KL}} = \mathrm{KL}\big(U \,\|\, p_i\big)
            = -\log k - \tfrac{1}{k}\sum_{j} \log \tilde p_{ij}, \qquad
            \mathrm{SC}(y) = \operatorname*{agg}_i C_i^{\mathrm{KL}},

        with :math:`\tilde p_i` the renormalized top-:math:`k` distribution.

    Examples:
        >>> # near one-hot position -> high self-certainty; flat -> ~0
        >>> peaked = [[0.0, -20.0, -20.0]]
        >>> self_certainty(peaked) > 5
        True
        >>> flat = [[-1.0986, -1.0986, -1.0986]]     # ~uniform over 3
        >>> abs(self_certainty(flat)) < 1e-3
        True
    """
    mat, rows = _as_topk(topk_logprobs)
    return _reduce(_per_token_self_certainty(mat, rows), aggregate)


def token_entropy(topk_logprobs: Any, *, aggregate: str = "mean") -> float:
    r"""
    Mean (or max) Shannon entropy of the top-k next-token distribution.

    The per-token entropy of the renormalized top-:math:`k` distribution, reduced
    over the trace. **Higher is less confident** (an *uncertainty*): its negation
    is Kang et al.'s negative-entropy confidence and it is also the token-level
    signal underlying DeepConf; negate it before using it as a selection score.
    With ``aggregate="max"`` it reports the single most-uncertain token.

    References:
        Malinin, A., & Gales, M. (2021). Uncertainty Estimation in Autoregressive
        Structured Prediction. *ICLR 2021*, *arXiv:2002.07650*.
        https://arxiv.org/abs/2002.07650

        Kang, Z., Zhao, X., & Song, D. (2025). Scalable Best-of-N Selection for
        Large Language Models via Self-Certainty (negative entropy, Eq. 7).
        *arXiv:2502.18581*. https://arxiv.org/abs/2502.18581

    Args:
        topk_logprobs: ``(T, k)`` array-like (or ragged list of rows) of
            per-position top-:math:`k` log-probabilities.
        aggregate: ``"mean"`` (default), ``"max"`` (most-uncertain token), or
            ``"min"``.

    Returns:
        The entropy in nats over the top-:math:`k` support; **lower** is more
        confident.

    Formula:
        .. math::

            H_i = -\sum_{j} \tilde p_{ij} \log \tilde p_{ij}, \qquad
            H(y) = \operatorname*{agg}_i H_i,

        with :math:`\tilde p_i` the renormalized top-:math:`k` distribution. The
        exact entropy sums over the full vocabulary; the top-:math:`k` value is a
        truncated lower bound.

    Examples:
        >>> round(token_entropy([[-1.0986, -1.0986, -1.0986]]), 4)  # uniform over 3
        1.0986
        >>> token_entropy([[0.0, -20.0, -20.0]]) < 1e-6             # near one-hot
        True
    """
    mat, rows = _as_topk(topk_logprobs)
    return _reduce(_per_token_entropy(mat, rows), aggregate)


def varentropy(topk_logprobs: Any, *, aggregate: str = "mean") -> float:
    r"""
    Varentropy: variance of the surprisal under the top-k next-token distribution.

    The probability-weighted variance of the token surprisals
    :math:`-\log \tilde p_{ij}` around the entropy -- a second-order dispersion
    signal that distinguishes a flatly-uncertain distribution (low varentropy)
    from one that is confident-but-multimodal (high varentropy). Popularized as a
    decoding confidence signal by the *entropix* project and used alongside
    entropy to flag branching / "forking" steps.

    References:
        Kontoyiannis, I., & Verdu, S. (2014). Optimal Lossless Data Compression:
        Non-Asymptotics and Asymptotics (source varentropy). *IEEE Trans. Inf.
        Theory 60(2)*.

        entropix (xjdr et al., 2024). Entropy-based sampling.
        https://github.com/xjdr-alt/entropix

    Args:
        topk_logprobs: ``(T, k)`` array-like (or ragged list of rows) of
            per-position top-:math:`k` log-probabilities.
        aggregate: ``"mean"`` (default), ``"max"``, or ``"min"``.

    Returns:
        The varentropy (``>= 0``) over the top-:math:`k` support.

    Formula:
        .. math::

            V\!H_i = \sum_{j} \tilde p_{ij} \big(-\log \tilde p_{ij} - H_i\big)^2
                   = \Big(\sum_{j} \tilde p_{ij} (\log \tilde p_{ij})^2\Big)
                     - H_i^2, \qquad
            V\!H(y) = \operatorname*{agg}_i V\!H_i.

    Examples:
        >>> round(varentropy([[-1.0986, -1.0986, -1.0986]]), 6)   # uniform -> 0
        0.0
    """
    mat, rows = _as_topk(topk_logprobs)
    return _reduce(_per_token_varentropy(mat, rows), aggregate)


def max_softmax_probability(topk_logprobs: Any, *, aggregate: str = "mean") -> float:
    r"""
    Maximum softmax probability (MSP), averaged over the trace.

    The per-token probability of the most likely next token, reduced over the
    trace -- the classic maximum-softmax-probability confidence / out-of-
    distribution baseline. The top-1 probability is exact (it is present in the
    top-:math:`k`), so no renormalization is needed. With ``aggregate="min"`` it
    reports the least-confident token, a worst-step signal.

    References:
        Hendrycks, D., & Gimpel, K. (2017). A Baseline for Detecting Misclassified
        and Out-of-Distribution Examples in Neural Networks. *ICLR 2017*,
        *arXiv:1610.02136*. https://arxiv.org/abs/1610.02136

    Args:
        topk_logprobs: ``(T, k)`` array-like (or ragged list of rows) of
            per-position top-:math:`k` log-probabilities.
        aggregate: ``"mean"`` (default), ``"min"`` (least-confident token), or
            ``"max"``.

    Returns:
        The mean (or min/max) top-1 probability in ``(0, 1]``; higher is more
        confident.

    Formula:
        .. math::

            \mathrm{MSP}(y) = \operatorname*{agg}_i \max_j p_{ij}.

    Examples:
        >>> round(max_softmax_probability([[0.0, -20.0], [-0.6931, -0.6931]]), 4)
        0.75
    """
    mat, rows = _as_topk(topk_logprobs)
    return _reduce(_per_token_max_prob(mat, rows), aggregate)


def logprob_margin(
    topk_logprobs: Any,
    *,
    use_prob: bool = False,
    aggregate: str = "mean",
) -> float:
    r"""
    Top1-top2 margin of the next-token distribution, averaged over the trace.

    The gap between the most and second-most likely next token at each position,
    reduced over the trace -- the margin-of-confidence / best-versus-second-best
    uncertainty measure. A large margin means the model had a clear preference at
    that step. Both ranks are within the top-:math:`k`, so the margin is exact.

    References:
        Scheffer, T., Decomain, C., & Wrobel, S. (2001). Active Hidden Markov
        Models for Information Extraction (margin sampling). *IDA 2001*, LNCS
        2189. https://doi.org/10.1007/3-540-44816-0_31

    Args:
        topk_logprobs: ``(T, k)`` array-like (or ragged list of rows) of
            per-position top-:math:`k` log-probabilities.
        use_prob: If ``True``, use the probability margin
            :math:`p_{(1)} - p_{(2)}`; if ``False`` (default), the log-prob
            margin :math:`\log p_{(1)} - \log p_{(2)}`.
        aggregate: ``"mean"`` (default), ``"min"`` (closest-call token), or
            ``"max"``.

    Returns:
        The mean (or min/max) top1-top2 margin (``>= 0``); higher is more
        confident. Positions with a single candidate contribute ``0``.

    Formula:
        .. math::

            m_i = \ell_{(1)}(i) - \ell_{(2)}(i), \qquad
            \mathrm{margin}(y) = \operatorname*{agg}_i m_i,

        with :math:`\ell_{(1)}, \ell_{(2)}` the two largest (log-)probabilities.

    Examples:
        >>> round(logprob_margin([[0.0, -1.0, -2.0], [-0.5, -0.7, -3.0]]), 4)
        0.6
    """
    mat, rows = _as_topk(topk_logprobs)
    return _reduce(_per_token_margin(mat, rows, use_prob), aggregate)


def token_confidence(topk_logprobs: Any) -> np.ndarray:
    r"""
    DeepConf per-token confidence :math:`C_i = -\tfrac1k \sum_j \log P_i(j)`.

    The negative mean of the top-:math:`k` log-probabilities at each position --
    high when the model's probability mass concentrates on a few tokens. This is
    the building block for every DeepConf trace-level reduction
    (:func:`deepconf_confidence`) and for the online early-termination rule
    :func:`scorio.aggregate.deepconf_online_stop`; unlike :func:`token_entropy`
    it uses the *raw* top-:math:`k` log-probabilities without renormalizing.

    References:
        Fu, Y., Wang, X., Tian, Y., & Zhao, J. (2025). Deep Think with Confidence
        (Eq. 2). *arXiv:2508.15260*. https://arxiv.org/abs/2508.15260

    Args:
        topk_logprobs: ``(T, k)`` array-like (or ragged list of rows) of
            per-position top-:math:`k` log-probabilities.

    Returns:
        A ``(T,)`` float array of per-token confidences (``>= 0``); higher is
        more confident.

    Formula:
        .. math::

            C_i = -\frac{1}{k} \sum_{j=1}^{k} \log P_i(j).

    Examples:
        >>> token_confidence([[0.0, -2.0], [-1.0, -3.0]]).tolist()
        [1.0, 2.0]
    """
    mat, rows = _as_topk(topk_logprobs)
    return _per_token_confidence(mat, rows)


def _group_confidences(conf: np.ndarray, window: int) -> np.ndarray:
    """Sliding-window means of a per-token confidence sequence.

    Returns the length-``(T - w + 1)`` array of window means for window size
    ``w = min(window, T)``; a trace shorter than ``window`` yields a single
    group equal to the whole-trace mean.
    """
    t = conf.size
    w = min(int(window), t)
    if w <= 0:
        raise ValueError(f"window must be positive; got {window}.")
    if w == t:
        return np.array([float(np.mean(conf))])
    csum = np.cumsum(np.insert(conf, 0, 0.0))
    return (csum[w:] - csum[:-w]) / w


def deepconf_confidence(
    topk_logprobs: Any,
    *,
    mode: str = "mean",
    window: int = 2048,
    tail_tokens: int = 2048,
    bottom_quantile: float = 0.10,
) -> float:
    r"""
    DeepConf trace confidence: average, tail, bottom-percentile, or lowest group.

    Reduces the DeepConf per-token confidences (:func:`token_confidence`) to one
    trace-level score, using the reduction that best separates correct from
    incorrect traces. All four reductions are confidences (*higher = more
    confident*) and drop into :func:`~scorio.aggregate.weighted_majority_vote`
    (confidence-weighted vote) or :func:`~scorio.aggregate.filtered_vote`
    (confidence filtering); the group-based reductions are the statistic behind
    DeepConf's online early stopping.

    References:
        Fu, Y., Wang, X., Tian, Y., & Zhao, J. (2025). Deep Think with Confidence.
        *arXiv:2508.15260*. https://arxiv.org/abs/2508.15260

    Args:
        topk_logprobs: ``(T, k)`` array-like (or ragged list of rows) of
            per-position top-:math:`k` log-probabilities.
        mode: Which reduction of the per-token confidences to return:

            - ``"mean"`` -- average trace confidence :math:`\tfrac1T\sum_i C_i`.
            - ``"tail"`` -- mean over the last ``tail_tokens`` tokens.
            - ``"bottom_group"`` -- mean of the lowest ``bottom_quantile``
              fraction of sliding-window (group) confidences.
            - ``"lowest_group"`` -- the minimum sliding-window confidence.
        window: Sliding-window (group) size in tokens for the group reductions
            (paper default ``2048``). Clamped to the trace length.
        tail_tokens: Tail length for ``mode="tail"`` (paper default ``2048``).
        bottom_quantile: Fraction of lowest groups averaged for
            ``mode="bottom_group"`` (paper default ``0.10``).

    Returns:
        The trace confidence (``>= 0``); higher is more confident.

    Formula:
        With :math:`C_i` the per-token confidence and :math:`C_{G_s}` the mean of
        :math:`C_i` over the sliding window :math:`G_s` of ``window`` tokens,

        .. math::

            C_{\mathrm{mean}} = \tfrac1T\sum_i C_i, \quad
            C_{\mathrm{tail}} = \operatorname{mean}_{i > T - \tau} C_i, \quad
            C_{\mathrm{lowest}} = \min_s C_{G_s}, \quad
            C_{\mathrm{bottom}} =
            \operatorname{mean}_{s:\,C_{G_s} \le Q_{\eta}(C_{G_\cdot})} C_{G_s}.

    Examples:
        >>> tk = [[0.0, -2.0]] * 3 + [[-1.0, -3.0]] * 3   # conf 1,1,1,2,2,2
        >>> round(deepconf_confidence(tk, mode="mean"), 4)
        1.5
        >>> round(deepconf_confidence(tk, mode="tail", tail_tokens=3), 4)
        2.0
        >>> round(deepconf_confidence(tk, mode="lowest_group", window=3), 4)
        1.0
    """
    conf = token_confidence(topk_logprobs)
    if mode == "mean":
        return float(np.mean(conf))
    if mode == "tail":
        tau = min(max(int(tail_tokens), 1), conf.size)
        return float(np.mean(conf[-tau:]))
    if mode in ("lowest_group", "bottom_group"):
        groups = _group_confidences(conf, window)
        if mode == "lowest_group":
            return float(np.min(groups))
        if not 0.0 < bottom_quantile <= 1.0:
            raise ValueError(
                f"bottom_quantile must be in (0, 1]; got {bottom_quantile}."
            )
        thresh = float(np.quantile(groups, bottom_quantile))
        return float(np.mean(groups[groups <= thresh]))
    raise ValueError(
        f"mode must be 'mean', 'tail', 'bottom_group', or 'lowest_group'; got {mode!r}."
    )
