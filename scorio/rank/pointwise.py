"""
Pointwise ranking methods.

Pointwise methods aggregate each model's item-level performance into one scalar
without explicit head-to-head modeling.

Notation
--------

Let :math:`R \\in \\{0,1\\}^{L \\times M \\times N}` and define
:math:`k_{lm}=\\sum_{n=1}^{N} R_{lmn}`,
:math:`\\widehat{p}_{lm}=k_{lm}/N`.

The pointwise score template is

.. math::
    s_l = \\sum_{m=1}^{M} w_m \\, g\\!\\left(\\widehat{p}_{lm}, \\phi_m\\right),

where :math:`\\phi_m` is an item statistic, :math:`g` is a fixed transform,
and :math:`w_m` are nonnegative weights.
"""

import numpy as np

from scorio.utils import rank_scores

from ._base import validate_input
from ._types import RankMethod, RankResult


def inverse_difficulty(
    R: np.ndarray,
    method: RankMethod = "competition",
    return_scores: bool = False,
    clip_range: tuple = (0.01, 0.99),
) -> RankResult:
    """
    Rank models by inverse-difficulty-weighted per-question accuracy.

    Method context:
        This pointwise method upweights hard questions by assigning each
        question an inverse weight proportional to the reciprocal global solve
        rate. It emphasizes rare successes while still aggregating all
        questions into one scalar score per model.

    Args:
        R: Binary outcome tensor with shape ``(L, M, N)`` or matrix
            ``(L, M)`` (treated as ``N=1``).
        method: Tie-handling rule passed to :func:`scorio.utils.rank_scores`.
            One of ``"competition"``, ``"competition_max"``, ``"dense"``,
            or ``"avg"``.
        return_scores: If ``True``, return ``(ranking, scores)``.
        clip_range: Two-sided clipping interval ``(a, b)`` applied to global
            solve rates before inversion. Must satisfy
            ``0 < a < b <= 1``.

    Returns:
        Ranking array of shape ``(L,)``.
        If ``return_scores=True``, also returns inverse-difficulty scores of
        shape ``(L,)``.

    Notation:
        ``k_lm = sum_{n=1}^N R_lmn`` and
        ``p_hat_lm = k_lm / N``.
        The global per-question solve rate is
        ``p_bar_m = (1/L) * sum_l p_hat_lm``.

    Formula:
        .. math::
            w_m \\propto \\frac{1}{\\operatorname{clip}(\\bar p_m, a, b)},
            \\quad \\sum_{m=1}^{M} w_m = 1

        .. math::
            s_l^{\\mathrm{inv\\text{-}diff}} =
            \\sum_{m=1}^{M} w_m \\, \\widehat{p}_{lm}

    Examples:
        >>> import numpy as np
        >>> from scorio import rank
        >>> R = np.array([
        ...     [[1, 1], [0, 0], [0, 0]],
        ...     [[0, 0], [1, 1], [0, 0]],
        ... ])
        >>> ranks, scores = rank.inverse_difficulty(R, return_scores=True)
        >>> ranks.tolist()
        [1, 1]
        >>> scores.shape
        (2,)

        >>> # tighter clipping is allowed
        >>> rank.inverse_difficulty(R, clip_range=(0.05, 0.95)).shape
        (2,)

    Notes:
        Very small clip lower bounds can make the weighting highly sensitive
        to a few rare solves. This is a question-difficulty reweighting
        heuristic, not a formal inverse-probability-weighting estimator.
    """
    R = validate_input(R)

    if len(clip_range) != 2:
        raise ValueError("clip_range must be a length-2 tuple (low, high).")
    low = float(clip_range[0])
    high = float(clip_range[1])
    if not np.isfinite(low) or not np.isfinite(high):
        raise ValueError("clip_range values must be finite.")
    if not (0.0 < low < high <= 1.0):
        raise ValueError("clip_range must satisfy 0 < low < high <= 1.")

    # Global difficulty: average across all models and trials for each question
    question_difficulty = R.mean(axis=(0, 2))  # Shape: (M,)

    # Clip to avoid extreme weights
    question_difficulty = np.clip(question_difficulty, low, high)

    # Inverse difficulty weights (normalized)
    weights = 1.0 / question_difficulty
    total_weight = float(np.sum(weights))
    if not np.isfinite(total_weight) or total_weight <= 0.0:
        raise ValueError(
            "inverse-difficulty weights are not finite; choose a stricter clip_range."
        )
    weights = weights / total_weight

    # Per-model accuracy on each question
    model_question_accuracy = R.mean(axis=2)  # Shape: (L, M)

    scores = model_question_accuracy @ weights

    ranking = rank_scores(scores)[method]

    return (ranking, scores) if return_scores else ranking


__all__ = ["inverse_difficulty"]
