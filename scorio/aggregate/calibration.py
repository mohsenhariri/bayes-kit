r"""KDE-calibrated weighted voting for scalar verifier scores.

This module implements the non-parametric weighted-voting method of Kuang et
al. (2025, Sec. 4.1).  Each candidate has one probability-like score for its
complete response.  Calibration fits class-conditional KDEs to logit-transformed
scores and estimates response correctness by binning the original scores.

The default estimator uses Gaussian kernels, separate Scott bandwidths for the
two correctness classes, and ten quantile bins.  Step-level score sequences
must be reduced to one score per response before fitting or voting.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.special import logsumexp

from ._base import _normalize, _pack, _valid_indices

__all__ = [
    "KDEVoteCalibration",
    "fit_kde_vote_calibration",
    "kde_weighted_vote",
]


def _readonly_vector(values: Any, *, name: str) -> np.ndarray:
    """Copy values into a one-dimensional, read-only float array."""
    array = np.array(values, dtype=float, copy=True)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional; got shape {array.shape}.")
    array.setflags(write=False)
    return array


def _probability_scores(values: Any, *, name: str) -> np.ndarray:
    scores = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(scores)) or not np.all((scores > 0.0) & (scores < 1.0)):
        raise ValueError(f"{name} must all be finite and strictly in (0, 1).")
    return scores


def _logit(values: np.ndarray) -> np.ndarray:
    return np.log(values) - np.log1p(-values)


def _log_gaussian_kde(
    query_logits: np.ndarray, samples: np.ndarray, bandwidth: float
) -> np.ndarray:
    """Evaluate a one-dimensional Gaussian KDE in log space."""
    query = np.asarray(query_logits, dtype=float)
    flat = query.reshape(-1)
    output = np.empty(flat.size, dtype=float)
    normalizer = math.log(samples.size * bandwidth * math.sqrt(2.0 * math.pi))
    for i, value in enumerate(flat):
        standardized = (value - samples) / bandwidth
        output[i] = float(logsumexp(-0.5 * standardized**2) - normalizer)
    return output.reshape(query.shape)


@dataclass(frozen=True, eq=False)
class KDEVoteCalibration:
    r"""Fitted state for non-parametric KDE weighted voting.

    This object models a *scalar response score*: ``correct_logits`` and
    ``incorrect_logits`` hold calibration scores transformed to logit space,
    while ``bin_edges`` and ``bin_probability`` define a binned estimator of
    final-answer correctness.  The stored arrays are defensive copies marked
    read-only.

    The correctness target is whether the response's extracted final answer is
    correct.  It is not a sequence of process-level correctness labels.

    References:
        Kuang, P., Wang, Y., Han, X., Liu, Y., Xu, K., & Wang, H. (2025).
        Optimal Aggregation of LLM and PRM Signals for Efficient Test-Time
        Scaling. *ICLR 2026*, *arXiv:2510.13918* (Sec. 4.1).
        https://arxiv.org/abs/2510.13918

    Attributes:
        correct_logits: Logit-transformed scalar calibration scores for
            responses with correct final answers.
        incorrect_logits: Logit-transformed scalar calibration scores for
            responses with incorrect final answers.
        correct_bandwidth: Gaussian KDE bandwidth for correct responses.
        incorrect_bandwidth: Gaussian KDE bandwidth for incorrect responses.
        bin_edges: Score-space edges of the binned correctness calibrator,
            including ``-inf`` and ``+inf``.
        bin_probability: Empirical final-answer correctness probability in
            each bin.
        kernel: KDE kernel identifier; currently always ``"gaussian"``.
        binning: Bin construction identifier; currently always ``"quantile"``.

    Notes:
        Construct this state with :func:`fit_kde_vote_calibration`.  Direct
        construction is supported mainly for inspection, serialization, and
        testing.
    """

    correct_logits: np.ndarray
    incorrect_logits: np.ndarray
    correct_bandwidth: float
    incorrect_bandwidth: float
    bin_edges: np.ndarray
    bin_probability: np.ndarray
    kernel: str = "gaussian"
    binning: str = "quantile"

    def __post_init__(self) -> None:
        correct = _readonly_vector(self.correct_logits, name="correct_logits")
        incorrect = _readonly_vector(self.incorrect_logits, name="incorrect_logits")
        edges = _readonly_vector(self.bin_edges, name="bin_edges")
        probabilities = _readonly_vector(self.bin_probability, name="bin_probability")

        if correct.size == 0 or incorrect.size == 0:
            raise ValueError("KDE calibration needs correct and incorrect samples.")
        if not np.all(np.isfinite(correct)) or not np.all(np.isfinite(incorrect)):
            raise ValueError("KDE logit samples must all be finite.")
        if (
            not math.isfinite(float(self.correct_bandwidth))
            or float(self.correct_bandwidth) <= 0.0
            or not math.isfinite(float(self.incorrect_bandwidth))
            or float(self.incorrect_bandwidth) <= 0.0
        ):
            raise ValueError("KDE bandwidths must be finite and > 0.")
        if edges.size != probabilities.size + 1 or edges.size < 2:
            raise ValueError("bin_edges must contain exactly one more value than bins.")
        if not math.isinf(edges[0]) or edges[0] >= 0.0:
            raise ValueError("bin_edges must start at -inf.")
        if not math.isinf(edges[-1]) or edges[-1] <= 0.0:
            raise ValueError("bin_edges must end at +inf.")
        if not np.all(np.diff(edges) > 0.0):
            raise ValueError("bin_edges must be strictly increasing.")
        if not np.all(np.isfinite(probabilities)) or not np.all(
            (probabilities >= 0.0) & (probabilities <= 1.0)
        ):
            raise ValueError("bin_probability values must be finite and in [0, 1].")
        if self.kernel != "gaussian":
            raise ValueError("only the implemented 'gaussian' kernel is valid.")
        if self.binning != "quantile":
            raise ValueError("only the implemented 'quantile' binning is valid.")

        object.__setattr__(self, "correct_logits", correct)
        object.__setattr__(self, "incorrect_logits", incorrect)
        object.__setattr__(self, "bin_edges", edges)
        object.__setattr__(self, "bin_probability", probabilities)
        object.__setattr__(self, "correct_bandwidth", float(self.correct_bandwidth))
        object.__setattr__(self, "incorrect_bandwidth", float(self.incorrect_bandwidth))

    @property
    def n_bins(self) -> int:
        """Number of fitted correctness-calibration bins."""
        return int(self.bin_probability.size)

    def calibrated_probability(self, scores: Any) -> np.ndarray:
        """Estimate final-answer correctness from scalar response scores.

        Args:
            scores: Scalar response-level verifier scores, finite and strictly
                in ``(0, 1)``.  Any array shape is accepted.

        Returns:
            An array with the same shape as ``scores`` containing each score's
            fitted-bin correctness probability.
        """
        values = _probability_scores(scores, name="scores")
        indices = np.searchsorted(self.bin_edges[1:-1], values, side="right")
        return self.bin_probability[indices]

    def log_density_ratio(self, scores: Any) -> np.ndarray:
        r"""Evaluate the fitted scalar-score log-density ratio.

        Args:
            scores: Scalar response-level verifier scores, finite and strictly
                in ``(0, 1)``.  Any array shape is accepted.

        Returns:
            An array with the same shape as ``scores`` containing
            :math:`\log\widehat f_1(p)-\log\widehat f_0(p)`.

        Notes:
            The two KDEs are evaluated over logit-transformed scores.  The
            Jacobian for conversion back to score-space densities is identical
            for both classes and cancels in their ratio.
        """
        values = _probability_scores(scores, name="scores")
        logits = _logit(values)
        log_correct = _log_gaussian_kde(
            logits, self.correct_logits, self.correct_bandwidth
        )
        log_incorrect = _log_gaussian_kde(
            logits, self.incorrect_logits, self.incorrect_bandwidth
        )
        return log_correct - log_incorrect

    def weights(self, scores: Any, *, n_answers: int) -> np.ndarray:
        r"""Evaluate KDE vote weights for one response pool.

        Args:
            scores: A nonempty one-dimensional pool of scalar response scores,
                finite and strictly in ``(0, 1)``.
            n_answers: Number :math:`m\ge2` of distinct valid answers in that
                same pool.

        Returns:
            One response weight per score.

        Formula:
            With :math:`\widehat q_M` equal to the mean fitted correctness
            probability over this one response pool,

            .. math::

                w(p)=\log\widehat f_1(p)-\log\widehat f_0(p)
                     +\log\widehat q_M+\log(m-1)
                     -\log(1-\widehat q_M).

        Notes:
            Exact empirical values :math:`\widehat q_M=0` or ``1`` produce
            extended weights ``-inf`` or ``+inf``.  :func:`kde_weighted_vote`
            compares answer groups using the exact corresponding limit.
        """
        if isinstance(n_answers, bool) or not isinstance(n_answers, (int, np.integer)):
            raise ValueError(f"n_answers must be an integer >= 2; got {n_answers!r}.")
        if int(n_answers) < 2:
            raise ValueError("n_answers must be >= 2 for the KDE weight formula.")

        values = _probability_scores(scores, name="scores")
        if values.ndim != 1 or values.size == 0:
            raise ValueError("scores must be a nonempty 1D response pool.")
        q_hat = float(np.mean(self.calibrated_probability(values)))
        if q_hat == 0.0:
            offset = float("-inf")
        elif q_hat == 1.0:
            offset = float("inf")
        else:
            offset = math.log(q_hat) + math.log(int(n_answers) - 1) - math.log1p(-q_hat)
        return self.log_density_ratio(values) + offset


def _resolve_bandwidth(samples: np.ndarray, specification: Any, label: str) -> float:
    if isinstance(specification, str):
        if specification != "scott":
            raise ValueError("bandwidth must be a positive number, pair, or 'scott'.")
        if samples.size < 2:
            raise ValueError(
                f"Scott bandwidth for {label} needs at least two samples; "
                "supply an explicit bandwidth instead."
            )
        standard_deviation = float(np.std(samples, ddof=1))
        if not math.isfinite(standard_deviation) or standard_deviation <= 0.0:
            raise ValueError(
                f"Scott bandwidth is undefined for constant {label} logits; "
                "supply an explicit positive bandwidth instead."
            )
        return standard_deviation * samples.size ** (-1.0 / 5.0)

    try:
        value = float(specification)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "bandwidth must be a positive number, pair, or 'scott'."
        ) from exc
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"bandwidth must be finite and > 0; got {specification!r}.")
    return value


def _bandwidth_specifications(bandwidth: Any) -> tuple[Any, Any]:
    """Split a scalar/shared setting or a two-class bandwidth setting."""
    if isinstance(bandwidth, np.ndarray) and bandwidth.ndim == 0:
        scalar = bandwidth.item()
        return scalar, scalar
    if isinstance(bandwidth, (tuple, list, np.ndarray)):
        if len(bandwidth) != 2:
            raise ValueError("a bandwidth sequence must be (correct, incorrect).")
        return bandwidth[0], bandwidth[1]
    return bandwidth, bandwidth


def fit_kde_vote_calibration(
    scores: Any,
    correct: Any,
    *,
    n_bins: int = 10,
    bandwidth: Any = "scott",
) -> KDEVoteCalibration:
    r"""Fit KDE score densities and a binned correctness calibrator.

    Fit once on labeled scalar response scores for a fixed generator, verifier,
    score construction, and target distribution, then reuse the returned state
    with :func:`kde_weighted_vote`.  The arrays may have any common shape and
    are flattened across calibration responses.

    ``scores`` contains one value per response.  Reduce step-level score
    sequences before fitting, and use the same reduction at inference time.

    References:
        Kuang, P., Wang, Y., Han, X., Liu, Y., Xu, K., & Wang, H. (2025).
        Optimal Aggregation of LLM and PRM Signals for Efficient Test-Time
        Scaling. *ICLR 2026*, *arXiv:2510.13918* (Sec. 4.1, Eq. 2).
        https://arxiv.org/abs/2510.13918

    Args:
        scores: Scalar response-level calibration scores, finite and strictly
            in ``(0, 1)``.
        correct: Same-shape boolean or 0/1 labels indicating whether each
            response's extracted final answer is correct.  Both classes must be
            present.
        n_bins: Requested number of equal-frequency bins.  Repeated quantile
            boundaries are collapsed, so the fitted count can be smaller.
        bandwidth: ``"scott"`` (default) applies Scott's rule separately to
            each class; one positive number is shared by both KDEs; or a
            ``(correct, incorrect)`` pair sets class-specific bandwidths.

    Returns:
        Fitted :class:`KDEVoteCalibration` state.

    Formula:
        With :math:`z_i=\operatorname{logit}(p_i)`, the Gaussian KDE for final
        correctness class :math:`c\in\{0,1\}` is

        .. math::

            \widehat f_c(p)=\frac{1}{|D_c|h_c}
              \sum_{i\in D_c}K\!\left(
                \frac{\operatorname{logit}(p)-z_i}{h_c}\right).

        A separate quantile-binned empirical calibrator :math:`g(p)` estimates
        final-answer correctness for the response-pool reliability term.

    Notes:
        The estimator uses Gaussian kernels, separate class bandwidths,
        quantile bins, and unsmoothed empirical correctness frequencies.
        ``bandwidth`` and ``n_bins`` control the configurable parts of this
        fit.

    Examples:
        >>> calibration = fit_kde_vote_calibration(
        ...     [0.8, 0.9, 0.1, 0.2], [1, 1, 0, 0],
        ...     n_bins=2, bandwidth=0.5,
        ... )
        >>> calibration.n_bins
        2
        >>> bool(calibration.log_density_ratio([0.85])[0] > 0)
        True
    """
    if isinstance(n_bins, bool) or not isinstance(n_bins, (int, np.integer)):
        raise ValueError(f"n_bins must be an integer >= 1; got {n_bins!r}.")
    if int(n_bins) < 1:
        raise ValueError(f"n_bins must be >= 1; got {n_bins}.")

    score_array = _probability_scores(scores, name="scores")
    correct_array = np.asarray(correct)
    if correct_array.shape != score_array.shape:
        raise ValueError(
            "scores and correct must have the same shape; got "
            f"{score_array.shape} and {correct_array.shape}."
        )
    try:
        numeric_correct = correct_array.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("correct must contain only boolean or 0/1 values.") from exc
    if not np.all(np.isfinite(numeric_correct)) or not np.all(
        (numeric_correct == 0.0) | (numeric_correct == 1.0)
    ):
        raise ValueError("correct must contain only boolean or 0/1 values.")

    flat_scores = score_array.reshape(-1)
    flat_correct = numeric_correct.reshape(-1).astype(bool)
    if flat_scores.size == 0:
        raise ValueError("need at least one calibration response.")
    if not np.any(flat_correct) or np.all(flat_correct):
        raise ValueError("KDE calibration needs correct and incorrect responses.")

    logits = _logit(flat_scores)
    correct_logits = logits[flat_correct]
    incorrect_logits = logits[~flat_correct]
    correct_spec, incorrect_spec = _bandwidth_specifications(bandwidth)
    correct_bandwidth = _resolve_bandwidth(
        correct_logits, correct_spec, "correct-class"
    )
    incorrect_bandwidth = _resolve_bandwidth(
        incorrect_logits, incorrect_spec, "incorrect-class"
    )

    # Observed-value boundaries avoid data-free bins on small or discrete sets.
    quantiles = np.quantile(
        flat_scores,
        np.linspace(0.0, 1.0, int(n_bins) + 1),
        method="nearest",
    )
    minimum = float(np.min(flat_scores))
    maximum = float(np.max(flat_scores))
    internal = np.unique(quantiles[1:-1])
    internal = internal[(internal > minimum) & (internal < maximum)]
    edges = np.concatenate(([-np.inf], internal, [np.inf]))
    indices = np.searchsorted(edges[1:-1], flat_scores, side="right")
    probabilities = np.empty(edges.size - 1, dtype=float)
    for bin_index in range(probabilities.size):
        in_bin = indices == bin_index
        if not np.any(in_bin):  # pragma: no cover - guarded by boundary construction
            raise RuntimeError("internal quantile construction produced an empty bin.")
        probabilities[bin_index] = float(np.mean(flat_correct[in_bin]))

    return KDEVoteCalibration(
        correct_logits=correct_logits,
        incorrect_logits=incorrect_logits,
        correct_bandwidth=correct_bandwidth,
        incorrect_bandwidth=incorrect_bandwidth,
        bin_edges=edges,
        bin_probability=probabilities,
    )


def _row_kde_vote(
    ans_row: Any, score_row: Any, calibration: KDEVoteCalibration
) -> tuple[Any, int]:
    part = _valid_indices(ans_row)
    if not part:
        return None, -1
    values = _probability_scores(
        [float(score_row[j]) for j in part], name="valid scores"
    )

    groups: dict[Any, dict[str, Any]] = {}
    for local_index, j in enumerate(part):
        answer = ans_row[j]
        if answer not in groups:
            groups[answer] = {
                "local": [],
                "first": j,
                "representative": j,
            }
        group = groups[answer]
        group["local"].append(local_index)
        if float(score_row[j]) > float(score_row[group["representative"]]):
            group["representative"] = j

    # With only one observed answer the selection is already determined, while
    # the theorem's log(m - 1) term is undefined.
    if len(groups) == 1:
        answer = next(iter(groups))
        return answer, int(groups[answer]["representative"])

    density_ratio = calibration.log_density_ratio(values)
    q_hat = float(np.mean(calibration.calibrated_probability(values)))
    n_answers = len(groups)

    # At q_hat boundaries, compare the exact limit of each group sum.  The
    # response count dominates; density evidence resolves equal-count groups.
    def key(answer: Any) -> tuple[float, float, int]:
        group = groups[answer]
        count = len(group["local"])
        ratio_sum = float(np.sum(density_ratio[group["local"]]))
        if q_hat == 1.0:
            primary = float(count)
        elif q_hat == 0.0:
            primary = float(-count)
        else:
            offset = math.log(q_hat) + math.log(n_answers - 1) - math.log1p(-q_hat)
            primary = ratio_sum + count * offset
            ratio_sum = 0.0
        return primary, ratio_sum, -int(group["first"])

    winner = max(groups, key=key)
    return winner, int(groups[winner]["representative"])


def kde_weighted_vote(
    answers: Any,
    scores: Any,
    calibration: KDEVoteCalibration,
    *,
    return_index: bool = False,
    return_score: bool = False,
) -> Any:
    r"""Select answers using fitted non-parametric KDE vote weights.

    Apply the KDE weighted-voting method of Kuang et al. to scalar
    response-level verification scores.  For each question,
    :math:`\widehat q_M` is the mean binned correctness estimate across valid
    responses, and :math:`m` is the number of distinct valid answer groups.

    Each candidate must have one score.  Reduce step-level score sequences
    before calling this function, using the same reduction as the calibration
    data.

    References:
        Kuang, P., Wang, Y., Han, X., Liu, Y., Xu, K., & Wang, H. (2025).
        Optimal Aggregation of LLM and PRM Signals for Efficient Test-Time
        Scaling. *ICLR 2026*, *arXiv:2510.13918* (Sec. 4.1).
        https://arxiv.org/abs/2510.13918

    Args:
        answers: ``(N,)`` or ``(M, N)`` sampled answer labels.  Unparsable
            entries are ignored.
        scores: Aligned scalar response-level verification probabilities,
            finite and strictly in ``(0, 1)`` wherever ``answers`` is valid.
        calibration: State returned by :func:`fit_kde_vote_calibration`, fitted
            using the same generator, verifier, scalar-score construction, and
            relevant target distribution.
        return_index: Also return the highest-scoring member index of the
            winning group; an all-unparsable row returns ``-1``.
        return_score: Also return that representative candidate's raw scalar
            score; an all-unparsable row returns ``NaN``.

    Returns:
        The selected answer per question: a scalar for one pool or an ``(M,)``
        object array for a batch.  Requested extras follow as
        ``(selected[, index][, score])``.

    Formula:
        Candidate :math:`i` receives

        .. math::

           w(p_i)=\log \widehat f_1(p_i)-\log \widehat f_0(p_i)
           +\log \widehat q_M+\log(m-1)-\log(1-\widehat q_M),

        and the selected answer is

        .. math::

            \widehat y=\arg\max_y\sum_{i:a_i=y}w(p_i).

        Ties are broken by earliest answer appearance.  Scorio's usual invalid
        answer convention defines the effective response pool: invalid answers
        neither vote nor enter :math:`\widehat q_M`.

    Examples:
        >>> calibration = fit_kde_vote_calibration(
        ...     [0.8, 0.9, 0.1, 0.2], [1, 1, 0, 0],
        ...     n_bins=2, bandwidth=0.5,
        ... )
        >>> kde_weighted_vote(
        ...     ["A", "A", "B"], [0.2, 0.2, 0.8], calibration
        ... )
        'B'
    """
    if not isinstance(calibration, KDEVoteCalibration):
        raise TypeError("calibration must be a KDEVoteCalibration.")

    Z, S, single = _normalize(answers, scores, require_scores=True)
    assert S is not None
    selected: list[Any] = []
    indices: list[int] = []
    selected_scores: list[float] = []
    for row, score_row in zip(Z, S, strict=True):
        answer, index = _row_kde_vote(row, score_row, calibration)
        selected.append(answer)
        indices.append(index)
        selected_scores.append(float(score_row[index]) if index >= 0 else float("nan"))
    return _pack(
        selected,
        indices,
        selected_scores,
        single,
        return_index=return_index,
        return_score=return_score,
    )
