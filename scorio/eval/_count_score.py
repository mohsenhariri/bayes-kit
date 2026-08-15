"""Finite-bank binary scores expressed as credit for each success count.

For a subset of ``k`` trials, a :class:`CountScore` stores the credit assigned
when that subset contains exactly ``j`` successes, for ``j = 0, ..., k``.  This
single representation covers Pass@k, unanimity, threshold scores, mG-Pass, and
weighted threshold spectra.

The numerical kernel evaluates the expectation of that credit under the exact
hypergeometric distribution.  It never forms ratios of binomial coefficients,
which keeps large valid banks finite.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import hypergeom

from ._inputs import BinaryBank, validate_finite_k, validate_latent_k

_MAX_PMF_ENTRIES = 1_000_000


def _real_vector(values: ArrayLike, *, name: str) -> NDArray[np.float64]:
    """Normalize a vector without silently discarding type information."""
    try:
        array = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric or boolean array") from exc
    if np.issubdtype(array.dtype, np.complexfloating):
        raise ValueError(f"{name} must contain real values, not complex values")
    if not (
        np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.bool_)
    ):
        raise ValueError(f"{name} must contain only numeric values")
    with np.errstate(over="ignore", invalid="ignore"):
        normalized = np.asarray(array, dtype=float)
    if not np.all(np.isfinite(normalized)):
        raise ValueError(f"{name} must be finite")
    return normalized


@dataclass(frozen=True, slots=True, eq=False)
class CountScore:
    """Credit assigned to each possible success count in ``k`` trials.

    ``values[j]`` is the score for a selected subset containing exactly ``j``
    successes.  Construction makes a defensive, read-only float64 copy so the
    score can be safely reused.
    """

    k: int
    values: NDArray[np.float64]

    def __post_init__(self) -> None:
        k = validate_latent_k(self.k)
        values = _real_vector(self.values, name="values")
        if values.ndim != 1 or values.shape[0] != k + 1:
            raise ValueError(
                f"values must be a length-{k + 1} 1D array; got shape {values.shape}"
            )
        stored = np.array(values, dtype=np.float64, copy=True)
        stored.setflags(write=False)
        object.__setattr__(self, "k", k)
        object.__setattr__(self, "values", stored)

    @classmethod
    def pass_at_k(cls, k: int) -> CountScore:
        """Return credit one for at least one success."""
        k = validate_latent_k(k)
        values = np.ones(k + 1, dtype=float)
        values[0] = 0.0
        return cls(k, values)

    @classmethod
    def unanimous_at_k(cls, k: int) -> CountScore:
        """Return credit one only when all ``k`` trials succeed."""
        return cls.threshold_at_k(k, k)

    @classmethod
    def threshold_at_k(cls, k: int, threshold: int) -> CountScore:
        """Return credit one for at least ``threshold`` successes."""
        k = validate_latent_k(k)
        if isinstance(threshold, (bool, np.bool_)) or not isinstance(
            threshold, (int, np.integer)
        ):
            raise ValueError(f"threshold must be an integer; got {threshold!r}")
        threshold = int(threshold)
        if not 1 <= threshold <= k:
            raise ValueError(f"threshold must satisfy 1 <= threshold <= k ({k})")
        values = np.zeros(k + 1, dtype=float)
        values[threshold:] = 1.0
        return cls(k, values)

    @classmethod
    def mg_at_k(cls, k: int) -> CountScore:
        r"""Return the published discrete mG-Pass credit levels.

        For ``m = ceil(k / 2)``, the exact-count credit is

        .. math::

            A_j = \frac{2}{k}\max(j-m, 0).
        """
        k = validate_latent_k(k)
        majority = (k + 1) // 2
        successes = np.arange(k + 1, dtype=float)
        values = (2.0 / k) * np.maximum(successes - majority, 0.0)
        return cls(k, values)

    @classmethod
    def auc_at_k(cls, k: int) -> CountScore:
        r"""Return the degree-``k`` AUC@k credit levels.

        AUC@k is a trapezoidal weighted average of Pass@j for
        ``j = 1, ..., k``.  For exactly ``s`` successes among ``k`` draws, its
        degree-elevated credit is the same weighted average of the finite-bank
        Pass@j values for a bank of size ``k`` containing ``s`` successes.
        """
        k = validate_latent_k(k)
        if k == 1:
            coefficients = np.ones(1, dtype=float)
        else:
            coefficients = np.full(k, 1.0 / (k - 1), dtype=float)
            coefficients[[0, -1]] = 0.5 / (k - 1)

        successes = np.arange(k + 1, dtype=int)[:, None]
        values = np.zeros(k + 1, dtype=float)
        chunk_size = max(1, _MAX_PMF_ENTRIES // (k + 1))
        for start in range(0, k, chunk_size):
            stop = min(start + chunk_size, k)
            budgets = np.arange(start + 1, stop + 1, dtype=int)[None, :]
            pass_values = hypergeom.sf(0, k, successes, budgets)
            values += pass_values @ coefficients[start:stop]
        return cls(k, values)

    @classmethod
    def spectrum(cls, weights: ArrayLike) -> CountScore:
        r"""Return cumulative credit from threshold weights ``w_1..w_k``.

        A subset with exactly ``j`` successes receives
        :math:`A_j = \sum_{r \le j} w_r`.  Weights must be finite,
        non-negative, and sum to at most one.
        """
        weight_array = _real_vector(weights, name="weights")
        if weight_array.ndim != 1 or weight_array.size == 0:
            raise ValueError("weights must be a non-empty 1D array")
        if np.any(weight_array < 0.0):
            raise ValueError("weights must be non-negative")
        weight_sum = float(np.sum(weight_array))
        if weight_sum > 1.0 + 1e-12:
            raise ValueError(
                f"weights must satisfy sum(weights) <= 1; got sum={weight_sum}"
            )

        values = np.concatenate(
            (np.array([0.0], dtype=float), np.cumsum(weight_array, dtype=float))
        )
        return cls(int(weight_array.size), values)

    def row_scores(self, bank: BinaryBank) -> NDArray[np.float64]:
        """Evaluate one finite-bank score per question."""
        validate_finite_k(bank.trial_count, self.k)
        return self._scores_from_successes(bank.successes, bank.trial_count)

    def state_scores(self, bank: BinaryBank) -> NDArray[np.float64]:
        """Evaluate scores aligned with ``bank.unique_successes``."""
        validate_finite_k(bank.trial_count, self.k)
        return self._scores_from_successes(
            bank.unique_successes,
            bank.trial_count,
        )

    def mean(self, bank: BinaryBank) -> float:
        """Evaluate the dataset mean using the bank's count histogram."""
        unique_scores = self.state_scores(bank)
        return float(
            np.dot(bank.frequencies.astype(float, copy=False), unique_scores)
            / bank.question_count
        )

    def _scores_from_successes(
        self,
        successes: NDArray[np.int64],
        trial_count: int,
    ) -> NDArray[np.float64]:
        """Evaluate scores for validated per-row success counts."""
        count_array = np.asarray(successes, dtype=np.int64)

        if np.all(self.values == self.values[0]):
            return np.full(count_array.shape, self.values[0], dtype=float)

        # A zero prefix followed by a one suffix is exactly a hypergeometric
        # survival probability.  This covers Pass, unanimity, and all binary
        # thresholds without summing a full PMF.
        nonzero = np.flatnonzero(self.values != 0.0)
        if nonzero.size:
            threshold = int(nonzero[0])
            if np.all(self.values[:threshold] == 0.0) and np.all(
                self.values[threshold:] == 1.0
            ):
                values = hypergeom.sf(
                    threshold - 1,
                    trial_count,
                    count_array,
                    self.k,
                )
                return np.asarray(values, dtype=float)

        output = np.empty(count_array.size, dtype=float)
        chunk_size = max(1, _MAX_PMF_ENTRIES // (self.k + 1))
        exact_successes = np.arange(self.k + 1, dtype=int)[:, None]
        for start in range(0, count_array.size, chunk_size):
            stop = min(start + chunk_size, count_array.size)
            probabilities = hypergeom.pmf(
                exact_successes,
                trial_count,
                count_array[None, start:stop],
                self.k,
            )
            output[start:stop] = self.values @ probabilities
        return output


def pass_curve(bank: BinaryBank, max_k: int) -> NDArray[np.float64]:
    """Return dataset Pass@k for every ``k`` from one through ``max_k``.

    The curve is evaluated in one stable broadcast over the prepared success
    histogram, making it suitable for AUC-style metrics.
    """
    max_k = validate_finite_k(bank.trial_count, max_k)
    curve = np.empty(max_k, dtype=float)
    unique_count = bank.unique_successes.size
    chunk_size = max(1, _MAX_PMF_ENTRIES // unique_count)
    frequencies = bank.frequencies.astype(float, copy=False)
    for start in range(0, max_k, chunk_size):
        stop = min(start + chunk_size, max_k)
        budgets = np.arange(start + 1, stop + 1, dtype=int)[None, :]
        probabilities = hypergeom.sf(
            0,
            bank.trial_count,
            bank.unique_successes[:, None],
            budgets,
        )
        curve[start:stop] = frequencies @ probabilities / bank.question_count
    return curve


__all__ = ["CountScore", "pass_curve"]
