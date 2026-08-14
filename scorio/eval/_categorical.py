"""Validated, immutable inputs for categorical evaluation metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ._inputs import _integer_matrix_view, _readonly_int64


def _readonly_float64(values: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Return a defensive, read-only float64 copy."""
    array = np.array(values, dtype=np.float64, order="C", copy=True)
    array.setflags(write=False)
    return array


def _scaled_rewards(
    values: npt.NDArray[np.float64],
) -> tuple[float, float, npt.NDArray[np.float64]]:
    """Return ``offset, scale, normalized`` without overflowing reward gaps."""
    offset = float(values[0])
    with np.errstate(over="ignore", invalid="ignore"):
        centered = values - offset
    if np.all(np.isfinite(centered)):
        scale = float(np.max(np.abs(centered)))
        if scale == 0.0:
            return offset, 0.0, np.zeros_like(values)
        return offset, scale, centered / scale

    scale = float(np.max(np.abs(values)))
    return 0.0, scale, values / scale


def _normalize_weights(
    weights: npt.ArrayLike | None,
    results: npt.NDArray[np.int64],
) -> npt.NDArray[np.float64]:
    if weights is None:
        if not np.all((results == 0) | (results == 1)):
            unique_values = ", ".join(map(str, np.unique(results)))
            raise ValueError(
                f"R is not binary (observed: {unique_values}), so weight "
                "vector 'w' must be provided."
            )
        return _readonly_float64([0.0, 1.0])

    try:
        array = np.asarray(weights)
    except (TypeError, ValueError) as exc:
        raise ValueError("w must be a finite, non-empty 1D numeric array.") from exc

    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"w must be a non-empty 1D array; got shape {array.shape}.")
    is_bool = np.issubdtype(array.dtype, np.bool_)
    is_numeric = np.issubdtype(array.dtype, np.number)
    if np.issubdtype(array.dtype, np.complexfloating):
        raise ValueError("w must contain real values, not complex values.")
    if not (is_bool or is_numeric):
        raise ValueError("w must contain only numeric values.")
    if not np.all(np.isfinite(array)):
        raise ValueError("w must contain only finite values.")

    with np.errstate(over="ignore", invalid="ignore"):
        normalized = np.array(array, dtype=np.float64, order="C", copy=True)
    if not np.all(np.isfinite(normalized)):
        raise ValueError("w values must be representable as finite float64 values.")
    normalized.setflags(write=False)
    return normalized


def _normalize_prior_results(
    prior_results: npt.ArrayLike | None,
    question_count: int,
) -> npt.NDArray[np.int64]:
    if prior_results is None:
        array = np.empty((question_count, 0), dtype=np.int64)
    else:
        try:
            array = np.asarray(prior_results)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "R0 must be a rectangular numeric or boolean array."
            ) from exc

        if array.ndim == 1:
            try:
                array = array.reshape(question_count, -1)
            except ValueError as exc:
                raise ValueError(
                    "A 1D R0 must contain a multiple of M entries so it can "
                    "be reshaped to one row per question."
                ) from exc
        elif array.ndim != 2:
            raise ValueError(
                f"R0 must be a 1D or 2D array; got {array.ndim} dimensions."
            )

        if array.shape[0] != question_count:
            raise ValueError("R0 must have the same number of rows (M) as R.")

    validated = _integer_matrix_view(array, name="R0", allow_empty_trials=True)
    return np.asarray(validated, dtype=np.int64, order="C")


def _validate_category_range(
    matrix: npt.NDArray[np.int64],
    category_count: int,
    name: str,
) -> None:
    if matrix.size and (np.any(matrix < 0) or np.any(matrix >= category_count)):
        raise ValueError(
            f"Entries of {name} must be integers in [0, {category_count - 1}]."
        )


def _row_category_counts(
    matrix: npt.NDArray[np.int64],
    category_count: int,
) -> npt.NDArray[np.int64]:
    """Count categories in every row with one vectorized bincount."""
    question_count = matrix.shape[0]
    row_offsets = category_count * np.arange(question_count, dtype=np.int64)
    flat_indices = (matrix + row_offsets[:, None]).ravel()
    counts = np.bincount(
        flat_indices,
        minlength=question_count * category_count,
    ).reshape(question_count, category_count)
    return _readonly_int64(counts)


@dataclass(frozen=True, slots=True, eq=False)
class CategoricalBank:
    """Prepared categorical outcomes and their per-row sufficient statistics."""

    weights: npt.NDArray[np.float64]
    counts: npt.NDArray[np.int64]
    prior_counts: npt.NDArray[np.int64]
    question_count: int
    trial_count: int
    prior_trial_count: int
    category_count: int

    def _group_counts(
        self,
        category_counts: npt.NDArray[np.int64],
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
        levels, inverse = np.unique(self.weights, return_inverse=True)
        grouped = np.zeros(
            (self.question_count, levels.size),
            dtype=np.int64,
        )
        for category, group in enumerate(inverse):
            grouped[:, group] += category_counts[:, category]
        return _readonly_int64(grouped), _readonly_float64(levels)

    def grouped_observed_counts(
        self,
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
        """Return observed counts grouped by ascending reward level."""
        return self._group_counts(self.counts)

    def grouped_posterior_counts(
        self,
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
        """Return Dirichlet(+1) counts grouped by ascending reward level."""
        posterior = self.counts + self.prior_counts + 1
        return self._group_counts(posterior)


def prepare_categorical_bank(
    R: npt.ArrayLike,
    w: npt.ArrayLike | None = None,
    R0: npt.ArrayLike | None = None,
) -> CategoricalBank:
    """Validate categorical inputs and prepare their sufficient statistics."""
    results = np.asarray(
        _integer_matrix_view(R, name="R"),
        dtype=np.int64,
        order="C",
    )
    weights = _normalize_weights(w, results)
    question_count = int(results.shape[0])
    prior_results = _normalize_prior_results(R0, question_count)
    category_count = int(weights.size)

    _validate_category_range(results, category_count, "R")
    _validate_category_range(prior_results, category_count, "R0")

    return CategoricalBank(
        weights=weights,
        counts=_row_category_counts(results, category_count),
        prior_counts=_row_category_counts(prior_results, category_count),
        question_count=question_count,
        trial_count=int(results.shape[1]),
        prior_trial_count=int(prior_results.shape[1]),
        category_count=category_count,
    )


def bayes_moments(bank: CategoricalBank) -> tuple[float, float]:
    """Compute Bayes@N mean and standard deviation from prepared counts."""
    posterior_counts = bank.counts + bank.prior_counts + 1
    total = bank.category_count + bank.prior_trial_count + bank.trial_count
    offset, scale, normalized_weights = _scaled_rewards(bank.weights)
    if scale == 0.0:
        return offset, 0.0

    probabilities = posterior_counts / total
    row_means = probabilities @ normalized_weights
    mean = offset + scale * float(np.mean(row_means))

    row_variances = probabilities @ (normalized_weights**2) - row_means**2
    tolerance = 64.0 * np.finfo(float).eps
    if np.any(row_variances < -tolerance):
        raise FloatingPointError("Bayes@N posterior variance is materially negative")
    row_variances = np.maximum(row_variances, 0.0)
    normalized_variance = float(np.sum(row_variances)) / (
        bank.question_count**2 * (total + 1)
    )
    sigma = scale * math.sqrt(normalized_variance)
    return float(mean), float(sigma)


def observed_mean(bank: CategoricalBank) -> float:
    """Compute the weighted mean of a prepared observed bank stably."""
    offset, scale, normalized_weights = _scaled_rewards(bank.weights)
    if scale == 0.0:
        return offset
    normalized_mean = float(np.sum(bank.counts @ normalized_weights)) / (
        bank.question_count * bank.trial_count
    )
    return float(offset + scale * normalized_mean)


__all__ = [
    "CategoricalBank",
    "bayes_moments",
    "observed_mean",
    "prepare_categorical_bank",
]
