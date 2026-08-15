"""Validated, immutable inputs shared by evaluation metric implementations."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


def _readonly_int64(values: np.ndarray) -> npt.NDArray[np.int64]:
    """Return a defensive, read-only int64 copy."""
    array = np.array(values, dtype=np.int64, order="C", copy=True)
    array.setflags(write=False)
    return array


def _integer_matrix_view(
    values: npt.ArrayLike,
    *,
    name: str = "R",
    allow_empty_trials: bool = False,
) -> np.ndarray:
    """Return a validated matrix view without copying its contents."""
    try:
        array = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must be a rectangular numeric or boolean array."
        ) from exc

    if array.ndim == 1:
        array = array.reshape(1, -1)
    elif array.ndim != 2:
        raise ValueError(
            f"{name} must be a 1D or 2D array; got {array.ndim} dimensions."
        )

    question_count, trial_count = array.shape
    if question_count == 0:
        raise ValueError(f"{name} must contain at least one question (M >= 1).")
    if trial_count == 0 and not allow_empty_trials:
        raise ValueError(f"{name} must contain at least one trial (N >= 1).")

    dtype = array.dtype
    is_bool = np.issubdtype(dtype, np.bool_)
    is_integer = np.issubdtype(dtype, np.integer)
    is_float = np.issubdtype(dtype, np.floating)
    if np.issubdtype(dtype, np.complexfloating):
        raise ValueError(f"{name} must contain real values, not complex values.")
    if not (is_bool or is_integer or is_float):
        raise ValueError(f"{name} must contain only numeric or boolean values.")

    if is_float:
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite values.")
        if not np.all(array == np.trunc(array)):
            raise ValueError(f"{name} must contain only integer-valued entries.")

    int64_info = np.iinfo(np.int64)
    if np.issubdtype(dtype, np.unsignedinteger):
        if array.size and np.max(array) > int64_info.max:
            raise ValueError(f"{name} entries must fit in a signed 64-bit integer.")
    elif is_float:
        promoted = array.astype(np.longdouble, copy=False)
        lower = np.longdouble(int64_info.min)
        upper = np.longdouble(2) ** 63
        if np.any(promoted < lower) or np.any(promoted >= upper):
            raise ValueError(f"{name} entries must fit in a signed 64-bit integer.")

    return array


def _is_binary_matrix(values: np.ndarray) -> bool:
    """Return whether an already validated, non-empty matrix is binary."""
    if np.issubdtype(values.dtype, np.bool_):
        return True
    if np.issubdtype(values.dtype, np.unsignedinteger):
        return bool(np.max(values) <= 1)
    return bool(np.min(values) >= 0 and np.max(values) <= 1)


def _integral_scalar(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer; got {value!r}.")
    return int(value)


def validate_finite_k(N: int, k: int) -> int:
    """Validate a finite-bank budget and return it as a Python ``int``."""
    trial_count = _integral_scalar(N, name="N")
    budget = _integral_scalar(k, name="k")
    if trial_count < 1:
        raise ValueError(f"N must be >= 1; got N={trial_count}.")
    if not 1 <= budget <= trial_count:
        raise ValueError(
            f"k must satisfy 1 <= k <= N (N={trial_count}); got k={budget}"
        )
    return budget


def validate_latent_k(k: int) -> int:
    """Validate a latent-resampling budget and return it as a Python ``int``."""
    budget = _integral_scalar(k, name="k")
    if budget < 1:
        raise ValueError(f"k must be >= 1; got k={budget}")
    return budget


def _finite_positive_scalar(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be a finite positive scalar; got {value!r}.")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a finite positive scalar; got {value!r}.")
    return result


def validate_beta_prior(alpha0: float, beta0: float) -> tuple[float, float]:
    """Validate Beta-prior parameters and return normalized Python floats."""
    return (
        _finite_positive_scalar(alpha0, name="alpha0"),
        _finite_positive_scalar(beta0, name="beta0"),
    )


@dataclass(frozen=True, slots=True, eq=False)
class BinaryBank:
    """Sufficient statistics prepared by :func:`prepare_binary_bank`."""

    question_count: int
    trial_count: int
    successes: npt.NDArray[np.int64]
    unique_successes: npt.NDArray[np.int64]
    frequencies: npt.NDArray[np.int64]


def prepare_binary_bank(results: npt.ArrayLike) -> BinaryBank:
    """Prepare binary statistics while preserving the public error contract."""
    matrix = _integer_matrix_view(results, name="R")
    if not _is_binary_matrix(matrix):
        raise ValueError("Entries of R must be integers in [0, 1].")

    question_count, trial_count = matrix.shape
    successes = np.sum(matrix, axis=1, dtype=np.int64)
    # Use the linear-time histogram when its O(N) storage stays modest;
    # sparse banks with many trials avoid allocating a mostly empty array.
    if trial_count + 1 <= 4 * question_count:
        histogram = np.bincount(successes, minlength=trial_count + 1)
        unique_successes = np.flatnonzero(histogram)
        frequencies = histogram[unique_successes]
    else:
        unique_successes, frequencies = np.unique(successes, return_counts=True)
    return BinaryBank(
        question_count=int(question_count),
        trial_count=int(trial_count),
        successes=_readonly_int64(successes),
        unique_successes=_readonly_int64(unique_successes),
        frequencies=_readonly_int64(frequencies),
    )


__all__ = [
    "BinaryBank",
    "prepare_binary_bank",
    "validate_beta_prior",
    "validate_finite_k",
    "validate_latent_k",
]
