from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from scorio.eval._inputs import (
    prepare_binary_bank,
    validate_beta_prior,
    validate_finite_k,
    validate_latent_k,
)


def test_binary_bank_contains_read_only_sufficient_statistics() -> None:
    results = np.array(
        [
            [0, 1, 1],
            [1, 1, 1],
            [0, 0, 0],
            [1, 0, 1],
        ],
        dtype=int,
    )

    bank = prepare_binary_bank(results)

    assert bank.question_count == 4
    assert bank.trial_count == 3
    assert np.array_equal(bank.successes, [2, 3, 0, 2])
    assert np.array_equal(bank.unique_successes, [0, 2, 3])
    assert np.array_equal(bank.frequencies, [1, 2, 1])
    for values in (bank.successes, bank.unique_successes, bank.frequencies):
        assert values.dtype == np.int64
        assert values.ndim == 1
        assert not values.flags.writeable
        with pytest.raises(ValueError, match="read-only"):
            values[0] = 10

    with pytest.raises(FrozenInstanceError):
        bank.trial_count = 10  # type: ignore[misc]


def test_binary_bank_accepts_one_question_and_does_not_alias_source() -> None:
    results = np.array([0, 1, 1, 0], dtype=int)
    bank = prepare_binary_bank(results)

    results[:] = 0

    assert bank.question_count == 1
    assert bank.trial_count == 4
    assert np.array_equal(bank.successes, [2])
    assert np.array_equal(bank.unique_successes, [2])
    assert np.array_equal(bank.frequencies, [1])


@pytest.mark.parametrize(
    ("results", "message"),
    (
        ([[0.0, 0.5]], "integer-valued"),
        ([[0.0, np.nan]], "finite"),
        ([[0, 2]], r"\[0, 1\]"),
    ),
)
def test_prepare_binary_bank_preserves_strict_validation(
    results: list[list[float]], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        prepare_binary_bank(results)


@pytest.mark.parametrize("value", (1, np.int32(2), np.int64(3)))
def test_k_validators_accept_integer_scalars(value: int | np.integer) -> None:
    assert validate_finite_k(3, value) == int(value)
    assert validate_latent_k(value) == int(value)


@pytest.mark.parametrize("value", (True, np.bool_(True), 1.0, np.float64(1.0), "1"))
def test_k_validators_reject_noninteger_types(value: object) -> None:
    with pytest.raises(ValueError, match="k must be an integer"):
        validate_finite_k(3, value)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="k must be an integer"):
        validate_latent_k(value)  # type: ignore[arg-type]


@pytest.mark.parametrize("N", (True, np.bool_(True), 3.0, np.float64(3.0)))
def test_finite_k_rejects_noninteger_trial_counts(N: object) -> None:
    with pytest.raises(ValueError, match="N must be an integer"):
        validate_finite_k(N, 1)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("N", "k"),
    (
        (0, 1),
        (3, 0),
        (3, -1),
        (3, 4),
    ),
)
def test_finite_k_rejects_values_outside_the_bank(N: int, k: int) -> None:
    with pytest.raises(ValueError):
        validate_finite_k(N, k)


@pytest.mark.parametrize("k", (0, -1))
def test_latent_k_rejects_nonpositive_values(k: int) -> None:
    with pytest.raises(ValueError, match="k must be >= 1"):
        validate_latent_k(k)


def test_validate_beta_prior_accepts_finite_positive_scalars() -> None:
    assert validate_beta_prior(np.int64(2), np.float32(0.5)) == (2.0, 0.5)


@pytest.mark.parametrize(
    ("alpha0", "beta0", "name"),
    (
        (0.0, 1.0, "alpha0"),
        (-1.0, 1.0, "alpha0"),
        (np.nan, 1.0, "alpha0"),
        (np.inf, 1.0, "alpha0"),
        (True, 1.0, "alpha0"),
        (1.0, 0.0, "beta0"),
        (1.0, -1.0, "beta0"),
        (1.0, np.nan, "beta0"),
        (1.0, np.inf, "beta0"),
        (1.0, np.bool_(True), "beta0"),
    ),
)
def test_validate_beta_prior_rejects_invalid_values(
    alpha0: object, beta0: object, name: str
) -> None:
    with pytest.raises(ValueError, match=name):
        validate_beta_prior(alpha0, beta0)  # type: ignore[arg-type]
