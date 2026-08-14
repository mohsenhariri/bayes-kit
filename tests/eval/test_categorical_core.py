from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from scorio.eval._categorical import prepare_categorical_bank


def test_categorical_bank_prepares_counts_and_reshapes_1d_prior() -> None:
    results = np.array([[0, 1, 2, 2], [2, 0, 1, 2]], dtype=int)
    weights = np.array([0.0, 0.5, 1.0])
    prior_results = np.array([2, 1, 0, 2], dtype=int)

    bank = prepare_categorical_bank(results, weights, prior_results)

    assert bank.question_count == 2
    assert bank.trial_count == 4
    assert bank.prior_trial_count == 2
    assert bank.category_count == 3
    assert np.array_equal(bank.weights, [0.0, 0.5, 1.0])
    assert np.array_equal(bank.counts, [[1, 1, 2], [1, 1, 2]])
    assert np.array_equal(bank.prior_counts, [[0, 1, 1], [1, 0, 1]])

    results[:] = 0
    weights[:] = 0.0
    prior_results[:] = 0
    assert np.array_equal(bank.weights, [0.0, 0.5, 1.0])
    assert np.array_equal(bank.counts, [[1, 1, 2], [1, 1, 2]])
    assert np.array_equal(bank.prior_counts, [[0, 1, 1], [1, 0, 1]])

    for values in (
        bank.weights,
        bank.counts,
        bank.prior_counts,
    ):
        assert not values.flags.writeable
        with pytest.raises(ValueError):
            values.flat[0] = 0

    with pytest.raises(FrozenInstanceError):
        bank.weights = np.array([0.0, 1.0])  # type: ignore[misc]


def test_categorical_bank_defaults_to_binary_weights_for_1d_results() -> None:
    bank = prepare_categorical_bank([False, True, True])

    assert bank.question_count == 1
    assert bank.trial_count == 3
    assert bank.prior_trial_count == 0
    assert np.array_equal(bank.weights, [0.0, 1.0])
    assert np.array_equal(bank.counts, [[1, 2]])
    assert np.array_equal(bank.prior_counts, [[0, 0]])


@pytest.mark.parametrize("results", ([[0, 2]], [[-1, 0]], [[2, 2]]))
def test_categorical_bank_requires_explicit_weights_for_nonbinary_results(
    results: list[list[int]],
) -> None:
    with pytest.raises(ValueError, match="weight vector 'w' must be provided"):
        prepare_categorical_bank(results)


@pytest.mark.parametrize(
    ("weights", "message"),
    (
        ([], "non-empty 1D"),
        (1.0, "non-empty 1D"),
        ([[0.0, 1.0]], "non-empty 1D"),
        ([0.0, np.nan], "finite"),
        ([0.0, np.inf], "finite"),
        ([0.0, 1.0j], "not complex"),
        (["zero", "one"], "numeric"),
    ),
)
def test_categorical_bank_rejects_invalid_weight_vectors(
    weights: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        prepare_categorical_bank([[0, 1]], weights)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("results", "message"),
    (
        ([[0.0, 0.5]], "integer-valued"),
        ([[0.0, np.nan]], "finite"),
        ([[0.0, np.inf]], "finite"),
        ([[0.0 + 0.0j, 1.0 + 0.0j]], "not complex"),
        ([["0", "1"]], "numeric or boolean"),
    ),
)
def test_categorical_bank_strictly_normalizes_results_before_casting(
    results: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        prepare_categorical_bank(results)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("prior_results", "message"),
    (
        ([0.0, 0.5], "integer-valued"),
        ([0.0, np.nan], "finite"),
        ([0.0, np.inf], "finite"),
        ([0.0 + 0.0j, 1.0 + 0.0j], "not complex"),
        (["0", "1"], "numeric or boolean"),
    ),
)
def test_categorical_bank_strictly_normalizes_prior_before_casting(
    prior_results: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        prepare_categorical_bank(
            [[0, 1], [1, 0]],
            [0.0, 1.0],
            prior_results,  # type: ignore[arg-type]
        )


def test_categorical_bank_validates_prior_shape() -> None:
    with pytest.raises(ValueError, match="multiple of M"):
        prepare_categorical_bank(
            [[0, 1], [1, 0]],
            R0=[0, 1, 0],
        )
    with pytest.raises(ValueError, match="same number of rows"):
        prepare_categorical_bank(
            [[0, 1], [1, 0]],
            R0=[[0, 1]],
        )
    with pytest.raises(ValueError, match="1D or 2D"):
        prepare_categorical_bank(
            [[0, 1], [1, 0]],
            R0=1,
        )


def test_categorical_bank_validates_category_ranges_from_weight_length() -> None:
    with pytest.raises(ValueError, match=r"Entries of R.*\[0, 1\]"):
        prepare_categorical_bank([[0, 2]], [0.0, 1.0])
    with pytest.raises(ValueError, match=r"Entries of R0.*\[0, 1\]"):
        prepare_categorical_bank([[0, 1]], [0.0, 1.0], [[2]])


def test_categorical_bank_groups_posterior_counts_by_reward() -> None:
    bank = prepare_categorical_bank(
        [[0, 1, 2, 2], [1, 1, 0, 2]],
        [1.0, 0.0, 1.0],
        [2, 0],
    )

    observed, observed_levels = bank.grouped_observed_counts()
    grouped, levels = bank.grouped_posterior_counts()

    assert np.array_equal(observed_levels, [0.0, 1.0])
    assert np.array_equal(observed, [[1, 3], [2, 2]])
    assert np.array_equal(levels, [0.0, 1.0])
    assert np.array_equal(grouped, [[2, 6], [3, 5]])
    assert not observed.flags.writeable
    assert not observed_levels.flags.writeable
    assert not grouped.flags.writeable
    assert not levels.flags.writeable
    with pytest.raises(ValueError):
        grouped[0, 0] = 0
    with pytest.raises(ValueError):
        levels[0] = 0.0
