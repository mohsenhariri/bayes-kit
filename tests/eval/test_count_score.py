import itertools

import numpy as np
import pytest
from scipy.stats import binom, hypergeom

from scorio import eval as eval_api
from scorio.eval._count_score import CountScore, pass_curve
from scorio.eval._inputs import BinaryBank, prepare_binary_bank


def _bank_with_every_success_count(trial_count: int) -> BinaryBank:
    rows = [
        np.concatenate(
            (
                np.ones(successes, dtype=np.int64),
                np.zeros(trial_count - successes, dtype=np.int64),
            )
        )
        for successes in range(trial_count + 1)
    ]
    return prepare_binary_bank(np.vstack(rows))


def _subset_score(
    trial_count: int,
    successes: int,
    score: CountScore,
) -> float:
    row = np.concatenate(
        (
            np.ones(successes, dtype=np.int64),
            np.zeros(trial_count - successes, dtype=np.int64),
        )
    )
    values = [
        score.values[int(np.sum(row[list(indices)]))]
        for indices in itertools.combinations(range(trial_count), score.k)
    ]
    return float(np.mean(values))


def _scores_for_k(k: int) -> list[CountScore]:
    weights = np.arange(1, k + 1, dtype=float)
    weights /= 2.0 * float(np.sum(weights))
    return [
        CountScore.pass_at_k(k),
        CountScore.unanimous_at_k(k),
        CountScore.mg_at_k(k),
        CountScore.auc_at_k(k),
        CountScore.spectrum(weights),
        *(CountScore.threshold_at_k(k, threshold) for threshold in range(1, k + 1)),
    ]


def test_count_score_stores_a_finite_read_only_copy() -> None:
    source = np.array([0.0, 0.25, 1.0])
    score = CountScore(2, source)
    source[1] = 0.75

    assert score.k == 2
    assert score.values.tolist() == [0.0, 0.25, 1.0]
    assert score.values.dtype == np.float64
    assert not score.values.flags.writeable
    with pytest.raises(ValueError):
        score.values[0] = 1.0


@pytest.mark.parametrize(
    ("k", "values", "message"),
    [
        (0, [0.0], "k must be >= 1"),
        (2, [0.0, 1.0], "length-3"),
        (2, [[0.0, 0.5, 1.0]], "1D"),
        (2, [0.0, np.nan, 1.0], "finite"),
        (2, [0.0, np.inf, 1.0], "finite"),
    ],
)
def test_count_score_rejects_invalid_values(
    k: int, values: list[float], message: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        CountScore(k, values)


def test_count_score_factories_have_expected_credit_levels() -> None:
    assert CountScore.pass_at_k(4).values.tolist() == [0.0, 1.0, 1.0, 1.0, 1.0]
    assert CountScore.unanimous_at_k(4).values.tolist() == [
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ]
    assert CountScore.threshold_at_k(4, 3).values.tolist() == [
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
    ]
    assert CountScore.mg_at_k(4).values == pytest.approx([0.0, 0.0, 0.0, 0.5, 1.0])
    assert CountScore.mg_at_k(3).values == pytest.approx([0.0, 0.0, 0.0, 2.0 / 3.0])
    assert CountScore.auc_at_k(2).values == pytest.approx([0.0, 0.75, 1.0])
    assert CountScore.spectrum([0.1, 0.2, 0.3, 0.4]).values == pytest.approx(
        [0.0, 0.1, 0.3, 0.6, 1.0]
    )


@pytest.mark.parametrize(
    "factory",
    [
        lambda: CountScore.threshold_at_k(3, 0),
        lambda: CountScore.threshold_at_k(3, 4),
        lambda: CountScore.spectrum([]),
        lambda: CountScore.spectrum([[0.5, 0.5]]),
        lambda: CountScore.spectrum([0.5, -0.1]),
        lambda: CountScore.spectrum([0.5, np.nan]),
        lambda: CountScore.spectrum([0.6, 0.5]),
    ],
)
def test_count_score_factories_reject_invalid_boundaries(factory) -> None:
    with pytest.raises((TypeError, ValueError)):
        factory()


def test_finite_scores_match_exhaustive_subset_enumeration() -> None:
    for trial_count in range(1, 8):
        bank = _bank_with_every_success_count(trial_count)
        for k in range(1, trial_count + 1):
            for score in _scores_for_k(k):
                expected = np.array(
                    [
                        _subset_score(trial_count, successes, score)
                        for successes in range(trial_count + 1)
                    ]
                )
                actual = score.row_scores(bank)
                assert actual == pytest.approx(expected, abs=2e-15)
                assert score.mean(bank) == pytest.approx(float(np.mean(expected)))


def test_histogram_mean_matches_expanded_question_scores() -> None:
    results = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 1, 1],
        ]
    )
    bank = prepare_binary_bank(results)
    score = CountScore.spectrum([0.1, 0.2, 0.3, 0.4])

    assert score.mean(bank) == pytest.approx(float(np.mean(score.row_scores(bank))))
    assert score.state_scores(bank) == pytest.approx(score.row_scores(bank)[[0, 2, 5]])


def test_pass_curve_matches_exhaustive_subset_enumeration() -> None:
    for trial_count in range(1, 8):
        bank = _bank_with_every_success_count(trial_count)
        expected = [
            np.mean(
                [
                    _subset_score(
                        trial_count,
                        successes,
                        CountScore.pass_at_k(k),
                    )
                    for successes in range(trial_count + 1)
                ]
            )
            for k in range(1, trial_count + 1)
        ]
        assert pass_curve(bank, trial_count) == pytest.approx(expected, abs=2e-15)


def test_auc_count_score_matches_finite_metric_and_latent_polynomial() -> None:
    rng = np.random.default_rng(20260814)
    for trial_count in range(1, 8):
        results = rng.integers(0, 2, size=(9, trial_count), dtype=np.int64)
        bank = prepare_binary_bank(results)
        for k in range(1, trial_count + 1):
            score = CountScore.auc_at_k(k)
            assert score.mean(bank) == pytest.approx(eval_api.auc_at_k(results, k))

            if k == 1:
                coefficients = np.ones(1, dtype=float)
            else:
                coefficients = np.full(k, 1.0 / (k - 1), dtype=float)
                coefficients[[0, -1]] = 0.5 / (k - 1)
            budgets = np.arange(1, k + 1, dtype=int)
            successes = np.arange(k + 1, dtype=int)
            for probability in (0.0, 0.1, 0.5, 0.9, 1.0):
                actual = float(
                    np.dot(score.values, binom.pmf(successes, k, probability))
                )
                expected = float(
                    np.dot(coefficients, 1.0 - (1.0 - probability) ** budgets)
                )
                assert actual == pytest.approx(expected, abs=2e-15)


def test_public_binary_point_metrics_match_count_scores() -> None:
    bank = _bank_with_every_success_count(6)
    results = np.vstack(
        [
            np.concatenate(
                (
                    np.ones(successes, dtype=np.int64),
                    np.zeros(6 - successes, dtype=np.int64),
                )
            )
            for successes in range(7)
        ]
    )

    for k in range(1, 7):
        assert eval_api.pass_at_k(results, k) == pytest.approx(
            CountScore.pass_at_k(k).mean(bank)
        )
        unanimous = CountScore.unanimous_at_k(k).mean(bank)
        assert eval_api.pass_hat_k(results, k) == pytest.approx(unanimous)
        assert eval_api.g_pass_at_k(results, k) == pytest.approx(unanimous)
        assert eval_api.mg_pass_at_k(results, k) == pytest.approx(
            CountScore.mg_at_k(k).mean(bank)
        )
        assert eval_api.maj_at_k(results, k) == pytest.approx(
            CountScore.threshold_at_k(k, k // 2 + 1).mean(bank)
        )
        for tau in (0.0, 0.5, 1.0):
            threshold = max(1, int(np.ceil(tau * k)))
            assert eval_api.g_pass_at_k_tau(results, k, tau) == pytest.approx(
                CountScore.threshold_at_k(k, threshold).mean(bank)
            )
        assert eval_api.auc_at_k(results, k) == pytest.approx(
            CountScore.auc_at_k(k).mean(bank)
        )


def test_large_bank_endpoints_and_general_score_stay_finite() -> None:
    trial_count = 2000
    k = 1000
    results = np.vstack(
        (
            np.zeros(trial_count, dtype=np.int64),
            np.concatenate(
                (
                    np.ones(1500, dtype=np.int64),
                    np.zeros(500, dtype=np.int64),
                )
            ),
            np.ones(trial_count, dtype=np.int64),
        )
    )
    bank = prepare_binary_bank(results)

    pass_scores = CountScore.pass_at_k(k).row_scores(bank)
    unanimous_scores = CountScore.unanimous_at_k(k).row_scores(bank)
    mg_scores = CountScore.mg_at_k(k).row_scores(bank)

    assert np.all(np.isfinite(pass_scores))
    assert np.all(np.isfinite(unanimous_scores))
    assert np.all(np.isfinite(mg_scores))
    assert pass_scores[[0, 2]].tolist() == [0.0, 1.0]
    assert unanimous_scores[[0, 2]].tolist() == [0.0, 1.0]
    assert mg_scores[[0, 2]].tolist() == [0.0, 1.0]
    assert pass_scores[1] == pytest.approx(hypergeom.sf(0, trial_count, 1500, k))
    assert unanimous_scores[1] == pytest.approx(hypergeom.pmf(k, trial_count, 1500, k))

    curve = pass_curve(bank, k)
    assert curve.shape == (k,)
    assert np.all(np.isfinite(curve))
    assert np.all(np.diff(curve) >= -1e-15)
