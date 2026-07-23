import numpy as np
import pytest

from scorio import rank


@pytest.mark.parametrize(
    ("fn", "kwargs"),
    [
        (rank.plackett_luce_map, {"prior": 1.0, "max_iter": 100}),
        (rank.davidson_luce_map, {"prior": 1.0, "max_iter": 100}),
        (rank.bradley_terry_luce_map, {"prior": 1.0, "max_iter": 100}),
    ],
)
def test_listwise_methods_smoke_and_ordering(
    ordered_binary_R: np.ndarray,
    rank_assertions,
    fn,
    kwargs: dict,
) -> None:
    ranking, _ = rank_assertions.assert_ranking_and_scores(
        fn(ordered_binary_R, return_scores=True, **kwargs)
    )
    rank_assertions.assert_ordering_sanity(ranking, best_idx=0, worst_idx=3)


@pytest.mark.parametrize("fn", [rank.plackett_luce, rank.bradley_terry_luce])
def test_listwise_ml_finite_cyclic_profile(fn) -> None:
    R = np.array(
        [
            [1, 0, 0, 1],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
        ]
    )
    ranking, scores = fn(R, return_scores=True)

    assert ranking.tolist() == [1, 2, 2]
    assert scores[0] > scores[1]
    assert scores[1] == scores[2]


@pytest.mark.parametrize("fn", [rank.plackett_luce, rank.bradley_terry_luce])
def test_listwise_ml_rejects_nonexistent_finite_estimate(
    ordered_binary_R: np.ndarray,
    fn,
) -> None:
    with pytest.raises(ValueError, match="no finite maximum-likelihood estimate"):
        fn(ordered_binary_R)


def test_plackett_luce_map_prior_coercion(ordered_binary_small_R: np.ndarray) -> None:
    _, scores_float = rank.plackett_luce_map(
        ordered_binary_small_R,
        prior=1.0,
        max_iter=80,
        return_scores=True,
    )
    _, scores_object = rank.plackett_luce_map(
        ordered_binary_small_R,
        prior=rank.GaussianPrior(mean=0.0, var=1.0),
        max_iter=80,
        return_scores=True,
    )

    assert scores_float.shape == scores_object.shape
    assert np.all(np.isfinite(scores_float))
    assert np.all(np.isfinite(scores_object))


def test_listwise_validation_errors(ordered_binary_small_R: np.ndarray) -> None:
    L = ordered_binary_small_R.shape[0]

    with pytest.raises(ValueError, match="max_iter must be >= 1"):
        rank.plackett_luce(ordered_binary_small_R, max_iter=0)

    with pytest.raises(ValueError, match="prior must be a finite scalar > 0"):
        rank.plackett_luce_map(ordered_binary_small_R, prior=0.0)

    with pytest.raises(
        ValueError, match=rf"max_tie_order must be <= number of models \({L}\)"
    ):
        rank.davidson_luce(ordered_binary_small_R, max_tie_order=L + 1)

    with pytest.raises(TypeError, match="prior must be a Prior object or float"):
        rank.bradley_terry_luce_map(ordered_binary_small_R, prior="bad")

    too_large_tie = np.array([[1], [1], [0]])
    with pytest.raises(ValueError, match="winner-set size exceeds max_tie_order"):
        rank.davidson_luce(too_large_tie, max_tie_order=1)
    with pytest.raises(ValueError, match="winner-set size exceeds max_tie_order"):
        rank.davidson_luce_map(too_large_tie, max_tie_order=1)


@pytest.mark.parametrize(
    "fn",
    [
        rank.plackett_luce,
        rank.plackett_luce_map,
        rank.davidson_luce,
        rank.davidson_luce_map,
        rank.bradley_terry_luce,
        rank.bradley_terry_luce_map,
    ],
)
def test_listwise_identical_models_tie_exactly(fn) -> None:
    R = np.array(
        [
            [1, 0, 1, 0],
            [1, 0, 1, 0],
            [0, 1, 1, 0],
        ]
    )
    ranking, scores = fn(R, return_scores=True)

    assert ranking[0] == ranking[1]
    assert scores[0] == scores[1]


@pytest.mark.parametrize(
    "fn",
    [
        rank.plackett_luce,
        rank.plackett_luce_map,
        rank.davidson_luce,
        rank.davidson_luce_map,
        rank.bradley_terry_luce,
        rank.bradley_terry_luce_map,
    ],
)
def test_listwise_symmetric_cycle_has_no_index_based_winner(fn) -> None:
    ranking, scores = fn(np.eye(3, dtype=int), return_scores=True)

    np.testing.assert_array_equal(ranking, np.ones(3, dtype=int))
    np.testing.assert_allclose(scores, scores[0], rtol=0.0, atol=0.0)


def test_plackett_luce_default_budget_converges_on_slow_finite_profile() -> None:
    events = [[1, 0, 0]] * 100 + [[1, 1, 0]] * 100 + [[0, 0, 1]]
    R = np.asarray(events, dtype=int).T
    _, scores = rank.plackett_luce(R, return_scores=True)
    assert np.all(np.isfinite(scores))


def test_davidson_luce_rejects_separated_single_winner_data() -> None:
    R = np.array([[1, 1, 1, 1], [0, 0, 0, 0]])
    with pytest.raises(ValueError, match="no finite maximum-likelihood strength"):
        rank.davidson_luce(R)


def test_davidson_luce_rejects_unbridged_co_winner_ties() -> None:
    R = np.array([[1, 1], [0, 1], [0, 0]])
    with pytest.raises(ValueError, match="not strongly connected"):
        rank.davidson_luce(R)


@pytest.mark.parametrize(
    "fn",
    [
        rank.plackett_luce_map,
        rank.davidson_luce_map,
        rank.bradley_terry_luce_map,
    ],
)
def test_listwise_uniform_prior_cannot_bypass_mle_checks(fn) -> None:
    R = np.array([[1, 1, 1, 1], [0, 0, 0, 0]])
    with pytest.raises(ValueError, match="no finite"):
        fn(R, prior=rank.UniformPrior())


def test_unknown_listwise_prior_subclass_is_not_assumed_exchangeable() -> None:
    class TargetPrior(rank.Prior):
        def penalty(self, theta: np.ndarray) -> float:
            target = np.array([2.0, -2.0, 0.0])
            return float(np.sum((theta - target) ** 2))

    R = np.array([[1, 0, 1], [1, 0, 1], [0, 1, 0]])
    _, scores = rank.plackett_luce_map(R, prior=TargetPrior(), return_scores=True)
    assert scores[0] != scores[1]
