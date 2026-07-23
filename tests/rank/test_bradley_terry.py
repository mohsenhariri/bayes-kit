import numpy as np
import pytest

from scorio import rank


@pytest.mark.parametrize(
    "fn",
    [
        rank.bradley_terry_map,
        rank.bradley_terry_davidson,
        rank.bradley_terry_davidson_map,
        rank.rao_kupper,
        rank.rao_kupper_map,
    ],
)
def test_bt_family_smoke_and_ordering(
    ordered_binary_R: np.ndarray,
    rank_assertions,
    fn,
) -> None:
    kwargs = {"max_iter": 100, "return_scores": True}
    if fn in {
        rank.bradley_terry_map,
        rank.bradley_terry_davidson_map,
        rank.rao_kupper_map,
    }:
        kwargs["prior"] = 1.0
    if fn in {rank.rao_kupper, rank.rao_kupper_map}:
        kwargs["tie_strength"] = 1.1

    ranking, _ = rank_assertions.assert_ranking_and_scores(
        fn(ordered_binary_R, **kwargs)
    )
    rank_assertions.assert_ordering_sanity(ranking, best_idx=0, worst_idx=3)


def test_bradley_terry_finite_mle() -> None:
    R = np.array(
        [
            [1, 0, 0, 1],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
        ]
    )
    ranking, scores = rank.bradley_terry(R, return_scores=True)

    assert ranking.tolist() == [1, 2, 2]
    assert scores[0] > scores[1]
    assert scores[1] == scores[2]


def test_bradley_terry_rejects_nonexistent_finite_mle(
    ordered_binary_R: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match="no finite maximum-likelihood estimate"):
        rank.bradley_terry(ordered_binary_R)


def test_bt_map_prior_coercion_float_and_object(ordered_binary_R: np.ndarray) -> None:
    _, scores_float = rank.bradley_terry_map(
        ordered_binary_R,
        prior=1.0,
        max_iter=80,
        return_scores=True,
    )
    _, scores_object = rank.bradley_terry_map(
        ordered_binary_R,
        prior=rank.GaussianPrior(mean=0.0, var=1.0),
        max_iter=80,
        return_scores=True,
    )

    assert scores_float.shape == scores_object.shape
    assert np.all(np.isfinite(scores_float))
    assert np.all(np.isfinite(scores_object))


def test_bt_family_validation_errors(
    ordered_binary_R: np.ndarray,
    tie_heavy_R: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match="max_iter must be > 0"):
        rank.bradley_terry(ordered_binary_R, max_iter=0)

    with pytest.raises(
        ValueError, match="prior must be a positive finite scalar variance"
    ):
        rank.bradley_terry_map(ordered_binary_R, prior=-1.0)

    with pytest.raises(ValueError, match="tie_strength must be >= 1.0"):
        rank.rao_kupper(ordered_binary_R, tie_strength=0.9)

    with pytest.raises(ValueError, match="tie_strength=1.0 implies no ties"):
        rank.rao_kupper(tie_heavy_R, tie_strength=1.0)

    with pytest.raises(TypeError, match="prior must be a Prior object or float"):
        rank.rao_kupper_map(ordered_binary_R, prior="bad")


@pytest.mark.parametrize(
    "fn",
    [
        rank.bradley_terry_map,
        rank.bradley_terry_davidson,
        rank.bradley_terry_davidson_map,
        rank.rao_kupper,
        rank.rao_kupper_map,
    ],
)
def test_bt_family_rejects_unfinished_optimizer_iterates(
    ordered_binary_R: np.ndarray,
    fn,
) -> None:
    with pytest.raises(RuntimeError, match="optimization failed"):
        fn(ordered_binary_R, max_iter=1)


@pytest.mark.parametrize(
    "fn",
    [
        rank.bradley_terry,
        rank.bradley_terry_map,
        rank.bradley_terry_davidson,
        rank.bradley_terry_davidson_map,
        rank.rao_kupper,
        rank.rao_kupper_map,
    ],
)
def test_bt_family_identical_models_tie_exactly(fn) -> None:
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
        rank.bradley_terry_davidson,
        rank.rao_kupper,
    ],
)
def test_bt_tie_models_reject_decisive_separation(fn) -> None:
    R = np.array([[1, 1, 1, 1], [0, 0, 0, 0]])
    with pytest.raises(ValueError, match="no finite maximum-likelihood strength"):
        fn(R)


@pytest.mark.parametrize("fn", [rank.bradley_terry_davidson, rank.rao_kupper])
def test_bt_tie_models_reject_unbridged_partial_ties(fn) -> None:
    R = np.array([[1, 1], [0, 0], [0, 0]])
    with pytest.raises(ValueError, match="not strongly connected"):
        fn(R)


@pytest.mark.parametrize(
    "fn",
    [
        rank.bradley_terry_map,
        rank.bradley_terry_davidson_map,
        rank.rao_kupper_map,
    ],
)
def test_uniform_prior_cannot_bypass_finite_mle_checks(fn) -> None:
    R = np.array([[1, 1, 1, 1], [0, 0, 0, 0]])
    with pytest.raises(ValueError, match="no finite"):
        fn(R, prior=rank.UniformPrior())


def test_unknown_prior_subclass_is_not_assumed_exchangeable() -> None:
    class TargetPrior(rank.Prior):
        def penalty(self, theta: np.ndarray) -> float:
            target = np.array([2.0, -2.0, 0.0])
            return float(np.sum((theta - target) ** 2))

    R = np.array(
        [
            [1, 0, 1, 0],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
        ]
    )
    _, scores = rank.bradley_terry_map(R, prior=TargetPrior(), return_scores=True)
    assert scores[0] != scores[1]
