import numpy as np
import pytest

from scorio import rank


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
