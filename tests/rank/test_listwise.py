import numpy as np
import pytest

from scorio import rank


@pytest.mark.parametrize(
    ("fn", "kwargs"),
    [
        (rank.plackett_luce, {"max_iter": 100}),
        (rank.plackett_luce_map, {"prior": 1.0, "max_iter": 100}),
        (rank.davidson_luce, {"max_iter": 100}),
        (rank.davidson_luce_map, {"prior": 1.0, "max_iter": 100}),
        (rank.bradley_terry_luce, {"max_iter": 100}),
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
