import numpy as np
import pytest

from scorio import rank


@pytest.mark.parametrize(
    ("fn", "kwargs"),
    [
        (rank.rasch, {"max_iter": 80}),
        (rank.rasch_map, {"prior": 1.0, "max_iter": 80}),
        (rank.rasch_2pl, {"max_iter": 80, "reg_discrimination": 0.01}),
        (
            rank.rasch_2pl_map,
            {"prior": 1.0, "max_iter": 80, "reg_discrimination": 0.01},
        ),
        (
            rank.rasch_3pl,
            {
                "max_iter": 60,
                "fix_guessing": 0.2,
                "reg_discrimination": 0.01,
                "reg_guessing": 0.1,
            },
        ),
        (
            rank.rasch_3pl_map,
            {
                "prior": 1.0,
                "max_iter": 60,
                "fix_guessing": 0.2,
                "reg_discrimination": 0.01,
                "reg_guessing": 0.1,
            },
        ),
        (
            rank.rasch_mml,
            {"max_iter": 12, "em_iter": 8, "n_quadrature": 9},
        ),
        (
            rank.rasch_mml_credible,
            {"quantile": 0.1, "max_iter": 12, "em_iter": 8, "n_quadrature": 9},
        ),
        (rank.dynamic_irt, {"variant": "linear", "max_iter": 80}),
    ],
)
def test_irt_family_fast_smoke_and_ordering(
    ordered_binary_small_R: np.ndarray,
    rank_assertions,
    fn,
    kwargs: dict,
) -> None:
    ranking, _ = rank_assertions.assert_ranking_and_scores(
        fn(ordered_binary_small_R, return_scores=True, **kwargs)
    )
    rank_assertions.assert_ordering_sanity(ranking, best_idx=0, worst_idx=3)


def test_irt_return_item_params_branches(
    ordered_binary_small_R: np.ndarray,
    rank_assertions,
) -> None:
    N = ordered_binary_small_R.shape[2]
    time_points = np.linspace(0.0, 1.0, num=N)

    ranking_rasch, scores_rasch, params_rasch = rank.rasch(
        ordered_binary_small_R,
        max_iter=60,
        return_item_params=True,
    )
    rank_assertions.assert_ranking(ranking_rasch)
    rank_assertions.assert_scores(
        scores_rasch, expected_len=ordered_binary_small_R.shape[0]
    )
    assert set(params_rasch) == {"difficulty"}

    ranking_2pl, scores_2pl, params_2pl = rank.rasch_2pl(
        ordered_binary_small_R,
        max_iter=60,
        return_item_params=True,
    )
    rank_assertions.assert_ranking(ranking_2pl)
    rank_assertions.assert_scores(
        scores_2pl, expected_len=ordered_binary_small_R.shape[0]
    )
    assert set(params_2pl) == {"difficulty", "discrimination"}

    ranking_3pl, scores_3pl, params_3pl = rank.rasch_3pl(
        ordered_binary_small_R,
        max_iter=50,
        fix_guessing=0.2,
        return_item_params=True,
    )
    rank_assertions.assert_ranking(ranking_3pl)
    rank_assertions.assert_scores(
        scores_3pl, expected_len=ordered_binary_small_R.shape[0]
    )
    assert set(params_3pl) == {"difficulty", "discrimination", "guessing"}

    ranking_growth, scores_growth, params_growth = rank.dynamic_irt(
        ordered_binary_small_R,
        variant="growth",
        score_target="gain",
        assume_time_axis=True,
        time_points=time_points,
        max_iter=60,
        return_item_params=True,
    )
    rank_assertions.assert_ranking(ranking_growth)
    rank_assertions.assert_scores(
        scores_growth, expected_len=ordered_binary_small_R.shape[0]
    )
    assert set(params_growth) == {
        "difficulty",
        "baseline",
        "slope",
        "ability_path",
        "time_points",
    }


def test_dynamic_irt_longitudinal_variants(
    ordered_binary_small_R: np.ndarray,
    rank_assertions,
) -> None:
    N = ordered_binary_small_R.shape[2]
    time_points = np.linspace(0.0, 1.0, num=N)

    out_growth = rank.dynamic_irt(
        ordered_binary_small_R,
        variant="growth",
        score_target="gain",
        assume_time_axis=True,
        time_points=time_points,
        max_iter=60,
        return_scores=True,
    )
    out_state = rank.dynamic_irt(
        ordered_binary_small_R,
        variant="state_space",
        score_target="mean",
        assume_time_axis=True,
        time_points=time_points,
        max_iter=60,
        return_scores=True,
    )

    rank_assertions.assert_ranking_and_scores(out_growth)
    rank_assertions.assert_ranking_and_scores(out_state)


def test_irt_validation_errors(ordered_binary_small_R: np.ndarray) -> None:
    with pytest.raises(ValueError, match=r"quantile must be in \(0, 1\)"):
        rank.rasch_mml_credible(ordered_binary_small_R, quantile=1.0)

    with pytest.raises(
        ValueError, match="interprets axis-2 as ordered longitudinal time"
    ):
        rank.dynamic_irt(ordered_binary_small_R, variant="growth")

    with pytest.raises(ValueError, match="Unknown variant"):
        rank.dynamic_irt(ordered_binary_small_R, variant="bad")

    with pytest.raises(
        ValueError, match="score_target is only used for longitudinal variants"
    ):
        rank.dynamic_irt(ordered_binary_small_R, variant="linear", score_target="gain")

    with pytest.raises(ValueError, match="score_target must be one of"):
        rank.dynamic_irt(
            ordered_binary_small_R,
            variant="growth",
            assume_time_axis=True,
            score_target="bad",
        )

    with pytest.raises(
        ValueError, match=r"guessing_upper must be in \(0, 1\) and finite"
    ):
        rank.rasch_3pl(ordered_binary_small_R, guessing_upper=0.0)
