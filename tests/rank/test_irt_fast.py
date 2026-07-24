import numpy as np
import pytest

from scorio import rank
from scorio.rank._base import average_event_exchangeable_scores


@pytest.mark.parametrize(
    ("fn", "kwargs"),
    [
        (rank.rasch, {"max_iter": 80}),
        (rank.rasch_map, {"prior": 1.0, "max_iter": 80}),
        (rank.rasch_2pl, {"max_iter": 300, "reg_discrimination": 0.01}),
        (
            rank.rasch_2pl_map,
            {"prior": 1.0, "max_iter": 300, "reg_discrimination": 0.01},
        ),
        (
            rank.rasch_3pl,
            {
                "max_iter": 500,
                "fix_guessing": 0.2,
                "reg_discrimination": 0.01,
                "reg_guessing": 0.1,
            },
        ),
        (
            rank.rasch_3pl_map,
            {
                "prior": 1.0,
                "max_iter": 500,
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
        max_iter=300,
        return_item_params=True,
    )
    rank_assertions.assert_ranking(ranking_2pl)
    rank_assertions.assert_scores(
        scores_2pl, expected_len=ordered_binary_small_R.shape[0]
    )
    assert set(params_2pl) == {"difficulty", "discrimination"}

    ranking_3pl, scores_3pl, params_3pl = rank.rasch_3pl(
        ordered_binary_small_R,
        max_iter=500,
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

    with pytest.raises(ValueError, match="ability/discrimination scale"):
        rank.rasch_2pl(ordered_binary_small_R, reg_discrimination=0.0)
    with pytest.raises(ValueError, match="ability/discrimination scale"):
        rank.rasch_3pl(
            ordered_binary_small_R,
            fix_guessing=0.2,
            reg_discrimination=0.0,
        )
    with pytest.raises(ValueError, match="reg_guessing must be positive"):
        rank.rasch_3pl(ordered_binary_small_R, reg_guessing=0.0)
    with pytest.raises(ValueError, match="reg_guessing must be positive"):
        rank.rasch_3pl_map(ordered_binary_small_R, reg_guessing=0.0)


def test_rasch_mml_uses_fixed_population_scale_for_item_calibration() -> None:
    rng = np.random.default_rng(20260723)
    easy = rng.binomial(1, 0.8, size=(40, 6, 10))
    hard = rng.binomial(1, 0.2, size=(40, 6, 10))

    _, _, easy_params = rank.rasch_mml(
        easy,
        max_iter=30,
        em_iter=25,
        n_quadrature=21,
        return_item_params=True,
    )
    _, _, hard_params = rank.rasch_mml(
        hard,
        max_iter=30,
        em_iter=25,
        n_quadrature=21,
        return_item_params=True,
    )

    assert easy_params["difficulty"].mean() < -0.5
    assert hard_params["difficulty"].mean() > 0.5


@pytest.mark.parametrize(
    ("fn", "kwargs"),
    [
        (rank.rasch, {"max_iter": 80}),
        (rank.rasch_map, {"max_iter": 80}),
        (rank.rasch_2pl, {"max_iter": 300}),
        (rank.rasch_2pl_map, {"max_iter": 300}),
        (rank.rasch_3pl, {"max_iter": 500, "fix_guessing": 0.2}),
        (rank.rasch_3pl_map, {"max_iter": 500, "fix_guessing": 0.2}),
        (
            rank.rasch_mml,
            {"max_iter": 12, "em_iter": 8, "n_quadrature": 9},
        ),
        (
            rank.rasch_mml_credible,
            {
                "quantile": 0.1,
                "max_iter": 12,
                "em_iter": 8,
                "n_quadrature": 9,
            },
        ),
    ],
)
def test_irt_exchangeable_models_receive_identical_scores(fn, kwargs: dict) -> None:
    R = np.array(
        [
            [[1, 0], [1, 1], [0, 1], [1, 0]],
            [[1, 0], [1, 1], [0, 1], [1, 0]],
            [[0, 0], [1, 0], [1, 0], [0, 1]],
        ]
    )

    ranking, scores = fn(R, return_scores=True, **kwargs)
    assert ranking[0] == ranking[1]
    assert scores[0] == scores[1]


@pytest.mark.parametrize(
    ("fn", "kwargs"),
    [
        (rank.rasch, {}),
        (rank.rasch_2pl, {"max_iter": 300}),
        (rank.rasch_3pl, {"max_iter": 500, "fix_guessing": 0.2}),
    ],
)
def test_joint_irt_rejects_infinite_person_and_item_mles(fn, kwargs: dict) -> None:
    extreme_person = np.array([[1, 1, 1, 1], [0, 1, 0, 1], [0, 0, 0, 0]])
    extreme_item = np.array([[1, 1, 0], [1, 0, 1], [1, 0, 0]])

    with pytest.raises(ValueError, match="no finite ability MLE"):
        fn(extreme_person, **kwargs)
    with pytest.raises(ValueError, match="no finite item-parameter estimate"):
        fn(extreme_item, **kwargs)


@pytest.mark.parametrize(
    ("fn", "kwargs"),
    [
        (rank.rasch, {}),
        (rank.rasch_2pl, {"max_iter": 300}),
        (rank.rasch_3pl, {"max_iter": 500, "fix_guessing": 0.2}),
    ],
)
def test_joint_irt_rejects_quasi_separation_without_extreme_margins(fn, kwargs) -> None:
    R = np.array(
        [
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 1, 1, 1],
            [1, 0, 1, 1],
        ]
    )
    with pytest.raises(ValueError, match="completely or quasi-separated"):
        fn(R, **kwargs)


@pytest.mark.parametrize(
    ("fn", "kwargs"),
    [
        (rank.rasch_map, {}),
        (rank.rasch_2pl_map, {"max_iter": 300}),
        (rank.rasch_3pl_map, {"max_iter": 500, "fix_guessing": 0.2}),
    ],
)
def test_uniform_ability_prior_cannot_bypass_person_mle_guard(fn, kwargs) -> None:
    R = np.array([[1, 1, 1, 1], [0, 1, 0, 1], [0, 0, 0, 0]])
    with pytest.raises(ValueError, match="no finite ability MLE"):
        fn(R, prior=rank.UniformPrior(), **kwargs)


def test_rasch_mml_handles_boundary_items_as_extended_mles() -> None:
    for value, expected_difficulty in [(1, -np.inf), (0, np.inf)]:
        R = np.full((8, 1, 3), value, dtype=int)
        ranking, scores, params = rank.rasch_mml(
            R,
            max_iter=5,
            em_iter=3,
            n_quadrature=9,
            return_item_params=True,
        )
        np.testing.assert_array_equal(ranking, np.ones(8, dtype=int))
        np.testing.assert_allclose(scores, 0.0, atol=1e-14)
        assert params["difficulty"][0] == expected_difficulty


def test_nonconvex_irt_is_model_and_item_permutation_equivariant(
    ordered_binary_small_R: np.ndarray,
) -> None:
    R = ordered_binary_small_R
    model_order = np.array([2, 0, 3, 1])
    item_order = np.array([7, 2, 9, 0, 5, 1, 8, 4, 6, 3])
    calls = [
        lambda X: rank.rasch_2pl(X, max_iter=500, return_scores=True),
        lambda X: rank.rasch_2pl_map(X, max_iter=500, return_scores=True),
        lambda X: rank.rasch_3pl(X, max_iter=500, fix_guessing=0.2, return_scores=True),
        lambda X: rank.rasch_3pl_map(
            X, max_iter=500, fix_guessing=0.2, return_scores=True
        ),
    ]

    for call in calls:
        base_ranks, base_scores = call(R)
        model_ranks, model_scores = call(R[model_order])
        item_ranks, item_scores = call(R[:, item_order])
        np.testing.assert_array_equal(model_ranks, base_ranks[model_order])
        np.testing.assert_allclose(
            model_scores, base_scores[model_order], rtol=1e-6, atol=1e-6
        )
        np.testing.assert_array_equal(item_ranks, base_ranks)
        np.testing.assert_allclose(item_scores, base_scores, rtol=1e-6, atol=1e-6)


def test_longitudinal_and_mirt_reject_boundary_item_parameters() -> None:
    base = np.array([[1, 1, 0], [1, 0, 1], [1, 0, 0]])
    R = np.repeat(base[:, :, None], 2, axis=2)
    with pytest.raises(ValueError, match="no finite item-parameter estimate"):
        rank.dynamic_irt(R, variant="growth", assume_time_axis=True)
    with pytest.raises(ValueError, match="no finite item-parameter estimate"):
        rank.mirt(R, n_factors=1, n_quadrature=7)


def test_unknown_irt_prior_subclass_is_not_assumed_exchangeable() -> None:
    class TargetPrior(rank.Prior):
        def penalty(self, theta: np.ndarray) -> float:
            target = np.array([2.0, -2.0, 0.0])
            return float(np.sum((theta - target) ** 2))

    R = np.array([[1, 0, 1, 0], [1, 0, 1, 0], [0, 1, 0, 1]])
    _, scores = rank.rasch_map(R, prior=TargetPrior(), return_scores=True)
    assert scores[0] != scores[1]


def test_exact_exchangeability_detects_joint_and_cyclic_automorphisms() -> None:
    joint_swaps = np.array(
        [
            [1, 0, 0, 1, 0],
            [0, 1, 0, 0, 1],
            [1, 1, 0, 0, 0],
            [0, 0, 1, 0, 1],
        ]
    )
    joint_scores = average_event_exchangeable_scores(
        np.array([0.0, 10.0, 20.0, 4.0]), joint_swaps
    )
    np.testing.assert_array_equal(joint_scores, [2.0, 15.0, 15.0, 2.0])

    cyclic = np.array([[0, 2, 1], [1, 0, 2], [2, 1, 0]])
    cyclic_scores = average_event_exchangeable_scores([1.0, 2.0, 6.0], cyclic)
    np.testing.assert_allclose(cyclic_scores, 3.0)


def test_nonconvex_irt_rejects_ambiguous_and_quasi_separated_joint_fits() -> None:
    ambiguous = np.array(
        [
            [1, 1, 0, 0, 1, 0],
            [1, 0, 1, 0, 0, 1],
            [0, 1, 1, 0, 1, 0],
            [0, 0, 0, 1, 1, 1],
        ]
    )
    with pytest.raises(ValueError, match="multiple equally good nonconvex"):
        rank.rasch_2pl(ambiguous, max_iter=500)

    quasi_separated = np.array(
        [
            [1, 0, 1, 1],
            [0, 1, 1, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    with pytest.raises(ValueError, match="no finite joint location estimate"):
        rank.rasch_3pl(quasi_separated, fix_guessing=0.2, max_iter=500)


def test_2pl_respects_item_order_on_asymmetric_profile() -> None:
    R = np.array(
        [
            [1, 0, 0, 0, 0, 1, 1],
            [0, 1, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 1, 0, 0],
            [1, 1, 1, 0, 0, 0, 0],
            [1, 0, 0, 1, 0, 0, 0],
        ]
    )
    item_order = np.array([6, 2, 3, 0, 5, 1, 4])
    base_ranks, base_scores = rank.rasch_2pl(R, max_iter=500, return_scores=True)
    item_ranks, item_scores = rank.rasch_2pl(
        R[:, item_order], max_iter=500, return_scores=True
    )
    np.testing.assert_array_equal(item_ranks, base_ranks)
    np.testing.assert_allclose(item_scores, base_scores, atol=1e-5)


def test_dynamic_growth_uses_the_correct_sufficient_statistic() -> None:
    R = np.array(
        [
            [[1, 1, 0], [0, 1, 0], [1, 0, 0], [0, 1, 1]],
            [[1, 0, 0], [1, 1, 1], [0, 0, 0], [1, 1, 0]],
            [[1, 1, 1], [1, 0, 1], [1, 0, 0], [0, 0, 0]],
            [[1, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0]],
        ]
    )
    item_order = np.array([0, 3, 1, 2])
    base_ranks, base_scores = rank.dynamic_irt(
        R,
        variant="growth",
        assume_time_axis=True,
        return_scores=True,
    )
    item_ranks, item_scores = rank.dynamic_irt(
        R[:, item_order],
        variant="growth",
        assume_time_axis=True,
        return_scores=True,
    )
    assert base_ranks[0] == base_ranks[2]
    assert base_scores[0] == base_scores[2]
    np.testing.assert_array_equal(item_ranks, base_ranks)
    np.testing.assert_allclose(item_scores, base_scores, atol=2e-4)

    _, gain_scores, params = rank.dynamic_irt(
        R,
        variant="growth",
        score_target="gain",
        assume_time_axis=True,
        return_item_params=True,
    )
    np.testing.assert_allclose(
        gain_scores, params["ability_path"][:, -1] - params["ability_path"][:, 0]
    )
    np.testing.assert_allclose(gain_scores, params["slope"])


def test_longitudinal_irt_rejects_unidentified_settings() -> None:
    R = np.array(
        [
            [[1, 1], [1, 1]],
            [[1, 0], [0, 1]],
            [[0, 1], [1, 0]],
        ]
    )
    with pytest.raises(ValueError, match="no finite ability MLE"):
        rank.dynamic_irt(R, variant="growth", assume_time_axis=True)
    with pytest.raises(ValueError, match="slope_reg must be positive"):
        rank.dynamic_irt(R[1:], variant="growth", assume_time_axis=True, slope_reg=0.0)
    with pytest.raises(ValueError, match="state_reg must be positive"):
        rank.dynamic_irt(
            R[1:], variant="state_space", assume_time_axis=True, state_reg=0.0
        )

    one_time = R[:, :, :1]
    with pytest.raises(ValueError, match="at least two time points"):
        rank.dynamic_irt(one_time, variant="growth", assume_time_axis=True)
