import numpy as np
import pytest

from scorio import rank

# Small, fast MIRT settings for CI smoke tests.
MIRT_KW = {
    "n_factors": 2,
    "n_quadrature": 7,
    "em_iter": 25,
    "max_iter": 40,
    "tol": 1e-3,
}


def test_mirt_2pl_smoke_and_ordering(
    ordered_binary_small_R: np.ndarray,
    rank_assertions,
) -> None:
    ranking, _ = rank_assertions.assert_ranking_and_scores(
        rank.mirt(ordered_binary_small_R, return_scores=True, **MIRT_KW)
    )
    # The fixture is dominance-ordered (model 0 dominates model 3 everywhere),
    # so the reference composite must place best first and worst last.
    rank_assertions.assert_ordering_sanity(ranking, best_idx=0, worst_idx=3)


def test_mirt_single_factor_reduces_cleanly(
    ordered_binary_small_R: np.ndarray,
    rank_assertions,
) -> None:
    ranking, _ = rank_assertions.assert_ranking_and_scores(
        rank.mirt(
            ordered_binary_small_R,
            n_factors=1,
            n_quadrature=11,
            em_iter=25,
            max_iter=40,
            tol=1e-3,
            return_scores=True,
        )
    )
    rank_assertions.assert_ordering_sanity(ranking, best_idx=0, worst_idx=3)


def test_mirt_3pl_fixed_guessing_smoke(
    ordered_binary_small_R: np.ndarray,
    rank_assertions,
) -> None:
    out = rank.mirt(
        ordered_binary_small_R,
        model="3pl",
        fix_guessing=0.2,
        return_scores=True,
        **MIRT_KW,
    )
    rank_assertions.assert_ranking_and_scores(out)


def test_mirt_3pl_estimated_guessing_smoke(
    ordered_binary_small_R: np.ndarray,
    rank_assertions,
) -> None:
    out = rank.mirt(
        ordered_binary_small_R,
        model="3pl",
        return_scores=True,
        **MIRT_KW,
    )
    rank_assertions.assert_ranking_and_scores(out)


def test_mirt_3pl_item_params(
    ordered_binary_small_R: np.ndarray,
    rank_assertions,
) -> None:
    L, M, _ = ordered_binary_small_R.shape
    ranking, scores, params = rank.mirt(
        ordered_binary_small_R,
        model="3pl",
        return_item_params=True,
        **MIRT_KW,
    )
    rank_assertions.assert_ranking(ranking)
    rank_assertions.assert_scores(scores, expected_len=L)

    assert set(params) == {
        "difficulty",
        "discrimination",
        "slopes",
        "intercept",
        "abilities",
        "ability_sd",
        "guessing",
    }
    D = MIRT_KW["n_factors"]
    assert params["slopes"].shape == (M, D)
    assert params["intercept"].shape == (M,)
    assert params["abilities"].shape == (L, D)
    assert params["ability_sd"].shape == (L, D)
    assert params["difficulty"].shape == (M,)
    assert params["discrimination"].shape == (M,)
    assert params["guessing"].shape == (M,)

    # MDISC is a norm (non-negative); guessing within configured bounds.
    assert np.all(params["discrimination"] >= 0.0)
    assert np.all((params["guessing"] >= 0.0) & (params["guessing"] <= 0.5))
    assert np.all(np.isfinite(params["abilities"]))
    assert np.all(params["ability_sd"] >= 0.0)
    # Sign convention orients each latent axis to a non-negative mean slope.
    assert np.all(params["slopes"].mean(axis=0) >= -1e-9)


def test_mirt_2pl_item_params_has_no_guessing(
    ordered_binary_small_R: np.ndarray,
) -> None:
    _, _, params = rank.mirt(
        ordered_binary_small_R,
        model="2pl",
        return_item_params=True,
        **MIRT_KW,
    )
    assert set(params) == {
        "difficulty",
        "discrimination",
        "slopes",
        "intercept",
        "abilities",
        "ability_sd",
    }


def test_mirt_validation_errors(ordered_binary_small_R: np.ndarray) -> None:
    with pytest.raises(ValueError, match="model must be '2pl' or '3pl'"):
        rank.mirt(ordered_binary_small_R, model="4pl")

    with pytest.raises(ValueError, match="fix_guessing is only valid"):
        rank.mirt(ordered_binary_small_R, model="2pl", fix_guessing=0.2)

    with pytest.raises(ValueError, match="Product quadrature grid"):
        rank.mirt(ordered_binary_small_R, n_factors=12, n_quadrature=15)

    with pytest.raises(ValueError, match="cannot exceed number of questions"):
        rank.mirt(ordered_binary_small_R, n_factors=11, n_quadrature=2)
