import numpy as np
import pytest
from scipy.stats import norm

from scorio import eval
from scorio.rank import eval_ranking
from scorio.utils import rank_scores


def _expected_scores(
    R: np.ndarray,
    metric_fn,
    *args,
    **kwargs,
) -> np.ndarray:
    return np.array(
        [metric_fn(R[model], *args, **kwargs) for model in range(R.shape[0])],
        dtype=float,
    )


@pytest.fixture(scope="module")
def top_p_subset(top_p_task_aime25: np.ndarray) -> np.ndarray:
    return top_p_task_aime25[:6, :10, :12]


@pytest.fixture(scope="module")
def greedy_subset(greedy_task_aime25: np.ndarray) -> np.ndarray:
    return greedy_task_aime25[:6, :10, :]


def test_avg_wrapper_matches_eval_scores_and_ranks(top_p_subset: np.ndarray) -> None:
    ranking, scores = eval_ranking.avg(top_p_subset, method="dense", return_scores=True)

    expected_scores = np.array(
        [eval.avg(top_p_subset[model])[0] for model in range(top_p_subset.shape[0])],
        dtype=float,
    )
    expected_ranking = rank_scores(expected_scores)["dense"]

    np.testing.assert_allclose(scores, expected_scores)
    np.testing.assert_allclose(ranking, expected_ranking)


def test_bayes_wrapper_matches_eval_scores_and_ranks(top_p_subset: np.ndarray) -> None:
    ranking, scores = eval_ranking.bayes(
        top_p_subset,
        method="competition_max",
        return_scores=True,
    )

    expected_scores = np.array(
        [eval.bayes(top_p_subset[model])[0] for model in range(top_p_subset.shape[0])],
        dtype=float,
    )
    expected_ranking = rank_scores(expected_scores)["competition_max"]

    np.testing.assert_allclose(scores, expected_scores)
    np.testing.assert_allclose(ranking, expected_ranking)


def test_bayes_quantile_uses_mu_plus_z_sigma(top_p_subset: np.ndarray) -> None:
    quantile = 0.1
    z = float(norm.ppf(quantile))

    _, scores = eval_ranking.bayes(top_p_subset, quantile=quantile, return_scores=True)
    expected_scores = np.empty(top_p_subset.shape[0], dtype=float)
    for model in range(top_p_subset.shape[0]):
        mu, sigma = eval.bayes(top_p_subset[model])
        expected_scores[model] = mu + z * sigma

    np.testing.assert_allclose(scores, expected_scores)


def test_bayes_R0_shared_and_per_model(top_p_subset: np.ndarray) -> None:
    shared_R0 = top_p_subset[0, :, :4]
    ranking_shared, scores_shared = eval_ranking.bayes(
        top_p_subset,
        R0=shared_R0,
        return_scores=True,
    )
    expected_shared = np.array(
        [
            eval.bayes(top_p_subset[model], R0=shared_R0)[0]
            for model in range(top_p_subset.shape[0])
        ],
        dtype=float,
    )
    expected_rank_shared = rank_scores(expected_shared)["competition"]
    np.testing.assert_allclose(scores_shared, expected_shared)
    np.testing.assert_allclose(ranking_shared, expected_rank_shared)

    per_model_R0 = top_p_subset[:, :, :4]
    ranking_per_model, scores_per_model = eval_ranking.bayes(
        top_p_subset,
        R0=per_model_R0,
        return_scores=True,
    )
    expected_per_model = np.array(
        [
            eval.bayes(top_p_subset[model], R0=per_model_R0[model])[0]
            for model in range(top_p_subset.shape[0])
        ],
        dtype=float,
    )
    expected_rank_per_model = rank_scores(expected_per_model)["competition"]
    np.testing.assert_allclose(scores_per_model, expected_per_model)
    np.testing.assert_allclose(ranking_per_model, expected_rank_per_model)


def test_eval_ranking_wrappers_are_model_permutation_equivariant(
    top_p_subset: np.ndarray,
) -> None:
    perm = np.array([3, 0, 5, 1, 4, 2], dtype=int)
    inv_perm = np.empty_like(perm)
    inv_perm[perm] = np.arange(perm.size)

    calls = [
        lambda X: eval_ranking.avg(X, return_scores=True),
        lambda X: eval_ranking.bayes(X, return_scores=True),
        lambda X: eval_ranking.pass_at_k(X, k=3, return_scores=True),
        lambda X: eval_ranking.pass_hat_k(X, k=3, return_scores=True),
        lambda X: eval_ranking.g_pass_at_k_tau(X, k=3, tau=0.7, return_scores=True),
        lambda X: eval_ranking.mg_pass_at_k(X, k=3, return_scores=True),
    ]

    for call in calls:
        ranking, scores = call(top_p_subset)
        ranking_perm, scores_perm = call(top_p_subset[perm])

        np.testing.assert_allclose(scores, scores_perm[inv_perm])
        np.testing.assert_allclose(ranking, ranking_perm[inv_perm])


def test_bayes_validation_errors(top_p_subset: np.ndarray) -> None:
    L, M, _ = top_p_subset.shape

    for bad_quantile in (-0.1, 0.0, 1.0, 1.1, np.nan):
        with pytest.raises(ValueError, match=r"quantile must be in \(0, 1\)"):
            eval_ranking.bayes(top_p_subset, quantile=bad_quantile)

    with pytest.raises(ValueError, match="Shared R0 must have shape"):
        eval_ranking.bayes(top_p_subset, R0=np.zeros((M - 1, 3), dtype=int))

    with pytest.raises(ValueError, match="Model-specific R0 must have shape"):
        eval_ranking.bayes(top_p_subset, R0=np.zeros((L - 1, M, 3), dtype=int))

    with pytest.raises(ValueError, match="R0 must be shape"):
        eval_ranking.bayes(top_p_subset, R0=np.zeros((3,), dtype=int))

    with pytest.raises(ValueError, match="real, finite integer-valued outcomes"):
        eval_ranking.bayes(top_p_subset, R0=np.full((M, 2), 0.5))

    with pytest.raises(ValueError, match="real, finite integer-valued outcomes"):
        eval_ranking.bayes(top_p_subset, R0=np.full((M, 2), np.nan))

    with pytest.raises(ValueError, match="real, finite integer-valued outcomes"):
        eval_ranking.bayes(top_p_subset, R0=np.ones((M, 2), dtype=complex))


def test_bayes_default_binary_weights_reward_success() -> None:
    R = np.array([[[1, 1]], [[0, 0]]], dtype=int)
    ranking, scores = eval_ranking.bayes(R, return_scores=True)

    assert scores[0] > scores[1]
    assert ranking[0] < ranking[1]


@pytest.mark.parametrize(
    ("rank_fn", "eval_fn", "kwargs", "method"),
    [
        (eval_ranking.pass_at_k, eval.pass_at_k, {"k": 3}, "competition"),
        (eval_ranking.pass_hat_k, eval.pass_hat_k, {"k": 3}, "avg"),
        (
            eval_ranking.g_pass_at_k_tau,
            eval.g_pass_at_k_tau,
            {"k": 3, "tau": 0.7},
            "dense",
        ),
        (eval_ranking.mg_pass_at_k, eval.mg_pass_at_k, {"k": 3}, "competition_max"),
    ],
)
def test_pass_family_wrappers_match_eval_scores_and_ranks(
    top_p_subset: np.ndarray,
    rank_fn,
    eval_fn,
    kwargs: dict[str, float | int],
    method: str,
) -> None:
    ranking, scores = rank_fn(top_p_subset, method=method, return_scores=True, **kwargs)
    expected_scores = _expected_scores(top_p_subset, eval_fn, **kwargs)
    expected_ranking = rank_scores(expected_scores)[method]

    np.testing.assert_allclose(scores, expected_scores)
    np.testing.assert_allclose(ranking, expected_ranking)


def test_g_pass_tau_edge_equivalences(top_p_subset: np.ndarray) -> None:
    _, scores_tau0 = eval_ranking.g_pass_at_k_tau(
        top_p_subset, k=3, tau=0.0, return_scores=True
    )
    _, scores_pass = eval_ranking.pass_at_k(top_p_subset, k=3, return_scores=True)
    np.testing.assert_allclose(scores_tau0, scores_pass)

    _, scores_tau1 = eval_ranking.g_pass_at_k_tau(
        top_p_subset, k=3, tau=1.0, return_scores=True
    )
    _, scores_hat = eval_ranking.pass_hat_k(top_p_subset, k=3, return_scores=True)
    np.testing.assert_allclose(scores_tau1, scores_hat)


def test_pass_family_rankings_remain_finite_for_large_n_and_k() -> None:
    N = 2000
    k = 1000
    R = np.zeros((2, 1, N), dtype=int)
    R[0] = 1

    calls = [
        lambda: eval_ranking.pass_at_k(R, k=k, return_scores=True),
        lambda: eval_ranking.pass_hat_k(R, k=k, return_scores=True),
        lambda: eval_ranking.g_pass_at_k_tau(R, k=k, tau=0.5, return_scores=True),
        lambda: eval_ranking.mg_pass_at_k(R, k=k, return_scores=True),
    ]

    for call in calls:
        ranking, scores = call()
        assert np.all(np.isfinite(ranking))
        assert np.all(np.isfinite(scores))
        np.testing.assert_allclose(scores, [1.0, 0.0])
        np.testing.assert_array_equal(ranking, [1, 2])


@pytest.mark.parametrize(
    ("fn", "kwargs"),
    [
        (eval_ranking.pass_at_k, {"k": 0}),
        (eval_ranking.pass_hat_k, {"k": 0}),
        (eval_ranking.g_pass_at_k_tau, {"k": 0, "tau": 0.5}),
        (eval_ranking.mg_pass_at_k, {"k": 0}),
    ],
)
def test_pass_family_invalid_k_raises(
    top_p_subset: np.ndarray, fn, kwargs: dict[str, float | int]
) -> None:
    with pytest.raises(ValueError, match="k must satisfy 1 <= k <= N"):
        fn(top_p_subset, **kwargs)


def test_g_pass_tau_invalid_tau_raises(top_p_subset: np.ndarray) -> None:
    with pytest.raises(ValueError, match="tau must be in \\[0, 1\\]"):
        eval_ranking.g_pass_at_k_tau(top_p_subset, k=3, tau=1.1)


def test_2d_input_promotion_matches_3d(greedy_subset: np.ndarray) -> None:
    matrix = greedy_subset[:, :, 0]

    ranking_3d, scores_3d = eval_ranking.pass_at_k(
        greedy_subset, k=1, return_scores=True
    )
    ranking_2d, scores_2d = eval_ranking.pass_at_k(matrix, k=1, return_scores=True)

    np.testing.assert_allclose(scores_2d, scores_3d)
    np.testing.assert_allclose(ranking_2d, ranking_3d)


def test_public_eval_ranking_api_exports_have_valid_smoke_calls(
    top_p_subset: np.ndarray,
) -> None:
    api_calls = {
        "avg": lambda: eval_ranking.avg(top_p_subset, return_scores=True),
        "bayes": lambda: eval_ranking.bayes(top_p_subset, return_scores=True),
        "pass_at_k": lambda: eval_ranking.pass_at_k(
            top_p_subset, k=3, return_scores=True
        ),
        "pass_hat_k": lambda: eval_ranking.pass_hat_k(
            top_p_subset, k=3, return_scores=True
        ),
        "g_pass_at_k_tau": lambda: eval_ranking.g_pass_at_k_tau(
            top_p_subset, k=3, tau=0.7, return_scores=True
        ),
        "mg_pass_at_k": lambda: eval_ranking.mg_pass_at_k(
            top_p_subset, k=3, return_scores=True
        ),
    }

    assert set(api_calls) == set(eval_ranking.__all__)

    L = top_p_subset.shape[0]
    for name, fn in api_calls.items():
        ranking, scores = fn()
        assert ranking.shape == (L,)
        assert scores.shape == (L,)
        assert np.all(np.isfinite(ranking))
        assert np.all(np.isfinite(scores))
        assert np.all(ranking >= 1)

        if name != "bayes":
            assert np.all((0.0 <= scores) & (scores <= 1.0))
