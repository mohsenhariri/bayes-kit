import math

import numpy as np
import pytest
from scipy.special import comb
from scipy.stats import norm

from scorio import eval as scorio_eval


@pytest.fixture(scope="module")
def binary_ref(top_p_task_aime25: np.ndarray) -> np.ndarray:
    return top_p_task_aime25[0, :12, :20]


@pytest.fixture(scope="module")
def multiclass_ref(
    top_p_task_aime25: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    R = (top_p_task_aime25[1, :12, :20] + top_p_task_aime25[2, :12, :20]).astype(
        int, copy=False
    )
    w = np.array([0.0, 0.5, 1.0], dtype=float)
    R0 = (top_p_task_aime25[3, :12, :6] + top_p_task_aime25[4, :12, :6]).astype(
        int, copy=False
    )
    return R, w, R0


@pytest.fixture(scope="module")
def top_p_model_slice(top_p_task_aime25: np.ndarray) -> np.ndarray:
    return top_p_task_aime25[0, :10, :12]


def _expected_normal_ci(
    mu: float,
    sigma: float,
    confidence: float,
    bounds: tuple[float, float] | None,
) -> tuple[float, float]:
    z = float(norm.ppf(0.5 + confidence / 2.0))
    lo = mu - z * sigma
    hi = mu + z * sigma
    if bounds is not None:
        lo = max(lo, bounds[0])
        hi = min(hi, bounds[1])
    return lo, hi


def _bayes_reference(
    R: np.ndarray,
    w: np.ndarray,
    R0: np.ndarray | None = None,
) -> tuple[float, float]:
    Rm = np.asarray(R, dtype=int)
    wv = np.asarray(w, dtype=float)
    M, N = Rm.shape
    C = int(wv.size - 1)

    if R0 is None:
        R0m = np.zeros((M, 0), dtype=int)
    else:
        R0m = np.asarray(R0, dtype=int)
        if R0m.ndim == 1:
            R0m = R0m.reshape(M, -1)
        if R0m.shape[0] != M:
            raise ValueError(
                "R0 must have same row count as R for reference computation."
            )

    D = int(R0m.shape[1])
    T = float(1 + C + D + N)
    delta_w = wv - wv[0]

    mu_rows = np.empty(M, dtype=float)
    var_rows = np.empty(M, dtype=float)

    for row in range(M):
        nu = np.ones(C + 1, dtype=float)

        for value in Rm[row]:
            nu[int(value)] += 1.0
        for value in R0m[row]:
            nu[int(value)] += 1.0

        row_mean_component = float(np.dot(nu / T, delta_w))
        second_moment_component = float(np.dot(nu / T, delta_w**2))

        mu_rows[row] = row_mean_component
        var_rows[row] = max(0.0, second_moment_component - row_mean_component**2)

    mu = float(wv[0] + np.mean(mu_rows))
    sigma = float(math.sqrt(np.sum(var_rows) / (M**2 * (T + 1.0))))
    return mu, sigma


def _pass_at_k_reference(R: np.ndarray, k: int) -> float:
    Rm = np.asarray(R, dtype=int)
    M, N = Rm.shape
    denom = float(comb(N, k))
    values = np.empty(M, dtype=float)
    for row in range(M):
        nu = int(np.sum(Rm[row]))
        values[row] = 1.0 - float(comb(N - nu, k)) / denom
    return float(np.mean(values))


def _pass_hat_k_reference(R: np.ndarray, k: int) -> float:
    Rm = np.asarray(R, dtype=int)
    M, N = Rm.shape
    denom = float(comb(N, k))
    values = np.empty(M, dtype=float)
    for row in range(M):
        nu = int(np.sum(Rm[row]))
        values[row] = float(comb(nu, k)) / denom
    return float(np.mean(values))


def _g_pass_at_k_tau_reference(R: np.ndarray, k: int, tau: float) -> float:
    if tau <= 0.0:
        return _pass_at_k_reference(R, k)

    Rm = np.asarray(R, dtype=int)
    M, N = Rm.shape
    denom = float(comb(N, k))
    j0 = int(math.ceil(tau * k))

    values = np.empty(M, dtype=float)
    for row in range(M):
        nu = int(np.sum(Rm[row]))
        total = 0.0
        for j in range(j0, k + 1):
            total += float(comb(nu, j) * comb(N - nu, k - j)) / denom
        values[row] = total
    return float(np.mean(values))


def _mg_pass_at_k_reference(R: np.ndarray, k: int) -> float:
    Rm = np.asarray(R, dtype=int)
    M, N = Rm.shape
    denom = float(comb(N, k))
    majority = int(math.ceil(0.5 * k))
    if majority >= k:
        return 0.0

    values = np.empty(M, dtype=float)
    for row in range(M):
        nu = int(np.sum(Rm[row]))
        total = 0.0
        for j in range(majority + 1, k + 1):
            total += (j - majority) * float(comb(nu, j) * comb(N - nu, k - j)) / denom
        values[row] = (2.0 / k) * total
    return float(np.mean(values))


def _auc_at_k_reference(R: np.ndarray, k: int) -> float:
    Rm = np.asarray(R, dtype=int)
    M, N = Rm.shape

    pass_vals = np.empty(k, dtype=float)
    for j in range(1, k + 1):
        denom = float(comb(N, j))
        row_vals = np.empty(M, dtype=float)
        for row in range(M):
            nu = int(np.sum(Rm[row]))
            row_vals[row] = 1.0 - float(comb(N - nu, j)) / denom
        pass_vals[j - 1] = float(np.mean(row_vals))

    if k == 1:
        return float(pass_vals[0])

    coeff = np.full(k, 1.0 / (k - 1), dtype=float)
    coeff[0] = 0.5 / (k - 1)
    coeff[-1] = 0.5 / (k - 1)
    return float(np.dot(coeff, pass_vals))


def _maj_at_k_reference(R: np.ndarray, k: int) -> float:
    Rm = np.asarray(R, dtype=int)
    M, N = Rm.shape
    denom = float(comb(N, k))
    j0 = (k // 2) + 1

    values = np.empty(M, dtype=float)
    for row in range(M):
        nu = int(np.sum(Rm[row]))
        total = 0.0
        for j in range(j0, k + 1):
            total += float(comb(nu, j) * comb(N - nu, k - j)) / denom
        values[row] = total
    return float(np.mean(values))


def _max_at_k_reference(
    R: np.ndarray,
    k: int,
    w: np.ndarray | None = None,
) -> float:
    Rm = np.asarray(R, dtype=int)
    if Rm.ndim == 1:
        Rm = Rm.reshape(1, -1)

    wv = np.array([0.0, 1.0], dtype=float) if w is None else np.asarray(w, dtype=float)

    M, N = Rm.shape
    coeff = comb(np.arange(k - 1, N, dtype=float), k - 1) / float(comb(N, k))

    values = np.empty(M, dtype=float)
    for row in range(M):
        sorted_rewards = np.sort(wv[Rm[row]])
        values[row] = float(np.dot(coeff, sorted_rewards[k - 1 :]))
    return float(np.mean(values))


def _threshold_spectrum_reference(
    R: np.ndarray,
    k: int,
    weights: np.ndarray,
) -> float:
    Rm = np.asarray(R, dtype=int)
    M, N = Rm.shape
    wv = np.asarray(weights, dtype=float)

    denom = float(comb(N, k))
    nu = np.sum(Rm, axis=1)
    levels = np.cumsum(wv, dtype=float)
    values = np.zeros(M, dtype=float)
    for j in range(1, k + 1):
        credit = float(levels[j - 1])
        if credit == 0.0:
            continue
        values += credit * comb(nu, j) * comb(N - nu, k - j) / denom
    return float(np.mean(values))


def _unanimous_spectrum_weights(k: int) -> np.ndarray:
    weights = np.zeros(k, dtype=float)
    weights[-1] = 1.0
    return weights


def _mg_spectrum_weights(k: int) -> np.ndarray:
    weights = np.zeros(k, dtype=float)
    weights[int(math.ceil(k / 2.0)) :] = 2.0 / k
    return weights


def test_bayes_multiclass_matches_closed_form_reference(
    multiclass_ref: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    R, w, R0 = multiclass_ref
    mu_prior, sigma_prior = scorio_eval.bayes(R, w, R0)
    exp_mu_prior, exp_sigma_prior = _bayes_reference(R, w, R0)

    mu_noprior, sigma_noprior = scorio_eval.bayes(R, w)
    exp_mu_noprior, exp_sigma_noprior = _bayes_reference(R, w)

    assert mu_prior == pytest.approx(exp_mu_prior)
    assert sigma_prior == pytest.approx(exp_sigma_prior)
    assert mu_noprior == pytest.approx(exp_mu_noprior)
    assert sigma_noprior == pytest.approx(exp_sigma_noprior)


def test_bayes_binary_default_weights_equal_explicit(binary_ref: np.ndarray) -> None:
    mu_auto, sigma_auto = scorio_eval.bayes(binary_ref)
    mu_explicit, sigma_explicit = scorio_eval.bayes(
        binary_ref, w=np.array([0.0, 1.0], dtype=float)
    )

    assert mu_auto == pytest.approx(mu_explicit)
    assert sigma_auto == pytest.approx(sigma_explicit)


def test_bayes_requires_weights_for_multiclass(
    multiclass_ref: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    R, _, _ = multiclass_ref
    with pytest.raises(ValueError, match="must be provided"):
        scorio_eval.bayes(R)


def test_bayes_validates_R0_row_count(
    multiclass_ref: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    R, w, _ = multiclass_ref
    bad_R0 = np.zeros((R.shape[0] + 1, 2), dtype=int)
    with pytest.raises(ValueError, match="same number of rows"):
        scorio_eval.bayes(R, w=w, R0=bad_R0)


def test_bayes_ci_matches_normal_interval_formula(binary_ref: np.ndarray) -> None:
    confidence = 0.9
    bounds = (0.0, 1.0)
    mu, sigma, lo, hi = scorio_eval.bayes_ci(
        binary_ref,
        confidence=confidence,
        bounds=bounds,
    )
    exp_lo, exp_hi = _expected_normal_ci(mu, sigma, confidence, bounds)

    assert lo == pytest.approx(exp_lo)
    assert hi == pytest.approx(exp_hi)
    assert lo <= mu <= hi


def test_avg_binary_and_weighted_match_manual_formulas(
    binary_ref: np.ndarray,
    multiclass_ref: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    a_binary, sigma_binary = scorio_eval.avg(binary_ref)
    assert a_binary == pytest.approx(float(np.mean(binary_ref)))
    assert sigma_binary >= 0.0

    R, w, _ = multiclass_ref
    a_weighted, sigma_weighted = scorio_eval.avg(R, w)
    assert a_weighted == pytest.approx(float(np.mean(w[R])))
    assert sigma_weighted >= 0.0

    _, sigma_bayes_weighted = scorio_eval.bayes(R, w)
    T = 1 + (w.size - 1) + R.shape[1]
    assert sigma_weighted == pytest.approx((T / R.shape[1]) * sigma_bayes_weighted)


def test_avg_requires_binary_when_weights_omitted(
    multiclass_ref: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    R, _, _ = multiclass_ref
    with pytest.raises(ValueError, match="Entries of R must be integers in \\[0, 1\\]"):
        scorio_eval.avg(R)


def test_avg_ci_matches_normal_interval_formula(binary_ref: np.ndarray) -> None:
    confidence = 0.8
    bounds = (0.0, 1.0)
    a, sigma, lo, hi = scorio_eval.avg_ci(
        binary_ref, confidence=confidence, bounds=bounds
    )
    exp_lo, exp_hi = _expected_normal_ci(a, sigma, confidence, bounds)

    assert lo == pytest.approx(exp_lo)
    assert hi == pytest.approx(exp_hi)
    assert lo <= a <= hi


def test_pass_point_metrics_match_closed_form_references(
    binary_ref: np.ndarray,
) -> None:
    k = 3
    assert scorio_eval.pass_at_k(binary_ref, k) == pytest.approx(
        _pass_at_k_reference(binary_ref, k)
    )
    assert scorio_eval.pass_hat_k(binary_ref, k) == pytest.approx(
        _pass_hat_k_reference(binary_ref, k)
    )
    assert scorio_eval.g_pass_at_k(binary_ref, k) == pytest.approx(
        _pass_hat_k_reference(binary_ref, k)
    )
    assert scorio_eval.g_pass_at_k_tau(binary_ref, k, 0.7) == pytest.approx(
        _g_pass_at_k_tau_reference(binary_ref, k, 0.7)
    )
    assert scorio_eval.mg_pass_at_k(binary_ref, k) == pytest.approx(
        _mg_pass_at_k_reference(binary_ref, k)
    )


def test_pass_point_metrics_remain_finite_for_large_n_and_k() -> None:
    N = 2000
    k = 1000
    R = np.zeros((2, N), dtype=int)
    R[0] = 1

    scores = [
        scorio_eval.pass_at_k(R, k),
        scorio_eval.pass_hat_k(R, k),
        scorio_eval.g_pass_at_k_tau(R, k, tau=0.5),
        scorio_eval.mg_pass_at_k(R, k),
    ]

    assert np.all(np.isfinite(scores))
    np.testing.assert_allclose(scores, 0.5)


def test_pass_family_monotonicity_and_bounds(binary_ref: np.ndarray) -> None:
    N = binary_ref.shape[1]
    k_values = list(range(1, min(N, 8) + 1))

    pass_vals = [scorio_eval.pass_at_k(binary_ref, k) for k in k_values]
    pass_hat_vals = [scorio_eval.pass_hat_k(binary_ref, k) for k in k_values]

    for idx in range(1, len(k_values)):
        assert pass_vals[idx] >= pass_vals[idx - 1]
        assert pass_hat_vals[idx] <= pass_hat_vals[idx - 1]

    for p, ph in zip(pass_vals, pass_hat_vals, strict=True):
        assert p >= ph
        assert 0.0 <= ph <= 1.0
        assert 0.0 <= p <= 1.0


def test_pass_aliases_and_tau_edge_equivalences(binary_ref: np.ndarray) -> None:
    k = 3
    assert scorio_eval.g_pass_at_k(binary_ref, k) == pytest.approx(
        scorio_eval.pass_hat_k(binary_ref, k)
    )

    np.testing.assert_allclose(
        scorio_eval.g_pass_at_k_ci(binary_ref, k),
        scorio_eval.pass_hat_k_ci(binary_ref, k),
    )

    assert scorio_eval.g_pass_at_k_tau(binary_ref, k, tau=0.0) == pytest.approx(
        scorio_eval.pass_at_k(binary_ref, k)
    )
    assert scorio_eval.g_pass_at_k_tau(binary_ref, k, tau=1.0) == pytest.approx(
        scorio_eval.pass_hat_k(binary_ref, k)
    )

    np.testing.assert_allclose(
        scorio_eval.g_pass_at_k_tau_ci(binary_ref, k, tau=0.0),
        scorio_eval.pass_at_k_ci(binary_ref, k),
    )
    np.testing.assert_allclose(
        scorio_eval.g_pass_at_k_tau_ci(binary_ref, k, tau=1.0),
        scorio_eval.pass_hat_k_ci(binary_ref, k),
    )


def test_pass_mg_k1_edge_case(binary_ref: np.ndarray) -> None:
    assert scorio_eval.mg_pass_at_k(binary_ref, 1) == pytest.approx(0.0)
    np.testing.assert_allclose(
        scorio_eval.mg_pass_at_k_ci(binary_ref, 1), (0.0, 0.0, 0.0, 0.0)
    )


def test_auc_matches_reference_and_k1_equivalence(binary_ref: np.ndarray) -> None:
    k = 3

    assert scorio_eval.auc_at_k(binary_ref, k) == pytest.approx(
        _auc_at_k_reference(binary_ref, k)
    )
    assert scorio_eval.auc_at_k(binary_ref, 1) == pytest.approx(
        scorio_eval.pass_at_k(binary_ref, 1)
    )

    np.testing.assert_allclose(
        scorio_eval.auc_at_k_ci(binary_ref, 1),
        scorio_eval.pass_at_k_ci(binary_ref, 1),
    )


def test_majority_matches_reference_and_threshold_equivalences(
    binary_ref: np.ndarray,
) -> None:
    k = 3
    tau = ((k // 2) + 1) / k

    assert scorio_eval.maj_at_k(binary_ref, k) == pytest.approx(
        _maj_at_k_reference(binary_ref, k)
    )
    assert scorio_eval.maj_at_k(binary_ref, k) == pytest.approx(
        scorio_eval.g_pass_at_k_tau(binary_ref, k, tau=tau)
    )

    np.testing.assert_allclose(
        scorio_eval.maj_at_k_ci(binary_ref, k),
        scorio_eval.g_pass_at_k_tau_ci(binary_ref, k, tau=tau),
    )

    assert scorio_eval.maj_at_k(binary_ref, 1) == pytest.approx(
        scorio_eval.pass_hat_k(binary_ref, 1)
    )
    assert scorio_eval.maj_at_k(binary_ref, 2) == pytest.approx(
        scorio_eval.pass_hat_k(binary_ref, 2)
    )


def test_max_reward_matches_reference_and_binary_pass_equivalence(
    binary_ref: np.ndarray,
) -> None:
    k = 3

    assert scorio_eval.max_at_k(binary_ref, k) == pytest.approx(
        _max_at_k_reference(binary_ref, k)
    )
    assert scorio_eval.max_at_k(binary_ref, k) == pytest.approx(
        scorio_eval.pass_at_k(binary_ref, k)
    )


def test_max_reward_weighted_categorical_support(
    multiclass_ref: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    R, w, R0 = multiclass_ref
    k = 2

    assert scorio_eval.max_at_k(R, k, w=w) == pytest.approx(
        _max_at_k_reference(R, k, w=w)
    )

    mu, sigma, lo, hi = scorio_eval.max_at_k_ci(R, k, w=w, R0=R0)
    assert np.isfinite(mu)
    assert np.isfinite(sigma)
    assert np.isfinite(lo)
    assert np.isfinite(hi)
    assert sigma >= 0.0
    assert lo <= mu <= hi
    assert lo >= np.min(w)
    assert hi <= np.max(w)


def test_max_reward_k1_matches_bayes_ci(
    multiclass_ref: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    R, w, R0 = multiclass_ref
    np.testing.assert_allclose(
        scorio_eval.max_at_k_ci(R, 1, w=w, R0=R0),
        scorio_eval.bayes_ci(R, w=w, R0=R0),
    )


def test_max_reward_requires_weights_for_non_binary(
    multiclass_ref: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    R, _, _ = multiclass_ref
    with pytest.raises(ValueError, match="weight vector 'w' must be provided"):
        scorio_eval.max_at_k(R, 2)


def test_threshold_spectrum_matches_reference_and_special_cases(
    binary_ref: np.ndarray,
) -> None:
    k = 3
    custom_weights = np.array([0.2, 0.1, 0.4], dtype=float)
    unanimous_weights = _unanimous_spectrum_weights(k)
    mg_weights = _mg_spectrum_weights(k)

    assert scorio_eval.threshold_spectrum_at_k(
        binary_ref, k, custom_weights
    ) == pytest.approx(_threshold_spectrum_reference(binary_ref, k, custom_weights))
    assert scorio_eval.threshold_spectrum_at_k(
        binary_ref, k, unanimous_weights
    ) == pytest.approx(_threshold_spectrum_reference(binary_ref, k, unanimous_weights))
    assert scorio_eval.threshold_spectrum_at_k(
        binary_ref, k, mg_weights
    ) == pytest.approx(_threshold_spectrum_reference(binary_ref, k, mg_weights))

    assert scorio_eval.threshold_spectrum_at_k(
        binary_ref, k, unanimous_weights
    ) == pytest.approx(scorio_eval.pass_hat_k(binary_ref, k))
    assert scorio_eval.threshold_spectrum_at_k(
        binary_ref, k, mg_weights
    ) == pytest.approx(scorio_eval.mg_pass_at_k(binary_ref, k))


def test_geometric_family_matches_closed_form_blends_and_power_controls(
    binary_ref: np.ndarray,
) -> None:
    k = 3
    custom_weights = np.array([0.2, 0.1, 0.4], dtype=float)

    pass_score = scorio_eval.pass_at_k(binary_ref, k)
    unanimous_score = scorio_eval.pass_hat_k(binary_ref, k)
    mg_score = scorio_eval.mg_pass_at_k(binary_ref, k)
    custom_spectrum = scorio_eval.threshold_spectrum_at_k(binary_ref, k, custom_weights)
    nu = np.sum(binary_ref, axis=1)
    denom = float(comb(binary_ref.shape[1], k))
    pass_vals = 1.0 - comb(binary_ref.shape[1] - nu, k) / denom
    unanimous_vals = comb(nu, k) / denom

    assert pass_score > 0.0

    assert scorio_eval.geom_at_k(binary_ref, k) == pytest.approx(
        float(np.mean(np.sqrt(pass_vals * unanimous_vals)))
    )
    assert scorio_eval.geom_ds_at_k(binary_ref, k) == pytest.approx(
        math.sqrt(pass_score * unanimous_score)
    )
    assert scorio_eval.geo_spectrum_at_k(binary_ref, k) == pytest.approx(
        math.sqrt(pass_score * mg_score)
    )
    assert scorio_eval.geo_spectrum_star_at_k(binary_ref, k) == pytest.approx(
        scorio_eval.geo_spectrum_at_k(binary_ref, k)
    )

    assert scorio_eval.geo_spectrum_at_k(
        binary_ref, k, lam=0.0, weights=custom_weights
    ) == pytest.approx(custom_spectrum)
    assert scorio_eval.geo_spectrum_at_k(
        binary_ref, k, lam=1.0, weights=custom_weights
    ) == pytest.approx(pass_score)
    assert scorio_eval.geo_spectrum_at_k(
        binary_ref, k, lam=0.25, weights=custom_weights
    ) == pytest.approx(
        scorio_eval.geo_spectrum_at_k(
            binary_ref, k, lambda_=0.25, weights=custom_weights
        )
    )

    assert scorio_eval.geom_at_k(
        binary_ref, k, pass_power=0.75, unanimous_power=0.25
    ) == pytest.approx(float(np.mean((pass_vals**0.75) * (unanimous_vals**0.25))))
    assert scorio_eval.geom_ds_at_k(
        binary_ref, k, pass_power=0.75, unanimous_power=0.25
    ) == pytest.approx((pass_score**0.75) * (unanimous_score**0.25))
    assert scorio_eval.geom_at_k(
        binary_ref, k, pass_power=0.25, unanimous_power=0.75
    ) == pytest.approx(float(np.mean((pass_vals**0.25) * (unanimous_vals**0.75))))
    assert scorio_eval.geom_ds_at_k(
        binary_ref, k, pass_power=0.25, unanimous_power=0.75
    ) == pytest.approx((pass_score**0.25) * (unanimous_score**0.75))
    assert scorio_eval.geom_ds_at_k(
        binary_ref, k, pass_power=-0.5, unanimous_power=0.5
    ) == pytest.approx(math.sqrt(unanimous_score / pass_score))

    zero_ref = np.zeros((3, 5), dtype=int)
    assert scorio_eval.geom_at_k(
        zero_ref, 2, pass_power=-0.5, unanimous_power=0.5
    ) == pytest.approx(0.0)

    with pytest.raises(ValueError, match="y_power must be non-negative when y is zero"):
        scorio_eval.geom_at_k(
            np.array([[1, 0, 0, 0, 0]], dtype=int),
            2,
            pass_power=0.5,
            unanimous_power=-0.5,
        )


def test_geometric_ci_special_cases_and_power_controls(binary_ref: np.ndarray) -> None:
    k = 3
    custom_weights = np.array([0.2, 0.1, 0.4], dtype=float)
    unanimous_weights = _unanimous_spectrum_weights(k)
    mg_weights = _mg_spectrum_weights(k)

    np.testing.assert_allclose(
        scorio_eval.threshold_spectrum_at_k_ci(binary_ref, k, unanimous_weights),
        scorio_eval.pass_hat_k_ci(binary_ref, k),
    )
    np.testing.assert_allclose(
        scorio_eval.threshold_spectrum_at_k_ci(binary_ref, k, mg_weights),
        scorio_eval.mg_pass_at_k_ci(binary_ref, k),
    )
    np.testing.assert_allclose(
        scorio_eval.geo_spectrum_at_k_ci(
            binary_ref, k, lam=0.0, weights=custom_weights
        ),
        scorio_eval.threshold_spectrum_at_k_ci(binary_ref, k, custom_weights),
    )
    np.testing.assert_allclose(
        scorio_eval.geo_spectrum_at_k_ci(
            binary_ref, k, lam=1.0, weights=custom_weights
        ),
        scorio_eval.pass_at_k_ci(binary_ref, k),
    )
    np.testing.assert_allclose(
        scorio_eval.geom_ds_at_k_ci(binary_ref, k),
        scorio_eval.geo_spectrum_at_k_ci(
            binary_ref, k, lam=0.5, weights=unanimous_weights
        ),
    )
    np.testing.assert_allclose(
        scorio_eval.geom_ds_at_k_ci(
            binary_ref, k, pass_power=0.75, unanimous_power=0.25
        ),
        scorio_eval.geo_spectrum_at_k_ci(
            binary_ref, k, lam=0.75, weights=unanimous_weights
        ),
    )
    np.testing.assert_allclose(
        scorio_eval.geo_spectrum_star_at_k_ci(binary_ref, k),
        scorio_eval.geo_spectrum_at_k_ci(binary_ref, k),
    )


def test_questionwise_and_dataset_geom_ci_are_distinct_targets() -> None:
    R = np.array([[1, 0, 0], [1, 1, 1]], dtype=int)
    assert scorio_eval.geom_at_k(R, 2) == pytest.approx(0.5)
    assert scorio_eval.geom_ds_at_k(R, 2) == pytest.approx(math.sqrt(5.0 / 12.0))

    question_mu, *_ = scorio_eval.geom_at_k_ci(R, 2)
    dataset_mu, *_ = scorio_eval.geom_ds_at_k_ci(R, 2)
    assert question_mu != pytest.approx(dataset_mu)


def test_spectrum_family_rejects_invalid_weights(binary_ref: np.ndarray) -> None:
    with pytest.raises(ValueError, match="length-3"):
        scorio_eval.threshold_spectrum_at_k(binary_ref, 3, np.array([1.0, 0.0]))
    with pytest.raises(ValueError, match="non-negative"):
        scorio_eval.threshold_spectrum_at_k(binary_ref, 3, np.array([0.2, -0.1, 0.3]))
    with pytest.raises(ValueError, match="sum\\(weights\\) <= 1"):
        scorio_eval.threshold_spectrum_at_k(binary_ref, 3, np.array([0.5, 0.4, 0.3]))
    with pytest.raises(ValueError, match="not complex"):
        scorio_eval.threshold_spectrum_at_k(
            binary_ref,
            3,
            np.array([0.2 + 0.1j, 0.1, 0.3]),
        )
    with pytest.raises(ValueError, match="numeric"):
        scorio_eval.threshold_spectrum_at_k(
            binary_ref,
            3,
            ["0.2", "0.1", "0.3"],
        )


@pytest.mark.parametrize(
    "fn", [scorio_eval.geo_spectrum_at_k, scorio_eval.geo_spectrum_at_k_ci]
)
def test_geo_spectrum_rejects_invalid_lambda(binary_ref: np.ndarray, fn) -> None:
    with pytest.raises(ValueError, match="lam must be in \\[0, 1\\]"):
        fn(binary_ref, 3, lam=1.1)
    with pytest.raises(TypeError, match="Specify at most one"):
        fn(binary_ref, 3, lam=0.25, lambda_=0.5)


@pytest.mark.parametrize(
    "fn",
    [
        scorio_eval.pass_at_k,
        scorio_eval.pass_hat_k,
        scorio_eval.g_pass_at_k,
        scorio_eval.mg_pass_at_k,
        scorio_eval.auc_at_k,
        scorio_eval.maj_at_k,
        scorio_eval.max_at_k,
        scorio_eval.pass_at_k_ci,
        scorio_eval.pass_hat_k_ci,
        scorio_eval.auc_at_k_ci,
        scorio_eval.maj_at_k_ci,
        scorio_eval.g_pass_at_k_ci,
        scorio_eval.mg_pass_at_k_ci,
    ],
)
def test_pass_family_invalid_k_raises(binary_ref: np.ndarray, fn) -> None:
    with pytest.raises(ValueError, match="k must satisfy 1 <= k <= N"):
        fn(binary_ref, 0)


@pytest.mark.parametrize(
    "fn", [scorio_eval.g_pass_at_k_tau, scorio_eval.g_pass_at_k_tau_ci]
)
def test_g_pass_tau_invalid_tau_raises(binary_ref: np.ndarray, fn) -> None:
    with pytest.raises(ValueError, match="tau must be in \\[0, 1\\]"):
        fn(binary_ref, 2, tau=1.1)


def test_pass_family_rejects_non_binary_values(binary_ref: np.ndarray) -> None:
    R_bad = binary_ref.copy()
    R_bad[0, 0] = 2
    with pytest.raises(ValueError, match="Entries of R must be integers in \\[0, 1\\]"):
        scorio_eval.pass_at_k(R_bad, 1)


def test_eval_apis_are_invariant_to_question_and_trial_permutations(
    top_p_model_slice: np.ndarray,
) -> None:
    R = top_p_model_slice
    R_perm = R[::-1, :][:, ::-1]

    assert scorio_eval.avg(R)[0] == pytest.approx(scorio_eval.avg(R_perm)[0])
    assert scorio_eval.bayes(R)[0] == pytest.approx(scorio_eval.bayes(R_perm)[0])
    assert scorio_eval.pass_at_k(R, 3) == pytest.approx(
        scorio_eval.pass_at_k(R_perm, 3)
    )
    assert scorio_eval.pass_hat_k(R, 3) == pytest.approx(
        scorio_eval.pass_hat_k(R_perm, 3)
    )
    assert scorio_eval.g_pass_at_k_tau(R, 3, 0.7) == pytest.approx(
        scorio_eval.g_pass_at_k_tau(R_perm, 3, 0.7)
    )
    assert scorio_eval.mg_pass_at_k(R, 3) == pytest.approx(
        scorio_eval.mg_pass_at_k(R_perm, 3)
    )
    assert scorio_eval.auc_at_k(R, 3) == pytest.approx(scorio_eval.auc_at_k(R_perm, 3))
    assert scorio_eval.maj_at_k(R, 3) == pytest.approx(scorio_eval.maj_at_k(R_perm, 3))
    assert scorio_eval.max_at_k(R, 3) == pytest.approx(scorio_eval.max_at_k(R_perm, 3))


def test_eval_apis_on_simulation_dataset_slice(top_p_model_slice: np.ndarray) -> None:
    R = top_p_model_slice

    a, a_sigma = scorio_eval.avg(R)
    assert 0.0 <= a <= 1.0
    assert a_sigma >= 0.0
    assert scorio_eval.pass_at_k(R, 1) == pytest.approx(a)

    b_mu, b_sigma = scorio_eval.bayes(R)
    assert 0.0 <= b_mu <= 1.0
    assert b_sigma >= 0.0

    p1 = scorio_eval.pass_at_k(R, 3)
    ph = scorio_eval.pass_hat_k(R, 3)
    auc = scorio_eval.auc_at_k(R, 3)
    maj = scorio_eval.maj_at_k(R, 3)
    gt = scorio_eval.g_pass_at_k_tau(R, 3, 0.7)
    mg = scorio_eval.mg_pass_at_k(R, 3)
    mx = scorio_eval.max_at_k(R, 3)
    assert p1 >= gt >= ph
    assert a <= auc <= p1
    assert p1 >= maj >= ph
    assert 0.0 <= mg <= 1.0
    assert 0.0 <= mx <= 1.0

    ci_outputs = [
        scorio_eval.bayes_ci(R),
        scorio_eval.avg_ci(R),
        scorio_eval.pass_at_k_ci(R, 3),
        scorio_eval.pass_hat_k_ci(R, 3),
        scorio_eval.auc_at_k_ci(R, 3),
        scorio_eval.maj_at_k_ci(R, 3),
        scorio_eval.g_pass_at_k_ci(R, 3),
        scorio_eval.g_pass_at_k_tau_ci(R, 3, 0.7),
        scorio_eval.mg_pass_at_k_ci(R, 3),
        scorio_eval.max_at_k_ci(R, 3),
    ]
    for mu, sigma, lo, hi in ci_outputs:
        assert np.isfinite(mu)
        assert np.isfinite(sigma)
        assert np.isfinite(lo)
        assert np.isfinite(hi)
        assert sigma >= 0.0
        assert lo <= hi
        assert lo <= mu <= hi


def test_public_eval_api_exports_have_valid_smoke_calls(binary_ref: np.ndarray) -> None:
    api_calls = {
        "bayes": lambda: scorio_eval.bayes(binary_ref),
        "bayes_ci": lambda: scorio_eval.bayes_ci(binary_ref),
        "avg": lambda: scorio_eval.avg(binary_ref),
        "avg_ci": lambda: scorio_eval.avg_ci(binary_ref),
        "pass_at_k": lambda: scorio_eval.pass_at_k(binary_ref, 2),
        "pass_hat_k": lambda: scorio_eval.pass_hat_k(binary_ref, 2),
        "unanimous_at_k": lambda: scorio_eval.unanimous_at_k(binary_ref, 2),
        "auc_at_k": lambda: scorio_eval.auc_at_k(binary_ref, 2),
        "maj_at_k": lambda: scorio_eval.maj_at_k(binary_ref, 2),
        "g_pass_at_k": lambda: scorio_eval.g_pass_at_k(binary_ref, 2),
        "g_pass_at_k_tau": lambda: scorio_eval.g_pass_at_k_tau(binary_ref, 2, tau=0.7),
        "mg_pass_at_k": lambda: scorio_eval.mg_pass_at_k(binary_ref, 2),
        "threshold_spectrum_at_k": lambda: scorio_eval.threshold_spectrum_at_k(
            binary_ref, 2, np.array([0.0, 1.0], dtype=float)
        ),
        "geom_at_k": lambda: scorio_eval.geom_at_k(binary_ref, 2),
        "geom_ds_at_k": lambda: scorio_eval.geom_ds_at_k(binary_ref, 2),
        "geo_spectrum_at_k_ci": lambda: scorio_eval.geo_spectrum_at_k_ci(binary_ref, 2),
        "geo_spectrum_at_k": lambda: scorio_eval.geo_spectrum_at_k(binary_ref, 2),
        "geo_spectrum_star_at_k": lambda: scorio_eval.geo_spectrum_star_at_k(
            binary_ref, 2
        ),
        "max_at_k": lambda: scorio_eval.max_at_k(binary_ref, 2),
        "pass_at_k_ci": lambda: scorio_eval.pass_at_k_ci(binary_ref, 2),
        "pass_hat_k_ci": lambda: scorio_eval.pass_hat_k_ci(binary_ref, 2),
        "unanimous_at_k_ci": lambda: scorio_eval.unanimous_at_k_ci(binary_ref, 2),
        "auc_at_k_ci": lambda: scorio_eval.auc_at_k_ci(binary_ref, 2),
        "maj_at_k_ci": lambda: scorio_eval.maj_at_k_ci(binary_ref, 2),
        "g_pass_at_k_ci": lambda: scorio_eval.g_pass_at_k_ci(binary_ref, 2),
        "g_pass_at_k_tau_ci": lambda: scorio_eval.g_pass_at_k_tau_ci(
            binary_ref, 2, tau=0.7
        ),
        "mg_pass_at_k_ci": lambda: scorio_eval.mg_pass_at_k_ci(binary_ref, 2),
        "max_at_k_ci": lambda: scorio_eval.max_at_k_ci(binary_ref, 2),
        "threshold_spectrum_at_k_ci": lambda: scorio_eval.threshold_spectrum_at_k_ci(
            binary_ref, 2, np.array([0.0, 1.0], dtype=float)
        ),
        "geom_at_k_ci": lambda: scorio_eval.geom_at_k_ci(binary_ref, 2),
        "geom_ds_at_k_ci": lambda: scorio_eval.geom_ds_at_k_ci(binary_ref, 2),
        "geo_spectrum_star_at_k_ci": lambda: scorio_eval.geo_spectrum_star_at_k_ci(
            binary_ref, 2
        ),
    }

    assert set(api_calls) == set(scorio_eval.__all__)

    for name, fn in api_calls.items():
        out = fn()
        if name.endswith("_ci"):
            mu, sigma, lo, hi = out
            assert np.isfinite(mu)
            assert np.isfinite(sigma)
            assert np.isfinite(lo)
            assert np.isfinite(hi)
            assert sigma >= 0.0
            assert lo <= hi
            assert lo <= mu <= hi
            continue

        if name in {"bayes", "avg"}:
            mu, sigma = out
            assert np.isfinite(mu)
            assert np.isfinite(sigma)
            assert sigma >= 0.0
            continue

        assert np.isfinite(out)
        assert 0.0 <= out <= 1.0
