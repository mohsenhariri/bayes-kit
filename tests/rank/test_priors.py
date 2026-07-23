import numpy as np
import pytest

from scorio import rank


def test_prior_is_abstract() -> None:
    with pytest.raises(TypeError):
        rank.Prior()


def test_prior_penalty_outputs_are_finite(ordered_binary_small_R: np.ndarray) -> None:
    theta = np.array([-0.3, -0.1, 0.1, 0.3], dtype=float)

    priors = [
        rank.GaussianPrior(mean=0.0, var=1.0),
        rank.LaplacePrior(loc=0.0, scale=1.0),
        rank.CauchyPrior(loc=0.0, scale=1.0),
        rank.UniformPrior(),
        rank.CustomPrior(lambda x: float(np.sum(np.abs(x)))),
        rank.EmpiricalPrior(ordered_binary_small_R, var=1.5),
    ]

    for prior in priors:
        penalty = float(prior.penalty(theta))
        assert np.isfinite(penalty)


def test_empirical_prior_centering_and_shape(
    ordered_binary_small_R: np.ndarray,
) -> None:
    prior = rank.EmpiricalPrior(ordered_binary_small_R, var=1.0)
    assert prior.prior_mean.shape == (ordered_binary_small_R.shape[0],)
    assert float(np.sum(prior.prior_mean)) == pytest.approx(0.0, abs=1e-10)


def test_empirical_prior_rejects_wrong_theta_length(
    ordered_binary_small_R: np.ndarray,
) -> None:
    prior = rank.EmpiricalPrior(ordered_binary_small_R)
    with pytest.raises(ValueError, match="must match number of models"):
        prior.penalty(np.array([0.0, 1.0], dtype=float))


@pytest.mark.parametrize(
    ("ctor", "kwargs", "match"),
    [
        (rank.GaussianPrior, {"var": 0.0}, "Variance must be positive"),
        (rank.LaplacePrior, {"scale": 0.0}, "Scale must be positive"),
        (rank.CauchyPrior, {"scale": 0.0}, "Scale must be positive"),
        (
            rank.EmpiricalPrior,
            {"R0": np.zeros((4, 3), dtype=int), "var": 0.0},
            "Variance must be positive",
        ),
    ],
)
def test_prior_constructor_validation_errors(ctor, kwargs, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        ctor(**kwargs)


@pytest.mark.parametrize(
    ("ctor", "kwargs"),
    [
        (rank.GaussianPrior, {"mean": np.nan}),
        (rank.GaussianPrior, {"var": np.inf}),
        (rank.LaplacePrior, {"loc": np.nan}),
        (rank.LaplacePrior, {"scale": np.inf}),
        (rank.CauchyPrior, {"loc": np.inf}),
        (rank.CauchyPrior, {"scale": np.nan}),
        (
            rank.EmpiricalPrior,
            {"R0": np.zeros((2, 2), dtype=int), "var": np.inf},
        ),
        (
            rank.EmpiricalPrior,
            {"R0": np.zeros((2, 2), dtype=int), "eps": np.nan},
        ),
    ],
)
def test_builtin_priors_reject_nonfinite_scalar_parameters(ctor, kwargs) -> None:
    with pytest.raises(ValueError, match="must be finite"):
        ctor(**kwargs)


@pytest.mark.parametrize("eps", [0.0, -0.1, 0.5, 1.0])
def test_empirical_prior_rejects_invalid_eps(eps: float) -> None:
    with pytest.raises(ValueError, match="strictly between 0 and 0.5"):
        rank.EmpiricalPrior(np.zeros((2, 2), dtype=int), eps=eps)


@pytest.mark.parametrize(
    ("R0", "match"),
    [
        (np.empty((2, 0), dtype=int), "non-empty"),
        (np.array([[0.0, np.nan], [1.0, 0.0]]), "NaN or Inf"),
        (np.array([[0, 2], [1, 0]]), "only binary values"),
    ],
)
def test_empirical_prior_rejects_invalid_outcomes(R0: np.ndarray, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        rank.EmpiricalPrior(R0)


def test_empirical_prior_rejects_invalid_R0_shape() -> None:
    with pytest.raises(ValueError, match=r"R0 must be 2D \(L, M\) or 3D \(L, M, D\)"):
        rank.EmpiricalPrior(np.zeros((2, 3, 4, 5), dtype=int))


def test_custom_prior_requires_callable() -> None:
    with pytest.raises(ValueError, match="penalty_fn must be callable"):
        rank.CustomPrior(penalty_fn=42)
