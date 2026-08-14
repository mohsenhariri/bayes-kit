"""Characterization tests for the public :mod:`scorio.eval` contract."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from scorio import eval as scorio_eval


EXPECTED_EXPORTS = (
    "bayes",
    "bayes_ci",
    "avg",
    "avg_ci",
    "pass_at_k",
    "pass_at_k_ci",
    "pass_hat_k",
    "pass_hat_k_ci",
    "unanimous_at_k",
    "unanimous_at_k_ci",
    "g_pass_at_k",
    "g_pass_at_k_ci",
    "g_pass_at_k_tau",
    "g_pass_at_k_tau_ci",
    "mg_pass_at_k",
    "mg_pass_at_k_ci",
    "maj_at_k",
    "maj_at_k_ci",
    "auc_at_k",
    "auc_at_k_ci",
    "max_at_k",
    "max_at_k_ci",
    "threshold_spectrum_at_k",
    "threshold_spectrum_at_k_ci",
    "geom_at_k",
    "geom_ds_at_k",
    "geom_at_k_ci",
    "geom_ds_at_k_ci",
    "geo_spectrum_at_k",
    "geo_spectrum_at_k_ci",
    "geo_spectrum_star_at_k",
    "geo_spectrum_star_at_k_ci",
)

REQUIRED = inspect.Parameter.empty
R = ("R", REQUIRED)
K = ("k", REQUIRED)
CONFIDENCE = ("confidence", 0.95)
BOUNDS_01 = ("bounds", (0.0, 1.0))
ALPHA0 = ("alpha0", 1.0)
BETA0 = ("beta0", 1.0)

RK = (R, K)
RK_CI = RK + (CONFIDENCE, BOUNDS_01, ALPHA0, BETA0)
GEOM = RK + (("pass_power", 0.5), ("unanimous_power", 0.5))
GEOM_CI = GEOM + (CONFIDENCE, BOUNDS_01, ALPHA0, BETA0)

EXPECTED_PARAMETERS: dict[str, tuple[tuple[str, Any], ...]] = {
    "bayes": (R, ("w", None), ("R0", None)),
    "bayes_ci": (
        R,
        ("w", None),
        ("R0", None),
        CONFIDENCE,
        ("bounds", None),
    ),
    "avg": (R, ("w", None)),
    "avg_ci": (R, ("w", None), CONFIDENCE, ("bounds", None)),
    **{
        name: RK
        for name in (
            "pass_at_k",
            "pass_hat_k",
            "unanimous_at_k",
            "g_pass_at_k",
            "mg_pass_at_k",
            "maj_at_k",
            "auc_at_k",
            "geo_spectrum_star_at_k",
        )
    },
    **{
        name: RK_CI
        for name in (
            "pass_at_k_ci",
            "pass_hat_k_ci",
            "unanimous_at_k_ci",
            "g_pass_at_k_ci",
            "mg_pass_at_k_ci",
            "maj_at_k_ci",
            "auc_at_k_ci",
            "geo_spectrum_star_at_k_ci",
        )
    },
    "g_pass_at_k_tau": RK + (("tau", REQUIRED),),
    "g_pass_at_k_tau_ci": RK
    + (("tau", REQUIRED), CONFIDENCE, BOUNDS_01, ALPHA0, BETA0),
    "max_at_k": RK + (("w", None),),
    "max_at_k_ci": RK
    + (("w", None), ("R0", None), CONFIDENCE, ("bounds", None)),
    "threshold_spectrum_at_k": RK + (("weights", REQUIRED),),
    "threshold_spectrum_at_k_ci": RK
    + (("weights", REQUIRED), CONFIDENCE, BOUNDS_01, ALPHA0, BETA0),
    "geom_at_k": GEOM,
    "geom_ds_at_k": GEOM,
    "geom_at_k_ci": GEOM_CI,
    "geom_ds_at_k_ci": GEOM_CI,
    "geo_spectrum_at_k": RK
    + (("lam", 0.5), ("weights", None), ("lambda_", None)),
    "geo_spectrum_at_k_ci": RK
    + (
        ("lam", 0.5),
        ("weights", None),
        ("lambda_", None),
        CONFIDENCE,
        BOUNDS_01,
        ALPHA0,
        BETA0,
    ),
}


def _api_calls(values: np.ndarray) -> dict[str, Callable[[], object]]:
    weights = np.array([0.25, 0.75], dtype=float)
    return {
        "bayes": lambda: scorio_eval.bayes(values),
        "bayes_ci": lambda: scorio_eval.bayes_ci(values),
        "avg": lambda: scorio_eval.avg(values),
        "avg_ci": lambda: scorio_eval.avg_ci(values),
        "pass_at_k": lambda: scorio_eval.pass_at_k(values, 2),
        "pass_at_k_ci": lambda: scorio_eval.pass_at_k_ci(values, 2),
        "pass_hat_k": lambda: scorio_eval.pass_hat_k(values, 2),
        "pass_hat_k_ci": lambda: scorio_eval.pass_hat_k_ci(values, 2),
        "unanimous_at_k": lambda: scorio_eval.unanimous_at_k(values, 2),
        "unanimous_at_k_ci": lambda: scorio_eval.unanimous_at_k_ci(values, 2),
        "g_pass_at_k": lambda: scorio_eval.g_pass_at_k(values, 2),
        "g_pass_at_k_ci": lambda: scorio_eval.g_pass_at_k_ci(values, 2),
        "g_pass_at_k_tau": lambda: scorio_eval.g_pass_at_k_tau(values, 2, 0.5),
        "g_pass_at_k_tau_ci": lambda: scorio_eval.g_pass_at_k_tau_ci(
            values, 2, 0.5
        ),
        "mg_pass_at_k": lambda: scorio_eval.mg_pass_at_k(values, 2),
        "mg_pass_at_k_ci": lambda: scorio_eval.mg_pass_at_k_ci(values, 2),
        "maj_at_k": lambda: scorio_eval.maj_at_k(values, 2),
        "maj_at_k_ci": lambda: scorio_eval.maj_at_k_ci(values, 2),
        "auc_at_k": lambda: scorio_eval.auc_at_k(values, 2),
        "auc_at_k_ci": lambda: scorio_eval.auc_at_k_ci(values, 2),
        "max_at_k": lambda: scorio_eval.max_at_k(values, 2),
        "max_at_k_ci": lambda: scorio_eval.max_at_k_ci(values, 2),
        "threshold_spectrum_at_k": lambda: scorio_eval.threshold_spectrum_at_k(
            values, 2, weights
        ),
        "threshold_spectrum_at_k_ci": lambda: scorio_eval.threshold_spectrum_at_k_ci(
            values, 2, weights
        ),
        "geom_at_k": lambda: scorio_eval.geom_at_k(values, 2),
        "geom_ds_at_k": lambda: scorio_eval.geom_ds_at_k(values, 2),
        "geom_at_k_ci": lambda: scorio_eval.geom_at_k_ci(values, 2),
        "geom_ds_at_k_ci": lambda: scorio_eval.geom_ds_at_k_ci(values, 2),
        "geo_spectrum_at_k": lambda: scorio_eval.geo_spectrum_at_k(values, 2),
        "geo_spectrum_at_k_ci": lambda: scorio_eval.geo_spectrum_at_k_ci(
            values, 2
        ),
        "geo_spectrum_star_at_k": lambda: scorio_eval.geo_spectrum_star_at_k(
            values, 2
        ),
        "geo_spectrum_star_at_k_ci": lambda: scorio_eval.geo_spectrum_star_at_k_ci(
            values, 2
        ),
    }


def _finite_k_call(name: str, values: np.ndarray, k: int) -> object:
    if name == "threshold_spectrum_at_k":
        return scorio_eval.threshold_spectrum_at_k(values, k, np.ones(k) / k)
    return getattr(scorio_eval, name)(values, k)


def _finite_k_ci_call(name: str, values: np.ndarray, k: int) -> object:
    if name == "g_pass_at_k_tau_ci":
        return scorio_eval.g_pass_at_k_tau_ci(values, k, 0.5)
    return getattr(scorio_eval, name)(values, k)


def _latent_k_ci_call(name: str, values: np.ndarray, k: int) -> object:
    if name == "threshold_spectrum_at_k_ci":
        return scorio_eval.threshold_spectrum_at_k_ci(
            values, k, np.full(k, 1.0 / k)
        )
    return getattr(scorio_eval, name)(values, k)


def test_public_exports_are_exact() -> None:
    assert scorio_eval.__all__ == list(EXPECTED_EXPORTS)
    assert set(EXPECTED_PARAMETERS) == set(EXPECTED_EXPORTS)


@pytest.mark.parametrize("name", EXPECTED_EXPORTS)
def test_public_signatures_and_defaults_are_stable(name: str) -> None:
    signature = inspect.signature(getattr(scorio_eval, name))
    actual = tuple(
        (parameter.name, parameter.default) for parameter in signature.parameters.values()
    )

    assert actual == EXPECTED_PARAMETERS[name]
    assert all(
        parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        for parameter in signature.parameters.values()
    )


@pytest.mark.parametrize("name", EXPECTED_EXPORTS)
def test_public_return_container_and_scalar_types(name: str) -> None:
    values = np.array([[0, 1, 1, 0], [1, 1, 0, 1]], dtype=int)
    result = _api_calls(values)[name]()

    if name in {"bayes", "avg"}:
        assert type(result) is tuple
        assert len(result) == 2
        assert all(type(value) is float for value in result)
    elif name.endswith("_ci"):
        assert type(result) is tuple
        assert len(result) == 4
        assert all(type(value) is float for value in result)
    else:
        assert type(result) is float


@pytest.mark.parametrize("name", EXPECTED_EXPORTS)
def test_one_dimensional_input_matches_a_single_row(name: str) -> None:
    one_dimensional = np.array([0, 1, 1, 0], dtype=int)
    single_row = one_dimensional.reshape(1, -1)

    actual = _api_calls(one_dimensional)[name]()
    expected = _api_calls(single_row)[name]()

    assert actual == pytest.approx(expected)


def test_named_aliases_and_equivalent_entry_points() -> None:
    values = np.array([[0, 1, 1, 0], [1, 1, 0, 1]], dtype=int)

    assert scorio_eval.unanimous_at_k is scorio_eval.pass_hat_k
    assert scorio_eval.unanimous_at_k_ci is scorio_eval.pass_hat_k_ci
    assert scorio_eval.g_pass_at_k(values, 2) == scorio_eval.pass_hat_k(values, 2)
    assert scorio_eval.g_pass_at_k_ci(values, 2) == pytest.approx(
        scorio_eval.pass_hat_k_ci(values, 2)
    )
    assert scorio_eval.geo_spectrum_star_at_k(
        values, 2
    ) == scorio_eval.geo_spectrum_at_k(values, 2)
    assert scorio_eval.geo_spectrum_star_at_k_ci(values, 2) == pytest.approx(
        scorio_eval.geo_spectrum_at_k_ci(values, 2)
    )
    assert scorio_eval.geo_spectrum_at_k(
        values, 2, lam=0.25
    ) == scorio_eval.geo_spectrum_at_k(values, 2, lambda_=0.25)


@pytest.mark.parametrize(
    "name",
    (
        "max_at_k",
        "threshold_spectrum_at_k",
        "geom_at_k",
        "geom_ds_at_k",
        "geo_spectrum_at_k",
        "geo_spectrum_star_at_k",
    ),
)
def test_finite_bank_point_metrics_reject_k_above_trial_count(name: str) -> None:
    values = np.array([0, 1, 1], dtype=int)

    with pytest.raises(ValueError, match="1 <= k <= N"):
        _finite_k_call(name, values, k=4)


@pytest.mark.parametrize(
    "name",
    (
        "pass_at_k_ci",
        "pass_hat_k_ci",
        "g_pass_at_k_ci",
        "g_pass_at_k_tau_ci",
        "mg_pass_at_k_ci",
        "maj_at_k_ci",
        "auc_at_k_ci",
    ),
)
def test_finite_bank_interval_metrics_reject_k_above_trial_count(name: str) -> None:
    values = np.array([0, 1, 1], dtype=int)

    with pytest.raises(ValueError, match="1 <= k <= N"):
        _finite_k_ci_call(name, values, k=4)


@pytest.mark.parametrize(
    "name",
    (
        "max_at_k_ci",
        "threshold_spectrum_at_k_ci",
        "geom_at_k_ci",
        "geom_ds_at_k_ci",
        "geo_spectrum_at_k_ci",
        "geo_spectrum_star_at_k_ci",
    ),
)
def test_latent_interval_metrics_allow_k_above_trial_count(name: str) -> None:
    values = np.array([0, 1, 1], dtype=int)

    result = _latent_k_ci_call(name, values, k=4)

    assert type(result) is tuple
    assert len(result) == 4
    assert all(type(value) is float for value in result)
