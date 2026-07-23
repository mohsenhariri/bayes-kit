"""Tests for scorio.aggregate.online (early-stopping / adaptive rules)."""

from __future__ import annotations

import numpy as np
import pytest

from scorio.aggregate import (
    adaptive_consistency_crp_stop,
    adaptive_consistency_dirichlet_stop,
    adaptive_consistency_stop,
    deepconf_online_stop,
    deepconf_stop_threshold,
    esc_stop,
)
from scorio.aggregate.online import _dirichlet_leader_probability


class TestAdaptiveConsistency:
    def test_decided_majority_stops(self) -> None:
        assert adaptive_consistency_stop(["A"] * 8 + ["B"] * 2) is True

    def test_tie_does_not_stop(self) -> None:
        assert adaptive_consistency_stop(["A", "A", "B", "B"]) is False

    def test_unanimous_five_probability(self) -> None:
        stop, p = adaptive_consistency_stop(["A"] * 5, return_prob=True)
        assert stop is True
        assert p == pytest.approx(1.0 - 0.5**6)  # Beta(6,1): 1 - 0.5^6

    def test_no_valid_answers_never_stops(self) -> None:
        stop, p = adaptive_consistency_stop([None, ""], return_prob=True)
        assert stop is False and p == pytest.approx(0.0)

    def test_empty_never_stops(self) -> None:
        assert adaptive_consistency_stop([]) is False

    def test_invalid_entries_ignored(self) -> None:
        # only the four valid A's count -> same as ["A"]*4
        a = adaptive_consistency_stop(["A", None, "A", "", "A", "A"], return_prob=True)[
            1
        ]
        b = adaptive_consistency_stop(["A"] * 4, return_prob=True)[1]
        assert a == pytest.approx(b)

    def test_more_separation_is_more_confident(self) -> None:
        p_small = adaptive_consistency_stop(["A"] * 6 + ["B"] * 4, return_prob=True)[1]
        p_large = adaptive_consistency_stop(["A"] * 10, return_prob=True)[1]
        assert p_large > p_small

    def test_higher_threshold_samples_more(self) -> None:
        answers = ["A"] * 8 + ["B"] * 2
        assert adaptive_consistency_stop(answers, threshold=0.95) is True
        assert adaptive_consistency_stop(answers, threshold=0.999) is False

    def test_bad_threshold_raises(self) -> None:
        with pytest.raises(ValueError):
            adaptive_consistency_stop(["A"], threshold=1.5)


class TestAdaptiveConsistencyDirichlet:
    @pytest.mark.parametrize(
        "answers",
        [
            ["A"] * 5,
            ["A"] * 5 + ["B"] * 2,
        ],
    )
    def test_fewer_than_three_categories_matches_beta(self, answers: list[str]) -> None:
        dirichlet = adaptive_consistency_dirichlet_stop(answers, return_prob=True)
        beta = adaptive_consistency_stop(answers, return_prob=True)
        assert dirichlet == pytest.approx(beta)

    def test_full_probability_uses_every_observed_category(self) -> None:
        _, probability = adaptive_consistency_dirichlet_stop(
            ["A"] * 5 + ["B"] * 2 + ["C"],
            return_prob=True,
        )
        beta_probability = adaptive_consistency_stop(
            ["A"] * 5 + ["B"] * 2,
            return_prob=True,
        )[1]
        assert probability == pytest.approx(0.8179649396053817)
        assert probability < beta_probability

    def test_large_counts_are_numerically_stable(self) -> None:
        _, probability = adaptive_consistency_dirichlet_stop(
            ["A"] * 1000 + ["B"] * 900 + ["C"],
            return_prob=True,
        )
        assert probability == pytest.approx(0.9891042503731576)

    def test_large_symmetric_counts_are_exact_by_exchangeability(self) -> None:
        _, probability = adaptive_consistency_dirichlet_stop(
            ["A"] * 1000 + ["B"] * 1000 + ["C"] * 1000,
            return_prob=True,
        )
        assert probability == pytest.approx(1.0 / 3.0)

    def test_generator_input_is_consumed_once(self) -> None:
        answers = (answer for answer in ["A"] * 5 + ["B"] * 2)
        result = adaptive_consistency_dirichlet_stop(answers, return_prob=True)
        expected = adaptive_consistency_stop(["A"] * 5 + ["B"] * 2, return_prob=True)
        assert result == pytest.approx(expected)

    def test_no_valid_answers_never_stops(self) -> None:
        assert adaptive_consistency_dirichlet_stop([None, ""], return_prob=True) == (
            False,
            0.0,
        )

    def test_bad_threshold_raises(self) -> None:
        with pytest.raises(ValueError):
            adaptive_consistency_dirichlet_stop(["A", "B", "C"], threshold=0.0)


class TestAdaptiveConsistencyCrp:
    def test_dominant_leader_stops(self) -> None:
        stop, probability = adaptive_consistency_crp_stop(
            ["A"] * 7 + ["B"],
            horizon=12,
            n_alpha=20,
            n_simulations=200,
            seed=7,
            return_prob=True,
        )
        assert stop is True
        assert probability >= 0.95

    def test_tied_counts_do_not_stop(self) -> None:
        stop, probability = adaptive_consistency_crp_stop(
            ["A", "B", "A", "B"],
            horizon=12,
            n_alpha=20,
            n_simulations=200,
            seed=7,
            return_prob=True,
        )
        assert stop is False
        assert 0.4 < probability < 0.6

    def test_one_step_probability_matches_crp_model(self) -> None:
        # For counts (1, 1) and one draw left, the first answer remains the
        # fixed-tie-broken leader unless the draw joins answer 2. Integrating
        # (1 + alpha) / (2 + alpha) under the paper's Gamma approximation gives
        # 0.6389217049692534.
        _, probability = adaptive_consistency_crp_stop(
            ["A", "B"],
            horizon=3,
            n_alpha=100,
            n_simulations=1000,
            seed=0,
            return_prob=True,
        )
        assert probability == pytest.approx(0.6389217049692534, abs=0.01)

    def test_seed_is_reproducible_and_invalid_entries_are_ignored(self) -> None:
        kwargs = {
            "horizon": 10,
            "n_alpha": 12,
            "n_simulations": 100,
            "seed": 23,
            "return_prob": True,
        }
        clean = adaptive_consistency_crp_stop(["A"] * 4 + ["B"], **kwargs)
        dirty = adaptive_consistency_crp_stop(
            ["A", None, "A", "", "A", "A", "B"], **kwargs
        )
        assert clean == dirty

    def test_no_valid_answers_never_stops(self) -> None:
        assert adaptive_consistency_crp_stop([None, ""], return_prob=True) == (
            False,
            0.0,
        )

    def test_budget_exhaustion_stops_without_simulation(self) -> None:
        assert adaptive_consistency_crp_stop(
            ["A", "B", "A"], horizon=3, return_prob=True
        ) == (True, 1.0)

    @pytest.mark.parametrize(
        ("parameter", "value"),
        [
            ("horizon", 0),
            ("horizon", 3.5),
            ("n_alpha", 0),
            ("n_simulations", False),
        ],
    )
    def test_bad_simulation_parameter_raises(
        self, parameter: str, value: object
    ) -> None:
        with pytest.raises(ValueError):
            adaptive_consistency_crp_stop(["A"], **{parameter: value})

    @pytest.mark.parametrize("seed", [-1, True, "bad"])
    def test_bad_seed_raises(self, seed: object) -> None:
        with pytest.raises(ValueError):
            adaptive_consistency_crp_stop(["A"], horizon=2, seed=seed)


class TestEsc:
    def test_unanimous_window_stops(self) -> None:
        assert esc_stop(["A", "A", "A"]) is True

    def test_mixed_window_continues(self) -> None:
        assert esc_stop(["A", "B", "A"]) is False

    def test_invalid_in_window_not_unanimous(self) -> None:
        assert esc_stop(["A", None, "A"]) is False

    def test_empty_window(self) -> None:
        assert esc_stop([]) is False

    def test_single_valid_answer(self) -> None:
        assert esc_stop(["A"]) is True

    def test_integer_answers(self) -> None:
        assert esc_stop([3, 3, 3]) is True and esc_stop([3, 3, 7]) is False


class TestDeepConfOnline:
    def test_threshold_is_upper_quantile(self) -> None:
        assert deepconf_stop_threshold([1, 2, 3, 4, 5], keep=0.2) == pytest.approx(4.2)

    def test_keep_all_is_min(self) -> None:
        assert deepconf_stop_threshold([1.0, 5.0, 3.0], keep=1.0) == pytest.approx(1.0)

    def test_threshold_bad_keep_raises(self) -> None:
        with pytest.raises(ValueError):
            deepconf_stop_threshold([1.0, 2.0], keep=0.0)

    def test_threshold_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            deepconf_stop_threshold([])

    def test_online_stop_returns_crossing_token(self) -> None:
        # conf 1,1,1,5,5,5 ; first 3-window mean = 1.0 < 2.0, window ends at token 2
        tk = [[0.0, -2.0]] * 3 + [[-4.0, -6.0]] * 3
        assert deepconf_online_stop(tk, threshold=2.0, window=3) == 2

    def test_online_stop_none_when_never_below(self) -> None:
        tk = [[0.0, -2.0]] * 3 + [[-4.0, -6.0]] * 3
        assert deepconf_online_stop(tk, threshold=0.5, window=3) is None

    def test_online_stop_high_threshold_terminates_first_window(self) -> None:
        tk = [[-4.0, -6.0]] * 6  # conf all 5.0
        # window 3, first window mean 5.0 < 10.0 -> stops at token index 2
        assert deepconf_online_stop(tk, threshold=10.0, window=3) == 2

    def test_online_stop_window_longer_than_trace(self) -> None:
        tk = [[0.0, -2.0], [-1.0, -3.0]]  # conf 1, 2 ; single group mean 1.5
        assert deepconf_online_stop(tk, threshold=2.0, window=999) == 1
        assert deepconf_online_stop(tk, threshold=1.0, window=999) is None

    def test_online_stop_rejects_nonpositive_window(self) -> None:
        with pytest.raises(ValueError, match="window must be positive"):
            deepconf_online_stop([[0.0, -1.0]], threshold=1.0, window=0)

    def test_threshold_rejects_nonfinite_warmup_confidence(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            deepconf_stop_threshold([1.0, np.nan])


def test_dirichlet_empty_probability_identity() -> None:
    assert _dirichlet_leader_probability([]) == 0.0


def test_esc_rejects_an_invalid_first_answer() -> None:
    assert esc_stop([None, "A"]) is False


def test_crp_rejects_boundary_threshold() -> None:
    with pytest.raises(ValueError, match="threshold"):
        adaptive_consistency_crp_stop(["A"], threshold=1.0)


def test_crp_single_leader_with_one_draw_left_remains_a_leader() -> None:
    assert adaptive_consistency_crp_stop(
        ["A"] * 10,
        horizon=11,
        n_alpha=1,
        n_simulations=1,
        seed=0,
        return_prob=True,
    ) == (True, 1.0)
