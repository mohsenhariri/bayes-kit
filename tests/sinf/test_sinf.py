"""Comprehensive tests for scorio.sinf sequential inference helpers."""

import math

import numpy as np
import pytest
from scipy.stats import norm

from scorio import eval as scorio_eval
from scorio import sinf


class TestRankingConfidence:
    def test_identical_scores_zero_sigma_returns_tie(self) -> None:
        rho, z = sinf.ranking_confidence(0.5, 0.0, 0.5, 0.0)
        assert rho == 0.5
        assert z == float("inf")

    def test_different_scores_zero_sigma_returns_certain(self) -> None:
        rho, z = sinf.ranking_confidence(0.7, 0.0, 0.3, 0.0)
        assert rho == 1.0
        assert z == float("inf")

    def test_well_separated_scores_high_confidence(self) -> None:
        rho, z = sinf.ranking_confidence(0.9, 0.01, 0.1, 0.01)
        assert rho > 0.999
        assert z > 3.0

    def test_overlapping_scores_low_confidence(self) -> None:
        rho, z = sinf.ranking_confidence(0.51, 0.1, 0.49, 0.1)
        assert rho < 0.6
        assert z < 0.5

    def test_symmetry(self) -> None:
        rho_ab, z_ab = sinf.ranking_confidence(0.6, 0.05, 0.4, 0.03)
        rho_ba, z_ba = sinf.ranking_confidence(0.4, 0.03, 0.6, 0.05)
        assert rho_ab == pytest.approx(rho_ba)
        assert z_ab == pytest.approx(z_ba)

    def test_manual_z_computation(self) -> None:
        mu_a, sigma_a = 0.7, 0.05
        mu_b, sigma_b = 0.4, 0.03
        expected_z = abs(mu_a - mu_b) / math.sqrt(sigma_a**2 + sigma_b**2)
        expected_rho = float(norm.cdf(expected_z))
        rho, z = sinf.ranking_confidence(mu_a, sigma_a, mu_b, sigma_b)
        assert z == pytest.approx(expected_z)
        assert rho == pytest.approx(expected_rho)

    def test_with_real_bayes_posteriors(self, top_p_task_aime25: np.ndarray) -> None:
        """Use real simulation data to compute Bayes posteriors, then check confidence."""
        mu_0, sigma_0 = scorio_eval.bayes(top_p_task_aime25[0])
        mu_1, sigma_1 = scorio_eval.bayes(top_p_task_aime25[1])
        rho, z = sinf.ranking_confidence(mu_0, sigma_0, mu_1, sigma_1)
        assert 0.5 <= rho <= 1.0
        assert z >= 0.0
        assert np.isfinite(rho)
        assert np.isfinite(z)


class TestCiFromMuSigma:
    def test_basic_interval(self) -> None:
        lo, hi = sinf.ci_from_mu_sigma(0.5, 0.1, confidence=0.95)
        z = float(norm.ppf(0.975))
        assert lo == pytest.approx(0.5 - z * 0.1)
        assert hi == pytest.approx(0.5 + z * 0.1)

    def test_clipping(self) -> None:
        lo, hi = sinf.ci_from_mu_sigma(0.05, 0.1, confidence=0.95, clip=(0.0, 1.0))
        assert lo >= 0.0
        assert hi <= 1.0

    def test_zero_sigma_gives_point_interval(self) -> None:
        lo, hi = sinf.ci_from_mu_sigma(0.5, 0.0, confidence=0.99)
        assert lo == pytest.approx(0.5)
        assert hi == pytest.approx(0.5)

    def test_higher_confidence_wider_interval(self) -> None:
        lo_90, hi_90 = sinf.ci_from_mu_sigma(0.5, 0.1, confidence=0.90)
        lo_99, hi_99 = sinf.ci_from_mu_sigma(0.5, 0.1, confidence=0.99)
        assert (hi_99 - lo_99) > (hi_90 - lo_90)

    def test_invalid_confidence_raises(self) -> None:
        with pytest.raises(ValueError, match="confidence must be in"):
            sinf.ci_from_mu_sigma(0.5, 0.1, confidence=0.0)
        with pytest.raises(ValueError, match="confidence must be in"):
            sinf.ci_from_mu_sigma(0.5, 0.1, confidence=1.0)

    def test_negative_sigma_raises(self) -> None:
        with pytest.raises(ValueError, match="sigma must be >= 0"):
            sinf.ci_from_mu_sigma(0.5, -0.1)

    def test_interval_contains_mu(self) -> None:
        for conf in [0.5, 0.8, 0.9, 0.95, 0.99]:
            lo, hi = sinf.ci_from_mu_sigma(0.42, 0.07, confidence=conf)
            assert lo <= 0.42 <= hi

    def test_with_real_bayes_posterior(self, top_p_task_aime25: np.ndarray) -> None:
        mu, sigma = scorio_eval.bayes(top_p_task_aime25[0])
        lo, hi = sinf.ci_from_mu_sigma(mu, sigma, confidence=0.95, clip=(0.0, 1.0))
        assert 0.0 <= lo <= mu <= hi <= 1.0


class TestShouldStop:
    def test_half_width_criterion(self) -> None:
        z95 = float(norm.ppf(0.975))
        # sigma=0.005, half_width = z95 * 0.005 ≈ 0.0098; threshold 0.02 => stop
        assert sinf.should_stop(0.005, confidence=0.95, max_half_width=0.02) is True
        # sigma=0.05, half_width = z95 * 0.05 ≈ 0.098; threshold 0.02 => don't stop
        assert sinf.should_stop(0.05, confidence=0.95, max_half_width=0.02) is False

    def test_ci_width_criterion(self) -> None:
        z95 = float(norm.ppf(0.975))
        # full width = 2 * z95 * sigma
        sigma = 0.005
        full_width = 2 * z95 * sigma
        assert (
            sinf.should_stop(sigma, confidence=0.95, max_ci_width=full_width + 0.001)
            is True
        )
        assert (
            sinf.should_stop(sigma, confidence=0.95, max_ci_width=full_width - 0.001)
            is False
        )

    def test_half_width_and_ci_width_consistency(self) -> None:
        sigma = 0.03
        z95 = float(norm.ppf(0.975))
        hw = z95 * sigma
        # Both criteria should agree
        result_hw = sinf.should_stop(sigma, confidence=0.95, max_half_width=hw + 1e-10)
        result_cw = sinf.should_stop(
            sigma, confidence=0.95, max_ci_width=2 * hw + 1e-10
        )
        assert result_hw == result_cw

    def test_must_provide_exactly_one_criterion(self) -> None:
        with pytest.raises(ValueError, match="Provide exactly one"):
            sinf.should_stop(0.01)  # neither
        with pytest.raises(ValueError, match="Provide exactly one"):
            sinf.should_stop(0.01, max_ci_width=0.1, max_half_width=0.05)  # both

    def test_zero_sigma_always_stops(self) -> None:
        assert sinf.should_stop(0.0, confidence=0.95, max_half_width=0.001) is True
        assert sinf.should_stop(0.0, confidence=0.95, max_ci_width=0.001) is True

    def test_with_real_data_progressive_precision(
        self, top_p_task_aime25: np.ndarray
    ) -> None:
        """More trials should reduce sigma, eventually meeting the threshold."""
        model_data = top_p_task_aime25[0]
        M = model_data.shape[0]
        sigmas = []
        for n_trials in [1, 5, 10, 20, 40, 80]:
            R_sub = model_data[:, :n_trials]
            _, sigma = scorio_eval.bayes(R_sub)
            sigmas.append(sigma)

        # Sigma should decrease with more trials
        for i in range(1, len(sigmas)):
            assert sigmas[i] <= sigmas[i - 1] + 1e-10

        # With enough trials, should eventually stop
        assert sinf.should_stop(sigmas[-1], max_half_width=0.1) is True


class TestShouldStopTop1:
    def test_clear_leader_stops_ci_overlap(self) -> None:
        # Model 0 is clearly the best
        mus = [0.9, 0.3, 0.2]
        sigmas = [0.01, 0.01, 0.01]
        result = sinf.should_stop_top1(
            mus, sigmas, confidence=0.95, method="ci_overlap"
        )
        assert result["stop"] is True
        assert result["leader"] == 0
        assert result["ambiguous"] == []

    def test_clear_leader_stops_zscore(self) -> None:
        mus = [0.9, 0.3, 0.2]
        sigmas = [0.01, 0.01, 0.01]
        result = sinf.should_stop_top1(mus, sigmas, confidence=0.95, method="zscore")
        assert result["stop"] is True
        assert result["leader"] == 0
        assert result["ambiguous"] == []

    def test_ambiguous_leader_does_not_stop(self) -> None:
        mus = [0.51, 0.49, 0.2]
        sigmas = [0.1, 0.1, 0.01]
        result = sinf.should_stop_top1(
            mus, sigmas, confidence=0.95, method="ci_overlap"
        )
        assert result["stop"] is False
        assert result["leader"] == 0
        assert 1 in result["ambiguous"]

    def test_zscore_ambiguous(self) -> None:
        mus = [0.51, 0.49, 0.2]
        sigmas = [0.1, 0.1, 0.01]
        result = sinf.should_stop_top1(mus, sigmas, confidence=0.95, method="zscore")
        assert result["stop"] is False
        assert 1 in result["ambiguous"]

    def test_all_equal_means_no_stop(self) -> None:
        mus = [0.5, 0.5, 0.5]
        sigmas = [0.1, 0.1, 0.1]
        result = sinf.should_stop_top1(mus, sigmas, confidence=0.95)
        assert result["stop"] is False

    def test_leader_is_argmax(self) -> None:
        mus = [0.3, 0.7, 0.5, 0.1]
        sigmas = [0.01, 0.01, 0.01, 0.01]
        result = sinf.should_stop_top1(mus, sigmas)
        assert result["leader"] == 1

    def test_validation_shape_mismatch(self) -> None:
        with pytest.raises(ValueError, match="1D and same shape"):
            sinf.should_stop_top1([0.5, 0.3], [0.1])

    def test_validation_empty_input(self) -> None:
        with pytest.raises(ValueError, match="Empty inputs"):
            sinf.should_stop_top1([], [])

    def test_invalid_method(self) -> None:
        with pytest.raises(ValueError, match="method must be"):
            sinf.should_stop_top1([0.5, 0.3], [0.1, 0.1], method="bad")

    def test_with_real_bayes_posteriors(
        self, top_p_data: dict[str, np.ndarray]
    ) -> None:
        """Use full simulation data across all models."""
        R = top_p_data["aime25"]  # (20, 30, 80)
        L = R.shape[0]
        mus = np.empty(L)
        sigmas = np.empty(L)
        for i in range(L):
            mus[i], sigmas[i] = scorio_eval.bayes(R[i])

        result_ci = sinf.should_stop_top1(
            mus, sigmas, confidence=0.95, method="ci_overlap"
        )
        result_zs = sinf.should_stop_top1(mus, sigmas, confidence=0.95, method="zscore")

        assert isinstance(result_ci["stop"], bool)
        assert isinstance(result_zs["stop"], bool)
        assert 0 <= result_ci["leader"] < L
        assert 0 <= result_zs["leader"] < L
        assert all(0 <= j < L for j in result_ci["ambiguous"])
        assert result_ci["leader"] not in result_ci["ambiguous"]

    def test_reducing_uncertainty_leads_to_stop(self) -> None:
        """As sigma shrinks, ambiguous competitors should vanish."""
        mus = [0.7, 0.5, 0.3]
        for sigma in [1.0, 0.1, 0.01, 0.001]:
            sigmas = [sigma] * 3
            result = sinf.should_stop_top1(
                mus, sigmas, confidence=0.95, method="ci_overlap"
            )
            if sigma <= 0.01:
                assert result["stop"] is True


class TestSuggestNextAllocation:
    def test_returns_leader_and_competitor(self) -> None:
        mus = [0.7, 0.5, 0.3]
        sigmas = [0.05, 0.05, 0.05]
        leader, competitor = sinf.suggest_next_allocation(mus, sigmas)
        assert leader == 0
        assert competitor != leader
        assert 0 <= competitor < 3

    def test_most_ambiguous_competitor_selected_ci_overlap(self) -> None:
        # Model 1 is closest to model 0 => most ambiguous
        mus = [0.7, 0.65, 0.2]
        sigmas = [0.05, 0.05, 0.05]
        leader, competitor = sinf.suggest_next_allocation(
            mus, sigmas, method="ci_overlap"
        )
        assert leader == 0
        assert competitor == 1

    def test_most_ambiguous_competitor_selected_zscore(self) -> None:
        mus = [0.7, 0.65, 0.2]
        sigmas = [0.05, 0.05, 0.05]
        leader, competitor = sinf.suggest_next_allocation(mus, sigmas, method="zscore")
        assert leader == 0
        assert competitor == 1

    def test_high_sigma_competitor_is_more_ambiguous(self) -> None:
        # Model 2 has same mean as model 1 but higher sigma => more ambiguous
        mus = [0.7, 0.4, 0.4]
        sigmas = [0.01, 0.01, 0.15]
        leader, competitor = sinf.suggest_next_allocation(
            mus, sigmas, method="ci_overlap"
        )
        assert leader == 0
        assert competitor == 2  # higher sigma makes it overlap more

    def test_validation_need_at_least_two(self) -> None:
        with pytest.raises(ValueError, match="at least two"):
            sinf.suggest_next_allocation([0.5], [0.1])

    def test_validation_shape_mismatch(self) -> None:
        with pytest.raises(ValueError, match="1D and same shape"):
            sinf.suggest_next_allocation([0.5, 0.3], [0.1])

    def test_invalid_method(self) -> None:
        with pytest.raises(ValueError, match="method must be"):
            sinf.suggest_next_allocation([0.5, 0.3], [0.1, 0.1], method="bad")

    def test_with_real_bayes_posteriors(
        self, top_p_data: dict[str, np.ndarray]
    ) -> None:
        """Use full simulation data to get real Bayes posteriors."""
        R = top_p_data["aime25"]
        L = R.shape[0]
        mus = np.empty(L)
        sigmas = np.empty(L)
        for i in range(L):
            mus[i], sigmas[i] = scorio_eval.bayes(R[i])

        leader, competitor = sinf.suggest_next_allocation(
            mus, sigmas, confidence=0.95, method="ci_overlap"
        )
        assert 0 <= leader < L
        assert 0 <= competitor < L
        assert leader != competitor
        assert leader == int(np.argmax(mus))

    def test_consistency_with_should_stop_top1(self) -> None:
        """When not stopped, the allocation target should be one of the ambiguous models."""
        mus = [0.6, 0.55, 0.3, 0.1]
        sigmas = [0.1, 0.1, 0.05, 0.05]

        stop_result = sinf.should_stop_top1(
            mus, sigmas, confidence=0.95, method="ci_overlap"
        )
        if not stop_result["stop"]:
            leader, competitor = sinf.suggest_next_allocation(
                mus, sigmas, confidence=0.95, method="ci_overlap"
            )
            assert leader == stop_result["leader"]
            # The suggested competitor should be in the ambiguous set
            assert competitor in stop_result["ambiguous"]


class TestAdaptiveWorkflowIntegration:
    """End-to-end test of a simulated adaptive evaluation loop using sinf helpers."""

    def test_sequential_evaluation_workflow(
        self, top_p_data: dict[str, np.ndarray]
    ) -> None:
        """Simulate sequential trial collection with stopping criteria."""
        R_full = top_p_data["aime25"]  # (20, 30, 80)
        L, M, N = R_full.shape

        # Start with 5 trials, add batches of 5
        for n_trials in range(5, N + 1, 5):
            R_sub = R_full[:, :, :n_trials]

            mus = np.empty(L)
            sigmas = np.empty(L)
            for i in range(L):
                mus[i], sigmas[i] = scorio_eval.bayes(R_sub[i])

            # Check if top-1 is resolved
            stop_result = sinf.should_stop_top1(
                mus, sigmas, confidence=0.95, method="ci_overlap"
            )

            if stop_result["stop"]:
                # Verify the leader has a meaningful advantage
                leader = stop_result["leader"]
                assert mus[leader] == np.max(mus)
                break

            # Get allocation suggestion
            leader, competitor = sinf.suggest_next_allocation(
                mus, sigmas, confidence=0.95, method="ci_overlap"
            )

            # Basic sanity on allocation
            assert leader == int(np.argmax(mus))
            assert competitor != leader

    def test_ci_width_convergence(self, top_p_task_aime25: np.ndarray) -> None:
        """CI width should monotonically decrease as trials increase."""
        model_data = top_p_task_aime25[0]  # (30, 80)
        prev_width = float("inf")

        for n_trials in [2, 5, 10, 20, 40, 80]:
            R_sub = model_data[:, :n_trials]
            mu, sigma = scorio_eval.bayes(R_sub)
            lo, hi = sinf.ci_from_mu_sigma(mu, sigma, confidence=0.95, clip=(0.0, 1.0))
            width = hi - lo
            assert width <= prev_width + 1e-10
            prev_width = width

    def test_pairwise_confidence_with_more_trials(
        self, top_p_task_aime25: np.ndarray
    ) -> None:
        """With enough trials, pairwise confidence should be reasonable."""
        model_a_data = top_p_task_aime25[0]
        model_b_data = top_p_task_aime25[5]

        z_values = []
        for n_trials in [5, 10, 20, 40, 80]:
            mu_a, sigma_a = scorio_eval.bayes(model_a_data[:, :n_trials])
            mu_b, sigma_b = scorio_eval.bayes(model_b_data[:, :n_trials])
            rho, z = sinf.ranking_confidence(mu_a, sigma_a, mu_b, sigma_b)
            z_values.append(z)
            assert np.isfinite(z) and z >= 0.0
            assert 0.5 <= rho <= 1.0

        # With max trials, z should be nontrivial (sigmas are small)
        assert z_values[-1] > 0.0


def test_sinf_public_api_exports() -> None:
    # The original fixed-look API must remain exported for backward compatibility
    # (the JSS manuscript and the Julia/JS ports depend on these names).
    legacy = {
        "ranking_confidence",
        "ci_from_mu_sigma",
        "should_stop",
        "should_stop_top1",
        "suggest_next_allocation",
    }
    assert legacy <= set(sinf.__all__)
    # The anytime-valid layer is exported alongside it.
    for name in (
        "confseq_mean",
        "score_confseq",
        "precision_stop",
        "compare_paired",
        "decide_better",
        "pairwise_confidence",
        "should_stop_top1_av",
        "should_stop_full_ranking",
        "suggest_next_allocation_stratified",
        "select_best_fixed_budget",
        "should_stop_sampling",
        "adaptive_consistency_stop",
    ):
        assert name in sinf.__all__, name
        assert hasattr(sinf, name), name
