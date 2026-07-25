using Test
using Scorio

@testset "eval/apis.jl" begin
    R = Int[
        0 1 1 0 1
        1 1 0 1 1
    ]
    R_multi = Int[
        0 1 2 2 1
        1 1 0 2 2
    ]
    w = [0.0, 0.5, 1.0]
    R0 = Int[
        0 2
        1 2
    ]

    @testset "Bayes family" begin
        mu_prior, sigma_prior = Scorio.Eval.bayes(R_multi, w, R0)
        @test mu_prior ≈ 0.575 atol = 1e-12
        @test sigma_prior ≈ 0.08427498280790524 atol = 1e-12

        mu_noprior, sigma_noprior = Scorio.Eval.bayes(R_multi, w)
        @test mu_noprior ≈ 0.5625 atol = 1e-12
        @test sigma_noprior ≈ 0.0919975090242484 atol = 1e-12

        mu_auto, sigma_auto = Scorio.Eval.bayes(R)
        mu_explicit, sigma_explicit = Scorio.Eval.bayes(R, [0.0, 1.0])
        @test mu_auto ≈ mu_explicit atol = 1e-12
        @test sigma_auto ≈ sigma_explicit atol = 1e-12

        # Python reshapes a flat R0 in row-major order. Keep the per-row
        # prior assignment identical rather than using Julia's column-major
        # reshape order.
        R_flat_prior = Int[0 0; 1 1]
        flat_R0 = [0, 1, 0, 1]
        explicit_R0 = Int[0 1; 0 1]
        flat_bayes = Scorio.Eval.bayes(R_flat_prior, nothing, flat_R0)
        @test flat_bayes == Scorio.Eval.bayes(R_flat_prior, nothing, explicit_R0)
        @test flat_bayes[1] ≈ 0.5 atol = 1e-12
        @test flat_bayes[2] ≈ 0.12598815766974242 atol = 1e-12

        flat_max_ci = Scorio.Eval.max_at_k_ci(R_flat_prior, 2, nothing, flat_R0)
        explicit_max_ci = Scorio.Eval.max_at_k_ci(R_flat_prior, 2, nothing, explicit_R0)
        @test all(isapprox.(collect(flat_max_ci), collect(explicit_max_ci); atol=1e-12))
        @test all(isapprox.(
            collect(flat_max_ci),
            [
                0.6904761904761905,
                0.13256581816261961,
                0.43065196129637023,
                0.9503004196560108,
            ];
            atol=1e-12,
        ))

        @test_throws ErrorException Scorio.Eval.bayes(R_multi)
        @test_throws ErrorException Scorio.Eval.bayes(R_multi, w, zeros(Int, 3, 2))

        mu, sigma, lo, hi = Scorio.Eval.bayes_ci(R, nothing, nothing, 0.9, (0.0, 1.0))
        z = Scorio._z_value(0.9; two_sided=true)
        expected_lo = max(mu - z * sigma, 0.0)
        expected_hi = min(mu + z * sigma, 1.0)
        @test lo ≈ expected_lo atol = 1e-12
        @test hi ≈ expected_hi atol = 1e-12
        @test lo <= mu <= hi
    end

    @testset "Avg family" begin
        a, sigma_a = Scorio.Eval.avg(R)
        @test a ≈ 0.7 atol = 1e-12
        @test sigma_a ≈ 0.16583123951776998 atol = 1e-12

        a_w, sigma_w = Scorio.Eval.avg(R_multi, w)
        @test a_w ≈ 0.6 atol = 1e-12
        @test sigma_w ≈ 0.14719601443879746 atol = 1e-12

        @test_throws ErrorException Scorio.Eval.avg(R_multi)

        a_ci, sigma_ci, lo, hi = Scorio.Eval.avg_ci(R, nothing, 0.8, (0.0, 1.0))
        z = Scorio._z_value(0.8; two_sided=true)
        expected_lo = max(a_ci - z * sigma_ci, 0.0)
        expected_hi = min(a_ci + z * sigma_ci, 1.0)
        @test lo ≈ expected_lo atol = 1e-12
        @test hi ≈ expected_hi atol = 1e-12
        @test lo <= a_ci <= hi
    end

    @testset "Pass family point metrics" begin
        @test Scorio.Eval.pass_at_k(R, 1) ≈ 0.7 atol = 1e-12
        @test Scorio.Eval.pass_at_k(R, 2) ≈ 0.95 atol = 1e-12
        @test Scorio.Eval.pass_hat_k(R, 1) ≈ 0.7 atol = 1e-12
        @test Scorio.Eval.pass_hat_k(R, 2) ≈ 0.45 atol = 1e-12
        @test Scorio.Eval.g_pass_at_k(R, 2) ≈ 0.45 atol = 1e-12
        @test Scorio.Eval.g_pass_at_k_tau(R, 2, 0.5) ≈ 0.95 atol = 1e-12
        @test Scorio.Eval.g_pass_at_k_tau(R, 2, 1.0) ≈ 0.45 atol = 1e-12
        @test Scorio.Eval.mg_pass_at_k(R, 2) ≈ 0.45 atol = 1e-12
        @test Scorio.Eval.mg_pass_at_k(R, 3) ≈ (1 / 6) atol = 1e-12
        @test Scorio.Eval.mg_pass_at_k(R, 1) ≈ 0.0 atol = 1e-12
    end

    @testset "Pass family CI and aliases" begin
        mu1, sigma1, lo1, hi1 = Scorio.Eval.pass_at_k_ci(R, 1)
        @test mu1 ≈ 0.6428571428571428 atol = 1e-12
        @test sigma1 ≈ 0.11845088536983578 atol = 1e-12
        @test lo1 ≈ 0.41069767359538273 atol = 1e-12
        @test hi1 ≈ 0.8750166121189029 atol = 1e-12

        mu2, sigma2, lo2, hi2 = Scorio.Eval.pass_hat_k_ci(R, 2)
        @test mu2 ≈ 0.4464285714285715 atol = 1e-12
        @test sigma2 ≈ 0.14616701378343605 atol = 1e-12
        @test lo2 ≈ 0.15994648868526573 atol = 1e-12
        @test hi2 ≈ 0.732910654171877 atol = 1e-12

        @test Scorio.Eval.g_pass_at_k(R, 3) ≈ Scorio.Eval.pass_hat_k(R, 3) atol = 1e-12
        @test Scorio.Eval.g_pass_at_k_tau(R, 3, 0.0) ≈ Scorio.Eval.pass_at_k(R, 3) atol = 1e-12
        @test Scorio.Eval.g_pass_at_k_tau(R, 3, 1.0) ≈ Scorio.Eval.pass_hat_k(R, 3) atol = 1e-12

        c1 = Scorio.Eval.g_pass_at_k_ci(R, 3)
        c2 = Scorio.Eval.pass_hat_k_ci(R, 3)
        @test all(isapprox.(collect(c1), collect(c2); atol=1e-12))

        c3 = Scorio.Eval.g_pass_at_k_tau_ci(R, 3, 0.0)
        c4 = Scorio.Eval.pass_at_k_ci(R, 3)
        @test all(isapprox.(collect(c3), collect(c4); atol=1e-12))

        c5 = Scorio.Eval.g_pass_at_k_tau_ci(R, 3, 1.0)
        c6 = Scorio.Eval.pass_hat_k_ci(R, 3)
        @test all(isapprox.(collect(c5), collect(c6); atol=1e-12))

        @test Scorio.Eval.mg_pass_at_k_ci(R, 1) == (0.0, 0.0, 0.0, 0.0)
    end

    @testset "Validation and vector input" begin
        for fn in (
            Scorio.Eval.pass_at_k,
            Scorio.Eval.pass_hat_k,
            Scorio.Eval.g_pass_at_k,
            Scorio.Eval.mg_pass_at_k,
            Scorio.Eval.pass_at_k_ci,
            Scorio.Eval.pass_hat_k_ci,
            Scorio.Eval.g_pass_at_k_ci,
            Scorio.Eval.mg_pass_at_k_ci,
        )
            @test_throws ErrorException fn(R, 0)
        end

        @test_throws ErrorException Scorio.Eval.g_pass_at_k_tau(R, 2, 1.1)
        @test_throws ErrorException Scorio.Eval.g_pass_at_k_tau_ci(R, 2, 1.1)
        @test_throws ErrorException Scorio.Eval.pass_at_k(Int[0 1 2; 1 0 1], 1)

        # The Eval namespace must retain Python's 1D/2D input contract even
        # though Rank exposes methods with several of the same names for 3D
        # response tensors. In particular, Rank dispatches must not leak into
        # `Scorio.Eval` through the parent-module generic functions.
        rank_shaped = zeros(Int, 2, 2, 2)
        @test_throws ErrorException Scorio.Eval.avg(rank_shaped)
        @test_throws ErrorException Scorio.Eval.bayes(rank_shaped)
        @test_throws ErrorException Scorio.Eval.pass_at_k(rank_shaped, 1)
        @test_throws ErrorException Scorio.Eval.pass_hat_k(rank_shaped, 1)
        @test_throws ErrorException Scorio.Eval.unanimous_at_k(rank_shaped, 1)
        @test_throws ErrorException Scorio.Eval.g_pass_at_k_tau(rank_shaped, 1, 0.5)
        @test_throws ErrorException Scorio.Eval.mg_pass_at_k(rank_shaped, 1)

        v = Int[0, 1, 1, 0, 1]
        @test Scorio.Eval.pass_at_k(v, 1) ≈ 0.6 atol = 1e-12
        @test Scorio.Eval.pass_hat_k(v, 1) ≈ 0.6 atol = 1e-12
        @test Scorio.Eval.bayes(v)[1] ≈ Scorio.Eval.bayes(reshape(v, 1, :))[1] atol = 1e-12

        # Python uses `np.asarray(..., dtype=int)`: rectangular nested
        # sequences are accepted and finite real values truncate toward zero.
        nested = ((0.9, 1.9), (1.2, 0.2))
        @test Scorio.Eval.pass_at_k(nested, 1) == 0.5
        @test Scorio.Eval.auc_at_k(nested, 2) ==
              Scorio.Eval.auc_at_k(Int[0 1; 1 0], 2)
        @test Scorio.Eval.bayes(((0, 1), (1, 0)), (0.0, 1.0), ((1,), (0,))) ==
              Scorio.Eval.bayes(
            Int[0 1; 1 0],
            [0.0, 1.0],
            reshape(Int[1, 0], 2, 1),
        )
        @test Scorio.Eval.avg_ci(((0, 1), (1, 0)), nothing, 0.95, [0.0, 1.0]) ==
              Scorio.Eval.avg_ci(Int[0 1; 1 0], nothing, 0.95, 0:1)
        @test_throws ErrorException Scorio.Eval.pass_at_k(((0, 1), (1,)), 1)
    end
end

@testset "eval/full_python_parity.jl" begin
    R = Int[
        0 1 1 0 1
        1 1 0 1 1
        0 0 1 0 0
    ]
    R_multi = Int[
        0 1 2 2 1
        1 1 0 2 2
        2 0 1 1 0
    ]
    category_weights = [-0.25, 0.5, 1.25]
    R0 = Int[
        0 2
        1 2
        2 0
    ]
    spectrum_weights = [0.2, 0.1, 0.4]

    # Values below were generated by the authoritative Python implementation.
    @testset "cross-language expected values" begin
        @test Scorio.Eval.auc_at_k(R, 3) ≈ 0.7333333333333333 atol = 2e-12
        @test Scorio.Eval.maj_at_k(R, 3) ≈ 0.5666666666666667 atol = 2e-12
        @test Scorio.Eval.max_at_k(R_multi, 2, category_weights) ≈ 0.9249999999999999 atol = 2e-12
        @test Scorio.Eval.threshold_spectrum_at_k(R, 3, spectrum_weights) ≈ 0.2966666666666667 atol = 2e-12
        @test Scorio.Eval.geom_at_k(R, 3) ≈ 0.31622776601683794 atol = 2e-12
        @test Scorio.Eval.geom_ds_at_k(R, 3) ≈ 0.3800584750330461 atol = 2e-12
        @test Scorio.Eval.geo_spectrum_at_k(R, 3) ≈ 0.3103164454170876 atol = 2e-12
        @test Scorio.Eval.geo_spectrum_star_at_k(R, 3) ≈ 0.3103164454170876 atol = 2e-12

        expected_cis = (
            (Scorio.Eval.auc_at_k_ci(R, 3),
             (0.689484126984127, 0.09311042112582836, 0.506991054992146, 0.8719771989761079)),
            (Scorio.Eval.maj_at_k_ci(R, 3),
             (0.5317460317460316, 0.12016074119411065, 0.29623530664993636, 0.7672567568421269)),
            (Scorio.Eval.max_at_k_ci(R_multi, 2, category_weights, R0),
             (0.8454545454545453, 0.10462903317721266, 0.6403854086899621, 1.0505236822191286)),
            (Scorio.Eval.threshold_spectrum_at_k_ci(R, 3, spectrum_weights),
             (0.3079365079365079, 0.06391252374889893, 0.18267026322760513, 0.43320275264541064)),
            (Scorio.Eval.geom_at_k_ci(R, 3),
             (0.4181975966056924, 0.10664873966935018, 0.2091699078571779, 0.6272252853542069)),
            (Scorio.Eval.geom_ds_at_k_ci(R, 3),
             (0.434283654733815, 0.10885372876335309, 0.22093426677475123, 0.6476330426928788)),
            (Scorio.Eval.geo_spectrum_at_k_ci(R, 3),
             (0.35459111924295705, 0.08887869735651185, 0.18039207343135852, 0.5287901650545556)),
            (Scorio.Eval.geo_spectrum_star_at_k_ci(R, 3),
             (0.35459111924295705, 0.08887869735651185, 0.18039207343135852, 0.5287901650545556)),
        )
        for (actual, expected) in expected_cis
            @test all(isapprox.(collect(actual), collect(expected); atol=5e-9, rtol=0.0))
        end
    end

    @testset "aliases and reductions" begin
        @test Scorio.Eval.unanimous_at_k === Scorio.Eval.pass_hat_k
        @test Scorio.Eval.unanimous_at_k_ci === Scorio.Eval.pass_hat_k_ci
        @test Scorio.Eval.auc_at_k(R, 1) ≈ Scorio.Eval.pass_at_k(R, 1) atol = 1e-12
        @test all(isapprox.(
            collect(Scorio.Eval.auc_at_k_ci(R, 1)),
            collect(Scorio.Eval.pass_at_k_ci(R, 1));
            atol=1e-12,
        ))
        @test Scorio.Eval.max_at_k(R, 3) ≈ Scorio.Eval.pass_at_k(R, 3) atol = 1e-12
        @test Scorio.Eval.max_at_k_ci(R_multi, 1, category_weights, R0) ==
              Scorio.Eval.bayes_ci(R_multi, category_weights, R0)

        unanimous_weights = [0.0, 0.0, 1.0]
        mg_weights = [0.0, 0.0, 2.0 / 3.0]
        @test Scorio.Eval.threshold_spectrum_at_k(R, 3, unanimous_weights) ≈
              Scorio.Eval.pass_hat_k(R, 3) atol = 1e-12
        @test Scorio.Eval.threshold_spectrum_at_k(R, 3, mg_weights) ≈
              Scorio.Eval.mg_pass_at_k(R, 3) atol = 1e-12
        @test all(isapprox.(
            collect(Scorio.Eval.threshold_spectrum_at_k_ci(R, 3, unanimous_weights)),
            collect(Scorio.Eval.pass_hat_k_ci(R, 3));
            atol=1e-12,
        ))
        @test all(isapprox.(
            collect(Scorio.Eval.threshold_spectrum_at_k_ci(R, 3, mg_weights)),
            collect(Scorio.Eval.mg_pass_at_k_ci(R, 3));
            atol=1e-12,
        ))
        @test all(isapprox.(
            collect(Scorio.Eval.geom_ds_at_k_ci(R, 3)),
            collect(Scorio.Eval.geo_spectrum_at_k_ci(R, 3, 0.5, unanimous_weights));
            atol=1e-12,
        ))
    end

    @testset "geometric controls and distinct targets" begin
        custom_spectrum = Scorio.Eval.threshold_spectrum_at_k(R, 3, spectrum_weights)
        @test Scorio.Eval.geo_spectrum_at_k(R, 3, 0.0, spectrum_weights) ≈ custom_spectrum atol = 1e-12
        @test Scorio.Eval.geo_spectrum_at_k(R, 3, 1.0, spectrum_weights) ≈
              Scorio.Eval.pass_at_k(R, 3) atol = 1e-12
        @test Scorio.Eval.geo_spectrum_at_k(R, 3, 0.5, spectrum_weights, 0.25) ≈
              Scorio.Eval.geo_spectrum_at_k(R, 3, 0.25, spectrum_weights) atol = 1e-12
        @test_throws ErrorException Scorio.Eval.geo_spectrum_at_k(
            R, 3, 0.25, spectrum_weights, 0.5,
        )

        zero_R = zeros(Int, 3, 5)
        @test Scorio.Eval.geom_at_k(zero_R, 2, -0.5, 0.5) == 0.0
        @test_throws ErrorException Scorio.Eval.geom_at_k(
            Int[1 0 0 0 0], 2, 0.5, -0.5,
        )
        @test_throws ErrorException Scorio.Eval.geom_at_k(R, 2, 0.0, 0.0)

        mixed_R = Int[1 0 0; 1 1 1]
        @test Scorio.Eval.geom_at_k(mixed_R, 2) ≈ 0.5 atol = 1e-12
        @test Scorio.Eval.geom_ds_at_k(mixed_R, 2) ≈ sqrt(5.0 / 12.0) atol = 1e-12
        @test Scorio.Eval.geom_at_k_ci(mixed_R, 2)[1] !=
              Scorio.Eval.geom_ds_at_k_ci(mixed_R, 2)[1]
    end

    @testset "validation and latent-k behavior" begin
        @test_throws ErrorException Scorio.Eval.auc_at_k(R, 0)
        @test_throws ErrorException Scorio.Eval.maj_at_k(R, 0)
        @test_throws ErrorException Scorio.Eval.max_at_k(R, 0)
        @test_throws ErrorException Scorio.Eval.threshold_spectrum_at_k(R, 3, [1.0, 0.0])
        @test_throws ErrorException Scorio.Eval.threshold_spectrum_at_k(R, 3, [0.2, -0.1, 0.3])
        @test_throws ErrorException Scorio.Eval.threshold_spectrum_at_k(R, 3, [0.5, 0.4, 0.3])
        @test_throws ErrorException Scorio.Eval.threshold_spectrum_at_k(R, 3, [0.2, Inf, 0.3])
        @test_throws ErrorException Scorio.Eval.geo_spectrum_at_k(R, 3, 1.1)
        @test_throws ErrorException Scorio.Eval.geom_at_k_ci(R, 3, 0.5, 0.5, 0.95, (0.0, 1.0), 0.0, 1.0)

        # Posterior targets are defined beyond the observed finite-bank size.
        @test_throws ErrorException Scorio.Eval.geom_at_k(R, 6)
        @test all(isfinite, Scorio.Eval.geom_at_k_ci(R, 6))
        @test all(isfinite, Scorio.Eval.max_at_k_ci(R, 6))
    end

    @testset "large finite bank remains stable" begin
        large_R = vcat(ones(Int, 1, 2000), zeros(Int, 1, 2000))
        scores = (
            Scorio.Eval.pass_at_k(large_R, 1000),
            Scorio.Eval.pass_hat_k(large_R, 1000),
            Scorio.Eval.g_pass_at_k_tau(large_R, 1000, 0.5),
            Scorio.Eval.mg_pass_at_k(large_R, 1000),
        )
        @test all(isfinite, scores)
        @test all(isapprox.(collect(scores), fill(0.5, 4); atol=1e-12))
    end

    @testset "complete public eval surface" begin
        expected = Set(Symbol[
            :bayes, :bayes_ci, :avg, :avg_ci,
            :pass_at_k, :pass_at_k_ci, :pass_hat_k, :pass_hat_k_ci,
            :unanimous_at_k, :unanimous_at_k_ci,
            :g_pass_at_k, :g_pass_at_k_ci,
            :g_pass_at_k_tau, :g_pass_at_k_tau_ci,
            :mg_pass_at_k, :mg_pass_at_k_ci,
            :maj_at_k, :maj_at_k_ci,
            :auc_at_k, :auc_at_k_ci,
            :max_at_k, :max_at_k_ci,
            :threshold_spectrum_at_k, :threshold_spectrum_at_k_ci,
            :geom_at_k, :geom_at_k_ci,
            :geom_ds_at_k, :geom_ds_at_k_ci,
            :geo_spectrum_at_k, :geo_spectrum_at_k_ci,
            :geo_spectrum_star_at_k, :geo_spectrum_star_at_k_ci,
        ])
        actual = Set(filter(name -> name != :Eval, names(Scorio.Eval)))
        @test actual == expected
        @test all(name -> isdefined(Scorio, name), expected)
    end
end

@testset "eval/positional_or_keyword_parity.jl" begin
    R = Int[0 1 1; 1 0 1]
    R_multi = Int[0 1 2; 2 1 0]
    category_weights = [-0.25, 0.5, 1.25]
    R0 = Int[0; 2;;]
    spectrum_weights = [0.2, 0.3]
    interval_bounds = (-1.0, 2.0)

    # Each specification lists required positional arguments followed by all
    # Python-optional parameters in their authoritative order.  Every split
    # point is exercised: all-keyword, every mixed positional/keyword prefix,
    # and all-positional.
    specifications = (
        (:bayes, Scorio.Eval.bayes, (R_multi,), (:w, :R0),
         (category_weights, R0)),
        (:bayes_ci, Scorio.Eval.bayes_ci, (R_multi,),
         (:w, :R0, :confidence, :bounds),
         (category_weights, R0, 0.8, interval_bounds)),
        (:avg, Scorio.Eval.avg, (R_multi,), (:w,), (category_weights,)),
        (:avg_ci, Scorio.Eval.avg_ci, (R_multi,),
         (:w, :confidence, :bounds), (category_weights, 0.8, interval_bounds)),
        (:pass_at_k, Scorio.Eval.pass_at_k, (R, 2), (), ()),
        (:pass_at_k_ci, Scorio.Eval.pass_at_k_ci, (R, 2),
         (:confidence, :bounds, :alpha0, :beta0),
         (0.8, interval_bounds, 0.7, 1.4)),
        (:pass_hat_k, Scorio.Eval.pass_hat_k, (R, 2), (), ()),
        (:pass_hat_k_ci, Scorio.Eval.pass_hat_k_ci, (R, 2),
         (:confidence, :bounds, :alpha0, :beta0),
         (0.8, interval_bounds, 0.7, 1.4)),
        (:unanimous_at_k, Scorio.Eval.unanimous_at_k, (R, 2), (), ()),
        (:unanimous_at_k_ci, Scorio.Eval.unanimous_at_k_ci, (R, 2),
         (:confidence, :bounds, :alpha0, :beta0),
         (0.8, interval_bounds, 0.7, 1.4)),
        (:g_pass_at_k, Scorio.Eval.g_pass_at_k, (R, 2), (), ()),
        (:g_pass_at_k_ci, Scorio.Eval.g_pass_at_k_ci, (R, 2),
         (:confidence, :bounds, :alpha0, :beta0),
         (0.8, interval_bounds, 0.7, 1.4)),
        (:g_pass_at_k_tau, Scorio.Eval.g_pass_at_k_tau, (R, 2, 0.6), (), ()),
        (:g_pass_at_k_tau_ci, Scorio.Eval.g_pass_at_k_tau_ci, (R, 2, 0.6),
         (:confidence, :bounds, :alpha0, :beta0),
         (0.8, interval_bounds, 0.7, 1.4)),
        (:mg_pass_at_k, Scorio.Eval.mg_pass_at_k, (R, 2), (), ()),
        (:mg_pass_at_k_ci, Scorio.Eval.mg_pass_at_k_ci, (R, 2),
         (:confidence, :bounds, :alpha0, :beta0),
         (0.8, interval_bounds, 0.7, 1.4)),
        (:maj_at_k, Scorio.Eval.maj_at_k, (R, 2), (), ()),
        (:maj_at_k_ci, Scorio.Eval.maj_at_k_ci, (R, 2),
         (:confidence, :bounds, :alpha0, :beta0),
         (0.8, interval_bounds, 0.7, 1.4)),
        (:auc_at_k, Scorio.Eval.auc_at_k, (R, 2), (), ()),
        (:auc_at_k_ci, Scorio.Eval.auc_at_k_ci, (R, 2),
         (:confidence, :bounds, :alpha0, :beta0),
         (0.8, interval_bounds, 0.7, 1.4)),
        (:max_at_k, Scorio.Eval.max_at_k, (R_multi, 2), (:w,),
         (category_weights,)),
        (:max_at_k_ci, Scorio.Eval.max_at_k_ci, (R_multi, 2),
         (:w, :R0, :confidence, :bounds),
         (category_weights, R0, 0.8, interval_bounds)),
        (:threshold_spectrum_at_k, Scorio.Eval.threshold_spectrum_at_k,
         (R, 2, spectrum_weights), (), ()),
        (:threshold_spectrum_at_k_ci, Scorio.Eval.threshold_spectrum_at_k_ci,
         (R, 2, spectrum_weights), (:confidence, :bounds, :alpha0, :beta0),
         (0.8, interval_bounds, 0.7, 1.4)),
        (:geom_at_k, Scorio.Eval.geom_at_k, (R, 2),
         (:pass_power, :unanimous_power), (0.3, 0.7)),
        (:geom_at_k_ci, Scorio.Eval.geom_at_k_ci, (R, 2),
         (:pass_power, :unanimous_power, :confidence, :bounds, :alpha0, :beta0),
         (0.3, 0.7, 0.8, interval_bounds, 0.7, 1.4)),
        (:geom_ds_at_k, Scorio.Eval.geom_ds_at_k, (R, 2),
         (:pass_power, :unanimous_power), (0.3, 0.7)),
        (:geom_ds_at_k_ci, Scorio.Eval.geom_ds_at_k_ci, (R, 2),
         (:pass_power, :unanimous_power, :confidence, :bounds, :alpha0, :beta0),
         (0.3, 0.7, 0.8, interval_bounds, 0.7, 1.4)),
        (:geo_spectrum_at_k, Scorio.Eval.geo_spectrum_at_k, (R, 2),
         (:lam, :weights, :lambda_), (0.5, spectrum_weights, 0.3)),
        (:geo_spectrum_at_k_ci, Scorio.Eval.geo_spectrum_at_k_ci, (R, 2),
         (:lam, :weights, :lambda_, :confidence, :bounds, :alpha0, :beta0),
         (0.5, spectrum_weights, 0.3, 0.8, interval_bounds, 0.7, 1.4)),
        (:geo_spectrum_star_at_k, Scorio.Eval.geo_spectrum_star_at_k, (R, 2),
         (), ()),
        (:geo_spectrum_star_at_k_ci, Scorio.Eval.geo_spectrum_star_at_k_ci,
         (R, 2), (:confidence, :bounds, :alpha0, :beta0),
         (0.8, interval_bounds, 0.7, 1.4)),
    )

    public_eval_names = filter(name -> name != :Eval, names(Scorio.Eval))
    @test Set(first.(specifications)) == Set(public_eval_names)
    for (name, fn, required, optional_names, optional_values) in specifications
        expected = fn(required..., optional_values...)
        for prefix_length in 0:length(optional_names)
            positional_values = optional_values[1:prefix_length]
            remaining_names = Tuple(optional_names[(prefix_length + 1):end])
            remaining_values = Tuple(optional_values[(prefix_length + 1):end])
            keywords = NamedTuple{remaining_names}(remaining_values)
            actual = fn(required..., positional_values...; keywords...)
            @test actual == expected
        end
    end
end

# ── Simulation-data-based tests (matching Python test_eval_apis.py) ──

if isdefined(@__MODULE__, :top_p_task_aime25)
@testset "eval/apis_simulation.jl" begin
    data = top_p_task_aime25()

    # Fixtures matching Python conftest
    # Python: binary_ref = top_p_task_aime25[0, :12, :20]
    binary_ref = data[1, 1:12, 1:20]
    # Python: multiclass: R = (data[1,:12,:20] + data[2,:12,:20])
    mc_R = Int.(data[2, 1:12, 1:20] .+ data[3, 1:12, 1:20])
    mc_w = [0.0, 0.5, 1.0]
    mc_R0 = Int.(data[4, 1:12, 1:6] .+ data[5, 1:12, 1:6])
    # Python: top_p_model_slice = top_p_task_aime25[0, :10, :12]
    top_p_model_slice = data[1, 1:10, 1:12]

    # Reference implementation for Bayes
    function _bayes_reference(Rm::AbstractMatrix{<:Integer}, wv::Vector{Float64},
                              R0m::Union{Nothing, AbstractMatrix{<:Integer}}=nothing)
        M, N = size(Rm)
        C = length(wv) - 1
        if isnothing(R0m)
            R0m_use = zeros(Int, M, 0)
        else
            R0m_use = R0m
        end
        D = size(R0m_use, 2)
        T = Float64(1 + C + D + N)
        delta_w = wv .- wv[1]

        mu_rows = zeros(Float64, M)
        var_rows = zeros(Float64, M)

        for row in 1:M
            nu = ones(Float64, C + 1)
            for val in Rm[row, :]
                nu[Int(val) + 1] += 1.0
            end
            for val in R0m_use[row, :]
                nu[Int(val) + 1] += 1.0
            end

            row_mean = sum(nu ./ T .* delta_w)
            second_moment = sum(nu ./ T .* (delta_w .^ 2))
            mu_rows[row] = row_mean
            var_rows[row] = max(0.0, second_moment - row_mean^2)
        end

        mu = wv[1] + sum(mu_rows) / M
        sigma = sqrt(sum(var_rows) / (M^2 * (T + 1.0)))
        return mu, sigma
    end

    # Reference for pass@k
    function _pass_at_k_reference(Rm::AbstractMatrix{<:Integer}, k::Int)
        M, N = size(Rm)
        denom = Float64(binomial(N, k))
        values = zeros(Float64, M)
        for row in 1:M
            nu = sum(Rm[row, :])
            values[row] = 1.0 - Float64(binomial(N - nu, k)) / denom
        end
        return sum(values) / M
    end

    function _pass_hat_k_reference(Rm::AbstractMatrix{<:Integer}, k::Int)
        M, N = size(Rm)
        denom = Float64(binomial(N, k))
        values = zeros(Float64, M)
        for row in 1:M
            nu = sum(Rm[row, :])
            values[row] = Float64(binomial(nu, k)) / denom
        end
        return sum(values) / M
    end

    function _g_pass_at_k_tau_reference(Rm::AbstractMatrix{<:Integer}, k::Int, tau::Float64)
        if tau <= 0.0
            return _pass_at_k_reference(Rm, k)
        end
        M, N = size(Rm)
        denom = Float64(binomial(N, k))
        j0 = Int(ceil(tau * k))
        values = zeros(Float64, M)
        for row in 1:M
            nu = sum(Rm[row, :])
            total = 0.0
            for j in j0:k
                total += Float64(binomial(nu, j) * binomial(N - nu, k - j)) / denom
            end
            values[row] = total
        end
        return sum(values) / M
    end

    function _mg_pass_at_k_reference(Rm::AbstractMatrix{<:Integer}, k::Int)
        M, N = size(Rm)
        denom = Float64(binomial(N, k))
        majority = Int(ceil(0.5 * k))
        if majority >= k
            return 0.0
        end
        values = zeros(Float64, M)
        for row in 1:M
            nu = sum(Rm[row, :])
            total = 0.0
            for j in (majority + 1):k
                total += (j - majority) * Float64(binomial(nu, j) * binomial(N - nu, k - j)) / denom
            end
            values[row] = (2.0 / k) * total
        end
        return sum(values) / M
    end

    @testset "Bayes multiclass matches reference" begin
        mu_prior, sigma_prior = Scorio.Eval.bayes(mc_R, mc_w, mc_R0)
        exp_mu_prior, exp_sigma_prior = _bayes_reference(mc_R, mc_w, mc_R0)
        @test mu_prior ≈ exp_mu_prior atol = 1e-10
        @test sigma_prior ≈ exp_sigma_prior atol = 1e-10

        mu_noprior, sigma_noprior = Scorio.Eval.bayes(mc_R, mc_w)
        exp_mu_noprior, exp_sigma_noprior = _bayes_reference(mc_R, mc_w)
        @test mu_noprior ≈ exp_mu_noprior atol = 1e-10
        @test sigma_noprior ≈ exp_sigma_noprior atol = 1e-10
    end

    @testset "Bayes binary default equals explicit" begin
        mu_auto, sigma_auto = Scorio.Eval.bayes(binary_ref)
        mu_explicit, sigma_explicit = Scorio.Eval.bayes(binary_ref, [0.0, 1.0])
        @test mu_auto ≈ mu_explicit atol = 1e-12
        @test sigma_auto ≈ sigma_explicit atol = 1e-12
    end

    @testset "Bayes requires weights for multiclass" begin
        @test_throws ErrorException Scorio.Eval.bayes(mc_R)
    end

    @testset "Bayes validates R0 row count" begin
        bad_R0 = zeros(Int, size(mc_R, 1) + 1, 2)
        @test_throws ErrorException Scorio.Eval.bayes(mc_R, mc_w, bad_R0)
    end

    @testset "Bayes CI matches normal interval" begin
        confidence = 0.9
        bounds = (0.0, 1.0)
        mu, sigma, lo, hi = Scorio.Eval.bayes_ci(binary_ref, nothing, nothing, confidence, bounds)
        z = Scorio._z_value(confidence; two_sided=true)
        expected_lo = max(mu - z * sigma, bounds[1])
        expected_hi = min(mu + z * sigma, bounds[2])
        @test lo ≈ expected_lo atol = 1e-12
        @test hi ≈ expected_hi atol = 1e-12
        @test lo <= mu <= hi
    end

    @testset "Avg binary and weighted match manual formulas" begin
        a_binary, sigma_binary = Scorio.Eval.avg(binary_ref)
        @test a_binary ≈ sum(binary_ref) / length(binary_ref) atol = 1e-12
        @test sigma_binary >= 0.0

        a_weighted, sigma_weighted = Scorio.Eval.avg(mc_R, mc_w)
        expected_a = sum(mc_w[mc_R[i] + 1] for i in eachindex(mc_R)) / length(mc_R)
        @test a_weighted ≈ expected_a atol = 1e-12
        @test sigma_weighted >= 0.0
    end

    @testset "Avg requires binary when weights omitted" begin
        @test_throws ErrorException Scorio.Eval.avg(mc_R)
    end

    @testset "Avg CI matches normal interval" begin
        confidence = 0.8
        bounds = (0.0, 1.0)
        a, sigma, lo, hi = Scorio.Eval.avg_ci(binary_ref, nothing, confidence, bounds)
        z = Scorio._z_value(confidence; two_sided=true)
        expected_lo = max(a - z * sigma, bounds[1])
        expected_hi = min(a + z * sigma, bounds[2])
        @test lo ≈ expected_lo atol = 1e-12
        @test hi ≈ expected_hi atol = 1e-12
        @test lo <= a <= hi
    end

    @testset "Pass point metrics match references" begin
        k = 3
        @test Scorio.Eval.pass_at_k(binary_ref, k) ≈ _pass_at_k_reference(binary_ref, k) atol = 1e-10
        @test Scorio.Eval.pass_hat_k(binary_ref, k) ≈ _pass_hat_k_reference(binary_ref, k) atol = 1e-10
        @test Scorio.Eval.g_pass_at_k(binary_ref, k) ≈ _pass_hat_k_reference(binary_ref, k) atol = 1e-10
        @test Scorio.Eval.g_pass_at_k_tau(binary_ref, k, 0.7) ≈ _g_pass_at_k_tau_reference(binary_ref, k, 0.7) atol = 1e-10
        @test Scorio.Eval.mg_pass_at_k(binary_ref, k) ≈ _mg_pass_at_k_reference(binary_ref, k) atol = 1e-10
    end

    @testset "Pass family monotonicity and bounds" begin
        N = size(binary_ref, 2)
        k_values = collect(1:min(N, 8))

        pass_vals = [Scorio.Eval.pass_at_k(binary_ref, k) for k in k_values]
        pass_hat_vals = [Scorio.Eval.pass_hat_k(binary_ref, k) for k in k_values]

        for idx in 2:length(k_values)
            @test pass_vals[idx] >= pass_vals[idx - 1] ||
                  isapprox(pass_vals[idx], pass_vals[idx - 1]; atol=1e-14, rtol=0.0)
            @test pass_hat_vals[idx] <= pass_hat_vals[idx - 1] ||
                  isapprox(pass_hat_vals[idx], pass_hat_vals[idx - 1]; atol=1e-14, rtol=0.0)
        end

        for (p, ph) in zip(pass_vals, pass_hat_vals)
            @test p >= ph || isapprox(p, ph; atol=1e-14, rtol=0.0)
            @test 0.0 <= ph <= 1.0
            @test 0.0 <= p <= 1.0
        end
    end

    @testset "Pass aliases and tau edge equivalences" begin
        k = 3
        @test Scorio.Eval.g_pass_at_k(binary_ref, k) ≈ Scorio.Eval.pass_hat_k(binary_ref, k) atol = 1e-12
        @test Scorio.Eval.g_pass_at_k_tau(binary_ref, k, 0.0) ≈ Scorio.Eval.pass_at_k(binary_ref, k) atol = 1e-12
        @test Scorio.Eval.g_pass_at_k_tau(binary_ref, k, 1.0) ≈ Scorio.Eval.pass_hat_k(binary_ref, k) atol = 1e-12

        c1 = Scorio.Eval.g_pass_at_k_ci(binary_ref, k)
        c2 = Scorio.Eval.pass_hat_k_ci(binary_ref, k)
        @test all(isapprox.(collect(c1), collect(c2); atol=1e-12))

        c3 = Scorio.Eval.g_pass_at_k_tau_ci(binary_ref, k, 0.0)
        c4 = Scorio.Eval.pass_at_k_ci(binary_ref, k)
        @test all(isapprox.(collect(c3), collect(c4); atol=1e-12))

        c5 = Scorio.Eval.g_pass_at_k_tau_ci(binary_ref, k, 1.0)
        c6 = Scorio.Eval.pass_hat_k_ci(binary_ref, k)
        @test all(isapprox.(collect(c5), collect(c6); atol=1e-12))
    end

    @testset "mg_pass_at_k k=1 edge case" begin
        @test Scorio.Eval.mg_pass_at_k(binary_ref, 1) ≈ 0.0 atol = 1e-12
        @test Scorio.Eval.mg_pass_at_k_ci(binary_ref, 1) == (0.0, 0.0, 0.0, 0.0)
    end

    @testset "Pass family invalid k raises" begin
        for fn in (
            Scorio.Eval.pass_at_k,
            Scorio.Eval.pass_hat_k,
            Scorio.Eval.g_pass_at_k,
            Scorio.Eval.mg_pass_at_k,
            Scorio.Eval.pass_at_k_ci,
            Scorio.Eval.pass_hat_k_ci,
            Scorio.Eval.g_pass_at_k_ci,
            Scorio.Eval.mg_pass_at_k_ci,
        )
            @test_throws ErrorException fn(binary_ref, 0)
        end
    end

    @testset "g_pass tau invalid tau raises" begin
        @test_throws ErrorException Scorio.Eval.g_pass_at_k_tau(binary_ref, 2, 1.1)
        @test_throws ErrorException Scorio.Eval.g_pass_at_k_tau_ci(binary_ref, 2, 1.1)
    end

    @testset "Pass family rejects non-binary" begin
        R_bad = copy(binary_ref)
        R_bad[1, 1] = 2
        @test_throws ErrorException Scorio.Eval.pass_at_k(R_bad, 1)
    end

    @testset "Eval APIs invariant to question/trial permutations" begin
        R = top_p_model_slice
        R_perm = R[end:-1:1, end:-1:1]

        @test Scorio.Eval.avg(R)[1] ≈ Scorio.Eval.avg(R_perm)[1] atol = 1e-12
        @test Scorio.Eval.bayes(R)[1] ≈ Scorio.Eval.bayes(R_perm)[1] atol = 1e-12
        @test Scorio.Eval.pass_at_k(R, 3) ≈ Scorio.Eval.pass_at_k(R_perm, 3) atol = 1e-12
        @test Scorio.Eval.pass_hat_k(R, 3) ≈ Scorio.Eval.pass_hat_k(R_perm, 3) atol = 1e-12
        @test Scorio.Eval.g_pass_at_k_tau(R, 3, 0.7) ≈ Scorio.Eval.g_pass_at_k_tau(R_perm, 3, 0.7) atol = 1e-12
        @test Scorio.Eval.mg_pass_at_k(R, 3) ≈ Scorio.Eval.mg_pass_at_k(R_perm, 3) atol = 1e-12
    end

    @testset "Eval APIs on simulation dataset slice" begin
        R = top_p_model_slice

        a, a_sigma = Scorio.Eval.avg(R)
        @test 0.0 <= a <= 1.0
        @test a_sigma >= 0.0
        @test Scorio.Eval.pass_at_k(R, 1) ≈ a atol = 1e-12

        b_mu, b_sigma = Scorio.Eval.bayes(R)
        @test 0.0 <= b_mu <= 1.0
        @test b_sigma >= 0.0

        p1 = Scorio.Eval.pass_at_k(R, 3)
        ph = Scorio.Eval.pass_hat_k(R, 3)
        gt = Scorio.Eval.g_pass_at_k_tau(R, 3, 0.7)
        mg = Scorio.Eval.mg_pass_at_k(R, 3)
        @test (p1 >= gt || isapprox(p1, gt; atol=1e-14, rtol=0.0)) &&
              (gt >= ph || isapprox(gt, ph; atol=1e-14, rtol=0.0))
        @test 0.0 <= mg <= 1.0

        ci_outputs = [
            Scorio.Eval.bayes_ci(R),
            Scorio.Eval.avg_ci(R),
            Scorio.Eval.pass_at_k_ci(R, 3),
            Scorio.Eval.pass_hat_k_ci(R, 3),
            Scorio.Eval.g_pass_at_k_ci(R, 3),
            Scorio.Eval.g_pass_at_k_tau_ci(R, 3, 0.7),
            Scorio.Eval.mg_pass_at_k_ci(R, 3),
        ]
        for (mu, sigma, lo, hi) in ci_outputs
            @test isfinite(mu)
            @test isfinite(sigma)
            @test isfinite(lo)
            @test isfinite(hi)
            @test sigma >= 0.0
            @test lo <= hi
            @test lo <= mu <= hi
        end
    end

    @testset "Public eval API exports smoke calls" begin
        api_calls = Dict{String,Function}(
            "bayes" => () -> Scorio.Eval.bayes(binary_ref),
            "bayes_ci" => () -> Scorio.Eval.bayes_ci(binary_ref),
            "avg" => () -> Scorio.Eval.avg(binary_ref),
            "avg_ci" => () -> Scorio.Eval.avg_ci(binary_ref),
            "pass_at_k" => () -> Scorio.Eval.pass_at_k(binary_ref, 2),
            "pass_hat_k" => () -> Scorio.Eval.pass_hat_k(binary_ref, 2),
            "g_pass_at_k" => () -> Scorio.Eval.g_pass_at_k(binary_ref, 2),
            "g_pass_at_k_tau" => () -> Scorio.Eval.g_pass_at_k_tau(binary_ref, 2, 0.7),
            "mg_pass_at_k" => () -> Scorio.Eval.mg_pass_at_k(binary_ref, 2),
            "pass_at_k_ci" => () -> Scorio.Eval.pass_at_k_ci(binary_ref, 2),
            "pass_hat_k_ci" => () -> Scorio.Eval.pass_hat_k_ci(binary_ref, 2),
            "g_pass_at_k_ci" => () -> Scorio.Eval.g_pass_at_k_ci(binary_ref, 2),
            "g_pass_at_k_tau_ci" => () -> Scorio.Eval.g_pass_at_k_tau_ci(binary_ref, 2, 0.7),
            "mg_pass_at_k_ci" => () -> Scorio.Eval.mg_pass_at_k_ci(binary_ref, 2),
        )

        expected_names = Set([
            "bayes", "bayes_ci", "avg", "avg_ci",
            "pass_at_k", "pass_hat_k", "g_pass_at_k", "g_pass_at_k_tau", "mg_pass_at_k",
            "pass_at_k_ci", "pass_hat_k_ci", "g_pass_at_k_ci", "g_pass_at_k_tau_ci", "mg_pass_at_k_ci",
        ])
        @test Set(keys(api_calls)) == expected_names

        for (name, fn) in api_calls
            out = fn()
            if endswith(name, "_ci")
                mu, sigma, lo, hi = out
                @test isfinite(mu)
                @test isfinite(sigma)
                @test isfinite(lo)
                @test isfinite(hi)
                @test sigma >= 0.0
                @test lo <= hi
                @test lo <= mu <= hi
                continue
            end

            if name in ("bayes", "avg")
                mu, sigma = out
                @test isfinite(mu)
                @test isfinite(sigma)
                @test sigma >= 0.0
                continue
            end

            @test isfinite(Float64(out))
            @test 0.0 <= Float64(out) <= 1.0
        end
    end
end
end
