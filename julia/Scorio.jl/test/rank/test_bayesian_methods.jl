using Test
using Scorio

@testset "rank/test_bayesian_methods.jl" begin
    R_small = ordered_binary_small_R()
    R_equal = equal_information_R()

    @testset "NumPy 2.4 default_rng golden vectors" begin
        scalar_state = UInt32[
            0xcd540ab7,
            0x9f1e2e6d,
            0x79fb94b6,
            0xd57873dc,
            0x64d420b7,
            0x7d282a1b,
            0x4692d5ff,
            0x33657971,
        ]
        sequence_state = UInt32[
            0xe3d1fc80,
            0xb45f843f,
            0x203807c7,
            0x42c16e36,
            0x0a344445,
            0x10f7f367,
            0xf4930f8c,
            0x988d461a,
        ]
        @test Scorio._numpy_seed_sequence_state(42, 8) == scalar_state
        @test Scorio._numpy_seed_sequence_state((1, 2, 3), 8) == sequence_state

        scalar_raw = UInt64[
            0xc621fbcd16d92688,
            0x705a5661a791ffc1,
            0xdbcd12c26eda1624,
            0xb286b60e1600888d,
            0x181c01b5339381eb,
            0xf9c262ed86c7538c,
            0xc2da0d2fbc5a4471,
            0xc93b82a3b7ac9740,
        ]
        sequence_raw = UInt64[
            0xaba411f8f6c9b990,
            0x1c6b7489df5024fb,
            0x82f2425c7e3229b3,
            0x5b2e60d5f3e5e266,
            0xb464d3e4360bf2f7,
            0x7ac1778cdd5f7170,
            0xb610e711f02badcf,
            0x617fa9a8e9b98e46,
        ]
        scalar_rng = Scorio._NumpyRNG(42)
        sequence_rng = Scorio._NumpyRNG((1, 2, 3))
        @test [Scorio._numpy_next_u64!(scalar_rng) for _ in 1:8] == scalar_raw
        @test [Scorio._numpy_next_u64!(sequence_rng) for _ in 1:8] == sequence_raw

        scalar_uniform = [
            0.7739560485559633,
            0.4388784397520523,
            0.8585979199113825,
            0.6973680290593639,
            0.09417734788764953,
        ]
        sequence_uniform = [
            0.6704722626516632,
            0.1110146366693816,
            0.5115090823948739,
            0.35617642615752443,
            0.7046635086208395,
        ]
        scalar_rng = Scorio._NumpyRNG(42)
        sequence_rng = Scorio._NumpyRNG((1, 2, 3))
        @test [Scorio._numpy_uniform!(scalar_rng) for _ in 1:5] == scalar_uniform
        @test [Scorio._numpy_uniform!(sequence_rng) for _ in 1:5] == sequence_uniform

        scalar_normal = [
            0.30471707975443135,
            -1.0399841062404955,
            0.7504511958064572,
            0.9405647163912139,
            -1.9510351886538364,
            -1.302179506862318,
            0.12784040316728537,
            -0.3162425923435822,
        ]
        sequence_normal = [
            -0.5986439813239683,
            2.7956942860102307,
            -0.17596600347685296,
            1.156005899397638,
            1.8700956817808088,
            -1.1938308236674509,
            -1.5016768187533203,
            0.05339828625618054,
        ]
        scalar_rng = Scorio._NumpyRNG(42)
        sequence_rng = Scorio._NumpyRNG((1, 2, 3))
        @test [Scorio._numpy_standard_normal!(scalar_rng) for _ in 1:8] ==
              scalar_normal
        @test [Scorio._numpy_standard_normal!(sequence_rng) for _ in 1:8] ==
              sequence_normal

        shapes = [0.2, 0.7, 1.0, 1.2, 2.5, 10.0]
        scalar_gamma = [
            0.2777035894591679,
            1.6496565623018467,
            0.08643739969837702,
            0.13178118659136595,
            1.7337108387343565,
            7.249612071848176,
        ]
        sequence_gamma = [
            0.1354890130626346,
            0.4277128115747893,
            0.8973426383557355,
            0.16265581084811392,
            2.2462211058942123,
            5.678532038911731,
        ]
        scalar_rng = Scorio._NumpyRNG(42)
        sequence_rng = Scorio._NumpyRNG((1, 2, 3))
        @test [Scorio._numpy_standard_gamma!(scalar_rng, x) for x in shapes] ==
              scalar_gamma
        @test [Scorio._numpy_standard_gamma!(sequence_rng, x) for x in shapes] ==
              sequence_gamma

        alphas = [0.2, 0.5, 1.0, 2.0, 5.0]
        betas = [0.7, 0.5, 3.0, 1.0, 2.0]
        scalar_beta = [
            0.47384252215869477,
            0.009232118044248207,
            0.3924379525501958,
            0.4204757000005991,
            0.6721979031278014,
        ]
        sequence_beta = [
            0.7579124520923216,
            0.6734604747295978,
            0.4375192607824704,
            0.8542951958955473,
            0.8088039870731756,
        ]
        scalar_rng = Scorio._NumpyRNG(42)
        sequence_rng = Scorio._NumpyRNG((1, 2, 3))
        @test [
            Scorio._numpy_beta!(scalar_rng, alpha, beta) for
            (alpha, beta) in zip(alphas, betas)
        ] == scalar_beta
        @test [
            Scorio._numpy_beta!(sequence_rng, alpha, beta) for
            (alpha, beta) in zip(alphas, betas)
        ] == sequence_beta

        # These later draws exercise platform-libm rounding paths that differ
        # by one or two ulps from Julia's default libopenlibm implementation.
        scalar_rng = Scorio._NumpyRNG(42)
        subunit_gamma = [
            Scorio._numpy_standard_gamma!(scalar_rng, 0.7) for _ in 1:26
        ]
        @test reinterpret(UInt64, subunit_gamma[26]) == 0x3febc245095f3fc8

        scalar_rng = Scorio._NumpyRNG(42)
        mixed_beta = [Scorio._numpy_beta!(scalar_rng, 0.7, 2.4) for _ in 1:65]
        @test reinterpret(UInt64, mixed_beta[65]) == 0x3fa3738b69a4ba36

        scalar_rng = Scorio._NumpyRNG(42)
        johnk_beta = [Scorio._numpy_beta!(scalar_rng, 0.7, 0.8) for _ in 1:89]
        @test reinterpret(UInt64, johnk_beta[89]) == 0x3feb35465b9c9c4c
    end

    @testset "Python-generated public Bayesian goldens" begin
        R_golden = [1 1 0 1 1; 0 1 0 0 1; 0 0 1 0 1]

        scalar_ts = Scorio.Rank.thompson(
            R_golden;
            n_samples=100,
            seed=42,
            return_scores=true,
        )
        sequence_ts = Scorio.Rank.thompson(
            R_golden;
            n_samples=100,
            seed=(1, 2, 3),
            return_scores=true,
        )
        @test scalar_ts == ([1, 2, 2], [-1.2, -2.4000000000000004, -2.4000000000000004])
        @test sequence_ts == ([1, 2, 2], [-1.31, -2.3449999999999998, -2.3449999999999998])

        scalar_mcmc = Scorio.Rank.bayesian_mcmc(
            R_golden;
            n_samples=100,
            burnin=20,
            seed=42,
            return_scores=true,
        )
        sequence_mcmc = Scorio.Rank.bayesian_mcmc(
            R_golden;
            n_samples=100,
            burnin=20,
            seed=(1, 2, 3),
            return_scores=true,
        )
        @test scalar_mcmc == (
            [2, 3, 1],
            [-0.00852987511517541, -0.23245954519611078, 0.3964082129584214],
        )
        @test sequence_mcmc == (
            [2, 1, 3],
            [0.1589370813208104, 0.41673292006604107, -0.3776367532766657],
        )
    end

    @testset "thompson seed determinism" begin
        out1 = Scorio.Rank.thompson(R_small; n_samples=1500, seed=11, return_scores=true)
        out2 = Scorio.Rank.thompson(R_small; n_samples=1500, seed=11, return_scores=true)

        ranking1, scores1 = assert_ranking_and_scores(out1)
        ranking2, scores2 = assert_ranking_and_scores(out2)

        @test scores1 ≈ scores2
        @test ranking1 ≈ ranking2

        sequence1 = Scorio.Rank.thompson(
            R_small;
            n_samples=100,
            seed=(1, 2, 3),
            return_scores=true,
        )
        sequence2 = Scorio.Rank.thompson(
            R_small;
            n_samples=100,
            seed=[1, 2, 3],
            return_scores=true,
        )
        @test sequence1 == sequence2
    end

    @testset "bayesian_mcmc seed determinism" begin
        out1 = Scorio.Rank.bayesian_mcmc(
            R_small;
            n_samples=800,
            burnin=200,
            seed=13,
            return_scores=true,
        )
        out2 = Scorio.Rank.bayesian_mcmc(
            R_small;
            n_samples=800,
            burnin=200,
            seed=13,
            return_scores=true,
        )

        ranking1, scores1 = assert_ranking_and_scores(out1)
        ranking2, scores2 = assert_ranking_and_scores(out2)

        @test scores1 ≈ scores2
        @test ranking1 ≈ ranking2
    end

    @testset "equal-information behavior" begin
        ranking_ts, scores_ts = Scorio.Rank.thompson(
            R_equal;
            n_samples=3000,
            seed=19,
            return_scores=true,
        )
        @test all(isapprox.(scores_ts, fill(first(scores_ts), length(scores_ts))))
        @test all(isapprox.(ranking_ts, fill(first(ranking_ts), length(ranking_ts))))

        ranking_mcmc, scores_mcmc = Scorio.Rank.bayesian_mcmc(
            R_equal;
            n_samples=700,
            burnin=100,
            seed=19,
            return_scores=true,
        )
        @test all(isapprox.(scores_mcmc, fill(first(scores_mcmc), length(scores_mcmc))))
        @test all(isapprox.(ranking_mcmc, fill(first(ranking_mcmc), length(ranking_mcmc))))
    end

    @testset "equal posterior subgroups tie exactly" begin
        R_thompson = zeros(Int, 3, 1, 4)
        R_thompson[:, 1, :] = [1 1 1 0; 1 0 1 1; 0 0 0 0]
        ranking_ts, scores_ts = Scorio.Rank.thompson(
            R_thompson;
            n_samples=1200,
            seed=31,
            return_scores=true,
        )
        @test ranking_ts[1] == ranking_ts[2]
        @test scores_ts[1] == scores_ts[2]

        R_mcmc = zeros(Int, 3, 1, 4)
        R_mcmc[:, 1, :] = [1 1 0 1; 1 1 0 1; 0 0 0 0]
        ranking_mcmc, scores_mcmc = Scorio.Rank.bayesian_mcmc(
            R_mcmc;
            n_samples=900,
            burnin=200,
            seed=37,
            return_scores=true,
        )
        @test ranking_mcmc[1] == ranking_mcmc[2]
        @test scores_mcmc[1] == scores_mcmc[2]
    end

    @testset "validation errors" begin
        @test_throws ErrorException Scorio.Rank.thompson(R_small; n_samples=0)
        @test_throws ErrorException Scorio.Rank.thompson(R_small; prior_alpha=0.0)
        @test_throws ErrorException Scorio.Rank.thompson(R_small; prior_beta=0.0)
        @test_throws ErrorException Scorio.Rank.thompson(R_small; seed=-1)
        @test_throws ErrorException Scorio.Rank.thompson(R_small; seed="bad")

        @test_throws ErrorException Scorio.Rank.bayesian_mcmc(R_small; n_samples=0)
        @test_throws ErrorException Scorio.Rank.bayesian_mcmc(R_small; burnin=-1)
        @test_throws ErrorException Scorio.Rank.bayesian_mcmc(R_small; prior_var=0.0)
        @test_throws ErrorException Scorio.Rank.bayesian_mcmc(R_small; seed=-1)
        @test_throws ErrorException Scorio.Rank.bayesian_mcmc(R_small; seed="bad")
    end


    @testset "nondeterministic seed accepted" begin
        assert_ranking_and_scores(
            Scorio.Rank.thompson(
                R_small;
                n_samples=10,
                seed=nothing,
                return_scores=true,
            ),
        )
        assert_ranking_and_scores(
            Scorio.Rank.bayesian_mcmc(
                R_small;
                n_samples=10,
                burnin=2,
                seed=nothing,
                return_scores=true,
            ),
        )
    end
end
