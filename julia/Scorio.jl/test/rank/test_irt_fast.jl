using Test
using Scorio

@testset "rank/test_irt_fast.jl" begin
    R_small = ordered_binary_small_R()

    @testset "IRT fast smoke and ordering" begin
        calls = [
            () -> Scorio.Rank.rasch(R_small; max_iter=80, return_scores=true),
            () -> Scorio.Rank.rasch_map(R_small; prior=1.0, max_iter=80, return_scores=true),
            () -> Scorio.Rank.rasch_2pl(
                R_small;
                max_iter=80,
                reg_discrimination=0.01,
                return_scores=true,
            ),
            () -> Scorio.Rank.rasch_2pl_map(
                R_small;
                prior=1.0,
                max_iter=80,
                reg_discrimination=0.01,
                return_scores=true,
            ),
            () -> Scorio.Rank.rasch_3pl(
                R_small;
                max_iter=60,
                fix_guessing=0.2,
                reg_discrimination=0.01,
                reg_guessing=0.1,
                return_scores=true,
            ),
            () -> Scorio.Rank.rasch_3pl_map(
                R_small;
                prior=1.0,
                max_iter=60,
                fix_guessing=0.2,
                reg_discrimination=0.01,
                reg_guessing=0.1,
                return_scores=true,
            ),
            () -> Scorio.Rank.rasch_mml(
                R_small;
                max_iter=12,
                em_iter=8,
                n_quadrature=9,
                return_scores=true,
            ),
            () -> Scorio.Rank.rasch_mml_credible(
                R_small;
                quantile=0.1,
                max_iter=12,
                em_iter=8,
                n_quadrature=9,
                return_scores=true,
            ),
            () -> Scorio.Rank.dynamic_irt(
                R_small;
                variant="linear",
                max_iter=80,
                return_scores=true,
            ),
            () -> Scorio.Rank.mirt(
                R_small;
                n_factors=2,
                n_quadrature=7,
                em_iter=25,
                max_iter=40,
                tol=1e-3,
                return_scores=true,
            ),
        ]

        for run in calls
            ranking, _ = assert_ranking_and_scores(run())
            assert_ordering_sanity(ranking; best_idx=1, worst_idx=4)
        end
    end

    @testset "mirt multidimensional" begin
        # 3PL with fixed and estimated guessing: smoke + valid ranking/scores.
        for opts in (
            (; model="3pl", fix_guessing=0.2),
            (; model="3pl"),
            (; n_factors=1, n_quadrature=11),
        )
            out = Scorio.Rank.mirt(
                R_small;
                n_quadrature=7,
                em_iter=25,
                max_iter=40,
                tol=1e-3,
                return_scores=true,
                opts...,
            )
            assert_ranking_and_scores(out)
        end

        # Item parameters: shapes, keys, and non-negativity.
        L, M, _ = size(R_small)
        ranking, scores, params = Scorio.Rank.mirt(
            R_small;
            n_factors=2,
            model="3pl",
            n_quadrature=7,
            em_iter=25,
            max_iter=40,
            tol=1e-3,
            return_item_params=true,
        )
        assert_ranking(ranking)
        assert_scores(scores; expected_len=L)
        @test Set(keys(params)) == Set([
            "difficulty",
            "discrimination",
            "slopes",
            "intercept",
            "abilities",
            "ability_sd",
            "guessing",
        ])
        @test size(params["slopes"]) == (M, 2)
        @test size(params["abilities"]) == (L, 2)
        @test size(params["ability_sd"]) == (L, 2)
        @test length(params["difficulty"]) == M
        @test all(params["discrimination"] .>= 0.0)
        @test all(params["ability_sd"] .>= 0.0)

        # 2PL item params omit guessing.
        _, _, params_2pl = Scorio.Rank.mirt(
            R_small;
            n_factors=2,
            model="2pl",
            n_quadrature=7,
            em_iter=25,
            max_iter=40,
            tol=1e-3,
            return_item_params=true,
        )
        @test Set(keys(params_2pl)) == Set([
            "difficulty",
            "discrimination",
            "slopes",
            "intercept",
            "abilities",
            "ability_sd",
        ])

        # Validation errors.
        @test_throws ErrorException Scorio.Rank.mirt(R_small; model="4pl")
        @test_throws ErrorException Scorio.Rank.mirt(R_small; model="2pl", fix_guessing=0.2)
        @test_throws ErrorException Scorio.Rank.mirt(R_small; n_factors=12, n_quadrature=15)
        @test_throws ErrorException Scorio.Rank.mirt(R_small; n_factors=11, n_quadrature=2)
    end

    @testset "return_item_params branches" begin
        N = size(R_small, 3)
        time_points = collect(range(0.0, 1.0; length=N))

        ranking_rasch, scores_rasch, params_rasch = Scorio.Rank.rasch(
            R_small;
            max_iter=60,
            return_item_params=true,
        )
        assert_ranking(ranking_rasch)
        assert_scores(scores_rasch; expected_len=size(R_small, 1))
        @test Set(keys(params_rasch)) == Set(["difficulty"])

        ranking_2pl, scores_2pl, params_2pl = Scorio.Rank.rasch_2pl(
            R_small;
            max_iter=60,
            return_item_params=true,
        )
        assert_ranking(ranking_2pl)
        assert_scores(scores_2pl; expected_len=size(R_small, 1))
        @test Set(keys(params_2pl)) == Set(["difficulty", "discrimination"])

        ranking_3pl, scores_3pl, params_3pl = Scorio.Rank.rasch_3pl(
            R_small;
            max_iter=50,
            fix_guessing=0.2,
            return_item_params=true,
        )
        assert_ranking(ranking_3pl)
        assert_scores(scores_3pl; expected_len=size(R_small, 1))
        @test Set(keys(params_3pl)) == Set(["difficulty", "discrimination", "guessing"])

        ranking_growth, scores_growth, params_growth = Scorio.Rank.dynamic_irt(
            R_small;
            variant="growth",
            score_target="gain",
            assume_time_axis=true,
            time_points=time_points,
            max_iter=60,
            return_item_params=true,
        )
        assert_ranking(ranking_growth)
        assert_scores(scores_growth; expected_len=size(R_small, 1))
        @test Set(keys(params_growth)) ==
              Set(["difficulty", "baseline", "slope", "ability_path", "time_points"])
    end

    @testset "dynamic_irt longitudinal variants" begin
        N = size(R_small, 3)
        time_points = collect(range(0.0, 1.0; length=N))

        out_growth = Scorio.Rank.dynamic_irt(
            R_small;
            variant="growth",
            score_target="gain",
            assume_time_axis=true,
            time_points=time_points,
            max_iter=60,
            return_scores=true,
        )
        out_state = Scorio.Rank.dynamic_irt(
            R_small;
            variant="state_space",
            score_target="mean",
            assume_time_axis=true,
            time_points=time_points,
            max_iter=60,
            return_scores=true,
        )

        assert_ranking_and_scores(out_growth)
        assert_ranking_and_scores(out_state)
    end

    @testset "validation errors" begin
        @test_throws ErrorException Scorio.Rank.rasch_mml_credible(R_small; quantile=1.0)

        @test_throws ErrorException Scorio.Rank.dynamic_irt(R_small; variant="growth")

        @test_throws ErrorException Scorio.Rank.dynamic_irt(R_small; variant="bad")

        @test_throws ErrorException Scorio.Rank.dynamic_irt(
            R_small;
            variant="linear",
            score_target="gain",
        )

        @test_throws ErrorException Scorio.Rank.dynamic_irt(
            R_small;
            variant="growth",
            assume_time_axis=true,
            score_target="bad",
        )

        @test_throws ErrorException Scorio.Rank.rasch_3pl(R_small; guessing_upper=0.0)
    end
end
