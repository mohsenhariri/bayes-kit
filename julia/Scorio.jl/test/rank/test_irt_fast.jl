using Test
using Scorio
using Random

@testset "rank/test_irt_fast.jl" begin
    R_small = ordered_binary_small_R()

    @testset "IRT fast smoke and ordering" begin
        calls = [
            () -> Scorio.Rank.rasch(R_small; max_iter=80, return_scores=true),
            () -> Scorio.Rank.rasch_map(R_small; prior=1.0, max_iter=80, return_scores=true),
            () -> Scorio.Rank.rasch_2pl(
                R_small;
                max_iter=300,
                reg_discrimination=0.01,
                return_scores=true,
            ),
            () -> Scorio.Rank.rasch_2pl_map(
                R_small;
                prior=1.0,
                max_iter=300,
                reg_discrimination=0.01,
                return_scores=true,
            ),
            () -> Scorio.Rank.rasch_3pl(
                R_small;
                max_iter=500,
                fix_guessing=0.2,
                reg_discrimination=0.01,
                reg_guessing=0.1,
                return_scores=true,
            ),
            () -> Scorio.Rank.rasch_3pl_map(
                R_small;
                prior=1.0,
                max_iter=500,
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
            max_iter=300,
            return_item_params=true,
        )
        assert_ranking(ranking_rasch)
        assert_scores(scores_rasch; expected_len=size(R_small, 1))
        @test Set(keys(params_rasch)) == Set(["difficulty"])

        ranking_2pl, scores_2pl, params_2pl = Scorio.Rank.rasch_2pl(
            R_small;
            max_iter=300,
            return_item_params=true,
        )
        assert_ranking(ranking_2pl)
        assert_scores(scores_2pl; expected_len=size(R_small, 1))
        @test Set(keys(params_2pl)) == Set(["difficulty", "discrimination"])

        ranking_3pl, scores_3pl, params_3pl = Scorio.Rank.rasch_3pl(
            R_small;
            max_iter=500,
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

@testset "IRT parity regressions" begin
    error_text(call) = try
        call()
        ""
    catch exception
        sprint(showerror, exception)
    end

    @testset "finite joint-estimate guards" begin
        extreme_person = [1 1 1 1; 0 1 0 1; 0 0 0 0]
        extreme_item = [1 1 0; 1 0 1; 1 0 0]
        quasi_separated = [0 0 0 1; 0 0 1 0; 0 1 1 1; 1 0 1 1]
        ml_calls = [
            R -> Scorio.Rank.rasch(R),
            R -> Scorio.Rank.rasch_2pl(R; max_iter=300),
            R -> Scorio.Rank.rasch_3pl(R; max_iter=500, fix_guessing=0.2),
        ]
        for call in ml_calls
            @test occursin("no finite ability MLE", error_text(() -> call(extreme_person)))
            @test occursin(
                "no finite item-parameter estimate",
                error_text(() -> call(extreme_item)),
            )
            @test occursin(
                "completely or quasi-separated",
                error_text(() -> call(quasi_separated)),
            )
        end

        uniform_calls = [
            R -> Scorio.Rank.rasch_map(R; prior=Scorio.Rank.UniformPrior()),
            R -> Scorio.Rank.rasch_2pl_map(
                R;
                prior=Scorio.Rank.UniformPrior(),
                max_iter=300,
            ),
            R -> Scorio.Rank.rasch_3pl_map(
                R;
                prior=Scorio.Rank.UniformPrior(),
                max_iter=500,
                fix_guessing=0.2,
            ),
        ]
        for call in uniform_calls
            @test occursin("no finite ability MLE", error_text(() -> call(extreme_person)))
        end
    end

    @testset "identified regularization settings" begin
        R = ordered_binary_small_R()
        @test occursin(
            "ability/discrimination scale",
            error_text(() -> Scorio.Rank.rasch_2pl(R; reg_discrimination=0.0)),
        )
        @test occursin(
            "ability/discrimination scale",
            error_text(
                () -> Scorio.Rank.rasch_3pl(
                    R;
                    fix_guessing=0.2,
                    reg_discrimination=0.0,
                ),
            ),
        )
        @test occursin(
            "reg_guessing must be positive",
            error_text(() -> Scorio.Rank.rasch_3pl(R; reg_guessing=0.0)),
        )
        @test occursin(
            "reg_guessing must be positive",
            error_text(() -> Scorio.Rank.rasch_3pl_map(R; reg_guessing=0.0)),
        )
    end

    @testset "exact exchangeability" begin
        R = zeros(Int, 3, 4, 2)
        R[1, :, :] = [1 0; 1 1; 0 1; 1 0]
        R[2, :, :] = R[1, :, :]
        R[3, :, :] = [0 0; 1 0; 1 0; 0 1]
        calls = [
            () -> Scorio.Rank.rasch(R; max_iter=80, return_scores=true),
            () -> Scorio.Rank.rasch_map(R; max_iter=80, return_scores=true),
            () -> Scorio.Rank.rasch_2pl(R; max_iter=300, return_scores=true),
            () -> Scorio.Rank.rasch_2pl_map(R; max_iter=300, return_scores=true),
            () -> Scorio.Rank.rasch_3pl(
                R;
                max_iter=500,
                fix_guessing=0.2,
                return_scores=true,
            ),
            () -> Scorio.Rank.rasch_3pl_map(
                R;
                max_iter=500,
                fix_guessing=0.2,
                return_scores=true,
            ),
            () -> Scorio.Rank.rasch_mml(
                R;
                max_iter=12,
                em_iter=8,
                n_quadrature=9,
                return_scores=true,
            ),
            () -> Scorio.Rank.rasch_mml_credible(
                R;
                quantile=0.1,
                max_iter=12,
                em_iter=8,
                n_quadrature=9,
                return_scores=true,
            ),
        ]
        for call in calls
            ranking, scores = call()
            @test ranking[1] == ranking[2]
            @test scores[1] == scores[2]
        end

        cyclic = [0 2 1; 1 0 2; 2 1 0]
        @test Scorio.average_event_exchangeable_scores([1.0, 2.0, 6.0], cyclic) ==
              fill(3.0, 3)

        target_prior = Scorio.Rank.CustomPrior(
            theta -> sum((theta .- [2.0, -2.0, 0.0]) .^ 2),
        )
        asymmetric = [1 0 1 0; 1 0 1 0; 0 1 0 1]
        _, scores = Scorio.Rank.rasch_map(
            asymmetric;
            prior=target_prior,
            return_scores=true,
        )
        @test scores[1] != scores[2]
    end

    @testset "MML population scale and boundary items" begin
        rng = MersenneTwister(20260723)
        easy = Int.(rand(rng, 40, 6, 10) .< 0.8)
        hard = Int.(rand(rng, 40, 6, 10) .< 0.2)
        _, _, easy_params = Scorio.Rank.rasch_mml(
            easy;
            max_iter=30,
            em_iter=25,
            n_quadrature=21,
            return_item_params=true,
        )
        _, _, hard_params = Scorio.Rank.rasch_mml(
            hard;
            max_iter=30,
            em_iter=25,
            n_quadrature=21,
            return_item_params=true,
        )
        @test sum(easy_params["difficulty"]) / 6 < -0.5
        @test sum(hard_params["difficulty"]) / 6 > 0.5

        for (value, expected_difficulty) in ((1, -Inf), (0, Inf))
            boundary = fill(value, 8, 1, 3)
            ranking, scores, params = Scorio.Rank.rasch_mml(
                boundary;
                max_iter=5,
                em_iter=3,
                n_quadrature=9,
                return_item_params=true,
            )
            @test ranking == ones(Int, 8)
            @test all(abs.(scores) .<= 1e-14)
            @test params["difficulty"][1] == expected_difficulty
        end
    end

    @testset "nonconvex and longitudinal identification" begin
        ambiguous = [
            1 1 0 0 1 0
            1 0 1 0 0 1
            0 1 1 0 1 0
            0 0 0 1 1 1
        ]
        @test occursin(
            "multiple equally good nonconvex",
            error_text(() -> Scorio.Rank.rasch_2pl(ambiguous; max_iter=500)),
        )

        quasi_separated = [1 0 1 1; 0 1 1 1; 0 0 1 0; 0 0 0 1]
        @test occursin(
            "no finite joint location estimate",
            error_text(
                () -> Scorio.Rank.rasch_3pl(
                    quasi_separated;
                    fix_guessing=0.2,
                    max_iter=500,
                ),
            ),
        )

        stationary = zeros(Int, 4, 4, 3)
        stationary[1, :, :] = [1 0 1; 1 1 0; 0 1 0; 1 0 0]
        stationary[2, :, :] = [0 1 0; 1 0 0; 0 0 1; 1 1 0]
        stationary[3, :, :] = [1 1 0; 0 0 1; 1 0 1; 0 0 1]
        stationary[4, :, :] = [0 0 1; 0 1 1; 1 1 0; 0 1 0]
        @test occursin(
            "stationary",
            error_text(
                () -> Scorio.Rank.rasch_3pl(
                    stationary;
                    fix_guessing=0.2,
                    max_iter=500,
                ),
            ),
        )

        longitudinal = zeros(Int, 4, 4, 3)
        longitudinal[1, :, :] = [1 1 0; 0 1 0; 1 0 0; 0 1 1]
        longitudinal[2, :, :] = [1 0 0; 1 1 1; 0 0 0; 1 1 0]
        longitudinal[3, :, :] = [1 1 1; 1 0 1; 1 0 0; 0 0 0]
        longitudinal[4, :, :] = [1 1 1; 1 0 0; 1 0 1; 1 1 0]
        _, base_scores = Scorio.Rank.dynamic_irt(
            longitudinal;
            variant="growth",
            assume_time_axis=true,
            return_scores=true,
        )
        _, item_scores = Scorio.Rank.dynamic_irt(
            longitudinal[:, [1, 4, 2, 3], :];
            variant="growth",
            assume_time_axis=true,
            return_scores=true,
        )
        @test base_scores[1] == base_scores[3]
        @test item_scores ≈ base_scores atol = 2e-4

        _, tuple_time_scores = Scorio.Rank.dynamic_irt(
            longitudinal;
            variant="growth",
            time_points=(0.0, 0.5, 1.0),
            assume_time_axis=true,
            return_scores=true,
        )
        @test all(isfinite, tuple_time_scores)

        _, gain_scores, params = Scorio.Rank.dynamic_irt(
            longitudinal;
            variant="growth",
            score_target="gain",
            assume_time_axis=true,
            return_item_params=true,
        )
        @test gain_scores ≈ params["ability_path"][:, end] .-
              params["ability_path"][:, 1]
        @test gain_scores ≈ params["slope"]

        unidentified = zeros(Int, 3, 2, 2)
        unidentified[1, :, :] = [1 1; 1 1]
        unidentified[2, :, :] = [1 0; 0 1]
        unidentified[3, :, :] = [0 1; 1 0]
        @test occursin(
            "no finite ability MLE",
            error_text(
                () -> Scorio.Rank.dynamic_irt(
                    unidentified;
                    variant="growth",
                    assume_time_axis=true,
                ),
            ),
        )
        @test occursin(
            "slope_reg must be positive",
            error_text(
                () -> Scorio.Rank.dynamic_irt(
                    unidentified[2:3, :, :];
                    variant="growth",
                    assume_time_axis=true,
                    slope_reg=0.0,
                ),
            ),
        )
        @test occursin(
            "state_reg must be positive",
            error_text(
                () -> Scorio.Rank.dynamic_irt(
                    unidentified[2:3, :, :];
                    variant="state_space",
                    assume_time_axis=true,
                    state_reg=0.0,
                ),
            ),
        )
        @test occursin(
            "at least two time points",
            error_text(
                () -> Scorio.Rank.dynamic_irt(
                    unidentified[:, :, 1:1];
                    variant="growth",
                    assume_time_axis=true,
                ),
            ),
        )

        boundary_item = [1 1 0; 1 0 1; 1 0 0]
        boundary_longitudinal = repeat(reshape(boundary_item, 3, 3, 1), 1, 1, 2)
        @test occursin(
            "no finite item-parameter estimate",
            error_text(
                () -> Scorio.Rank.dynamic_irt(
                    boundary_longitudinal;
                    variant="growth",
                    assume_time_axis=true,
                ),
            ),
        )
        @test occursin(
            "no finite item-parameter estimate",
            error_text(
                () -> Scorio.Rank.mirt(
                    boundary_longitudinal;
                    n_factors=1,
                    n_quadrature=7,
                ),
            ),
        )
    end
end
