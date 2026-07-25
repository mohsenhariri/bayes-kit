using Test
using Scorio

@testset "rank/test_bradley_terry.jl" begin
    R = ordered_binary_R()
    R_tie = tie_heavy_R()

    @testset "BT family smoke and ordering" begin
        calls = [
            () -> Scorio.Rank.bradley_terry_map(R; prior=1.0, max_iter=100, return_scores=true),
            () -> Scorio.Rank.bradley_terry_davidson(R; max_iter=100, return_scores=true),
            () -> Scorio.Rank.bradley_terry_davidson_map(
                R;
                prior=1.0,
                max_iter=100,
                return_scores=true,
            ),
            () -> Scorio.Rank.rao_kupper(
                R;
                tie_strength=1.1,
                max_iter=100,
                return_scores=true,
            ),
            () -> Scorio.Rank.rao_kupper_map(
                R;
                tie_strength=1.1,
                prior=1.0,
                max_iter=100,
                return_scores=true,
            ),
        ]

        for run in calls
            ranking, _ = assert_ranking_and_scores(run())
            assert_ordering_sanity(ranking; best_idx=1, worst_idx=4)
        end
    end

    @testset "BT finite MLE identification" begin
        finite_R = [1 0 0 1; 0 1 0 0; 0 0 1 0]
        ranking, scores = Scorio.Rank.bradley_terry(finite_R; return_scores=true)
        @test ranking == [1, 2, 2]
        @test scores[1] > scores[2]
        @test scores[2] == scores[3]

        err = try
            Scorio.Rank.bradley_terry(R)
            nothing
        catch caught
            caught
        end
        @test err isa ErrorException
        @test occursin("no finite maximum-likelihood estimate", sprint(showerror, err))
    end

    @testset "MAP prior coercion" begin
        _, scores_float = Scorio.Rank.bradley_terry_map(
            R;
            prior=1.0,
            max_iter=80,
            return_scores=true,
        )
        _, scores_object = Scorio.Rank.bradley_terry_map(
            R;
            prior=Scorio.Rank.GaussianPrior(0.0, 1.0),
            max_iter=80,
            return_scores=true,
        )

        @test length(scores_float) == length(scores_object)
        @test all(isfinite, scores_float)
        @test all(isfinite, scores_object)
    end

    @testset "SciPy L-BFGS-B near-tie parity" begin
        R_parity = optimizer_parity_R()
        expected_ranking = [1, 4, 3, 2]

        ranking, scores = Scorio.Rank.bradley_terry_davidson(
            R_parity;
            return_scores=true,
        )
        @test ranking == expected_ranking
        @test scores ≈ [
            1.0661859976160217,
            0.8250917872442711,
            1.066183101350352,
            1.0661859773886564,
        ] atol = 1e-7 rtol = 1e-7

        ranking, scores = Scorio.Rank.bradley_terry_davidson_map(
            R_parity;
            return_scores=true,
        )
        @test ranking == expected_ranking
        @test scores ≈ [
            1.0584047665491454,
            0.843421892274795,
            1.0584042965334972,
            1.0584046743266902,
        ] atol = 1e-7 rtol = 1e-7
    end

    @testset "Validation errors" begin
        @test_throws ErrorException Scorio.Rank.bradley_terry(R; max_iter=0)
        @test_throws ErrorException Scorio.Rank.bradley_terry_map(R; prior=-1.0)
        @test_throws ErrorException Scorio.Rank.rao_kupper(R; tie_strength=0.9)
        @test_throws ErrorException Scorio.Rank.rao_kupper(R_tie; tie_strength=1.0)
        @test_throws ErrorException Scorio.Rank.rao_kupper_map(R; prior="bad")
    end
end
