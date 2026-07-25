using Test
using Scorio

@testset "rank/test_listwise.jl" begin
    R = ordered_binary_R()
    R_small = ordered_binary_small_R()

    @testset "listwise smoke and ordering" begin
        calls = [
            () -> Scorio.Rank.plackett_luce_map(R; prior=1.0, max_iter=100, return_scores=true),
            () -> Scorio.Rank.davidson_luce_map(R; prior=1.0, max_iter=100, return_scores=true),
            () -> Scorio.Rank.bradley_terry_luce_map(
                R;
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

    @testset "listwise finite MLE identification" begin
        finite_R = [1 0 0 1; 0 1 0 0; 0 0 1 0]
        for fn in (Scorio.Rank.plackett_luce, Scorio.Rank.bradley_terry_luce)
            ranking, scores = fn(finite_R; return_scores=true)
            @test ranking == [1, 2, 2]
            @test scores[1] > scores[2]
            @test scores[2] == scores[3]

            err = try
                fn(R)
                nothing
            catch caught
                caught
            end
            @test err isa ErrorException
            @test occursin(
                "no finite maximum-likelihood estimate",
                sprint(showerror, err),
            )
        end
    end

    @testset "plackett_luce_map prior coercion" begin
        _, scores_float = Scorio.Rank.plackett_luce_map(
            R_small;
            prior=1.0,
            max_iter=80,
            return_scores=true,
        )
        _, scores_object = Scorio.Rank.plackett_luce_map(
            R_small;
            prior=Scorio.Rank.GaussianPrior(0.0, 1.0),
            max_iter=80,
            return_scores=true,
        )

        @test length(scores_float) == length(scores_object)
        @test all(isfinite, scores_float)
        @test all(isfinite, scores_object)
    end

    @testset "SciPy L-BFGS-B near-tie parity" begin
        ranking, scores = Scorio.Rank.bradley_terry_luce(
            optimizer_parity_R();
            return_scores=true,
        )
        @test ranking == [1, 4, 3, 2]
        @test scores ≈ [
            1.0864041029481568,
            0.8335613204496499,
            1.016435314626585,
            1.0864038683692245,
        ] atol = 1e-7 rtol = 1e-7
    end

    @testset "validation errors" begin
        L = size(R_small, 1)

        @test_throws ErrorException Scorio.Rank.plackett_luce(R_small; max_iter=0)
        @test_throws ErrorException Scorio.Rank.plackett_luce_map(R_small; prior=0.0)
        @test_throws ErrorException Scorio.Rank.davidson_luce(R_small; max_tie_order=L + 1)
        @test_throws ErrorException Scorio.Rank.bradley_terry_luce_map(R_small; prior="bad")
    end
end
