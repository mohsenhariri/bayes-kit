using Test
using Scorio

@testset "rank/test_rank_slow.jl" begin
    R_small = ordered_binary_small_R()
    identity_R = zeros(Int, size(R_small, 1), size(R_small, 1))
    for i in axes(identity_R, 1)
        identity_R[i, i] = 1
    end

    @testset "default-parameter smoke" begin
        calls = [
            () -> Scorio.Rank.thompson(R_small; return_scores=true),
            () -> Scorio.Rank.bayesian_mcmc(R_small; return_scores=true),
            () -> Scorio.Rank.rasch_3pl(
                R_small;
                fix_guessing=0.2,
                return_scores=true,
            ),
            () -> Scorio.Rank.rasch_3pl_map(R_small; return_scores=true),
            () -> Scorio.Rank.rasch_mml(R_small; return_scores=true),
            () -> Scorio.Rank.rasch_mml_credible(R_small; return_scores=true),
            () -> Scorio.Rank.davidson_luce(identity_R; return_scores=true),
            () -> Scorio.Rank.davidson_luce_map(R_small; return_scores=true),
        ]

        for run in calls
            ranking, scores = assert_ranking_and_scores(run())
            @test length(ranking) == size(R_small, 1)
            @test length(scores) == size(R_small, 1)
            @test all(isfinite, Float64.(scores))
        end
    end
end
