using Test
using Scorio

@testset "rank/test_graph_seriation_hodge.jl" begin
    R = ordered_binary_R()
    R_small = ordered_binary_small_R()

    @testset "graph/seriation/hodge smoke and ordering" begin
        calls = [
            () -> Scorio.Rank.pagerank(R; return_scores=true),
            () -> Scorio.Rank.spectral(R; return_scores=true),
            () -> Scorio.Rank.alpharank(
                R;
                population_size=20,
                max_iter=20_000,
                return_scores=true,
            ),
            () -> Scorio.Rank.nash(R; return_scores=true),
            () -> Scorio.Rank.rank_centrality(R; return_scores=true),
            () -> Scorio.Rank.serial_rank(R; return_scores=true),
            () -> Scorio.Rank.hodge_rank(R; return_scores=true),
        ]

        for run in calls
            ranking, _ = assert_ranking_and_scores(run())
            assert_ordering_sanity(ranking; best_idx=1, worst_idx=4)
        end
    end

    @testset "nash return_equilibrium branch" begin
        ranking, scores, equilibrium = Scorio.Rank.nash(
            R_small;
            return_scores=true,
            return_equilibrium=true,
        )

        assert_ranking(ranking; expected_len=size(R_small, 1))
        assert_scores(scores; expected_len=size(R_small, 1))
        @test length(equilibrium) == size(R_small, 1)
        @test all(isfinite, equilibrium)
        @test all(equilibrium .>= 0.0)
        @test sum(equilibrium) ≈ 1.0 atol = 1e-10
    end

    @testset "pagerank sequence teleport" begin
        _, vector_scores = Scorio.Rank.pagerank(
            R_small;
            teleport=fill(0.25, 4),
            return_scores=true,
        )
        _, tuple_scores = Scorio.Rank.pagerank(
            R_small;
            teleport=(0.25, 0.25, 0.25, 0.25),
            return_scores=true,
        )
        @test tuple_scores ≈ vector_scores
    end

    @testset "hodge return_diagnostics branch" begin
        ranking, scores, diagnostics = Scorio.Rank.hodge_rank(
            R_small;
            pairwise_stat="log_odds",
            weight_method="decisive",
            return_scores=true,
            return_diagnostics=true,
        )

        assert_ranking(ranking; expected_len=size(R_small, 1))
        assert_scores(scores; expected_len=size(R_small, 1))
        @test keys(diagnostics) == Set(["residual_l2", "relative_residual_l2"])
        @test isfinite(Float64(diagnostics["residual_l2"]))
        @test isfinite(Float64(diagnostics["relative_residual_l2"]))
    end

    @testset "validation errors" begin
        @test_throws ErrorException Scorio.Rank.pagerank(R_small; damping=1.0)
        @test_throws ErrorException Scorio.Rank.pagerank(R_small; teleport=[1.0, 2.0])
        @test_throws ErrorException Scorio.Rank.alpharank(R_small; alpha=-0.1)
        @test_throws ErrorException Scorio.Rank.nash(R_small; solver="bad")
        @test_throws ErrorException Scorio.Rank.rank_centrality(R_small; tie_handling="bad")
        @test_throws ErrorException Scorio.Rank.serial_rank(R_small; comparison="bad")
        @test_throws ErrorException Scorio.Rank.hodge_rank(R_small; pairwise_stat="bad")
    end
end
