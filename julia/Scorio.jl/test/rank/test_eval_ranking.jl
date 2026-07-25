using Test
using Scorio

@testset "rank/eval_ranking.jl" begin
    @testset "avg rank wrapper" begin
        R = zeros(Int, 2, 2, 3)
        R[1, :, :] = [1 1 0; 1 0 1]
        R[2, :, :] = [0 0 1; 1 0 0]

        ranking, scores = Scorio.Rank.avg(R; return_scores=true)
        expected_scores = zeros(Float64, 2)
        for l in 1:2
            mu, _ = Scorio.Eval.avg(@view R[l, :, :])
            expected_scores[l] = mu
        end

        @test scores ≈ expected_scores
        @test ranking == Scorio.rank_scores(expected_scores)["competition"]

        nested_R = [
            [collect(R[l, i, :]) for i in axes(R, 2)] for l in axes(R, 1)
        ]
        nested_ranking, nested_scores = Scorio.Rank.avg(nested_R; return_scores=true)
        @test nested_scores ≈ scores
        @test nested_ranking == ranking
    end

    @testset "bayes rank wrapper with R0 variants and quantile" begin
        R = zeros(Int, 2, 3, 2)
        R[1, :, :] = [1 0; 1 1; 0 0]
        R[2, :, :] = [0 0; 1 0; 1 1]
        w = [0.0, 1.0]

        R0_shared = Int[
            1 1
            0 1
            0 0
        ]

        ranking_shared, scores_shared = Scorio.Rank.bayes(R, w; R0=R0_shared, return_scores=true)
        expected_shared = zeros(Float64, 2)
        for l in 1:2
            mu, _ = Scorio.Eval.bayes(@view(R[l, :, :]), w, R0_shared)
            expected_shared[l] = mu
        end
        @test scores_shared ≈ expected_shared
        @test ranking_shared == Scorio.rank_scores(expected_shared)["competition"]

        nested_R0 = [collect(row) for row in eachrow(R0_shared)]
        nested_ranking, nested_scores = Scorio.Rank.bayes(
            R,
            w;
            R0=nested_R0,
            return_scores=true,
        )
        @test nested_scores ≈ scores_shared
        @test nested_ranking == ranking_shared

        R0_per = zeros(Int, 2, 3, 2)
        R0_per[1, :, :] = R0_shared
        R0_per[2, :, :] = [0 0; 1 0; 1 1]
        ranking_per, scores_per = Scorio.Rank.bayes(R, w; R0=R0_per, return_scores=true)
        expected_per = zeros(Float64, 2)
        for l in 1:2
            mu, _ = Scorio.Eval.bayes(@view(R[l, :, :]), w, @view(R0_per[l, :, :]))
            expected_per[l] = mu
        end
        @test scores_per ≈ expected_per
        @test ranking_per == Scorio.rank_scores(expected_per)["competition"]

        q = 0.05
        z = Scorio._norm_ppf(q)
        _, q_scores = Scorio.Rank.bayes(R, w; R0=R0_shared, quantile=q, return_scores=true)
        expected_q = zeros(Float64, 2)
        for l in 1:2
            mu, sigma = Scorio.Eval.bayes(@view(R[l, :, :]), w, R0_shared)
            expected_q[l] = mu + z * sigma
        end
        @test q_scores ≈ expected_q atol = 1e-8

        @test_throws ErrorException Scorio.Rank.bayes(R, w; quantile=-0.1)
        @test_throws ErrorException Scorio.Rank.bayes(R, w; quantile=1.1)
        @test_throws ErrorException Scorio.Rank.bayes(R, w; R0=zeros(Int, 2, 2))
        @test_throws ErrorException Scorio.Rank.bayes(R, w; R0=zeros(Int, 3, 3, 1))
        @test_throws ErrorException Scorio.Rank.bayes(R, w; R0=ones(Int, 2))
    end

    @testset "pass@k family rank wrappers" begin
        R = zeros(Int, 2, 2, 3)
        R[1, :, :] = [1 1 0; 0 1 0]
        R[2, :, :] = [1 0 0; 0 0 0]

        r_pass, s_pass = Scorio.Rank.pass_at_k(R, 2; return_scores=true)
        expected_pass = [
            Scorio.Eval.pass_at_k(@view(R[1, :, :]), 2),
            Scorio.Eval.pass_at_k(@view(R[2, :, :]), 2),
        ]
        @test s_pass ≈ expected_pass
        @test r_pass == Scorio.rank_scores(expected_pass)["competition"]

        r_phat, s_phat = Scorio.Rank.pass_hat_k(R, 2; return_scores=true)
        expected_phat = [
            Scorio.Eval.pass_hat_k(@view(R[1, :, :]), 2),
            Scorio.Eval.pass_hat_k(@view(R[2, :, :]), 2),
        ]
        @test s_phat ≈ expected_phat
        @test r_phat == Scorio.rank_scores(expected_phat)["competition"]

        _, s_tau = Scorio.Rank.g_pass_at_k_tau(R, 2, 1.0; return_scores=true)
        @test s_tau ≈ s_phat

        r_mg, s_mg = Scorio.Rank.mg_pass_at_k(R, 2; return_scores=true)
        expected_mg = [
            Scorio.Eval.mg_pass_at_k(@view(R[1, :, :]), 2),
            Scorio.Eval.mg_pass_at_k(@view(R[2, :, :]), 2),
        ]
        @test s_mg ≈ expected_mg
        @test r_mg == Scorio.rank_scores(expected_mg)["competition"]

        @test_throws ErrorException Scorio.Rank.pass_at_k(R, 0)
        @test_throws ErrorException Scorio.Rank.pass_hat_k(R, 0)
        @test_throws ErrorException Scorio.Rank.g_pass_at_k_tau(R, 0, 0.5)
        @test_throws ErrorException Scorio.Rank.mg_pass_at_k(R, 0)
    end
end
