using Test
using Scorio

@testset "rank/test_priors.jl" begin
    R_small = ordered_binary_small_R()

    @testset "Prior abstract" begin
        @test_throws MethodError Scorio.Rank.Prior()
    end

    @testset "Finite penalties" begin
        theta = [-0.3, -0.1, 0.1, 0.3]

        priors = [
            Scorio.Rank.GaussianPrior(0.0, 1.0),
            Scorio.Rank.LaplacePrior(0.0, 1.0),
            Scorio.Rank.CauchyPrior(0.0, 1.0),
            Scorio.Rank.UniformPrior(),
            Scorio.Rank.CustomPrior(x -> sum(abs.(x))),
            Scorio.Rank.EmpiricalPrior(R_small; var=1.5),
        ]

        for prior in priors
            penalty = Scorio.penalty(prior, theta)
            @test isfinite(Float64(penalty))
        end
    end

    @testset "Empirical prior shape and centering" begin
        prior = Scorio.Rank.EmpiricalPrior(R_small; var=1.0)
        @test length(prior.prior_mean) == size(R_small, 1)
        @test sum(prior.prior_mean) ≈ 0.0 atol = 1e-10

        positional = Scorio.Rank.EmpiricalPrior(R_small, 1.0, 1e-6)
        @test positional.prior_mean ≈ prior.prior_mean
        @test positional.penalty(zeros(size(R_small, 1))) ==
              Scorio.penalty(positional, zeros(size(R_small, 1)))

        nested_R = [
            [collect(R_small[l, i, :]) for i in axes(R_small, 2)] for
            l in axes(R_small, 1)
        ]
        nested = Scorio.Rank.EmpiricalPrior(nested_R; var=1.0)
        @test nested.prior_mean ≈ prior.prior_mean
    end

    @testset "Empirical prior theta length validation" begin
        prior = Scorio.Rank.EmpiricalPrior(R_small)
        @test_throws ErrorException Scorio.penalty(prior, [0.0, 1.0])
    end

    @testset "Constructor validation" begin
        @test_throws ErrorException Scorio.Rank.GaussianPrior(0.0, 0.0)
        @test_throws ErrorException Scorio.Rank.LaplacePrior(0.0, 0.0)
        @test_throws ErrorException Scorio.Rank.CauchyPrior(0.0, 0.0)
        @test_throws ErrorException Scorio.Rank.EmpiricalPrior(zeros(Int, 4, 3); var=0.0)
        @test_throws ErrorException Scorio.Rank.EmpiricalPrior(zeros(Int, 2, 3, 4, 5))
        @test_throws ErrorException Scorio.Rank.CustomPrior(42)
    end
end
