using Test
using Scorio

if !isdefined(Main, :ordered_binary_small_R)
    include(joinpath(@__DIR__, "helpers.jl"))
end

@testset "rank/test_rank_public_api.jl" begin
    R_small = ordered_binary_small_R()
    R_matrix = ordered_binary_matrix()
    R_multi, w, R0_shared, _ = multiclass_rank_data()

    expected_exports = Symbol[
        :Prior,
        :GaussianPrior,
        :LaplacePrior,
        :CauchyPrior,
        :UniformPrior,
        :CustomPrior,
        :EmpiricalPrior,
        :avg,
        :bayes,
        :pass_at_k,
        :pass_hat_k,
        :g_pass_at_k_tau,
        :mg_pass_at_k,
        :inverse_difficulty,
        :elo,
        :glicko,
        :trueskill,
        :bradley_terry,
        :bradley_terry_map,
        :bradley_terry_davidson,
        :bradley_terry_davidson_map,
        :rao_kupper,
        :rao_kupper_map,
        :thompson,
        :bayesian_mcmc,
        :borda,
        :copeland,
        :win_rate,
        :minimax,
        :schulze,
        :ranked_pairs,
        :kemeny_young,
        :nanson,
        :baldwin,
        :majority_judgment,
        :rasch,
        :rasch_map,
        :rasch_2pl,
        :rasch_2pl_map,
        :rasch_3pl,
        :rasch_3pl_map,
        :rasch_mml,
        :rasch_mml_credible,
        :dynamic_irt,
        :mirt,
        :pagerank,
        :spectral,
        :alpharank,
        :nash,
        :rank_centrality,
        :serial_rank,
        :hodge_rank,
        :plackett_luce,
        :plackett_luce_map,
        :davidson_luce,
        :davidson_luce_map,
        :bradley_terry_luce,
        :bradley_terry_luce_map,
    ]

    @test length(expected_exports) == 58

    actual_exports = Set(filter(name -> name != :Rank, names(Scorio.Rank; all=false, imported=true)))
    @test actual_exports == Set(expected_exports)

    function_calls = Dict{Symbol,Function}(
        :avg => () -> Scorio.Rank.avg(R_small; return_scores=true),
        :bayes => () -> Scorio.Rank.bayes(R_multi, w; R0=R0_shared, return_scores=true),
        :pass_at_k => () -> Scorio.Rank.pass_at_k(R_small, 2; return_scores=true),
        :pass_hat_k => () -> Scorio.Rank.pass_hat_k(R_small, 2; return_scores=true),
        :g_pass_at_k_tau =>
            () -> Scorio.Rank.g_pass_at_k_tau(R_small, 2, 0.7; return_scores=true),
        :mg_pass_at_k => () -> Scorio.Rank.mg_pass_at_k(R_small, 2; return_scores=true),
        :inverse_difficulty =>
            () -> Scorio.Rank.inverse_difficulty(R_small; return_scores=true),
        :elo => () -> Scorio.Rank.elo(R_small; return_scores=true),
        :glicko => () -> Scorio.Rank.glicko(R_small; return_scores=true),
        :trueskill => () -> Scorio.Rank.trueskill(R_small; return_scores=true),
        :bradley_terry =>
            () -> Scorio.Rank.bradley_terry(R_small; max_iter=80, return_scores=true),
        :bradley_terry_map =>
            () -> Scorio.Rank.bradley_terry_map(R_small; prior=1.0, max_iter=80, return_scores=true),
        :bradley_terry_davidson =>
            () -> Scorio.Rank.bradley_terry_davidson(R_small; max_iter=80, return_scores=true),
        :bradley_terry_davidson_map => () -> Scorio.Rank.bradley_terry_davidson_map(
            R_small;
            prior=1.0,
            max_iter=80,
            return_scores=true,
        ),
        :rao_kupper =>
            () -> Scorio.Rank.rao_kupper(R_small; tie_strength=1.1, max_iter=80, return_scores=true),
        :rao_kupper_map => () -> Scorio.Rank.rao_kupper_map(
            R_small;
            tie_strength=1.1,
            prior=1.0,
            max_iter=80,
            return_scores=true,
        ),
        :thompson =>
            () -> Scorio.Rank.thompson(R_small; n_samples=700, seed=7, return_scores=true),
        :bayesian_mcmc => () -> Scorio.Rank.bayesian_mcmc(
            R_small;
            n_samples=400,
            burnin=100,
            seed=7,
            return_scores=true,
        ),
        :borda => () -> Scorio.Rank.borda(R_small; return_scores=true),
        :copeland => () -> Scorio.Rank.copeland(R_small; return_scores=true),
        :win_rate => () -> Scorio.Rank.win_rate(R_small; return_scores=true),
        :minimax => () -> Scorio.Rank.minimax(R_small; return_scores=true),
        :schulze => () -> Scorio.Rank.schulze(R_small; return_scores=true),
        :ranked_pairs => () -> Scorio.Rank.ranked_pairs(R_small; return_scores=true),
        :kemeny_young =>
            () -> Scorio.Rank.kemeny_young(R_small; time_limit=1.0, return_scores=true),
        :nanson => () -> Scorio.Rank.nanson(R_small; return_scores=true),
        :baldwin => () -> Scorio.Rank.baldwin(R_small; return_scores=true),
        :majority_judgment => () -> Scorio.Rank.majority_judgment(R_small; return_scores=true),
        :rasch => () -> Scorio.Rank.rasch(R_small; max_iter=60, return_scores=true),
        :rasch_map => () -> Scorio.Rank.rasch_map(R_small; prior=1.0, max_iter=60, return_scores=true),
        :rasch_2pl => () -> Scorio.Rank.rasch_2pl(R_small; max_iter=60, return_scores=true),
        :rasch_2pl_map =>
            () -> Scorio.Rank.rasch_2pl_map(R_small; prior=1.0, max_iter=60, return_scores=true),
        :rasch_3pl =>
            () -> Scorio.Rank.rasch_3pl(R_small; max_iter=50, fix_guessing=0.2, return_scores=true),
        :rasch_3pl_map => () -> Scorio.Rank.rasch_3pl_map(
            R_small;
            prior=1.0,
            max_iter=50,
            fix_guessing=0.2,
            return_scores=true,
        ),
        :rasch_mml => () -> Scorio.Rank.rasch_mml(
            R_small;
            max_iter=10,
            em_iter=6,
            n_quadrature=9,
            return_scores=true,
        ),
        :rasch_mml_credible => () -> Scorio.Rank.rasch_mml_credible(
            R_small;
            quantile=0.1,
            max_iter=10,
            em_iter=6,
            n_quadrature=9,
            return_scores=true,
        ),
        :dynamic_irt =>
            () -> Scorio.Rank.dynamic_irt(R_matrix; variant="linear", max_iter=60, return_scores=true),
        :mirt => () -> Scorio.Rank.mirt(
            R_small;
            n_factors=2,
            n_quadrature=7,
            em_iter=10,
            max_iter=30,
            return_scores=true,
        ),
        :pagerank => () -> Scorio.Rank.pagerank(R_small; return_scores=true),
        :spectral => () -> Scorio.Rank.spectral(R_small; return_scores=true),
        :alpharank =>
            () -> Scorio.Rank.alpharank(R_small; population_size=20, max_iter=10_000, return_scores=true),
        :nash => () -> Scorio.Rank.nash(R_small; return_scores=true),
        :rank_centrality => () -> Scorio.Rank.rank_centrality(R_small; return_scores=true),
        :serial_rank => () -> Scorio.Rank.serial_rank(R_small; return_scores=true),
        :hodge_rank => () -> Scorio.Rank.hodge_rank(R_small; return_scores=true),
        :plackett_luce =>
            () -> Scorio.Rank.plackett_luce(R_small; max_iter=80, return_scores=true),
        :plackett_luce_map =>
            () -> Scorio.Rank.plackett_luce_map(R_small; prior=1.0, max_iter=80, return_scores=true),
        :davidson_luce =>
            () -> Scorio.Rank.davidson_luce(R_small; max_iter=80, return_scores=true),
        :davidson_luce_map =>
            () -> Scorio.Rank.davidson_luce_map(R_small; prior=1.0, max_iter=80, return_scores=true),
        :bradley_terry_luce =>
            () -> Scorio.Rank.bradley_terry_luce(R_small; max_iter=80, return_scores=true),
        :bradley_terry_luce_map => () -> Scorio.Rank.bradley_terry_luce_map(
            R_small;
            prior=1.0,
            max_iter=80,
            return_scores=true,
        ),
    )

    class_calls = Dict{Symbol,Function}(
        :Prior => () -> Scorio.Rank.Prior(),
        :GaussianPrior => () -> Scorio.Rank.GaussianPrior(),
        :LaplacePrior => () -> Scorio.Rank.LaplacePrior(),
        :CauchyPrior => () -> Scorio.Rank.CauchyPrior(),
        :UniformPrior => () -> Scorio.Rank.UniformPrior(),
        :CustomPrior => () -> Scorio.Rank.CustomPrior(x -> sum(abs2, x)),
        :EmpiricalPrior => () -> Scorio.Rank.EmpiricalPrior(R_small),
    )

    @test Set(keys(function_calls)) ∪ Set(keys(class_calls)) == Set(expected_exports)

    for name in sort!(collect(keys(function_calls)); by=String)
        ranking, scores = assert_ranking_and_scores(function_calls[name]())
        assert_ordering_sanity(ranking; best_idx=1, worst_idx=length(ranking))
        assert_scores(scores; expected_len=length(ranking))
    end

    for (name, build) in class_calls
        if name == :Prior
            @test_throws MethodError build()
            continue
        end

        prior = build()
        theta = collect(range(-0.5, 0.5; length=size(R_small, 1)))
        value = Scorio.penalty(prior, theta)
        @test isfinite(Float64(value))
    end
end
