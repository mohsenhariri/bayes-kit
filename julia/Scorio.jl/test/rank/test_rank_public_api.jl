using Test
using Scorio

if !isdefined(Main, :ordered_binary_small_R)
    include(joinpath(@__DIR__, "helpers.jl"))
end

@testset "rank/test_rank_public_api.jl" begin
    R_small = ordered_binary_small_R()
    R_matrix = ordered_binary_matrix()
    R_multi, w, R0_shared, _ = multiclass_rank_data()
    finite_mle_R = [1 0 0 1; 0 1 0 0; 0 0 1 0]

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

    actual_exports = Set(
        filter(
            name -> name ∉ (:Rank, :Scorio),
            names(Scorio.Rank; all=false, imported=true),
        ),
    )
    @test actual_exports == Set(expected_exports)

    # Independent transcription of Python's inspect.signature order.  For
    # every positional prefix, the generated Julia method must expose exactly
    # the unconsumed suffix as keywords.
    python_optional_parameters = Dict{Symbol,Tuple}(
        :avg => (:method, :return_scores),
        :bayes => (:w, :R0, :quantile, :method, :return_scores),
        :pass_at_k => (:method, :return_scores),
        :pass_hat_k => (:method, :return_scores),
        :g_pass_at_k_tau => (:method, :return_scores),
        :mg_pass_at_k => (:method, :return_scores),
        :inverse_difficulty => (:method, :return_scores, :clip_range),
        :elo => (:K, :initial_rating, :tie_handling, :method, :return_scores),
        :glicko => (
            :initial_rating,
            :initial_rd,
            :c,
            :rd_max,
            :tie_handling,
            :return_deviation,
            :method,
            :return_scores,
        ),
        :trueskill => (
            :mu_initial,
            :sigma_initial,
            :beta,
            :tau,
            :method,
            :return_scores,
            :tie_handling,
            :draw_margin,
        ),
        :bradley_terry => (:method, :return_scores, :max_iter),
        :bradley_terry_map => (:prior, :method, :return_scores, :max_iter),
        :bradley_terry_davidson => (:method, :return_scores, :max_iter),
        :bradley_terry_davidson_map => (:prior, :method, :return_scores, :max_iter),
        :rao_kupper => (:tie_strength, :method, :return_scores, :max_iter),
        :rao_kupper_map => (:tie_strength, :prior, :method, :return_scores, :max_iter),
        :thompson =>
            (:n_samples, :prior_alpha, :prior_beta, :seed, :method, :return_scores),
        :bayesian_mcmc =>
            (:n_samples, :burnin, :prior_var, :seed, :method, :return_scores),
        :borda => (:method, :return_scores),
        :copeland => (:method, :return_scores),
        :win_rate => (:method, :return_scores),
        :minimax => (:variant, :tie_policy, :method, :return_scores),
        :schulze => (:tie_policy, :method, :return_scores),
        :ranked_pairs => (:strength, :tie_policy, :method, :return_scores),
        :kemeny_young =>
            (:tie_policy, :method, :return_scores, :time_limit, :tie_aware),
        :nanson => (:rank_ties, :method, :return_scores),
        :baldwin => (:rank_ties, :method, :return_scores),
        :majority_judgment => (:method, :return_scores),
        :rasch => (:method, :return_scores, :max_iter, :return_item_params),
        :rasch_map =>
            (:prior, :method, :return_scores, :max_iter, :return_item_params),
        :rasch_2pl => (
            :method,
            :return_scores,
            :max_iter,
            :return_item_params,
            :reg_discrimination,
        ),
        :rasch_2pl_map => (
            :prior,
            :method,
            :return_scores,
            :max_iter,
            :return_item_params,
            :reg_discrimination,
        ),
        :rasch_3pl => (
            :method,
            :return_scores,
            :max_iter,
            :fix_guessing,
            :return_item_params,
            :reg_discrimination,
            :reg_guessing,
            :guessing_upper,
        ),
        :rasch_3pl_map => (
            :prior,
            :method,
            :return_scores,
            :max_iter,
            :fix_guessing,
            :return_item_params,
            :reg_discrimination,
            :reg_guessing,
            :guessing_upper,
        ),
        :rasch_mml => (
            :method,
            :return_scores,
            :max_iter,
            :em_iter,
            :n_quadrature,
            :return_item_params,
        ),
        :rasch_mml_credible =>
            (:quantile, :method, :return_scores, :max_iter, :em_iter, :n_quadrature),
        :dynamic_irt => (
            :variant,
            :method,
            :return_scores,
            :max_iter,
            :return_item_params,
            :time_points,
            :score_target,
            :slope_reg,
            :state_reg,
            :assume_time_axis,
        ),
        :mirt => (
            :n_factors,
            :model,
            :method,
            :return_scores,
            :max_iter,
            :em_iter,
            :n_quadrature,
            :fix_guessing,
            :reg_discrimination,
            :reg_guessing,
            :guessing_upper,
            :tol,
            :return_item_params,
        ),
        :pagerank => (:damping, :max_iter, :tol, :method, :return_scores, :teleport),
        :spectral => (:max_iter, :tol, :method, :return_scores),
        :alpharank =>
            (:alpha, :population_size, :max_iter, :tol, :method, :return_scores),
        :nash => (
            :n_iter,
            :temperature,
            :solver,
            :score_type,
            :return_equilibrium,
            :method,
            :return_scores,
        ),
        :rank_centrality => (
            :method,
            :return_scores,
            :tie_handling,
            :smoothing,
            :teleport,
            :max_iter,
            :tol,
        ),
        :serial_rank => (:comparison, :method, :return_scores),
        :hodge_rank => (
            :pairwise_stat,
            :weight_method,
            :epsilon,
            :method,
            :return_scores,
            :return_diagnostics,
        ),
        :plackett_luce => (:method, :return_scores, :max_iter, :tol),
        :plackett_luce_map => (:prior, :method, :return_scores, :max_iter),
        :davidson_luce => (:method, :return_scores, :max_iter, :max_tie_order),
        :davidson_luce_map =>
            (:prior, :method, :return_scores, :max_iter, :max_tie_order),
        :bradley_terry_luce => (:method, :return_scores, :max_iter),
        :bradley_terry_luce_map => (:prior, :method, :return_scores, :max_iter),
    )
    python_required_arity = Dict(
        :pass_at_k => 2,
        :pass_hat_k => 2,
        :g_pass_at_k_tau => 3,
        :mg_pass_at_k => 2,
    )

    @test length(python_optional_parameters) == 51
    for (name, option_names) in python_optional_parameters
        fn = getfield(Scorio.Rank, name)
        required_arity = get(python_required_arity, name, 1)
        reflected = Dict(method.nargs - 1 => method for method in methods(fn))
        expected_arities = required_arity:(required_arity + length(option_names))

        @test Set(keys(reflected)) == Set(expected_arities)
        for consumed in 0:length(option_names)
            method = reflected[required_arity + consumed]
            @test Tuple(Base.kwarg_decl(method)) == option_names[(consumed + 1):end]
        end
    end

    # Exercise every exported function with all Python-optional parameters in
    # positional order.  Values that shorten optimizers preserve the original
    # public-API smoke-test runtime.
    function_calls = Dict{Symbol,Function}(
        :avg => () -> Scorio.Rank.avg(R_small, "competition", true),
        :bayes =>
            () -> Scorio.Rank.bayes(R_multi, w, R0_shared, nothing, "competition", true),
        :pass_at_k => () -> Scorio.Rank.pass_at_k(R_small, 2, "competition", true),
        :pass_hat_k => () -> Scorio.Rank.pass_hat_k(R_small, 2, "competition", true),
        :g_pass_at_k_tau =>
            () -> Scorio.Rank.g_pass_at_k_tau(R_small, 2, 0.7, "competition", true),
        :mg_pass_at_k => () -> Scorio.Rank.mg_pass_at_k(R_small, 2, "competition", true),
        :inverse_difficulty =>
            () -> Scorio.Rank.inverse_difficulty(R_small, "competition", true, (0.01, 0.99)),
        :elo => () -> Scorio.Rank.elo(
            R_small,
            32.0,
            1500.0,
            "correct_draw_only",
            "competition",
            true,
        ),
        :glicko => () -> Scorio.Rank.glicko(
            R_small,
            1500.0,
            350.0,
            0.0,
            350.0,
            "correct_draw_only",
            false,
            "competition",
            true,
        ),
        :trueskill => () -> Scorio.Rank.trueskill(
            R_small,
            25.0,
            25.0 / 3.0,
            25.0 / 6.0,
            25.0 / 300.0,
            "competition",
            true,
            "skip",
            0.0,
        ),
        :bradley_terry =>
            () -> Scorio.Rank.bradley_terry(finite_mle_R, "competition", true, 80),
        :bradley_terry_map =>
            () -> Scorio.Rank.bradley_terry_map(R_small, 1.0, "competition", true, 80),
        :bradley_terry_davidson =>
            () -> Scorio.Rank.bradley_terry_davidson(R_small, "competition", true, 80),
        :bradley_terry_davidson_map => () ->
            Scorio.Rank.bradley_terry_davidson_map(
                R_small,
                1.0,
                "competition",
                true,
                80,
            ),
        :rao_kupper =>
            () -> Scorio.Rank.rao_kupper(R_small, 1.1, "competition", true, 80),
        :rao_kupper_map =>
            () -> Scorio.Rank.rao_kupper_map(R_small, 1.1, 1.0, "competition", true, 80),
        :thompson =>
            () -> Scorio.Rank.thompson(R_small, 700, 1.0, 1.0, 7, "competition", true),
        :bayesian_mcmc => () ->
            Scorio.Rank.bayesian_mcmc(R_small, 400, 100, 1.0, 7, "competition", true),
        :borda => () -> Scorio.Rank.borda(R_small, "competition", true),
        :copeland => () -> Scorio.Rank.copeland(R_small, "competition", true),
        :win_rate => () -> Scorio.Rank.win_rate(R_small, "competition", true),
        :minimax => () ->
            Scorio.Rank.minimax(R_small, "margin", "half", "competition", true),
        :schulze => () -> Scorio.Rank.schulze(R_small, "half", "competition", true),
        :ranked_pairs => () ->
            Scorio.Rank.ranked_pairs(R_small, "margin", "half", "competition", true),
        :kemeny_young => () ->
            Scorio.Rank.kemeny_young(R_small, "half", "competition", true, 1.0, true),
        :nanson => () -> Scorio.Rank.nanson(R_small, "average", "competition", true),
        :baldwin => () -> Scorio.Rank.baldwin(R_small, "average", "competition", true),
        :majority_judgment =>
            () -> Scorio.Rank.majority_judgment(R_small, "competition", true),
        :rasch => () -> Scorio.Rank.rasch(R_small, "competition", true, 60, false),
        :rasch_map =>
            () -> Scorio.Rank.rasch_map(R_small, 1.0, "competition", true, 60, false),
        :rasch_2pl =>
            () -> Scorio.Rank.rasch_2pl(R_small, "competition", true, 300, false, 0.01),
        :rasch_2pl_map => () ->
            Scorio.Rank.rasch_2pl_map(
                R_small,
                1.0,
                "competition",
                true,
                300,
                false,
                0.01,
            ),
        :rasch_3pl => () -> Scorio.Rank.rasch_3pl(
            R_small,
            "competition",
            true,
            500,
            0.2,
            false,
            0.01,
            0.1,
            0.5,
        ),
        :rasch_3pl_map => () -> Scorio.Rank.rasch_3pl_map(
            R_small,
            1.0,
            "competition",
            true,
            500,
            0.2,
            false,
            0.01,
            0.1,
            0.5,
        ),
        :rasch_mml =>
            () -> Scorio.Rank.rasch_mml(R_small, "competition", true, 10, 6, 9, false),
        :rasch_mml_credible => () ->
            Scorio.Rank.rasch_mml_credible(R_small, 0.1, "competition", true, 10, 6, 9),
        :dynamic_irt => () -> Scorio.Rank.dynamic_irt(
            finite_mle_R,
            "linear",
            "competition",
            true,
            60,
            false,
            nothing,
            "final",
            0.01,
            1.0,
            false,
        ),
        :mirt => () -> Scorio.Rank.mirt(
            R_small,
            2,
            "2pl",
            "competition",
            true,
            30,
            10,
            7,
            nothing,
            0.01,
            0.1,
            0.5,
            1e-4,
            false,
        ),
        :pagerank =>
            () -> Scorio.Rank.pagerank(R_small, 0.85, 100, 1e-6, "competition", true, nothing),
        :spectral =>
            () -> Scorio.Rank.spectral(R_small, 10_000, 1e-12, "competition", true),
        :alpharank => () -> Scorio.Rank.alpharank(
            R_small,
            1.0,
            20,
            10_000,
            1e-12,
            "competition",
            true,
        ),
        :nash => () -> Scorio.Rank.nash(
            R_small,
            100,
            0.1,
            "lp",
            "vs_equilibrium",
            false,
            "competition",
            true,
        ),
        :rank_centrality => () -> Scorio.Rank.rank_centrality(
            R_small,
            "competition",
            true,
            "half",
            0.0,
            0.0,
            10_000,
            1e-12,
        ),
        :serial_rank =>
            () -> Scorio.Rank.serial_rank(R_small, "prob_diff", "competition", true),
        :hodge_rank => () -> Scorio.Rank.hodge_rank(
            R_small,
            "binary",
            "total",
            0.5,
            "competition",
            true,
            false,
        ),
        :plackett_luce =>
            () -> Scorio.Rank.plackett_luce(finite_mle_R, "competition", true, 80, 1e-8),
        :plackett_luce_map =>
            () -> Scorio.Rank.plackett_luce_map(R_small, 1.0, "competition", true, 80),
        :davidson_luce => () ->
            Scorio.Rank.davidson_luce(finite_mle_R, "competition", true, 80, nothing),
        :davidson_luce_map => () ->
            Scorio.Rank.davidson_luce_map(
                R_small,
                1.0,
                "competition",
                true,
                80,
                nothing,
            ),
        :bradley_terry_luce => () ->
            Scorio.Rank.bradley_terry_luce(finite_mle_R, "competition", true, 80),
        :bradley_terry_luce_map => () ->
            Scorio.Rank.bradley_terry_luce_map(R_small, 1.0, "competition", true, 80),
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

    @testset "Python-compatible prior constructors" begin
        # Python's ABC rejects Prior() because `penalty` is abstract; Julia's
        # abstract type rejects it at construction as well.
        @test_throws MethodError Scorio.Rank.Prior()

        gaussian_default = Scorio.Rank.GaussianPrior()
        gaussian_one = Scorio.Rank.GaussianPrior(2.0)
        gaussian_two = Scorio.Rank.GaussianPrior(2.0, 3.0)
        @test (gaussian_default.mean, gaussian_default.var) == (0.0, 1.0)
        @test (gaussian_one.mean, gaussian_one.var) == (2.0, 1.0)
        @test (gaussian_two.mean, gaussian_two.var) == (2.0, 3.0)

        laplace_default = Scorio.Rank.LaplacePrior()
        laplace_one = Scorio.Rank.LaplacePrior(2.0)
        laplace_two = Scorio.Rank.LaplacePrior(2.0, 3.0)
        @test (laplace_default.loc, laplace_default.scale) == (0.0, 1.0)
        @test (laplace_one.loc, laplace_one.scale) == (2.0, 1.0)
        @test (laplace_two.loc, laplace_two.scale) == (2.0, 3.0)

        cauchy_default = Scorio.Rank.CauchyPrior()
        cauchy_one = Scorio.Rank.CauchyPrior(2.0)
        cauchy_two = Scorio.Rank.CauchyPrior(2.0, 3.0)
        @test (cauchy_default.loc, cauchy_default.scale) == (0.0, 1.0)
        @test (cauchy_one.loc, cauchy_one.scale) == (2.0, 1.0)
        @test (cauchy_two.loc, cauchy_two.scale) == (2.0, 3.0)

        uniform = Scorio.Rank.UniformPrior()
        @test Scorio.penalty(uniform, [-1.0, 2.0]) == 0.0
        @test_throws MethodError Scorio.Rank.UniformPrior(1.0)

        custom = Scorio.Rank.CustomPrior(theta -> sum(abs2, theta))
        @test Scorio.penalty(custom, [-1.0, 2.0]) == 5.0
        @test_throws MethodError Scorio.Rank.CustomPrior()

        empirical_default = Scorio.Rank.EmpiricalPrior(R_small)
        empirical_two = Scorio.Rank.EmpiricalPrior(R_small, 2.0)
        empirical_three = Scorio.Rank.EmpiricalPrior(R_small, 2.0, 1e-4)
        @test (empirical_default.var, empirical_default.eps) == (1.0, 1e-6)
        @test (empirical_two.var, empirical_two.eps) == (2.0, 1e-6)
        @test (empirical_three.var, empirical_three.eps) == (2.0, 1e-4)
        @test_throws MethodError Scorio.Rank.EmpiricalPrior()
    end

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
