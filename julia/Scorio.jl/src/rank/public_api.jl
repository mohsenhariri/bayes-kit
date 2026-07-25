"""
Define a Rank namespace wrapper whose optional arguments may be supplied either
positionally (in Python signature order) or by keyword.

Julia's implementation functions intentionally use keywords for readability.
Python, however, declares every public Rank parameter as positional-or-keyword.
For a signature with `n` optional parameters this macro emits `n + 1` concrete
methods: the keyword-only form and one form for every positional prefix.  Each
method forwards to the implementation with keywords, so validation, defaults,
and numerical behavior continue to have a single source of truth.
"""
macro _rank_positional_api(name, required_expr, options_expr, target_expr=name)
    required =
        required_expr isa Expr && required_expr.head == :tuple ?
        required_expr.args : Any[required_expr]
    options =
        options_expr isa Expr && options_expr.head == :tuple ?
        options_expr.args : Any[options_expr]

    option_names = Any[]
    option_defaults = Any[]
    for option in options
        if !(option isa Expr && option.head == :(=) && option.args[1] isa Symbol)
            error("Rank positional API options must be `name=default` expressions")
        end
        push!(option_names, option.args[1])
        push!(option_defaults, option.args[2])
    end

    definitions = Any[]
    for positional_count in 0:length(options)
        signature_args = Any[name]
        remaining_keywords = Any[
            Expr(:kw, option_names[i], option_defaults[i]) for
            i in (positional_count + 1):length(options)
        ]
        isempty(remaining_keywords) ||
            push!(signature_args, Expr(:parameters, remaining_keywords...))
        append!(signature_args, required)
        append!(signature_args, option_names[1:positional_count])

        forwarded_keywords = Any[
            Expr(:kw, option_names[i], option_names[i]) for i in eachindex(option_names)
        ]
        target_args = Any[target_expr]
        isempty(forwarded_keywords) ||
            push!(target_args, Expr(:parameters, forwarded_keywords...))
        append!(target_args, required)

        push!(
            definitions,
            Expr(
                :function,
                Expr(:call, signature_args...),
                Expr(:call, target_args...),
            ),
        )
    end

    return esc(Expr(:block, definitions...))
end

# Eval-ranking implementations need private targets because their public names
# also exist in `Scorio.Eval` at the parent-module level.
function _rank_bayes_api_target(
    R;
    w=nothing,
    R0=nothing,
    quantile=nothing,
    method="competition",
    return_scores=false,
)
    return Scorio._rank_bayes(
        R,
        w;
        R0=R0,
        quantile=quantile,
        method=method,
        return_scores=return_scores,
    )
end

@doc (@doc Scorio._rank_bayes) _rank_bayes_api_target

@_rank_positional_api avg (R,) (method="competition", return_scores=false) Scorio._rank_avg
@_rank_positional_api bayes (R,) (
    w=nothing,
    R0=nothing,
    quantile=nothing,
    method="competition",
    return_scores=false,
) _rank_bayes_api_target
@_rank_positional_api pass_at_k (R, k) (
    method="competition",
    return_scores=false,
) Scorio._rank_pass_at_k
@_rank_positional_api pass_hat_k (R, k) (
    method="competition",
    return_scores=false,
) Scorio._rank_pass_hat_k
@_rank_positional_api g_pass_at_k_tau (R, k, tau) (
    method="competition",
    return_scores=false,
) Scorio._rank_g_pass_at_k_tau
@_rank_positional_api mg_pass_at_k (R, k) (
    method="competition",
    return_scores=false,
) Scorio._rank_mg_pass_at_k

# These six names collide with scalar Eval methods at the package root, so the
# manual documents their Rank-local wrappers. Reuse the implementation docs.
@doc (@doc Scorio._rank_avg) avg
@doc (@doc Scorio._rank_bayes) bayes
@doc (@doc Scorio._rank_pass_at_k) pass_at_k
@doc (@doc Scorio._rank_pass_hat_k) pass_hat_k
@doc (@doc Scorio._rank_g_pass_at_k_tau) g_pass_at_k_tau
@doc (@doc Scorio._rank_mg_pass_at_k) mg_pass_at_k

@_rank_positional_api inverse_difficulty (R,) (
    method="competition",
    return_scores=false,
    clip_range=(0.01, 0.99),
) Scorio.inverse_difficulty

@_rank_positional_api elo (R,) (
    K=32.0,
    initial_rating=1500.0,
    tie_handling="correct_draw_only",
    method="competition",
    return_scores=false,
) Scorio.elo
@_rank_positional_api glicko (R,) (
    initial_rating=1500.0,
    initial_rd=350.0,
    c=0.0,
    rd_max=350.0,
    tie_handling="correct_draw_only",
    return_deviation=false,
    method="competition",
    return_scores=false,
) Scorio.glicko
@_rank_positional_api trueskill (R,) (
    mu_initial=25.0,
    sigma_initial=25.0 / 3.0,
    beta=25.0 / 6.0,
    tau=25.0 / 300.0,
    method="competition",
    return_scores=false,
    tie_handling="skip",
    draw_margin=0.0,
) Scorio.trueskill

@_rank_positional_api bradley_terry (R,) (
    method="competition",
    return_scores=false,
    max_iter=500,
) Scorio.bradley_terry
@_rank_positional_api bradley_terry_map (R,) (
    prior=1.0,
    method="competition",
    return_scores=false,
    max_iter=500,
) Scorio.bradley_terry_map
@_rank_positional_api bradley_terry_davidson (R,) (
    method="competition",
    return_scores=false,
    max_iter=500,
) Scorio.bradley_terry_davidson
@_rank_positional_api bradley_terry_davidson_map (R,) (
    prior=1.0,
    method="competition",
    return_scores=false,
    max_iter=500,
) Scorio.bradley_terry_davidson_map
@_rank_positional_api rao_kupper (R,) (
    tie_strength=1.1,
    method="competition",
    return_scores=false,
    max_iter=500,
) Scorio.rao_kupper
@_rank_positional_api rao_kupper_map (R,) (
    tie_strength=1.1,
    prior=1.0,
    method="competition",
    return_scores=false,
    max_iter=500,
) Scorio.rao_kupper_map

@_rank_positional_api thompson (R,) (
    n_samples=10_000,
    prior_alpha=1.0,
    prior_beta=1.0,
    seed=42,
    method="competition",
    return_scores=false,
) Scorio.thompson
@_rank_positional_api bayesian_mcmc (R,) (
    n_samples=5_000,
    burnin=1_000,
    prior_var=1.0,
    seed=42,
    method="competition",
    return_scores=false,
) Scorio.bayesian_mcmc

@_rank_positional_api borda (R,) (method="competition", return_scores=false) Scorio.borda
@_rank_positional_api copeland (R,) (method="competition", return_scores=false) Scorio.copeland
@_rank_positional_api win_rate (R,) (method="competition", return_scores=false) Scorio.win_rate
@_rank_positional_api minimax (R,) (
    variant="margin",
    tie_policy="half",
    method="competition",
    return_scores=false,
) Scorio.minimax
@_rank_positional_api schulze (R,) (
    tie_policy="half",
    method="competition",
    return_scores=false,
) Scorio.schulze
@_rank_positional_api ranked_pairs (R,) (
    strength="margin",
    tie_policy="half",
    method="competition",
    return_scores=false,
) Scorio.ranked_pairs
@_rank_positional_api kemeny_young (R,) (
    tie_policy="half",
    method="competition",
    return_scores=false,
    time_limit=nothing,
    tie_aware=true,
) Scorio.kemeny_young
@_rank_positional_api nanson (R,) (
    rank_ties="average",
    method="competition",
    return_scores=false,
) Scorio.nanson
@_rank_positional_api baldwin (R,) (
    rank_ties="average",
    method="competition",
    return_scores=false,
) Scorio.baldwin
@_rank_positional_api majority_judgment (R,) (
    method="competition",
    return_scores=false,
) Scorio.majority_judgment

@_rank_positional_api rasch (R,) (
    method="competition",
    return_scores=false,
    max_iter=500,
    return_item_params=false,
) Scorio.rasch
@_rank_positional_api rasch_map (R,) (
    prior=1.0,
    method="competition",
    return_scores=false,
    max_iter=500,
    return_item_params=false,
) Scorio.rasch_map
@_rank_positional_api rasch_2pl (R,) (
    method="competition",
    return_scores=false,
    max_iter=500,
    return_item_params=false,
    reg_discrimination=0.01,
) Scorio.rasch_2pl
@_rank_positional_api rasch_2pl_map (R,) (
    prior=1.0,
    method="competition",
    return_scores=false,
    max_iter=500,
    return_item_params=false,
    reg_discrimination=0.01,
) Scorio.rasch_2pl_map
@_rank_positional_api rasch_3pl (R,) (
    method="competition",
    return_scores=false,
    max_iter=500,
    fix_guessing=nothing,
    return_item_params=false,
    reg_discrimination=0.01,
    reg_guessing=0.1,
    guessing_upper=0.5,
) Scorio.rasch_3pl
@_rank_positional_api rasch_3pl_map (R,) (
    prior=1.0,
    method="competition",
    return_scores=false,
    max_iter=500,
    fix_guessing=nothing,
    return_item_params=false,
    reg_discrimination=0.01,
    reg_guessing=0.1,
    guessing_upper=0.5,
) Scorio.rasch_3pl_map
@_rank_positional_api rasch_mml (R,) (
    method="competition",
    return_scores=false,
    max_iter=100,
    em_iter=20,
    n_quadrature=21,
    return_item_params=false,
) Scorio.rasch_mml
@_rank_positional_api rasch_mml_credible (R,) (
    quantile=0.05,
    method="competition",
    return_scores=false,
    max_iter=100,
    em_iter=20,
    n_quadrature=21,
) Scorio.rasch_mml_credible
@_rank_positional_api dynamic_irt (R,) (
    variant="linear",
    method="competition",
    return_scores=false,
    max_iter=500,
    return_item_params=false,
    time_points=nothing,
    score_target="final",
    slope_reg=0.01,
    state_reg=1.0,
    assume_time_axis=false,
) Scorio.dynamic_irt
@_rank_positional_api mirt (R,) (
    n_factors=2,
    model="2pl",
    method="competition",
    return_scores=false,
    max_iter=50,
    em_iter=100,
    n_quadrature=15,
    fix_guessing=nothing,
    reg_discrimination=0.01,
    reg_guessing=0.1,
    guessing_upper=0.5,
    tol=1e-4,
    return_item_params=false,
) Scorio.mirt

@_rank_positional_api pagerank (R,) (
    damping=0.85,
    max_iter=100,
    tol=1e-6,
    method="competition",
    return_scores=false,
    teleport=nothing,
) Scorio.pagerank
@_rank_positional_api spectral (R,) (
    max_iter=10_000,
    tol=1e-12,
    method="competition",
    return_scores=false,
) Scorio.spectral
@_rank_positional_api alpharank (R,) (
    alpha=1.0,
    population_size=50,
    max_iter=100_000,
    tol=1e-12,
    method="competition",
    return_scores=false,
) Scorio.alpharank
@_rank_positional_api nash (R,) (
    n_iter=100,
    temperature=0.1,
    solver="lp",
    score_type="vs_equilibrium",
    return_equilibrium=false,
    method="competition",
    return_scores=false,
) Scorio.nash
@_rank_positional_api rank_centrality (R,) (
    method="competition",
    return_scores=false,
    tie_handling="half",
    smoothing=0.0,
    teleport=0.0,
    max_iter=10_000,
    tol=1e-12,
) Scorio.rank_centrality
@_rank_positional_api serial_rank (R,) (
    comparison="prob_diff",
    method="competition",
    return_scores=false,
) Scorio.serial_rank
@_rank_positional_api hodge_rank (R,) (
    pairwise_stat="binary",
    weight_method="total",
    epsilon=0.5,
    method="competition",
    return_scores=false,
    return_diagnostics=false,
) Scorio.hodge_rank

@_rank_positional_api plackett_luce (R,) (
    method="competition",
    return_scores=false,
    max_iter=10_000,
    tol=1e-8,
) Scorio.plackett_luce
@_rank_positional_api plackett_luce_map (R,) (
    prior=1.0,
    method="competition",
    return_scores=false,
    max_iter=500,
) Scorio.plackett_luce_map
@_rank_positional_api davidson_luce (R,) (
    method="competition",
    return_scores=false,
    max_iter=500,
    max_tie_order=nothing,
) Scorio.davidson_luce
@_rank_positional_api davidson_luce_map (R,) (
    prior=1.0,
    method="competition",
    return_scores=false,
    max_iter=500,
    max_tie_order=nothing,
) Scorio.davidson_luce_map
@_rank_positional_api bradley_terry_luce (R,) (
    method="competition",
    return_scores=false,
    max_iter=500,
) Scorio.bradley_terry_luce
@_rank_positional_api bradley_terry_luce_map (R,) (
    prior=1.0,
    method="competition",
    return_scores=false,
    max_iter=500,
) Scorio.bradley_terry_luce_map
