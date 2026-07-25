"""Listwise and setwise Luce-family ranking methods."""

function _validate_positive_float(name::AbstractString, value, minimum::Real)::Float64
    fvalue = Float64(value)
    minimum_f = Float64(minimum)
    if !isfinite(fvalue) || fvalue <= minimum_f
        error("$name must be a finite scalar > $minimum_f, got $value")
    end
    return fvalue
end

function _coerce_listwise_prior(prior)
    if prior isa Bool
        error("prior must be a Prior object or positive finite float")
    end
    if prior isa Real
        variance = _validate_positive_float("prior", prior, 0.0)
        return GaussianPrior(0.0, variance)
    end
    prior isa Prior ||
        error("prior must be a Prior object or float, got $(typeof(prior))")
    return prior
end

function _logaddexp(a::Real, b::Real)::Float64
    af = Float64(a)
    bf = Float64(b)
    if af == -Inf
        return bf
    end
    if bf == -Inf
        return af
    end
    m = max(af, bf)
    return m + log(exp(af - m) + exp(bf - m))
end

function _logsumexp(values)::Float64
    vals = Float64.(values)
    if isempty(vals)
        return -Inf
    end
    max_v = maximum(vals)
    if max_v == -Inf
        return -Inf
    end
    return max_v + log(sum(exp.(vals .- max_v)))
end

function _log_elementary_symmetric_sum(log_x, k::Integer)::Float64
    if k < 0
        error("k must be >= 0")
    end
    if k == 0
        return 0.0
    end

    lx = Float64.(log_x)
    n = length(lx)
    if k > n
        return -Inf
    end

    log_e = fill(-Inf, k + 1)
    log_e[1] = 0.0

    for i in 1:n
        upper = min(k, i)
        for j in upper:-1:1
            log_e[j + 1] = _logaddexp(log_e[j + 1], log_e[j] + lx[i])
        end
    end

    return log_e[k + 1]
end

function _extract_winners_losers_events(
    R::AbstractArray{<:Integer,3},
)::Vector{Tuple{Vector{Int},Vector{Int}}}
    L, M, N = size(R)
    events = Vector{Tuple{Vector{Int},Vector{Int}}}()

    for m in 1:M
        for n in 1:N
            outcomes = @view R[:, m, n]
            winners = Int.(findall(==(1), outcomes))
            if length(winners) == 0 || length(winners) == L
                continue
            end
            losers = Int.(findall(==(0), outcomes))
            push!(events, (winners, losers))
        end
    end

    return events
end

function _davidson_equivalence_statistics(
    events::Vector{Tuple{Vector{Int},Vector{Int}}},
    n_models::Integer,
    max_tie_order::Integer,
)::Matrix{Int}
    counts = zeros(Int, Int(n_models), Int(max_tie_order))
    for (winners, _) in events
        tie_order = length(winners)
        counts[winners, tie_order] .+= 1
    end
    return counts
end

function _mm_plackett_luce(
    wins::AbstractMatrix{<:Real};
    max_iter::Integer=500,
    tol::Real=1e-8,
)::Vector{Float64}
    L = size(wins, 1)
    W = vec(sum(wins; dims=2))
    total_wins = sum(W)

    if total_wins == 0.0
        return fill(1.0 / L, L)
    end
    if !is_strongly_connected(wins .> 0.0)
        error(
            "Plackett-Luce has no finite maximum-likelihood estimate because the directed win graph is not strongly connected; use plackett_luce_map for a regularized estimate.",
        )
    end

    pi = max.(W ./ total_wins, 1e-10)
    n_comparisons = wins .+ transpose(wins)

    converged = false
    for _ in 1:Int(max_iter)
        pi_old = copy(pi)

        for i in 1:L
            denom = 0.0
            for j in 1:L
                if i == j
                    continue
                end
                if n_comparisons[i, j] > 0.0
                    denom += n_comparisons[i, j] / (pi_old[i] + pi_old[j])
                end
            end

            if denom > 0.0
                pi[i] = W[i] / denom
            else
                pi[i] = pi_old[i]
            end
        end

        pi_sum = sum(pi)
        if pi_sum > 0.0
            pi ./= pi_sum
        end
        pi = max.(pi, 1e-10)

        if maximum(abs.(pi .- pi_old)) < tol
            converged = true
            break
        end
    end

    converged ||
        error("plackett_luce MM optimization did not converge within max_iter")

    return pi
end

function _estimate_pl_map(
    wins::AbstractMatrix{<:Real},
    prior::Prior;
    max_iter::Integer=500,
)::Vector{Float64}
    if prior isa UniformPrior
        return _mm_plackett_luce(wins; max_iter=max_iter)
    end
    L = size(wins, 1)

    function negative_log_posterior(log_pi::Vector{Float64})
        centered = log_pi .- (sum(log_pi) / L)
        nll = 0.0

        for i in 1:L
            for j in 1:L
                if i == j
                    continue
                end
                n_ij = wins[i, j]
                if n_ij > 0.0
                    nll -= n_ij * (centered[i] - _logaddexp(centered[i], centered[j]))
                end
            end
        end

        return Float64(nll + penalty(prior, centered))
    end

    total_wins = max.(vec(sum(wins; dims=2)), 1.0)
    log_pi_init = log.(total_wins ./ sum(total_wins))

    x = _minimize_or_error(
        negative_log_posterior,
        log_pi_init,
        Int(max_iter),
        "plackett_luce_map",
    )

    log_pi = x .- (sum(x) / L)
    return exp.(clamp.(log_pi, -30.0, 30.0))
end

function _log_denominator_davidson_luce(
    log_alpha::AbstractVector{<:Real},
    log_delta_params::AbstractVector{<:Real},
    comparison_set::AbstractVector{<:Integer},
    max_tie_order::Integer,
)::Float64
    items = Int.(comparison_set)
    if isempty(items)
        return -Inf
    end

    D = min(Int(max_tie_order), length(items))
    terms = Float64[]

    for t in 1:D
        log_delta_t = if t == 1
            0.0
        else
            idx = t - 1
            idx <= length(log_delta_params) ? Float64(log_delta_params[idx]) : 0.0
        end

        log_x = Float64.(log_alpha[items]) ./ Float64(t)
        log_e_t = _log_elementary_symmetric_sum(log_x, t)
        if log_e_t == -Inf
            continue
        end
        push!(terms, log_delta_t + log_e_t)
    end

    return _logsumexp(terms)
end

function _estimate_davidson_luce_ml(
    events::Vector{Tuple{Vector{Int},Vector{Int}}};
    n_models::Integer,
    max_tie_order::Integer,
    max_iter::Integer=500,
)
    L = Int(n_models)
    if isempty(events)
        return fill(1.0 / L, L), ones(Float64, max(Int(max_tie_order) - 1, 1))
    end

    directed_choices = falses(L, L)
    for (winners, losers) in events
        directed_choices[winners, losers] .= true
        directed_choices[winners, winners] .= true
    end
    for model in 1:L
        directed_choices[model, model] = false
    end
    if !is_strongly_connected(directed_choices)
        error(
            "Davidson-Luce has no finite maximum-likelihood strength estimate because its win/co-winner graph is not strongly connected; use davidson_luce_map with a proper prior.",
        )
    end

    comparison_set = collect(1:L)

    function negative_log_likelihood(params::Vector{Float64})
        log_alpha = params[1:L]
        log_delta_params = params[(L + 1):end]
        centered = log_alpha .- (sum(log_alpha) / L)

        nll = 0.0
        for (winners, _) in events
            t = length(winners)
            (t < 1 || t > max_tie_order) &&
                error("Observed winner-set size exceeds max_tie_order")

            log_delta_t = t == 1 ? 0.0 : Float64(log_delta_params[t - 1])
            log_numerator = log_delta_t + (sum(centered[winners]) / Float64(t))
            log_denom = _log_denominator_davidson_luce(
                centered,
                log_delta_params,
                comparison_set,
                max_tie_order,
            )

            nll -= (log_numerator - log_denom)
        end

        return Float64(nll)
    end

    log_alpha0 = zeros(Float64, L)
    log_delta0 = zeros(Float64, max(Int(max_tie_order) - 1, 0))
    params0 = vcat(log_alpha0, log_delta0)

    x = _minimize_or_error(
        negative_log_likelihood,
        params0,
        Int(max_iter),
        "davidson_luce",
    )

    log_alpha_hat = x[1:L] .- (sum(@view x[1:L]) / L)
    alpha = exp.(clamp.(log_alpha_hat, -30.0, 30.0))

    log_delta_hat = x[(L + 1):end]
    delta = isempty(log_delta_hat) ? [1.0] : exp.(log_delta_hat)

    return alpha, delta
end

function _estimate_davidson_luce_map(
    events::Vector{Tuple{Vector{Int},Vector{Int}}};
    n_models::Integer,
    prior::Prior,
    max_tie_order::Integer,
    max_iter::Integer=500,
)
    if prior isa UniformPrior
        return _estimate_davidson_luce_ml(
            events;
            n_models=n_models,
            max_tie_order=max_tie_order,
            max_iter=max_iter,
        )
    end
    L = Int(n_models)
    if isempty(events)
        return fill(1.0 / L, L), ones(Float64, max(Int(max_tie_order) - 1, 1))
    end

    comparison_set = collect(1:L)

    function negative_log_posterior(params::Vector{Float64})
        log_alpha = params[1:L]
        log_delta_params = params[(L + 1):end]
        centered = log_alpha .- (sum(log_alpha) / L)

        nll = 0.0
        for (winners, _) in events
            t = length(winners)
            (t < 1 || t > max_tie_order) &&
                error("Observed winner-set size exceeds max_tie_order")

            log_delta_t = t == 1 ? 0.0 : Float64(log_delta_params[t - 1])
            log_numerator = log_delta_t + (sum(centered[winners]) / Float64(t))
            log_denom = _log_denominator_davidson_luce(
                centered,
                log_delta_params,
                comparison_set,
                max_tie_order,
            )

            nll -= (log_numerator - log_denom)
        end

        return Float64(nll + penalty(prior, centered))
    end

    log_alpha0 = zeros(Float64, L)
    log_delta0 = zeros(Float64, max(Int(max_tie_order) - 1, 0))
    params0 = vcat(log_alpha0, log_delta0)

    x = _minimize_or_error(
        negative_log_posterior,
        params0,
        Int(max_iter),
        "davidson_luce_map",
    )

    log_alpha_hat = x[1:L] .- (sum(@view x[1:L]) / L)
    alpha = exp.(clamp.(log_alpha_hat, -30.0, 30.0))

    log_delta_hat = x[(L + 1):end]
    delta = isempty(log_delta_hat) ? [1.0] : exp.(log_delta_hat)

    return alpha, delta
end

function _estimate_btl_ml(
    events::Vector{Tuple{Vector{Int},Vector{Int}}};
    n_models::Integer,
    max_iter::Integer=500,
)::Vector{Float64}
    L = Int(n_models)
    if isempty(events)
        return fill(1.0 / L, L)
    end

    directed_choices = falses(L, L)
    for (winners, losers) in events
        directed_choices[winners, losers] .= true
    end
    if !is_strongly_connected(directed_choices)
        error(
            "Bradley-Terry-Luce has no finite maximum-likelihood estimate because the directed choice graph is not strongly connected; use bradley_terry_luce_map for a regularized estimate.",
        )
    end

    function negative_log_likelihood(log_pi::Vector{Float64})
        centered = log_pi .- (sum(log_pi) / L)
        nll = 0.0

        for (winners, losers) in events
            log_sum_losers = _logsumexp(centered[losers])
            nll -= sum(centered[winners])
            for w in winners
                nll += _logaddexp(centered[w], log_sum_losers)
            end
        end

        return Float64(nll)
    end

    log_pi0 = zeros(Float64, L)
    x = _minimize_or_error(
        negative_log_likelihood,
        log_pi0,
        Int(max_iter),
        "bradley_terry_luce",
    )

    log_pi_hat = x .- (sum(x) / L)
    return exp.(clamp.(log_pi_hat, -30.0, 30.0))
end

function _estimate_btl_map(
    events::Vector{Tuple{Vector{Int},Vector{Int}}};
    n_models::Integer,
    prior::Prior,
    max_iter::Integer=500,
)::Vector{Float64}
    if prior isa UniformPrior
        return _estimate_btl_ml(events; n_models=n_models, max_iter=max_iter)
    end
    L = Int(n_models)
    if isempty(events)
        return fill(1.0 / L, L)
    end

    function negative_log_posterior(log_pi::Vector{Float64})
        centered = log_pi .- (sum(log_pi) / L)
        nll = 0.0

        for (winners, losers) in events
            log_sum_losers = _logsumexp(centered[losers])
            nll -= sum(centered[winners])
            for w in winners
                nll += _logaddexp(centered[w], log_sum_losers)
            end
        end

        return Float64(nll + penalty(prior, centered))
    end

    log_pi0 = zeros(Float64, L)
    x = _minimize_or_error(
        negative_log_posterior,
        log_pi0,
        Int(max_iter),
        "bradley_terry_luce_map",
    )

    log_pi_hat = x .- (sum(x) / L)
    return exp.(clamp.(log_pi_hat, -30.0, 30.0))
end

"""
    plackett_luce(
        R;
        method="competition",
        return_scores=false,
        max_iter=10000,
        tol=1e-8,
    )

Rank models with Plackett-Luce ML on decisive pairwise-reduced outcomes.

This implementation uses pairwise decisive counts and Hunter's MM update:

```math
\\pi_i^{(k+1)} =
\\frac{\\sum_j W_{ij}}
{\\sum_{j\\ne i}(W_{ij}+W_{ji})/(\\pi_i^{(k)}+\\pi_j^{(k)})}
```

followed by normalization of ``\\pi``.

# References
Plackett, R. L. (1975). *The Analysis of Permutations*.
Hunter, D. R. (2004). MM algorithms for generalized Bradley-Terry models.
"""
function plackett_luce(
    R;
    method="competition",
    return_scores=false,
    max_iter=10000,
    tol=1e-8,
)
    Rv = validate_input(R)
    max_iter_i = _validate_positive_int("max_iter", max_iter)
    tol_f = _validate_positive_float("tol", tol, 0.0)

    wins = build_pairwise_wins(Rv)
    scores = _mm_plackett_luce(wins; max_iter=max_iter_i, tol=tol_f)
    scores = average_equivalent_scores(scores, Rv)
    ranking = rank_scores(scores)[string(method)]
    return return_scores ? (ranking, scores) : ranking
end

"""
    plackett_luce_map(
        R;
        prior=1.0,
        method="competition",
        return_scores=false,
        max_iter=500,
    )

Rank models with Plackett-Luce MAP using a prior on centered log-strengths.

With ``theta_i = \\log \\pi_i``:

```math
\\hat\\theta \\in \\arg\\min_\\theta
\\left[
-\\sum_{i\\ne j}W_{ij}\\left(\\theta_i-\\log(e^{\\theta_i}+e^{\\theta_j})\\right)
+ \\operatorname{penalty}(\\theta-\\bar\\theta)
\\right]
```
"""
function plackett_luce_map(
    R;
    prior=1.0,
    method="competition",
    return_scores=false,
    max_iter=500,
)
    Rv = validate_input(R)
    max_iter_i = _validate_positive_int("max_iter", max_iter)
    prior_obj = _coerce_listwise_prior(prior)

    wins = build_pairwise_wins(Rv)
    scores = _estimate_pl_map(wins, prior_obj; max_iter=max_iter_i)
    if _prior_is_exchangeable(prior_obj)
        scores = average_equivalent_scores(scores, Rv)
    end
    ranking = rank_scores(scores)[string(method)]
    return return_scores ? (ranking, scores) : ranking
end

"""
    davidson_luce(
        R;
        method="competition",
        return_scores=false,
        max_iter=500,
        max_tie_order=nothing,
)

Rank models with Davidson-Luce setwise tie likelihood (ML).

For event comparison set ``S=W\\cup L``, winner set size `t=|W|`,
``g_t(W)=\\left(\\prod_{i\\in W}\\alpha_i\\right)^{1/t}``, and tie-order parameters
`delta_t`:

```math
\\Pr(W\\mid S)=
\\frac{\\delta_t g_t(W)}
{\\sum_{t'=1}^{\\min(D,|S|)}\\delta_{t'}
\\sum_{|T|=t'} g_{t'}(T)}
```

# Reference
Firth, D., Kosmidis, I., & Turner, H. L. (2019).
Davidson-Luce model for multi-item choice with ties.
"""
function davidson_luce(
    R;
    method="competition",
    return_scores=false,
    max_iter=500,
    max_tie_order=nothing,
)
    Rv = validate_input(R)
    max_iter_i = _validate_positive_int("max_iter", max_iter)

    events = _extract_winners_losers_events(Rv)
    L = size(Rv, 1)

    if isnothing(max_tie_order)
        max_tie_order = max(L - 1, 1)
    end
    max_tie_order_i = _validate_positive_int("max_tie_order", max_tie_order)
    if max_tie_order_i > L
        error("max_tie_order must be <= number of models ($L)")
    end
    if any(length(winners) > max_tie_order_i for (winners, _) in events)
        error(
            "Observed winner-set size exceeds max_tie_order; increase max_tie_order so every observed event has positive model probability.",
        )
    end

    scores, _ = _estimate_davidson_luce_ml(
        events;
        n_models=L,
        max_tie_order=max_tie_order_i,
        max_iter=max_iter_i,
    )
    scores = average_equivalent_scores(
        scores,
        _davidson_equivalence_statistics(events, L, max_tie_order_i),
    )
    ranking = rank_scores(scores)[string(method)]
    return return_scores ? (ranking, scores) : ranking
end

"""
    davidson_luce_map(
        R;
        prior=1.0,
        method="competition",
        return_scores=false,
        max_iter=500,
        max_tie_order=nothing,
)

Rank models with Davidson-Luce MAP estimation.

```math
\\hat\\theta \\in \\arg\\min_\\theta
\\left[-\\log p(\\text{events}\\mid\\theta,\\delta) + \\operatorname{penalty}(\\theta)\\right]
```
"""
function davidson_luce_map(
    R;
    prior=1.0,
    method="competition",
    return_scores=false,
    max_iter=500,
    max_tie_order=nothing,
)
    Rv = validate_input(R)
    max_iter_i = _validate_positive_int("max_iter", max_iter)
    prior_obj = _coerce_listwise_prior(prior)

    L = size(Rv, 1)
    events = _extract_winners_losers_events(Rv)

    if isnothing(max_tie_order)
        max_tie_order = max(L - 1, 1)
    end
    max_tie_order_i = _validate_positive_int("max_tie_order", max_tie_order)
    if max_tie_order_i > L
        error("max_tie_order must be <= number of models ($L)")
    end
    if any(length(winners) > max_tie_order_i for (winners, _) in events)
        error(
            "Observed winner-set size exceeds max_tie_order; increase max_tie_order so every observed event has positive model probability.",
        )
    end

    scores, _ = _estimate_davidson_luce_map(
        events;
        n_models=L,
        prior=prior_obj,
        max_tie_order=max_tie_order_i,
        max_iter=max_iter_i,
    )
    if _prior_is_exchangeable(prior_obj)
        scores = average_equivalent_scores(
            scores,
            _davidson_equivalence_statistics(events, L, max_tie_order_i),
        )
    end
    ranking = rank_scores(scores)[string(method)]
    return return_scores ? (ranking, scores) : ranking
end

"""
    bradley_terry_luce(
        R;
        method="competition",
        return_scores=false,
        max_iter=500,
    )

Rank models with Bradley-Terry-Luce composite-likelihood ML from setwise
winner/loser events.

For each event `(W,L)`, each winner ``i\\in W`` is treated as a Luce choice from
``{i}\\cup L``, yielding composite log-likelihood:

```math
\\ell_{\\mathrm{comp}}(\\pi)
= \\sum_{(W,L)}\\sum_{i\\in W}
\\left[
\\log\\pi_i - \\log\\!\\left(\\pi_i+\\sum_{j\\in L}\\pi_j\\right)
\\right]
```
"""
function bradley_terry_luce(
    R;
    method="competition",
    return_scores=false,
    max_iter=500,
)
    Rv = validate_input(R)
    max_iter_i = _validate_positive_int("max_iter", max_iter)

    events = _extract_winners_losers_events(Rv)
    scores = _estimate_btl_ml(events; n_models=size(Rv, 1), max_iter=max_iter_i)
    scores = average_event_exchangeable_scores(scores, Rv)
    ranking = rank_scores(scores)[string(method)]
    return return_scores ? (ranking, scores) : ranking
end

"""
    bradley_terry_luce_map(
        R;
        prior=1.0,
        method="competition",
        return_scores=false,
        max_iter=500,
)

Rank models with Bradley-Terry-Luce composite-likelihood MAP estimation.

```math
\\hat\\theta \\in \\arg\\min_\\theta
\\left[-\\ell_{\\mathrm{comp}}(\\theta)+\\operatorname{penalty}(\\theta)\\right]
```
"""
function bradley_terry_luce_map(
    R;
    prior=1.0,
    method="competition",
    return_scores=false,
    max_iter=500,
)
    Rv = validate_input(R)
    max_iter_i = _validate_positive_int("max_iter", max_iter)
    prior_obj = _coerce_listwise_prior(prior)

    events = _extract_winners_losers_events(Rv)
    scores = _estimate_btl_map(
        events;
        n_models=size(Rv, 1),
        prior=prior_obj,
        max_iter=max_iter_i,
    )
    if _prior_is_exchangeable(prior_obj)
        scores = average_event_exchangeable_scores(scores, Rv)
    end
    ranking = rank_scores(scores)[string(method)]
    return return_scores ? (ranking, scores) : ranking
end
