"""Bradley-Terry family ranking methods."""

function _validate_max_iter(max_iter)
    if max_iter isa Bool || !(max_iter isa Integer)
        error("max_iter must be an integer, got $(typeof(max_iter))")
    end
    if max_iter <= 0
        error("max_iter must be > 0, got $max_iter")
    end
    return Int(max_iter)
end

function _validate_tie_strength(tie_strength)
    if tie_strength isa Bool || !(tie_strength isa Real)
        error(
            "tie_strength must be a finite scalar >= 1.0, got $(typeof(tie_strength))",
        )
    end
    kappa = Float64(tie_strength)
    if !isfinite(kappa)
        error("tie_strength must be finite.")
    end
    if kappa < 1.0
        error("tie_strength must be >= 1.0 for Rao-Kupper")
    end
    return kappa
end

function _coerce_prior(prior)
    if prior isa Bool
        error("prior must be a Prior object or positive finite float")
    end

    if prior isa Real
        var = Float64(prior)
        if !isfinite(var) || var <= 0.0
            error("prior must be a positive finite scalar variance")
        end
        return GaussianPrior(0.0, var)
    end

    if !(prior isa Prior)
        error("prior must be a Prior object or float, got $(typeof(prior))")
    end

    return prior
end

_prior_is_exchangeable(prior::Prior) =
    typeof(prior) in (GaussianPrior, LaplacePrior, CauchyPrior, UniformPrior)

function _finite_diff_gradient(
    f,
    x::Vector{Float64},
    fx::Float64=Float64(f(x)),
)
    # scipy.optimize.minimize(..., method="L-BFGS-B", jac=nothing) uses an
    # absolute forward-difference step of 1e-8.  Use the representable step
    # after floating-point addition, matching scipy.optimize._numdiff.
    g = zeros(Float64, length(x))
    for k in eachindex(x)
        stepped = copy(x)
        stepped[k] += 1e-8
        step = stepped[k] - x[k]
        if step == 0.0
            sign = x[k] >= 0.0 ? 1.0 : -1.0
            fallback = sqrt(eps(Float64)) * sign * max(1.0, abs(x[k]))
            stepped[k] = x[k] + fallback
            step = stepped[k] - x[k]
        end
        g[k] = (Float64(f(stepped)) - fx) / step
    end
    return g
end

function _minimize_objective(f, x0; max_iter::Int=500, return_status::Bool=false)
    x = Float64.(x0)
    fx = Float64(f(x))
    if !isfinite(fx)
        return return_status ? (x, false) : x
    end

    first_evaluation = true
    function objective_and_gradient(candidate)
        value = if first_evaluation
            first_evaluation = false
            fx
        else
            Float64(f(candidate))
        end
        gradient = _finite_diff_gradient(f, candidate, value)
        return value, gradient
    end

    dimension = length(x)
    optimizer = LBFGSB.L_BFGS_B(dimension, 10)
    bounds = zeros(Float64, 3, dimension)
    # LBFGSB counts function-and-gradient requests, whereas SciPy's maxfun
    # counts the objective call plus one call per forward-difference column.
    max_fg_requests = fld(15000, dimension + 1) + 1
    _, solution = optimizer(
        objective_and_gradient,
        x,
        bounds;
        m=10,
        factr=1e7,
        pgtol=1e-5,
        iprint=-1,
        maxfun=max_fg_requests,
        maxiter=max_iter,
    )

    task = rstrip(String(optimizer.task))
    converged = startswith(task, "CONVERGENCE")
    return return_status ? (solution, converged) : solution
end

function _minimize_or_error(f, x0, max_iter::Int, model_name::AbstractString)
    x, converged =
        _minimize_objective(f, x0; max_iter=max_iter, return_status=true)
    converged || error("$model_name optimization failed: iteration limit reached")
    return x
end

function _estimate_bt_ml(wins; max_iter::Int=500)
    n = size(wins, 1)

    if sum(wins) <= 0.0
        return ones(Float64, n)
    end
    if !is_strongly_connected(wins .> 0.0)
        error(
            "Bradley-Terry has no finite maximum-likelihood estimate because the directed win graph is not strongly connected; use bradley_terry_map for a regularized estimate.",
        )
    end

    function negative_log_likelihood(log_pi)
        centered = log_pi .- (sum(log_pi) / n)
        capped = clamp.(centered, -30.0, 30.0)
        pi = exp.(capped)

        nll = 0.0
        for i in 1:n
            for j in 1:n
                if i == j
                    continue
                end
                n_ij = wins[i, j]
                if n_ij > 0
                    nll -= n_ij * (capped[i] - log(pi[i] + pi[j]))
                end
            end
        end
        return Float64(nll)
    end

    total_wins = max.(vec(sum(wins, dims=2)), 1.0)
    log_pi_init = log.(total_wins ./ sum(total_wins))

    x = _minimize_or_error(negative_log_likelihood, log_pi_init, max_iter, "bradley_terry")
    log_pi = x .- (sum(x) / n)
    return exp.(clamp.(log_pi, -30.0, 30.0))
end

function _estimate_bt_map(wins, prior::Prior; max_iter::Int=500)
    if prior isa UniformPrior
        return _estimate_bt_ml(wins; max_iter=max_iter)
    end
    n = size(wins, 1)
    no_decisive_outcomes = sum(wins) <= 0.0

    if no_decisive_outcomes && prior isa GaussianPrior && prior.mean == 0.0
        return ones(Float64, n)
    end

    function negative_log_posterior(log_pi)
        centered = log_pi .- (sum(log_pi) / n)
        capped = clamp.(centered, -30.0, 30.0)
        pi = exp.(capped)

        nll = 0.0
        for i in 1:n
            for j in 1:n
                if i == j
                    continue
                end
                n_ij = wins[i, j]
                if n_ij > 0
                    nll -= n_ij * (capped[i] - log(pi[i] + pi[j]))
                end
            end
        end

        return Float64(nll + penalty(prior, centered))
    end

    total_wins = max.(vec(sum(wins, dims=2)), 1.0)
    log_pi_init = log.(total_wins ./ sum(total_wins))

    x = _minimize_or_error(
        negative_log_posterior,
        log_pi_init,
        max_iter,
        "bradley_terry_map",
    )
    log_pi = x .- (sum(x) / n)
    scores = exp.(clamp.(log_pi, -30.0, 30.0))

    if no_decisive_outcomes && (maximum(scores) - minimum(scores) <= 1e-5)
        return ones(Float64, n)
    end

    return scores
end

function _estimate_bt_davidson(wins, ties; max_iter::Int=500)
    n = size(wins, 1)
    eps = 1e-10

    if sum(wins) <= 0.0
        return ones(Float64, n)
    end
    existence_graph = (wins .> 0.0) .| (ties .> 0.0)
    if !is_strongly_connected(existence_graph)
        error(
            "Bradley-Terry-Davidson has no finite maximum-likelihood strength estimate because its win/tie graph is not strongly connected; use bradley_terry_davidson_map with a proper prior.",
        )
    end

    function negative_log_likelihood(params)
        log_pi = params[1:n]
        log_theta = params[n+1]

        centered = log_pi .- (sum(log_pi) / n)
        capped = clamp.(centered, -30.0, 30.0)
        pi = exp.(capped)
        theta = exp(clamp(log_theta, -10.0, 10.0))

        nll = 0.0
        for i in 1:n
            for j in (i + 1):n
                n_ij = wins[i, j]
                n_ji = wins[j, i]
                n_tie = ties[i, j]

                denom = max(pi[i] + pi[j] + theta * sqrt(pi[i] * pi[j]), eps)

                if n_ij > 0
                    nll -= n_ij * log(max(pi[i] / denom, eps))
                end
                if n_ji > 0
                    nll -= n_ji * log(max(pi[j] / denom, eps))
                end
                if n_tie > 0
                    tie_prob = theta * sqrt(pi[i] * pi[j]) / denom
                    nll -= n_tie * log(max(tie_prob, eps))
                end
            end
        end

        return Float64(nll)
    end

    total_wins = max.(vec(sum(wins, dims=2)), 1.0)
    log_pi_init = log.(total_wins ./ sum(total_wins))
    params_init = vcat(log_pi_init, 0.0)

    x = _minimize_or_error(
        negative_log_likelihood,
        params_init,
        max_iter,
        "bradley_terry_davidson",
    )
    log_pi = x[1:n] .- (sum(x[1:n]) / n)
    return exp.(clamp.(log_pi, -30.0, 30.0))
end

function _estimate_bt_davidson_map(wins, ties, prior::Prior; max_iter::Int=500)
    if prior isa UniformPrior
        return _estimate_bt_davidson(wins, ties; max_iter=max_iter)
    end
    n = size(wins, 1)
    eps = 1e-10
    no_decisive_outcomes = sum(wins) <= 0.0

    if no_decisive_outcomes && prior isa GaussianPrior && prior.mean == 0.0
        return ones(Float64, n)
    end

    function negative_log_posterior(params)
        log_pi = params[1:n]
        log_theta = params[n+1]

        centered = log_pi .- (sum(log_pi) / n)
        capped = clamp.(centered, -30.0, 30.0)
        pi = exp.(capped)
        theta = exp(clamp(log_theta, -10.0, 10.0))

        nll = 0.0
        for i in 1:n
            for j in (i + 1):n
                n_ij = wins[i, j]
                n_ji = wins[j, i]
                n_tie = ties[i, j]

                denom = max(pi[i] + pi[j] + theta * sqrt(pi[i] * pi[j]), eps)

                if n_ij > 0
                    nll -= n_ij * log(max(pi[i] / denom, eps))
                end
                if n_ji > 0
                    nll -= n_ji * log(max(pi[j] / denom, eps))
                end
                if n_tie > 0
                    tie_prob = theta * sqrt(pi[i] * pi[j]) / denom
                    nll -= n_tie * log(max(tie_prob, eps))
                end
            end
        end

        return Float64(nll + penalty(prior, centered))
    end

    total_wins = max.(vec(sum(wins, dims=2)), 1.0)
    log_pi_init = log.(total_wins ./ sum(total_wins))
    params_init = vcat(log_pi_init, 0.0)

    x = _minimize_or_error(
        negative_log_posterior,
        params_init,
        max_iter,
        "bradley_terry_davidson_map",
    )
    log_pi = x[1:n] .- (sum(x[1:n]) / n)
    scores = exp.(clamp.(log_pi, -30.0, 30.0))

    if no_decisive_outcomes && (maximum(scores) - minimum(scores) <= 1e-5)
        return ones(Float64, n)
    end

    return scores
end

function _estimate_rao_kupper_ml(wins, ties, tie_strength; max_iter::Int=500)
    n = size(wins, 1)
    eps = 1e-12
    kappa = _validate_tie_strength(tie_strength)

    total_ties = 0.0
    for i in 1:n
        for j in (i + 1):n
            total_ties += ties[i, j]
        end
    end
    if kappa == 1.0 && total_ties > 0.0
        error("tie_strength=1.0 implies no ties, but tie counts exist")
    end
    if sum(wins) <= 0.0
        return ones(Float64, n)
    end
    existence_graph = (wins .> 0.0) .| (ties .> 0.0)
    if !is_strongly_connected(existence_graph)
        error(
            "Rao-Kupper has no finite maximum-likelihood strength estimate because its win/tie graph is not strongly connected; use rao_kupper_map with a proper prior.",
        )
    end

    function negative_log_likelihood(log_pi)
        centered = log_pi .- (sum(log_pi) / n)
        pi = exp.(clamp.(centered, -30.0, 30.0))

        nll = 0.0
        for i in 1:n
            for j in (i + 1):n
                n_ij = wins[i, j]
                n_ji = wins[j, i]
                n_tie = ties[i, j]

                denom_ij = pi[i] + kappa * pi[j]
                denom_ji = kappa * pi[i] + pi[j]

                p_ij = max(pi[i] / denom_ij, eps)
                p_ji = max(pi[j] / denom_ji, eps)

                p_tie = 0.0
                if kappa > 1.0
                    p_tie =
                        (kappa^2 - 1.0) * pi[i] * pi[j] / (denom_ij * denom_ji)
                    p_tie = max(p_tie, eps)
                end

                if n_ij > 0
                    nll -= n_ij * log(p_ij)
                end
                if n_ji > 0
                    nll -= n_ji * log(p_ji)
                end
                if n_tie > 0
                    if kappa == 1.0
                        return Inf
                    end
                    nll -= n_tie * log(p_tie)
                end
            end
        end

        return Float64(nll)
    end

    log_pi0 = zeros(Float64, n)
    x = _minimize_or_error(negative_log_likelihood, log_pi0, max_iter, "rao_kupper")
    log_pi = x .- (sum(x) / n)
    return exp.(clamp.(log_pi, -30.0, 30.0))
end

function _estimate_rao_kupper_map(wins, ties, tie_strength, prior::Prior; max_iter::Int=500)
    if prior isa UniformPrior
        return _estimate_rao_kupper_ml(
            wins,
            ties,
            tie_strength;
            max_iter=max_iter,
        )
    end
    n = size(wins, 1)
    eps = 1e-12
    kappa = _validate_tie_strength(tie_strength)

    total_ties = 0.0
    for i in 1:n
        for j in (i + 1):n
            total_ties += ties[i, j]
        end
    end
    if kappa == 1.0 && total_ties > 0.0
        error("tie_strength=1.0 implies no ties, but tie counts exist")
    end

    no_decisive_outcomes = sum(wins) <= 0.0
    if no_decisive_outcomes && prior isa GaussianPrior && prior.mean == 0.0
        return ones(Float64, n)
    end

    function negative_log_posterior(log_pi)
        centered = log_pi .- (sum(log_pi) / n)
        pi = exp.(clamp.(centered, -30.0, 30.0))

        nll = 0.0
        for i in 1:n
            for j in (i + 1):n
                n_ij = wins[i, j]
                n_ji = wins[j, i]
                n_tie = ties[i, j]

                denom_ij = pi[i] + kappa * pi[j]
                denom_ji = kappa * pi[i] + pi[j]

                p_ij = max(pi[i] / denom_ij, eps)
                p_ji = max(pi[j] / denom_ji, eps)

                p_tie = 0.0
                if kappa > 1.0
                    p_tie =
                        (kappa^2 - 1.0) * pi[i] * pi[j] / (denom_ij * denom_ji)
                    p_tie = max(p_tie, eps)
                end

                if n_ij > 0
                    nll -= n_ij * log(p_ij)
                end
                if n_ji > 0
                    nll -= n_ji * log(p_ji)
                end
                if n_tie > 0
                    if kappa == 1.0
                        return Inf
                    end
                    nll -= n_tie * log(p_tie)
                end
            end
        end

        return Float64(nll + penalty(prior, centered))
    end

    log_pi0 = zeros(Float64, n)
    x = _minimize_or_error(
        negative_log_posterior,
        log_pi0,
        max_iter,
        "rao_kupper_map",
    )
    log_pi = x .- (sum(x) / n)
    scores = exp.(clamp.(log_pi, -30.0, 30.0))

    if no_decisive_outcomes && (maximum(scores) - minimum(scores) <= 1e-5)
        return ones(Float64, n)
    end

    return scores
end

"""
    bradley_terry(R; method="competition", return_scores=false, max_iter=500)

Rank models with Bradley-Terry maximum likelihood on decisive pairwise wins.

Let ``W_{ij}`` be decisive wins of model `i` over `j` and strengths `pi_i > 0`.

```math
\\Pr(i \\succ j) = \\frac{\\pi_i}{\\pi_i + \\pi_j}
```

```math
\\log p(W\\mid \\pi)
= \\sum_{i\\ne j} W_{ij}\\left[\\log \\pi_i - \\log(\\pi_i+\\pi_j)\\right]
```

# References
Bradley, R. A., & Terry, M. E. (1952). Rank Analysis of Incomplete
Block Designs: I. The Method of Paired Comparisons. *Biometrika*.
https://doi.org/10.1093/biomet/39.3-4.324
"""
function bradley_terry(R; method="competition", return_scores=false, max_iter=500)
    Rv = validate_input(R)
    max_iter_i = _validate_max_iter(max_iter)
    wins = build_pairwise_wins(Rv)
    scores = _estimate_bt_ml(wins; max_iter=max_iter_i)
    scores = average_equivalent_scores(scores, Rv)
    ranking = rank_scores(scores)[string(method)]
    return return_scores ? (ranking, scores) : ranking
end

"""
    bradley_terry_map(
        R;
        prior=1.0,
        method="competition",
        return_scores=false,
        max_iter=500,
    )

Rank models with Bradley-Terry MAP estimation using the given prior on centered
log-strengths.

With `theta_i = log(pi_i)`:

```math
\\hat\\theta
= \\arg\\min_{\\theta}
\\left[-\\log p(W\\mid \\theta) + \\operatorname{penalty}(\\theta)\\right]
```

```math
\\hat\\pi_i = \\exp(\\hat\\theta_i)
```

# Reference
Caron, F., & Doucet, A. (2012). Efficient Bayesian inference for generalized
Bradley-Terry models. https://doi.org/10.1080/10618600.2012.638220
"""
function bradley_terry_map(
    R;
    prior=1.0,
    method="competition",
    return_scores=false,
    max_iter=500,
)
    Rv = validate_input(R)
    max_iter_i = _validate_max_iter(max_iter)
    prior_obj = _coerce_prior(prior)
    wins = build_pairwise_wins(Rv)
    scores = _estimate_bt_map(wins, prior_obj; max_iter=max_iter_i)
    if _prior_is_exchangeable(prior_obj)
        scores = average_equivalent_scores(scores, Rv)
    end
    ranking = rank_scores(scores)[string(method)]
    return return_scores ? (ranking, scores) : ranking
end

"""
    bradley_terry_davidson(R; method="competition", return_scores=false, max_iter=500)

Rank models with Bradley-Terry-Davidson ML, incorporating explicit tie mass.

The Davidson tie extension introduces `nu > 0`:

```math
\\Pr(i\\succ j)
= \\frac{\\pi_i}{\\pi_i+\\pi_j+\\nu\\sqrt{\\pi_i\\pi_j}},
\\quad
\\Pr(i\\sim j)
= \\frac{\\nu\\sqrt{\\pi_i\\pi_j}}{\\pi_i+\\pi_j+\\nu\\sqrt{\\pi_i\\pi_j}}
```

# Reference
Davidson, R. R. (1970). On extending the Bradley-Terry model to accommodate
ties in paired comparison experiments. https://doi.org/10.1080/01621459.1970.10481082
"""
function bradley_terry_davidson(
    R;
    method="competition",
    return_scores=false,
    max_iter=500,
)
    Rv = validate_input(R)
    max_iter_i = _validate_max_iter(max_iter)
    wins, ties = build_pairwise_counts(Rv)
    scores = _estimate_bt_davidson(wins, ties; max_iter=max_iter_i)
    scores = average_equivalent_scores(scores, Rv)
    ranking = rank_scores(scores)[string(method)]
    return return_scores ? (ranking, scores) : ranking
end

"""
    bradley_terry_davidson_map(
        R;
        prior=1.0,
        method="competition",
        return_scores=false,
        max_iter=500,
)

Rank models with Bradley-Terry-Davidson MAP estimation.

```math
(\\hat\\theta,\\hat\\nu)
= \\arg\\min_{\\theta,\\nu>0}
\\left[-\\log p(W,T\\mid \\theta,\\nu) + \\operatorname{penalty}(\\theta)\\right]
```
"""
function bradley_terry_davidson_map(
    R;
    prior=1.0,
    method="competition",
    return_scores=false,
    max_iter=500,
)
    Rv = validate_input(R)
    max_iter_i = _validate_max_iter(max_iter)
    prior_obj = _coerce_prior(prior)
    wins, ties = build_pairwise_counts(Rv)
    scores = _estimate_bt_davidson_map(wins, ties, prior_obj; max_iter=max_iter_i)
    if _prior_is_exchangeable(prior_obj)
        scores = average_equivalent_scores(scores, Rv)
    end
    ranking = rank_scores(scores)[string(method)]
    return return_scores ? (ranking, scores) : ranking
end

"""
    rao_kupper(
        R;
        tie_strength=1.1,
        method="competition",
        return_scores=false,
        max_iter=500,
)

Rank models with the Rao-Kupper tie model (ML).

With fixed ``kappa \\ge 1``:

```math
\\Pr(i\\succ j)=\\frac{\\pi_i}{\\pi_i+\\kappa\\pi_j}, \\quad
\\Pr(j\\succ i)=\\frac{\\pi_j}{\\kappa\\pi_i+\\pi_j}
```

```math
\\Pr(i\\sim j)=
\\frac{(\\kappa^2-1)\\pi_i\\pi_j}
{(\\pi_i+\\kappa\\pi_j)(\\kappa\\pi_i+\\pi_j)}
```

# Reference
Rao, P. V., & Kupper, L. L. (1967). Ties in paired-comparison experiments:
A generalization of the Bradley-Terry model.
https://doi.org/10.1080/01621459.1967.10482901
"""
function rao_kupper(
    R;
    tie_strength=1.1,
    method="competition",
    return_scores=false,
    max_iter=500,
)
    Rv = validate_input(R)
    max_iter_i = _validate_max_iter(max_iter)
    kappa = _validate_tie_strength(tie_strength)
    wins, ties = build_pairwise_counts(Rv)
    scores = _estimate_rao_kupper_ml(wins, ties, kappa; max_iter=max_iter_i)
    scores = average_equivalent_scores(scores, Rv)
    ranking = rank_scores(scores)[string(method)]
    return return_scores ? (ranking, scores) : ranking
end

"""
    rao_kupper_map(
        R;
        tie_strength=1.1,
        prior=1.0,
        method="competition",
        return_scores=false,
        max_iter=500,
)

Rank models with the Rao-Kupper tie model under MAP estimation.

```math
\\hat\\theta
= \\arg\\min_{\\theta}
\\left[-\\log p(W,T\\mid \\theta,\\kappa) + \\operatorname{penalty}(\\theta)\\right]
```
"""
function rao_kupper_map(
    R;
    tie_strength=1.1,
    prior=1.0,
    method="competition",
    return_scores=false,
    max_iter=500,
)
    Rv = validate_input(R)
    max_iter_i = _validate_max_iter(max_iter)
    kappa = _validate_tie_strength(tie_strength)
    prior_obj = _coerce_prior(prior)
    wins, ties = build_pairwise_counts(Rv)
    scores = _estimate_rao_kupper_map(wins, ties, kappa, prior_obj; max_iter=max_iter_i)
    if _prior_is_exchangeable(prior_obj)
        scores = average_equivalent_scores(scores, Rv)
    end
    ranking = rank_scores(scores)[string(method)]
    return return_scores ? (ranking, scores) : ranking
end
