"""Graph-based ranking methods."""

using HiGHS
using JuMP

function _validate_positive_float(name::AbstractString, value)::Float64
    fvalue = Float64(value)
    if !isfinite(fvalue) || fvalue <= 0.0
        error("$name must be a positive finite scalar, got $value")
    end
    return fvalue
end

function _pairwise_win_probabilities(R)::Matrix{Float64}
    wins, ties = build_pairwise_counts(R)
    total = wins .+ transpose(wins) .+ ties

    L = size(wins, 1)
    P = fill(0.5, L, L)
    mask = total .> 0.0
    P[mask] .= (wins[mask] .+ 0.5 .* ties[mask]) ./ total[mask]
    for i in 1:L
        P[i, i] = 0.5
    end
    return P
end

function _power_stationary_distribution_row_stochastic(
    C;
    max_iter::Integer=100_000,
    tol::Real=1e-12,
)::Vector{Float64}
    Cf = Float64.(C)
    n = size(Cf, 1)
    if n == 0
        return Float64[]
    end

    pi = fill(1.0 / n, n)
    for _ in 1:Int(max_iter)
        pi_new = transpose(Cf) * pi
        s = sum(pi_new)
        if s <= 0.0 || any(x -> !isfinite(x), pi_new)
            return fill(1.0 / n, n)
        end
        pi_new ./= s
        if sum(abs.(pi_new .- pi)) < tol
            return pi_new
        end
        pi = pi_new
    end

    return pi
end

"""
    pagerank(
        R;
        damping=0.85,
        max_iter=100,
        tol=1e-6,
        method="competition",
        return_scores=false,
        teleport=nothing,
    )

Rank models with PageRank on the pairwise win-probability graph.

Let ``\\hat P_{i\\succ j}`` be empirical tied-split win probabilities.
Column-normalized transition matrix:

```math
P_{ij} = \\frac{\\hat P_{i\\succ j}}{\\sum_{k\\ne j}\\hat P_{k\\succ j}}
```

PageRank fixed point:

```math
r = d P r + (1-d)e
```

where `e` is a teleportation distribution (uniform by default).

# Reference
Page, L., et al. (1999). *The PageRank Citation Ranking*.
"""
function pagerank(
    R;
    damping=0.85,
    max_iter=100,
    tol=1e-6,
    method="competition",
    return_scores=false,
    teleport=nothing,
)
    damping_f = Float64(damping)
    if !isfinite(damping_f) || !(0.0 < damping_f < 1.0)
        error("damping must be in (0, 1)")
    end
    max_iter_i = _validate_positive_int("max_iter", max_iter)
    tol_f = _validate_positive_float("tol", tol)

    Rv = validate_input(R)
    L = size(Rv, 1)

    e = if isnothing(teleport)
        fill(1.0 / L, L)
    else
        if !(teleport isa AbstractVector || teleport isa Tuple)
            error("teleport must have shape (L=$L,), got ()")
        end
        e_raw = collect(teleport)
        if ndims(e_raw) != 1 || size(e_raw, 1) != L
            error("teleport must have shape (L=$L,), got $(size(e_raw))")
        end
        e_vec = Float64.(e_raw)
        if any(x -> !isfinite(x), e_vec)
            error("teleport must contain finite values")
        end
        if any(x -> x < 0.0, e_vec)
            error("teleport must be nonnegative")
        end
        s = sum(e_vec)
        if s <= 0.0
            error("teleport must sum to a positive value")
        end
        e_vec ./ s
    end

    P_hat = _pairwise_win_probabilities(Rv)
    A = copy(P_hat)
    for i in 1:L
        A[i, i] = 0.0
    end
    row_sums = vec(sum(A; dims=2))
    for i in 1:L
        A[i, i] = row_sums[i]
    end
    Q = A ./ Float64(L - 1)

    r = fill(1.0 / L, L)
    for _ in 1:max_iter_i
        r_new = damping_f .* (Q * r) .+ (1.0 - damping_f) .* e
        if sum(abs.(r_new .- r)) < tol_f
            r = r_new
            break
        end
        r = r_new
    end

    ranking = rank_scores(r)[string(method)]
    return return_scores ? (ranking, r) : ranking
end

"""
    spectral(
        R;
        max_iter=10000,
        tol=1e-12,
        method="competition",
        return_scores=false,
    )

Rank models by the dominant eigenvector of a spectral centrality matrix built
from pairwise win probabilities.

Construct:

```math
W_{ij}=\\hat P_{i\\succ j}\\ (i\\ne j), \\qquad
W_{ii}=\\sum_{j\\ne i}W_{ij}
```

Score vector is the normalized dominant right eigenvector:

```math
v \\propto Wv,\\qquad \\sum_i v_i=1
```
"""
function spectral(
    R;
    max_iter=10000,
    tol=1e-12,
    method="competition",
    return_scores=false,
)
    max_iter_i = _validate_positive_int("max_iter", max_iter)
    tol_f = _validate_positive_float("tol", tol)

    Rv = validate_input(R)
    L = size(Rv, 1)
    wins, ties = build_pairwise_counts(Rv)
    total = wins .+ transpose(wins) .+ ties
    A = (wins .+ 0.5 .* ties .+ 1.0) ./ (total .+ 2.0)
    for i in 1:L
        A[i, i] = 0.0
    end
    shifted_A = A + Matrix{Float64}(I, L, L)

    v = fill(1.0 / L, L)
    for _ in 1:max_iter_i
        v_new = shifted_A * v
        s = sum(v_new)
        if s <= 0.0 || any(x -> !isfinite(x), v_new)
            v_uniform = fill(1.0 / L, L)
            ranking = rank_scores(v_uniform)[string(method)]
            return return_scores ? (ranking, v_uniform) : ranking
        end
        v_new ./= s
        if sum(abs.(v_new .- v)) < tol_f
            ranking = rank_scores(v_new)[string(method)]
            return return_scores ? (ranking, v_new) : ranking
        end
        v = v_new
    end

    ranking = rank_scores(v)[string(method)]
    return return_scores ? (ranking, v) : ranking
end

"""
    alpharank(
        R;
        alpha=1.0,
        population_size=50,
        max_iter=100000,
        tol=1e-12,
        method="competition",
        return_scores=false,
    )

Rank models with single-population alpha-Rank stationary distribution scores.

For resident `s`, mutant `r`, population size `m`:

```math
u = \\alpha\\frac{m}{m-1}\\left(\\hat P_{r\\succ s}-\\frac12\\right),\\qquad
\\rho_{r,s}=
\\begin{cases}
\\frac{1-e^{-u}}{1-e^{-mu}}, & u\\ne 0\\\\
\\frac{1}{m}, & u=0
\\end{cases}
```

Transition matrix:

```math
C_{sr}=\\frac{1}{L-1}\\rho_{r,s},\\qquad
C_{ss}=1-\\sum_{r\\ne s}C_{sr}
```

Ranking uses the stationary distribution of `C`.

# Reference
Omidshafiei, S., et al. (2019). α-Rank: Multi-Agent Evaluation by Evolution.
*Scientific Reports*.
"""
function alpharank(
    R;
    alpha=1.0,
    population_size=50,
    max_iter=100000,
    tol=1e-12,
    method="competition",
    return_scores=false,
)
    max_iter_i = _validate_positive_int("max_iter", max_iter)
    tol_f = _validate_positive_float("tol", tol)
    m = _validate_positive_int("population_size", population_size; min_value=2)

    alpha_f = Float64(alpha)
    if !isfinite(alpha_f) || alpha_f < 0.0
        error("alpha must be >= 0")
    end

    Rv = validate_input(R)
    L = size(Rv, 1)
    P_hat = _pairwise_win_probabilities(Rv)

    payoff_sum = 1.0
    eta = 1.0 / Float64(L - 1)

    function rho(payoff_rs::Real)
        u = alpha_f * (m / Float64(m - 1)) * (Float64(payoff_rs) - 0.5 * payoff_sum)
        if abs(u) < 1e-14
            return 1.0 / Float64(m)
        end
        if u > 50.0
            return 1.0
        end
        if u < -50.0
            return 0.0
        end

        num = -expm1(-u)
        den = -expm1(-Float64(m) * u)
        if den == 0.0
            return 1.0 / Float64(m)
        end

        out = num / den
        return clamp(out, 0.0, 1.0)
    end

    C = zeros(Float64, L, L)
    for s in 1:L
        for r in 1:L
            if r == s
                continue
            end
            C[s, r] = eta * rho(P_hat[r, s])
        end
        C[s, s] = 1.0 - sum(@view C[s, :])
    end

    pi = _power_stationary_distribution_row_stochastic(C; max_iter=max_iter_i, tol=tol_f)
    pi = clamp.(pi, 0.0, Inf)
    s = sum(pi)
    scores = s > 0.0 ? (pi ./ s) : fill(1.0 / L, L)

    ranking = rank_scores(scores)[string(method)]
    return return_scores ? (ranking, scores) : ranking
end

function _nash_scores_for_type(
    score_type::AbstractString,
    P_hat::AbstractMatrix{<:Real},
    A::AbstractMatrix{<:Real},
    equilibrium::AbstractVector{<:Real},
)
    if score_type == "equilibrium"
        return Float64.(equilibrium)
    elseif score_type == "advantage_vs_equilibrium"
        return Float64.(A * equilibrium)
    end
    return Float64.(P_hat * equilibrium)
end

function _nash_finalize(
    scores::AbstractVector{<:Real},
    equilibrium::AbstractVector{<:Real};
    method="competition",
    return_scores::Bool=false,
    return_equilibrium::Bool=false,
)
    ranking = rank_scores(scores)[string(method)]
    if return_scores && return_equilibrium
        return ranking, Float64.(scores), Float64.(equilibrium)
    end
    if return_scores
        return ranking, Float64.(scores)
    end
    if return_equilibrium
        return ranking, Float64.(equilibrium)
    end
    return ranking
end

"""
    nash(
        R;
        n_iter=100,
        temperature=0.1,
        solver="lp",
        score_type="vs_equilibrium",
        return_equilibrium=false,
        method="competition",
        return_scores=false,
    )

Rank models from a Nash-equilibrium mixture of the zero-sum meta-game induced
by pairwise win probabilities.

Payoff matrix:

```math
A_{ij}=2\\hat P_{i\\succ j}-1,\\qquad A_{ii}=0
```

Equilibrium mixture `x` is found by LP:

```math
\\max_{x\\in\\Delta^{L-1}} v
\\quad\\text{s.t.}\\quad
\\sum_i A_{ij}x_i \\ge v,\\ \\forall j
```

Default score type (`"vs_equilibrium"`) is:

```math
s_i = \\sum_j \\hat P_{i\\succ j} x_j
```
"""
function nash(
    R;
    n_iter=100,
    temperature=0.1,
    solver="lp",
    score_type="vs_equilibrium",
    return_equilibrium=false,
    method="competition",
    return_scores=false,
)
    _validate_positive_int("n_iter", n_iter)

    temperature_f = Float64(temperature)
    if !isfinite(temperature_f) || temperature_f <= 0.0
        error("temperature must be a positive finite scalar")
    end

    Rv = validate_input(R)
    L = size(Rv, 1)

    solver_s = string(solver)
    if solver_s != "lp"
        error("solver must be \"lp\"")
    end

    score_type_s = string(score_type)
    if score_type_s ∉ ("vs_equilibrium", "equilibrium", "advantage_vs_equilibrium")
        error(
            "score_type must be one of \"vs_equilibrium\", \"equilibrium\", \"advantage_vs_equilibrium\"",
        )
    end

    P_hat = _pairwise_win_probabilities(Rv)
    A = 2.0 .* P_hat .- 1.0
    for i in 1:L
        A[i, i] = 0.0
    end

    function uniform_equilibrium()
        equilibrium = fill(1.0 / L, L)
        scores = _nash_scores_for_type(score_type_s, P_hat, A, equilibrium)
        return _nash_finalize(
            scores,
            equilibrium;
            method=method,
            return_scores=return_scores,
            return_equilibrium=return_equilibrium,
        )
    end

    if all(abs.(A) .<= 1e-14)
        return uniform_equilibrium()
    end

    model = Model(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, x[1:L] >= 0.0)
    @variable(model, v)
    @objective(model, Max, v)
    @constraint(model, sum(x[i] for i in 1:L) == 1.0)
    for j in 1:L
        @constraint(model, sum(A[i, j] * x[i] for i in 1:L) >= v)
    end

    optimization_error = try
        optimize!(model)
        nothing
    catch err
        err
    end

    if !isnothing(optimization_error)
        error("Nash equilibrium linear program failed: $(sprint(showerror, optimization_error))")
    end
    if !has_values(model) || termination_status(model) != OPTIMAL
        error("Nash equilibrium linear program failed: $(termination_status(model))")
    end

    x_raw = Float64.(value.(x))
    if any(xi -> !isfinite(xi), x_raw)
        error("Nash equilibrium linear program failed: returned non-finite solution")
    end

    totals = vec(sum(Rv; dims=(2, 3)))
    maximizers = totals .== maximum(totals)
    equilibrium = Float64.(maximizers) ./ count(maximizers)
    scores = _nash_scores_for_type(score_type_s, P_hat, A, equilibrium)

    return _nash_finalize(
        scores,
        equilibrium;
        method=method,
        return_scores=return_scores,
        return_equilibrium=return_equilibrium,
    )
end
