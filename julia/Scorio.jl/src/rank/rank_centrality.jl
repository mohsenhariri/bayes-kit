"""Rank Centrality method."""

function _stationary_distribution_power(
    P::AbstractMatrix{<:Real};
    max_iter::Integer=10_000,
    tol::Real=1e-12,
)::Vector{Float64}
    n = size(P, 1)
    if n == 0
        return Float64[]
    end

    pi = fill(1.0 / n, n)
    Pf = Float64.(P)
    for _ in 1:Int(max_iter)
        pi_new = transpose(Pf) * pi
        s = sum(pi_new)
        if s <= 0.0
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
    rank_centrality(
        R;
        method="competition",
        return_scores=false,
        tie_handling="half",
        smoothing=0.0,
        teleport=0.0,
        max_iter=10000,
        tol=1e-12,
    )

Rank models with Rank Centrality using stationary distribution of a
pairwise-comparison Markov chain.

Let `d_max` be the maximum degree of the undirected comparison graph and
``\\hat P_{j\\succ i}`` the empirical probability that `j` beats `i`.
For ``i \\ne j``:

```math
P_{ij} = \\frac{1}{d_{\\max}}\\,\\hat P_{j\\succ i},
\\qquad
P_{ii} = 1 - \\sum_{j\\ne i} P_{ij}
```

Scores are stationary probabilities ``\\pi`` with:

```math
\\pi^\\top P = \\pi^\\top,\\qquad \\sum_i \\pi_i = 1
```

# Reference
Negahban, S., Oh, S., & Shah, D. (2017). Rank Centrality:
Ranking from Pairwise Comparisons. *Operations Research*.
"""
function rank_centrality(
    R;
    method="competition",
    return_scores=false,
    tie_handling="half",
    smoothing=0.0,
    teleport=0.0,
    max_iter=10000,
    tol=1e-12,
)
    Rv = validate_input(R)
    L = size(Rv, 1)

    tie_mode = string(tie_handling)
    if tie_mode ∉ ("ignore", "half")
        error("tie_handling must be \"ignore\" or \"half\"")
    end

    smoothing_f = Float64(smoothing)
    if !isfinite(smoothing_f) || smoothing_f < 0.0
        error("smoothing must be >= 0")
    end

    teleport_f = Float64(teleport)
    if !isfinite(teleport_f) || !(0.0 <= teleport_f < 1.0)
        error("teleport must be in [0, 1)")
    end

    max_iter_i = _validate_positive_int("max_iter", max_iter)
    tol_f = _validate_positive_float("tol", tol)

    wins = if tie_mode == "ignore"
        build_pairwise_wins(Rv)
    else
        w, t = build_pairwise_counts(Rv)
        w .+ 0.5 .* t
    end

    wins_s = wins .+ smoothing_f
    denom = wins_s .+ transpose(wins_s)

    eye_mask = falses(L, L)
    for i in 1:L
        eye_mask[i, i] = true
    end
    adj = (denom .> 0.0) .& .!eye_mask
    deg = vec(sum(adj; dims=2))
    d_max = isempty(deg) ? 0 : Int(maximum(deg))

    p_ji = zeros(Float64, L, L)
    mask = adj
    p_ji[mask] .= transpose(wins_s)[mask] ./ denom[mask]
    if teleport_f == 0.0 && smoothing_f == 0.0 && tie_mode == "ignore"
        if !is_strongly_connected(p_ji .> 0.0)
            error(
                "Rank Centrality requires strongly connected positive transition support when tie_handling='ignore'; use teleport>0, smoothing>0, or tie_handling='half'.",
            )
        end
    end

    if d_max == 0
        scores = fill(1.0 / L, L)
        ranking = rank_scores(scores)[string(method)]
        return return_scores ? (ranking, scores) : ranking
    end

    P = zeros(Float64, L, L)
    P[mask] .= p_ji[mask] ./ Float64(d_max)
    for i in 1:L
        P[i, i] = 1.0 - sum(@view P[i, :])
    end

    if teleport_f > 0.0
        P = (1.0 - teleport_f) .* P .+ teleport_f .* (ones(Float64, L, L) ./ L)
    end

    scores = _stationary_distribution_power(P; max_iter=max_iter_i, tol=tol_f)
    ranking = rank_scores(scores)[string(method)]
    return return_scores ? (ranking, scores) : ranking
end
