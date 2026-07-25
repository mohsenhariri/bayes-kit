"""Voting/social-choice ranking methods."""

using HiGHS
using JuMP

function _per_question_correct_counts(R::AbstractArray{<:Integer, 3})
    return dropdims(sum(R; dims=3); dims=3)
end

function _pairwise_preference_counts(k; tie_policy="half")
    tie_mode = string(tie_policy)
    if tie_mode ∉ ("ignore", "half")
        error("tie_policy must be one of {'ignore','half'}")
    end

    L, M = size(k)
    P = zeros(Float64, L, L)

    for i in 1:L
        for j in (i + 1):L
            i_over_j = 0.0
            j_over_i = 0.0
            for m in 1:M
                if k[i, m] > k[j, m]
                    i_over_j += 1.0
                elseif k[j, m] > k[i, m]
                    j_over_i += 1.0
                end
            end

            if tie_mode == "half"
                ties = Float64(M) - i_over_j - j_over_i
                i_over_j += 0.5 * ties
                j_over_i += 0.5 * ties
            end

            P[i, j] = i_over_j
            P[j, i] = j_over_i
        end
    end

    return P
end

function _rankdata_desc(values::AbstractVector; method="average")
    method_s = string(method)
    if method_s ∉ ("average", "min", "max", "dense", "ordinal")
        error("unknown method \"$method_s\"")
    end

    n = length(values)
    if n == 0
        return Float64[]
    end

    order = sortperm(values; rev=true, alg=MergeSort)
    sorted_values = values[order]

    ranks_sorted = zeros(Float64, n)
    i = 1
    dense_rank = 1.0
    while i <= n
        j = i
        while (j + 1) <= n && sorted_values[j + 1] == sorted_values[i]
            j += 1
        end

        if method_s == "ordinal"
            for t in i:j
                ranks_sorted[t] = Float64(t)
            end
        else
            rank_value = if method_s == "average"
                (Float64(i) + Float64(j)) / 2.0
            elseif method_s == "min"
                Float64(i)
            elseif method_s == "max"
                Float64(j)
            else
                dense_rank
            end

            for t in i:j
                ranks_sorted[t] = rank_value
            end
        end

        dense_rank += 1.0
        i = j + 1
    end

    ranks = zeros(Float64, n)
    for idx in 1:n
        ranks[order[idx]] = ranks_sorted[idx]
    end
    return ranks
end

function _topological_level_scores(adj::AbstractMatrix{Bool})
    if ndims(adj) != 2 || size(adj, 1) != size(adj, 2)
        error("adj must be square (L, L)")
    end

    L = size(adj, 1)
    remaining = trues(L)
    indeg = zeros(Int, L)
    for v in 1:L
        c = 0
        for u in 1:L
            if adj[u, v]
                c += 1
            end
        end
        indeg[v] = c
    end

    scores = zeros(Float64, L)
    current_score = Float64(L)

    while any(remaining)
        zero_indeg = [remaining[i] && indeg[i] == 0 for i in 1:L]
        if !any(zero_indeg)
            for i in 1:L
                if remaining[i]
                    scores[i] = current_score
                end
            end
            break
        end

        nodes = findall(zero_indeg)
        for u in nodes
            scores[u] = current_score
        end
        current_score -= 1.0

        for u in nodes
            remaining[u] = false
            for v in 1:L
                if adj[u, v]
                    indeg[v] -= 1
                end
            end
        end
    end

    return scores
end

"""
    borda(R; method="competition", return_scores=false)

Rank models with Borda count from per-question model orderings.

Let ``k_{lm} = \\sum_{n=1}^{N} R_{lmn}`` and ``r_{lm}`` be the descending
tie-averaged rank of model `l` on question `m` (rank 1 is best):

```math
s_l^{\\mathrm{Borda}} = \\sum_{m=1}^{M} (L - r_{lm})
```

# Reference
de Borda, J.-C. (1781/1784). *Mémoire sur les élections au scrutin*.
"""
function borda(R; method="competition", return_scores=false)
    Rv = validate_input(R)
    k = _per_question_correct_counts(Rv)
    L, M = size(k)

    scores = zeros(Float64, L)
    for m in 1:M
        r = _rankdata_desc(@view(k[:, m]); method="average")
        scores .+= Float64(L) .- r
    end

    ranking = rank_scores(scores)[string(method)]
    return return_scores ? (ranking, scores) : ranking
end

"""
    copeland(R; method="competition", return_scores=false)

Rank models by Copeland score over pairwise question-level majorities.

Let ``W^{(q)}_{ij}`` be the number of questions where ``k_{im} > k_{jm}``:

```math
s_i^{\\mathrm{Copeland}}
= \\sum_{j\\ne i}\\operatorname{sign}\\!\\left(W^{(q)}_{ij} - W^{(q)}_{ji}\\right)
```
"""
function copeland(R; method="competition", return_scores=false)
    Rv = validate_input(R)
    k = _per_question_correct_counts(Rv)
    L, M = size(k)

    scores = zeros(Float64, L)
    for i in 1:L
        for j in (i + 1):L
            i_over_j = 0.0
            j_over_i = 0.0
            for m in 1:M
                if k[i, m] > k[j, m]
                    i_over_j += 1.0
                elseif k[j, m] > k[i, m]
                    j_over_i += 1.0
                end
            end

            if i_over_j > j_over_i
                scores[i] += 1.0
                scores[j] -= 1.0
            elseif j_over_i > i_over_j
                scores[i] -= 1.0
                scores[j] += 1.0
            end
        end
    end

    ranking = rank_scores(scores)[string(method)]
    return return_scores ? (ranking, scores) : ranking
end

"""
    win_rate(R; method="competition", return_scores=false)

Rank models by aggregate pairwise win rate.

With the same ``W^{(q)}_{ij}`` counts:

```math
s_i^{\\mathrm{winrate}}
= \\frac{\\sum_{j\\ne i} W^{(q)}_{ij}}
{\\sum_{j\\ne i}\\left(W^{(q)}_{ij}+W^{(q)}_{ji}\\right)}
```

Models with no decisive pairwise outcomes receive score `0.5`.
"""
function win_rate(R; method="competition", return_scores=false)
    Rv = validate_input(R)
    k = _per_question_correct_counts(Rv)
    L, M = size(k)

    wins = zeros(Float64, L, L)
    for i in 1:L
        for j in (i + 1):L
            i_over_j = 0.0
            j_over_i = 0.0
            for m in 1:M
                if k[i, m] > k[j, m]
                    i_over_j += 1.0
                elseif k[j, m] > k[i, m]
                    j_over_i += 1.0
                end
            end
            wins[i, j] = i_over_j
            wins[j, i] = j_over_i
        end
    end

    total_wins = vec(sum(wins; dims=2))
    total_comparisons = vec(sum(wins; dims=2)) .+ vec(sum(wins; dims=1))

    scores = fill(0.5, L)
    for i in 1:L
        if total_comparisons[i] > 0.0
            scores[i] = total_wins[i] / total_comparisons[i]
        end
    end

    ranking = rank_scores(scores)[string(method)]
    return return_scores ? (ranking, scores) : ranking
end

"""
    minimax(
        R;
        variant="margin",
        tie_policy="half",
        method="competition",
        return_scores=false,
    )

Rank models with Minimax (Simpson-Kramer), using worst defeat strength.

Let ``P_{ij}`` be pairwise preference counts and ``\\Delta_{ij}=P_{ij}-P_{ji}``.

Margin variant:

```math
s_i^{\\mathrm{minimax}} = -\\max_{j\\ne i}\\max(0,\\Delta_{ji})
```

Winning-votes variant:

```math
s_i^{\\mathrm{wv}} = -\\max_{j:\\,P_{ji}>P_{ij}} P_{ji}
```
"""
function minimax(
    R;
    variant="margin",
    tie_policy="half",
    method="competition",
    return_scores=false,
)
    variant_s = string(variant)
    if variant_s ∉ ("margin", "winning_votes")
        error("variant must be one of {'margin','winning_votes'}")
    end

    Rv = validate_input(R)
    k = _per_question_correct_counts(Rv)
    P = _pairwise_preference_counts(k; tie_policy=tie_policy)
    margin = P .- transpose(P)

    L = size(P, 1)
    scores = zeros(Float64, L)
    for i in 1:L
        max_defeat = 0.0
        for j in 1:L
            if i == j
                continue
            end
            if margin[j, i] > 0.0
                defeat_strength = variant_s == "margin" ? margin[j, i] : P[j, i]
                if defeat_strength > max_defeat
                    max_defeat = defeat_strength
                end
            end
        end
        scores[i] = -max_defeat
    end

    ranking = rank_scores(scores)[string(method)]
    return return_scores ? (ranking, scores) : ranking
end

"""
    schulze(R; tie_policy="half", method="competition", return_scores=false)

Rank models using the Schulze strongest-path method.

Initialize:

```math
p_{ij} =
\\begin{cases}
P_{ij}, & P_{ij}>P_{ji} \\\\
0, & \\text{otherwise}
\\end{cases}
```

Then apply strongest-path closure:

```math
p_{jk} \\leftarrow \\max\\!\\left(p_{jk}, \\min(p_{ji}, p_{ik})\\right)
```

Model `i` beats `j` if ``p_{ij} > p_{ji}``.
"""
function schulze(R; tie_policy="half", method="competition", return_scores=false)
    Rv = validate_input(R)
    k = _per_question_correct_counts(Rv)
    P = _pairwise_preference_counts(k; tie_policy=tie_policy)
    L = size(P, 1)

    p = zeros(Float64, L, L)
    for i in 1:L
        for j in 1:L
            if i == j
                continue
            end
            if P[i, j] > P[j, i]
                p[i, j] = P[i, j]
            end
        end
    end

    for i in 1:L
        for j in 1:L
            if i == j
                continue
            end
            for k_ in 1:L
                if i == k_ || j == k_
                    continue
                end
                p[j, k_] = max(p[j, k_], min(p[j, i], p[i, k_]))
            end
        end
    end

    beats = p .> transpose(p)
    scores = _topological_level_scores(beats)

    ranking = rank_scores(scores)[string(method)]
    return return_scores ? (ranking, scores) : ranking
end

"""
    ranked_pairs(
        R;
        strength="margin",
        tie_policy="half",
        method="competition",
        return_scores=false,
    )

Rank models with Ranked Pairs (Tideman) by locking pairwise victories without
creating directed cycles.

Victories are sorted by strength (margin or winning-votes), then each edge
`winner -> loser` is locked only if it does not create a directed cycle in the
current locked graph.
"""
function ranked_pairs(
    R;
    strength="margin",
    tie_policy="half",
    method="competition",
    return_scores=false,
)
    strength_s = string(strength)
    if strength_s ∉ ("margin", "winning_votes")
        error("strength must be one of {'margin','winning_votes'}")
    end

    Rv = validate_input(R)
    k = _per_question_correct_counts(Rv)
    P = _pairwise_preference_counts(k; tie_policy=tie_policy)
    L = size(P, 1)
    margin = P .- transpose(P)

    victories = Tuple{Float64, Float64, Int, Int}[]
    for i in 1:L
        for j in (i + 1):L
            if margin[i, j] == 0.0
                continue
            end

            winner = margin[i, j] > 0.0 ? i : j
            loser = margin[i, j] > 0.0 ? j : i
            m = abs(margin[i, j])
            wv = P[winner, loser]
            primary = strength_s == "margin" ? m : wv
            push!(victories, (primary, wv, winner, loser))
        end
    end

    sort!(victories; by=t -> (-t[1], -t[2], t[3], t[4]))

    locked = falses(L, L)
    function has_path(src::Int, dst::Int)
        stack = Int[src]
        seen = falses(L)
        seen[src] = true

        while !isempty(stack)
            u = pop!(stack)
            if u == dst
                return true
            end
            for v in 1:L
                if locked[u, v] && !seen[v]
                    seen[v] = true
                    push!(stack, v)
                end
            end
        end
        return false
    end

    for (_, _, winner, loser) in victories
        if has_path(loser, winner)
            continue
        end
        locked[winner, loser] = true
    end

    scores = _topological_level_scores(locked)
    ranking = rank_scores(scores)[string(method)]
    return return_scores ? (ranking, scores) : ranking
end

function _kemeny_var_index(i::Int, j::Int, L::Int)
    if i == j
        error("No variable for i==j")
    end
    return (i - 1) * (L - 1) + (j > i ? (j - 1) : j)
end

function _solve_kemeny_milp(
    c::Vector{Float64},
    L::Int;
    time_limit=nothing,
    fixed_var=nothing,
)
    n_vars = L * (L - 1)

    model = Model(HiGHS.Optimizer)
    set_silent(model)
    if !isnothing(time_limit)
        set_optimizer_attribute(model, "time_limit", Float64(time_limit))
    end

    @variable(model, y[1:n_vars], Bin)
    @objective(model, Min, sum(c[t] * y[t] for t in 1:n_vars))

    for i in 1:L
        for j in (i + 1):L
            @constraint(model, y[_kemeny_var_index(i, j, L)] + y[_kemeny_var_index(j, i, L)] == 1)
        end
    end

    for i in 1:L
        for j in (i + 1):L
            for k in (j + 1):L
                @constraint(
                    model,
                    y[_kemeny_var_index(i, j, L)] +
                    y[_kemeny_var_index(j, k, L)] +
                    y[_kemeny_var_index(k, i, L)] <= 2,
                )
                @constraint(
                    model,
                    y[_kemeny_var_index(i, k, L)] +
                    y[_kemeny_var_index(k, j, L)] +
                    y[_kemeny_var_index(j, i, L)] <= 2,
                )
            end
        end
    end

    if !isnothing(fixed_var)
        idx = Int(fixed_var)
        fix(y[idx], 1.0; force=true)
    end

    optimize!(model)

    status = termination_status(model)
    if !has_values(model)
        return nothing, status, nothing
    end

    x = value.(y)
    obj = objective_value(model)
    return x, status, obj
end

"""
    kemeny_young(
        R;
        tie_policy="half",
        method="competition",
        return_scores=false,
        time_limit=nothing,
        tie_aware=true,
    )

Rank models with Kemeny-Young via MILP optimization. With `tie_aware=true`,
the routine analyzes forced pairwise orders among optimal solutions.

Binary variables ``y_{ij}`` indicate whether model `i` is above `j`:

```math
\\max_y \\sum_{i\\ne j} P_{ij} y_{ij}
```

subject to:

```math
y_{ij}+y_{ji}=1,\\qquad
y_{ij}+y_{jk}+y_{ki}\\le 2 \\quad (\\forall i,j,k\\ \\text{distinct})
```

`tie_aware=true` checks which pairwise orders are forced across all optimal
solutions and ranks by that forced-order DAG.
"""
function kemeny_young(
    R;
    tie_policy="half",
    method="competition",
    return_scores=false,
    time_limit=nothing,
    tie_aware=true,
)
    Rv = validate_input(R)
    if !isnothing(time_limit)
        time_limit_f = Float64(time_limit)
        if !isfinite(time_limit_f) || time_limit_f <= 0.0
            error("time_limit must be a positive finite scalar.")
        end
        time_limit = time_limit_f
    end

    k = _per_question_correct_counts(Rv)
    P = _pairwise_preference_counts(k; tie_policy=tie_policy)
    L = size(P, 1)

    n_vars = L * (L - 1)
    c = zeros(Float64, n_vars)
    for i in 1:L
        for j in 1:L
            if i == j
                continue
            end
            c[_kemeny_var_index(i, j, L)] = -P[i, j]
        end
    end

    x, status, opt_obj = _solve_kemeny_milp(c, L; time_limit=time_limit)
    if status != OPTIMAL || x === nothing || isnothing(opt_obj) || !isfinite(Float64(opt_obj))
        error("Kemeny-Young MILP did not prove an optimal solution: $status")
    end

    y = zeros(Float64, L, L)
    for i in 1:L
        for j in 1:L
            if i == j
                continue
            end
            y[i, j] = x[_kemeny_var_index(i, j, L)]
        end
    end

    if !tie_aware
        scores = vec(sum(y; dims=2))
        ranking = rank_scores(scores)[string(method)]
        return return_scores ? (ranking, scores) : ranking
    end

    opt_value = Float64(opt_obj)
    opt_tol = 1e-9 * max(1.0, abs(opt_value))

    function can_be_optimal_with(i_above_j::Int, j_below_i::Int)
        idx = _kemeny_var_index(i_above_j, j_below_i, L)
        x_fix, status_fix, obj_fix =
            _solve_kemeny_milp(c, L; time_limit=time_limit, fixed_var=idx)

        if status_fix == INFEASIBLE
            return false
        end
        if status_fix != OPTIMAL || x_fix === nothing || isnothing(obj_fix) ||
           !isfinite(Float64(obj_fix))
            error(
                "Tie-aware Kemeny-Young MILP did not prove an optimal subproblem solution: $status_fix",
            )
        end

        return Float64(obj_fix) <= opt_value + opt_tol
    end

    forced = falses(L, L)
    for i in 1:L
        for j in (i + 1):L
            local winner::Int
            local loser::Int
            local reverse_optimal::Bool
            if y[i, j] >= 0.5
                winner = i
                loser = j
                reverse_optimal = can_be_optimal_with(j, i)
            else
                winner = j
                loser = i
                reverse_optimal = can_be_optimal_with(i, j)
            end

            if !reverse_optimal
                forced[winner, loser] = true
            end
        end
    end

    scores = _topological_level_scores(forced)
    ranking = rank_scores(scores)[string(method)]
    return return_scores ? (ranking, scores) : ranking
end

"""
    nanson(R; rank_ties="average", method="competition", return_scores=false)

Rank models with Nanson's elimination rule (iterative Borda with below-mean
elimination).

At round `t`, with active set `A_t` and Borda scores ``s_i^{(t)}``:

```math
E_t = \\{ i\\in A_t : s_i^{(t)} < \\overline{s}^{(t)} \\},
\\qquad
A_{t+1} = A_t \\setminus E_t
```
"""
function nanson(
    R;
    rank_ties="average",
    method="competition",
    return_scores=false,
)
    Rv = validate_input(R)
    k = _per_question_correct_counts(Rv)
    L, M = size(k)

    alive = trues(L)
    survival = zeros(Float64, L)
    round_idx = 0.0

    while count(alive) > 1
        idx = findall(alive)
        k_sub = k[idx, :]
        n_alive = length(idx)

        borda_sub = zeros(Float64, n_alive)
        for m in 1:M
            r = _rankdata_desc(@view(k_sub[:, m]); method=rank_ties)
            borda_sub .+= Float64(n_alive) .- r
        end

        mean_score = sum(borda_sub) / Float64(n_alive)
        to_eliminate = borda_sub .<= mean_score
        if all(to_eliminate)
            break
        end

        eliminated = idx[to_eliminate]
        survival[eliminated] .= round_idx
        alive[eliminated] .= false
        round_idx += 1.0
    end

    survival[alive] .= round_idx
    ranking = rank_scores(survival)[string(method)]
    return return_scores ? (ranking, survival) : ranking
end

"""
    baldwin(R; rank_ties="average", method="competition", return_scores=false)

Rank models with Baldwin's elimination rule (iterative elimination of minimum
Borda score).

At round `t`:

```math
E_t = \\arg\\min_{i\\in A_t} s_i^{(t)},
\\qquad
A_{t+1} = A_t \\setminus E_t
```

This implementation removes all models tied at the minimum in a round.
"""
function baldwin(
    R;
    rank_ties="average",
    method="competition",
    return_scores=false,
)
    Rv = validate_input(R)
    k = _per_question_correct_counts(Rv)
    L, M = size(k)

    alive = trues(L)
    survival = zeros(Float64, L)
    round_idx = 0.0

    while count(alive) > 1
        idx = findall(alive)
        k_sub = k[idx, :]
        n_alive = length(idx)

        borda_sub = zeros(Float64, n_alive)
        for m in 1:M
            r = _rankdata_desc(@view(k_sub[:, m]); method=rank_ties)
            borda_sub .+= Float64(n_alive) .- r
        end

        min_score = minimum(borda_sub)
        to_eliminate = borda_sub .== min_score
        if all(to_eliminate)
            break
        end

        eliminated = idx[to_eliminate]
        survival[eliminated] .= round_idx
        alive[eliminated] .= false
        round_idx += 1.0
    end

    survival[alive] .= round_idx
    ranking = rank_scores(survival)[string(method)]
    return return_scores ? (ranking, survival) : ranking
end

"""
    majority_judgment(R; method="competition", return_scores=false)

Rank models using Majority Judgment with recursive median-grade tie-breaking.

Each question assigns grade ``k_{lm}\\in\\{0,\\dots,N\\}``.
Models are compared by lower median grade; ties are broken by recursively
removing one occurrence of the current median grade from tied models and
repeating the comparison.

# Reference
Balinski, M., & Laraki, R. (2011). *Majority Judgment*.
"""
function majority_judgment(R; method="competition", return_scores=false)
    Rv = validate_input(R)
    k = _per_question_correct_counts(Rv)
    L, M = size(k)
    N = size(Rv, 3)

    counts = zeros(Int, L, N + 1)
    for i in 1:L
        for m in 1:M
            counts[i, k[i, m] + 1] += 1
        end
    end

    function lower_median_grade(hist::AbstractVector{<:Integer}, total::Int)
        target = (total - 1) ÷ 2
        cum = 0
        for g in eachindex(hist)
            cum += Int(hist[g])
            if cum > target
                return g - 1
            end
        end
        return length(hist) - 1
    end

    function compare(i::Int, j::Int)
        hi = copy(@view counts[i, :])
        hj = copy(@view counts[j, :])
        ti = M
        tj = M
        while ti > 0 && tj > 0
            gi = lower_median_grade(hi, ti)
            gj = lower_median_grade(hj, tj)
            if gi != gj
                return gi > gj ? -1 : 1
            end

            hi[gi + 1] -= 1
            hj[gj + 1] -= 1
            ti -= 1
            tj -= 1
        end

        return 0
    end

    order = collect(1:L)
    for i in 2:L
        x = order[i]
        j = i - 1
        while j >= 1 && compare(x, order[j]) < 0
            order[j + 1] = order[j]
            j -= 1
        end
        order[j + 1] = x
    end

    scores = zeros(Float64, L)
    current_score = Float64(L)
    start = 1
    while start <= L
        stop = start + 1
        while stop <= L && compare(order[start], order[stop]) == 0
            stop += 1
        end

        for idx in start:(stop - 1)
            scores[order[idx]] = current_score
        end
        current_score -= 1.0
        start = stop
    end

    ranking = rank_scores(scores)[string(method)]
    return return_scores ? (ranking, scores) : ranking
end
