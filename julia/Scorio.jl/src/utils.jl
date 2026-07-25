"""Ranking utilities for score conversion, comparison, and hashing."""

function _as_float_vector(x, name::AbstractString)::Vector{Float64}
    v = Float64.(collect(x))
    if ndims(v) != 1
        error("$name must be a 1D sequence.")
    end
    return v
end

function _group_adjacent_scores(sorted_scores::Vector{Float64}, tol::Float64)::Vector{Float64}
    grouped = copy(sorted_scores)
    n = length(grouped)
    for i in 2:n
        if abs(grouped[i] - grouped[i - 1]) <= tol
            grouped[i] = grouped[i - 1]
        end
    end
    return grouped
end

function _rank_arrays_from_grouped(
    grouped_sorted::Vector{Float64},
    order::Vector{Int},
)::Dict{String,Any}
    n = length(grouped_sorted)

    competition_sorted = zeros(Int, n)
    competition_max_sorted = zeros(Int, n)
    dense_sorted = zeros(Int, n)
    avg_sorted = zeros(Float64, n)

    i = 1
    dense_rank = 1
    while i <= n
        j = i
        while j + 1 <= n && grouped_sorted[j + 1] == grouped_sorted[i]
            j += 1
        end

        min_rank = i
        max_rank = j
        avg_rank = (i + j) / 2.0

        for t in i:j
            competition_sorted[t] = min_rank
            competition_max_sorted[t] = max_rank
            dense_sorted[t] = dense_rank
            avg_sorted[t] = avg_rank
        end

        dense_rank += 1
        i = j + 1
    end

    competition = similar(competition_sorted)
    competition_max = similar(competition_max_sorted)
    dense = similar(dense_sorted)
    avg = similar(avg_sorted)

    for i in eachindex(order)
        idx = order[i]
        competition[idx] = competition_sorted[i]
        competition_max[idx] = competition_max_sorted[i]
        dense[idx] = dense_sorted[i]
        avg[idx] = avg_sorted[i]
    end

    return Dict{String,Any}(
        "competition" => competition,
        "competition_max" => competition_max,
        "dense" => dense,
        "avg" => avg,
    )
end

"""
    rank_scores(scores_in_id_order; tol=1e-12, sigmas_in_id_order=nothing,
                confidence=0.95, ci_tie_method="zscore_adjacent")
        -> Dict{String, Any}

Convert scores into rank arrays under multiple tie-handling conventions.

Returns base keys:
- `"competition"`: min-rank competition (1,2,2,4,...) as `Vector{Int}`
- `"competition_max"`: max-rank competition as `Vector{Int}`
- `"dense"`: dense ranking as `Vector{Int}`
- `"avg"`: fractional/average ranking as `Vector{Float64}`

If `sigmas_in_id_order` is provided, uncertainty-aware tie grouping is applied
between adjacent sorted items and the following keys are added:
- `"competition_ci"`, `"competition_max_ci"`, `"dense_ci"`, `"avg_ci"`.
"""
function rank_scores(
    scores_in_id_order;
    tol::Real=1e-12,
    sigmas_in_id_order=nothing,
    confidence::Real=0.95,
    ci_tie_method::AbstractString="zscore_adjacent",
)::Dict{String,Any}
    scores = _as_float_vector(scores_in_id_order, "scores_in_id_order")
    order = sortperm(scores, rev=true)
    sorted_scores = scores[order]
    grouped_scores = _group_adjacent_scores(sorted_scores, Float64(tol))

    out = _rank_arrays_from_grouped(grouped_scores, order)

    if !isnothing(sigmas_in_id_order)
        sigmas = _as_float_vector(sigmas_in_id_order, "sigmas_in_id_order")
        if length(sigmas) != length(scores)
            error("sigmas_in_id_order must have the same length as scores.")
        end

        mus_s = scores[order]
        sig_s = sigmas[order]
        ci_grouped = copy(grouped_scores)

        if ci_tie_method == "zscore_adjacent"
            z_thresh = _z_value(confidence; two_sided=false)
            for i in 2:length(ci_grouped)
                if abs(ci_grouped[i] - ci_grouped[i - 1]) <= Float64(tol)
                    ci_grouped[i] = ci_grouped[i - 1]
                    continue
                end

                denom = sqrt(sig_s[i - 1]^2 + sig_s[i]^2)
                if denom == 0.0
                    continue
                end

                z = abs(mus_s[i - 1] - mus_s[i]) / denom
                if z < z_thresh
                    ci_grouped[i] = ci_grouped[i - 1]
                end
            end
        elseif ci_tie_method == "ci_overlap_adjacent"
            z = _z_value(confidence; two_sided=true)
            for i in 2:length(ci_grouped)
                if abs(ci_grouped[i] - ci_grouped[i - 1]) <= Float64(tol)
                    ci_grouped[i] = ci_grouped[i - 1]
                    continue
                end

                lo_prev = mus_s[i - 1] - z * sig_s[i - 1]
                hi_prev = mus_s[i - 1] + z * sig_s[i - 1]
                lo_cur = mus_s[i] - z * sig_s[i]
                hi_cur = mus_s[i] + z * sig_s[i]

                if lo_prev <= hi_cur
                    ci_grouped[i] = ci_grouped[i - 1]
                end
            end
        else
            error("Unknown ci_tie_method.")
        end

        ci_out = _rank_arrays_from_grouped(ci_grouped, order)
        out["competition_ci"] = ci_out["competition"]
        out["competition_max_ci"] = ci_out["competition_max"]
        out["dense_ci"] = ci_out["dense"]
        out["avg_ci"] = ci_out["avg"]
    end

    return out
end

"""
    competition_ranks_from_scores(scores_in_id_order; tol=1e-12) -> Vector{Int}

Return competition (min) ranks from scores.

Higher scores receive better (smaller) ranks. Ties are grouped when adjacent
sorted scores differ by at most `tol`, using the same tie rule as
[`rank_scores`](@ref). Returned ranks are 1-based and aligned to the original
input order.

# Arguments
- `scores_in_id_order`: 1D score sequence in model-id order.
- `tol::Real=1e-12`: absolute tolerance for tie grouping in sorted-score order.

# Returns
- `Vector{Int}`: competition ranks in the same order as `scores_in_id_order`.

# Examples
```julia
scores = [0.9, 0.8, 0.8, 0.1]
ranks = competition_ranks_from_scores(scores)
# ranks == [1, 2, 2, 4]
```
"""
function competition_ranks_from_scores(
    scores_in_id_order;
    tol::Real=1e-12,
)::Vector{Int}
    ranks = rank_scores(scores_in_id_order; tol=tol)["competition"]
    return Int.(round.(ranks))
end

function _rankdata_average(values::Vector{Float64})::Vector{Float64}
    n = length(values)
    if n == 0
        return Float64[]
    end

    order = sortperm(values)
    out = zeros(Float64, n)

    i = 1
    while i <= n
        j = i
        while j + 1 <= n && values[order[j + 1]] == values[order[i]]
            j += 1
        end

        avg_rank = (Float64(i) + Float64(j)) / 2.0
        for t in i:j
            out[order[t]] = avg_rank
        end

        i = j + 1
    end

    return out
end

function _pearson_corr(x::Vector{Float64}, y::Vector{Float64})::Float64
    n = length(x)
    if n != length(y) || n == 0
        return NaN
    end

    mx = sum(x) / n
    my = sum(y) / n

    xx = 0.0
    yy = 0.0
    xy = 0.0
    for i in 1:n
        dx = x[i] - mx
        dy = y[i] - my
        xx += dx * dx
        yy += dy * dy
        xy += dx * dy
    end

    denom = sqrt(xx * yy)
    if denom == 0.0
        return NaN
    end

    return xy / denom
end

function _kendall_tau(g::Vector{Float64}, t::Vector{Float64})::Tuple{Float64, Float64}
    n = length(g)
    if n < 2
        return NaN, NaN
    end

    concordant = 0
    discordant = 0
    tie_x_only = 0
    tie_y_only = 0
    tie_both = 0

    for i in 1:(n - 1)
        for j in (i + 1):n
            dx = g[i] - g[j]
            dy = t[i] - t[j]

            if dx == 0.0 && dy == 0.0
                tie_both += 1
            elseif dx == 0.0
                tie_x_only += 1
            elseif dy == 0.0
                tie_y_only += 1
            elseif dx * dy > 0.0
                concordant += 1
            else
                discordant += 1
            end
        end
    end

    n0 = n * (n - 1) / 2
    n1 = tie_x_only + tie_both
    n2 = tie_y_only + tie_both
    denom = sqrt((n0 - n1) * (n0 - n2))

    if denom == 0.0
        return NaN, NaN
    end

    tau = (concordant - discordant) / denom

    var_tau = (2.0 * (2.0 * n + 5.0)) / (9.0 * n * (n - 1.0))
    if var_tau <= 0.0
        return Float64(tau), NaN
    end

    z = Float64(tau) / sqrt(var_tau)
    pvalue = 2.0 * (1.0 - _normal_cdf(abs(z)))
    pvalue = clamp(pvalue, 0.0, 1.0)

    return Float64(tau), Float64(pvalue)
end

function _spearman_rho(g::Vector{Float64}, t::Vector{Float64})::Tuple{Float64, Float64}
    n = length(g)
    if n < 2
        return NaN, NaN
    end

    rg = _rankdata_average(g)
    rt = _rankdata_average(t)

    rho = _pearson_corr(rg, rt)
    if !isfinite(rho)
        return NaN, NaN
    end

    if n <= 2
        return Float64(rho), NaN
    end

    if abs(rho) >= 1.0
        return sign(rho), 0.0
    end

    t_stat = rho * sqrt((n - 2.0) / max(1e-15, 1.0 - rho * rho))
    pvalue = 2.0 * (1.0 - _normal_cdf(abs(t_stat)))
    pvalue = clamp(pvalue, 0.0, 1.0)

    return Float64(rho), Float64(pvalue)
end

function _weighted_kendall_tau(g::Vector{Float64}, t::Vector{Float64})::Tuple{Float64, Float64}
    n = length(g)
    if n < 2
        return NaN, NaN
    end

    # Hyperbolic additive weights by positions in g (best ranks get higher weights).
    order = sortperm(g)
    pos = zeros(Int, n)
    for i in 1:n
        pos[order[i]] = i
    end
    weights = [1.0 / pos[i] for i in 1:n]

    concordant = 0.0
    discordant = 0.0

    for i in 1:(n - 1)
        for j in (i + 1):n
            dx = g[i] - g[j]
            dy = t[i] - t[j]
            if dx == 0.0 || dy == 0.0
                continue
            end

            w = weights[i] + weights[j]
            if dx * dy > 0.0
                concordant += w
            else
                discordant += w
            end
        end
    end

    denom = concordant + discordant
    if denom == 0.0
        return NaN, NaN
    end

    tau = (concordant - discordant) / denom
    return Float64(tau), NaN
end

function compare_rankings(
    ranked_list_a,
    ranked_list_b;
    method::AbstractString="all",
)
    allowed_methods = Set(["kendall", "spearman", "weighted_kendall", "all"])
    if !(method in allowed_methods)
        error("method must be one of ['all', 'kendall', 'spearman', 'weighted_kendall']; got '$method'")
    end

    g = _as_float_vector(ranked_list_a, "ranked_list_a")
    t = _as_float_vector(ranked_list_b, "ranked_list_b")

    n = length(g)
    if n == 0 || n != length(t)
        error("Ranked lists must have the same non-zero length.")
    end

    if any(x -> !isfinite(x), g) || any(x -> !isfinite(x), t)
        error("ranked lists must not contain NaN or inf.")
    end

    diffs = t .- g
    fraction_mismatched = Float64(sum(diffs .!= 0.0) / n)
    max_disp = n > 1 ? Float64(maximum(abs.(diffs)) / (n - 1)) : 0.0

    kendall_stat, kendall_p = _kendall_tau(g, t)
    spearman_stat, spearman_p = _spearman_rho(g, t)
    weighted_stat, weighted_p = _weighted_kendall_tau(g, t)

    if method == "kendall"
        return (Float64(kendall_stat), Float64(kendall_p))
    elseif method == "spearman"
        return (Float64(spearman_stat), Float64(spearman_p))
    elseif method == "weighted_kendall"
        return (Float64(weighted_stat), Float64(weighted_p))
    end

    return Dict(
        "kendalltau" => (Float64(kendall_stat), Float64(kendall_p)),
        "spearmanr" => (Float64(spearman_stat), Float64(spearman_p)),
        "weighted_kendalltau" => (Float64(weighted_stat), Float64(weighted_p)),
        "fraction_mismatched" => fraction_mismatched,
        "max_disp" => max_disp,
    )
end

function lehmer_hash(ranked_list)
    perm = collect(ranked_list)
    n = length(perm)

    if any(x -> !(x isa Integer), perm)
        error("ranked_list must be a permutation of integers 0..n-1.")
    end

    perm_int = Int.(perm)
    if Set(perm_int) != Set(0:(n - 1))
        error("ranked_list must be a permutation of 0..n-1 with no ties.")
    end

    factorials = Vector{BigInt}(undef, n)
    if n > 0
        factorials[1] = big(1)
        for i in 2:n
            factorials[i] = factorials[i - 1] * (i - 1)
        end
    end

    hash_value = big(0)
    for i in 1:n
        inversions = 0
        for j in (i + 1):n
            if perm_int[j] < perm_int[i]
                inversions += 1
            end
        end
        hash_value += BigInt(inversions) * factorials[n - i + 1]
    end

    return hash_value
end

function lehmer_unhash(hash_value, n::Integer)
    n_int = Int(n)
    if n_int < 0
        error("n must be >= 0")
    end

    h = BigInt(hash_value)
    max_hash = factorial(BigInt(n_int))

    if h < 0 || h >= max_hash
        error("hash_value must be in range 0..$(n_int)!-1 = $(max_hash - 1); got $hash_value")
    end

    factorials = Vector{BigInt}(undef, n_int)
    if n_int > 0
        factorials[1] = big(1)
        for i in 2:n_int
            factorials[i] = factorials[i - 1] * (i - 1)
        end
    end

    available = collect(0:(n_int - 1))
    result = Int[]

    for i in 1:n_int
        f = factorials[n_int - i + 1]
        idx = Int(div(h, f))
        h = mod(h, f)
        push!(result, popat!(available, idx + 1))
    end

    return result
end

function ordered_bell(n::Int)
    if n < 0
        error("n must be >= 0")
    end

    F = fill(big(0), n + 1)
    F[1] = big(1)
    for m in 1:n
        s = big(0)
        for k in 1:m
            s += binomial(BigInt(m), BigInt(k)) * F[m - k + 1]
        end
        F[m + 1] = s
    end
    return F
end

function comb_rank_lex(indices, n::Int, k::Int)
    idx = Int.(collect(indices))
    r = big(0)
    prev = -1

    for pos in 1:k
        start_x = prev + 1
        end_x = idx[pos] - 1
        remaining = k - pos

        for x in start_x:end_x
            if n - 1 - x >= remaining && remaining >= 0
                r += binomial(BigInt(n - 1 - x), BigInt(remaining))
            end
        end

        prev = idx[pos]
    end

    return r
end

function comb_unrank_lex(r, n::Int, k::Int)
    if k == 0
        return Int[]
    end

    r_big = BigInt(r)
    total = binomial(BigInt(n), BigInt(k))
    if r_big < 0 || r_big >= total
        error("Combination rank out of range.")
    end

    combo = Int[]
    x = 0
    for pos in 1:k
        rem = k - pos
        while true
            cnt = if n - 1 - x >= rem
                binomial(BigInt(n - 1 - x), BigInt(rem))
            else
                big(0)
            end

            if r_big < cnt
                push!(combo, x)
                x += 1
                break
            end

            r_big -= cnt
            x += 1
        end
    end

    return combo
end

function blocks_from_rank_list(rank_list; tol::Real=1e-12)
    r = _as_float_vector(rank_list, "rank_list")
    n = length(r)
    if n == 0
        return Vector{Vector{Int}}()
    end

    ids = collect(0:(n - 1))
    order = sortperm(1:n, by=i -> (r[i], ids[i]))

    r_sorted = r[order]
    ids_sorted = ids[order]

    blocks = Vector{Vector{Int}}()
    cur = Int[ids_sorted[1]]

    for i in 2:n
        if abs(r_sorted[i] - r_sorted[i - 1]) <= Float64(tol)
            push!(cur, ids_sorted[i])
        else
            push!(blocks, sort(cur))
            cur = Int[ids_sorted[i]]
        end
    end

    push!(blocks, sort(cur))
    return blocks
end

function ranking_hash(rank_list; tol::Real=1e-12)
    blocks = blocks_from_rank_list(rank_list; tol=tol)
    n = length(rank_list)
    F = ordered_bell(n)

    remaining = collect(0:(n - 1))
    remaining_set = Set(remaining)
    h = big(0)

    for block in blocks
        m = length(remaining)
        k = length(block)

        for s in 1:(k - 1)
            h += binomial(BigInt(m), BigInt(s)) * F[m - s + 1]
        end

        pos = Dict{Int, Int}()
        for (i, v) in enumerate(remaining)
            pos[v] = i - 1
        end
        idx = [pos[v] for v in block]
        subset_rank = comb_rank_lex(idx, m, k)
        h += subset_rank * F[m - k + 1]

        for v in block
            delete!(remaining_set, v)
        end
        remaining = [x for x in remaining if x in remaining_set]
    end

    return h
end

function unhash_ranking(h, n::Integer)
    n_int = Int(n)
    F = ordered_bell(n_int)
    h_big = BigInt(h)

    if h_big < 0 || h_big >= F[n_int + 1]
        error("h out of range for n=$n_int. Must be 0..$(F[n_int + 1] - 1).")
    end

    remaining = collect(0:(n_int - 1))
    rank_list = fill(0, n_int)
    cur_rank = 1

    while !isempty(remaining)
        m = length(remaining)

        offset = big(0)
        k_chosen = 0
        for k in 1:m
            cnt = binomial(BigInt(m), BigInt(k)) * F[m - k + 1]
            if h_big < offset + cnt
                h_big -= offset
                k_chosen = k
                break
            end
            offset += cnt
        end

        if k_chosen == 0
            error("Unhashing failed.")
        end

        suffix = F[m - k_chosen + 1]
        subset_rank = div(h_big, suffix)
        h_big = mod(h_big, suffix)

        idx = comb_unrank_lex(subset_rank, m, k_chosen)
        group_ids = [remaining[i + 1] for i in idx]
        for item in group_ids
            rank_list[item + 1] = cur_rank
        end

        chosen = Set(group_ids)
        remaining = [x for x in remaining if !(x in chosen)]
        cur_rank += k_chosen
    end

    return rank_list
end

export competition_ranks_from_scores,
    rank_scores,
    compare_rankings,
    lehmer_hash,
    lehmer_unhash,
    ranking_hash,
    unhash_ranking
