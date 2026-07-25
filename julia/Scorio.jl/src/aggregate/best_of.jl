# Reward-based fixed-pool selection rules.

function _row_best_of_n(ans_row, score_row)
    best_index = 0
    best_score = -Inf
    for j in eachindex(ans_row)
        _is_valid_answer(ans_row[j]) || continue
        score = Float64(score_row[j])
        if score > best_score
            best_score = score
            best_index = j
        end
    end
    best_index == 0 && return nothing, 0
    return ans_row[best_index], best_index
end

function _row_majority_of_the_bests(ans_row, score_row, m)
    part = _valid_indices(ans_row)
    isempty(part) && return nothing, 0
    n = length(part)
    n == 1 && return ans_row[part[1]], part[1]

    mm = m === nothing ? _default_m(n) : _python_int(m)
    mm = min(max(mm, 1), n)
    order = sort(part; by=j -> (-Float64(score_row[j]), j))

    weights = Dict{Any, BigInt}()
    representatives = Dict{Any, Int}()
    for (rank, j) in enumerate(order)
        upper = BigInt(n - rank + 1)
        lower = BigInt(n - rank)
        weight = upper^mm - lower^mm
        a = ans_row[j]
        if !haskey(weights, a)
            weights[a] = BigInt(0)
            representatives[a] = j
        end
        weights[a] += weight
    end

    # Scan in original order so exact mass ties go to earliest appearance.
    best = nothing
    for j in part
        a = ans_row[j]
        if best === nothing || weights[a] > weights[best]
            best = a
        end
    end
    return best, representatives[best]
end

"""Best-of-N: select the valid candidate with the largest score."""
function best_of_n(
    answers,
    scores;
    return_index::Bool=false,
    return_score::Bool=false,
)
    Z, S, single = _normalize_candidates(answers, scores; require_scores=true)
    return _run_score_rule(
        _row_best_of_n,
        Z,
        S,
        single;
        return_index=return_index,
        return_score=return_score,
    )
end

"""
    majority_of_the_bests(answers, scores; m=nothing, ...)

Exact mode of the Best-of-N answer under size-`m` bootstrap resampling.  The
default is `floor(sqrt(n))` over valid candidates.
"""
function majority_of_the_bests(
    answers,
    scores;
    m=nothing,
    return_index::Bool=false,
    return_score::Bool=false,
)
    if m !== nothing
        (m isa Integer && !(m isa Bool) && m >= 1) ||
            error("m must be a positive integer or nothing; got $m.")
    end
    Z, S, single = _normalize_candidates(answers, scores; require_scores=true)
    return _run_score_rule(
        (row, score_row) -> _row_majority_of_the_bests(row, score_row, m),
        Z,
        S,
        single;
        return_index=return_index,
        return_score=return_score,
    )
end

const mob = majority_of_the_bests

function _row_best_of_majority(ans_row, score_row, alpha::Float64, aggregate)
    part = _valid_indices(ans_row)
    isempty(part) && return nothing, 0

    groups = Dict{Any, Vector{Int}}()
    representatives = Dict{Any, Int}()
    labels = Any[]
    for j in part
        a = ans_row[j]
        if !haskey(groups, a)
            push!(labels, a)
            groups[a] = Int[j]
            representatives[a] = j
        else
            push!(groups[a], j)
            if Float64(score_row[j]) > Float64(score_row[representatives[a]])
                representatives[a] = j
            end
        end
    end

    gated = Any[a for a in labels if length(groups[a]) / length(part) >= alpha]
    isempty(gated) && (gated = copy(labels))

    function group_reward(a)::Float64
        idx = groups[a]
        if aggregate == "mean"
            return sum(Float64(score_row[j]) for j in idx) / length(idx)
        elseif aggregate == "sum"
            return sum(Float64(score_row[j]) for j in idx)
        end
        return maximum(Float64(score_row[j]) for j in idx)
    end

    best = gated[1]
    best_reward = group_reward(best)
    for a in @view gated[2:end]
        reward = group_reward(a)
        if reward > best_reward
            best = a
            best_reward = reward
        end
    end
    return best, representatives[best]
end

"""Frequency-gated reward selection (Best-of-Majority)."""
function best_of_majority(
    answers,
    scores;
    alpha::Real=0.0,
    aggregate="mean",
    return_index::Bool=false,
    return_score::Bool=false,
)
    alpha_f = Float64(alpha)
    0.0 <= alpha_f <= 1.0 || error("alpha must be in [0, 1]; got $alpha.")
    aggregate in ("mean", "sum", "max") ||
        error("aggregate must be 'mean', 'sum', or 'max'; got $aggregate.")
    Z, S, single = _normalize_candidates(answers, scores; require_scores=true)
    return _run_score_rule(
        (row, score_row) ->
            _row_best_of_majority(row, score_row, alpha_f, aggregate),
        Z,
        S,
        single;
        return_index=return_index,
        return_score=return_score,
    )
end
