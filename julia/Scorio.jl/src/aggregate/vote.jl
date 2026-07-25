# Vote-based fixed-pool aggregation rules.

function _row_majority(ans_row)
    counts = Dict{Any, Int}()
    labels = Any[]
    first_indices = Dict{Any, Int}()
    for j in eachindex(ans_row)
        a = ans_row[j]
        _is_valid_answer(a) || continue
        if !haskey(counts, a)
            push!(labels, a)
            counts[a] = 0
            first_indices[a] = j
        end
        counts[a] += 1
    end
    isempty(labels) && return nothing, 0
    best = labels[1]
    for a in @view labels[2:end]
        counts[a] > counts[best] && (best = a)
    end
    return best, first_indices[best]
end

function _row_weighted(ans_row, score_row, aggregate)
    sums = Dict{Any, Float64}()
    counts = Dict{Any, Int}()
    representatives = Dict{Any, Int}()
    labels = Any[]
    for j in eachindex(ans_row)
        a = ans_row[j]
        _is_valid_answer(a) || continue
        score = Float64(score_row[j])
        if !haskey(sums, a)
            push!(labels, a)
            sums[a] = score
            counts[a] = 1
            representatives[a] = j
        else
            sums[a] += score
            counts[a] += 1
            if score > Float64(score_row[representatives[a]])
                representatives[a] = j
            end
        end
    end
    isempty(labels) && return nothing, 0
    weight(a) = aggregate == "mean" ? sums[a] / counts[a] : sums[a]
    best = labels[1]
    for a in @view labels[2:end]
        weight(a) > weight(best) && (best = a)
    end
    return best, representatives[best]
end

"""Plain majority vote with ties broken by earliest appearance."""
function majority_vote(answers; return_index::Bool=false)
    Z, _, single = _normalize_candidates(answers)
    selected = Any[]
    indices = Int[]
    for i in axes(Z, 1)
        a, idx = _row_majority(@view Z[i, :])
        push!(selected, a)
        push!(indices, idx)
    end
    selection = _finalize(selected, single)
    public_indices = Int[index - 1 for index in indices]
    return return_index ? (selection, _finalize(public_indices, single)) : selection
end

"""Raw-score weighted majority vote using group sums or means."""
function weighted_majority_vote(
    answers,
    scores;
    aggregate="sum",
    return_index::Bool=false,
    return_score::Bool=false,
)
    aggregate in ("sum", "mean") ||
        error("aggregate must be 'sum' or 'mean'; got $aggregate.")
    Z, S, single = _normalize_candidates(answers, scores; require_scores=true)
    return _run_score_rule(
        (row, score_row) -> _row_weighted(row, score_row, aggregate),
        Z,
        S,
        single;
        return_index=return_index,
        return_score=return_score,
    )
end

function _row_softmax(ans_row, score_row, temperature::Float64)
    part = _valid_indices(ans_row)
    isempty(part) && return nothing, 0
    max_score = maximum(Float64(score_row[j]) for j in part)
    weights = Dict{Int, Float64}(
        j => exp((Float64(score_row[j]) - max_score) / temperature) for j in part
    )
    return _plurality(
        ans_row,
        part,
        j -> weights[j],
        j -> Float64(score_row[j]),
    )
end

function _row_rank_weighted(ans_row, score_row, p::Float64)
    part = _valid_indices(ans_row)
    isempty(part) && return nothing, 0
    n = length(part)
    order = sort(part; by=j -> (-Float64(score_row[j]), j))
    if isinteger(p)
        exponent = Int(p)
        weights = Dict{Int, BigInt}(
            j => BigInt(n - rank + 1)^exponent for (rank, j) in enumerate(order)
        )
        return _plurality(
            ans_row,
            part,
            j -> weights[j],
            j -> Float64(score_row[j]),
        )
    end
    weights = Dict{Int, Float64}(
        j => ((n - rank + 1) / n)^p for (rank, j) in enumerate(order)
    )
    return _plurality(
        ans_row,
        part,
        j -> weights[j],
        j -> Float64(score_row[j]),
    )
end

function _row_logit_weighted(
    ans_row,
    score_row,
    threshold::Float64,
    transform,
)
    part = _valid_indices(ans_row)
    isempty(part) && return nothing, 0
    weight = if transform == "logit"
        threshold_logit = log(threshold / (1.0 - threshold))
        j -> begin
            score = Float64(score_row[j])
            log(score / (1.0 - score)) - threshold_logit
        end
    else
        j -> Float64(score_row[j]) - threshold
    end
    return _plurality(ans_row, part, weight, j -> Float64(score_row[j]))
end

function _row_filtered(ans_row, score_row, keep, weighted::Bool)
    part = _valid_indices(ans_row)
    isempty(part) && return nothing, 0
    k = _keep_count(keep, length(part))
    order = sort(part; by=j -> (-Float64(score_row[j]), j))
    kept = order[1:k]
    weight(j) = weighted ? Float64(score_row[j]) : 1.0
    return _plurality(ans_row, kept, weight, j -> Float64(score_row[j]))
end

"""Temperature-softmax-weighted majority vote (CISC)."""
function softmax_weighted_vote(
    answers,
    scores;
    temperature::Real=1.0,
    return_index::Bool=false,
    return_score::Bool=false,
)
    temperature_f = Float64(temperature)
    temperature_f > 0.0 || error("temperature must be > 0; got $temperature.")
    Z, S, single = _normalize_candidates(answers, scores; require_scores=true)
    return _run_score_rule(
        (row, score_row) -> _row_softmax(row, score_row, temperature_f),
        Z,
        S,
        single;
        return_index=return_index,
        return_score=return_score,
    )
end

"""Rank/Borda-weighted vote, invariant to monotone score transforms."""
function rank_weighted_vote(
    answers,
    scores;
    p::Real=1.0,
    return_index::Bool=false,
    return_score::Bool=false,
)
    p_f = Float64(p)
    p_f >= 0.0 && isfinite(p_f) ||
        error("p must be a finite non-negative number; got $p.")
    Z, S, single = _normalize_candidates(answers, scores; require_scores=true)
    return _run_score_rule(
        (row, score_row) -> _row_rank_weighted(row, score_row, p_f),
        Z,
        S,
        single;
        return_index=return_index,
        return_score=return_score,
    )
end

"""Threshold-shifted log-odds or linear weighted majority vote."""
function logit_weighted_vote(
    answers,
    scores;
    threshold::Real=0.5,
    transform="logit",
    return_index::Bool=false,
    return_score::Bool=false,
)
    transform in ("logit", "linear") ||
        error("transform must be 'logit' or 'linear'; got $transform.")
    threshold_f = Float64(threshold)
    Z, S, single = _normalize_candidates(answers, scores; require_scores=true)
    if transform == "logit"
        0.0 < threshold_f < 1.0 || error(
            "threshold must be in (0, 1) for transform='logit'; got $threshold.",
        )
        for i in axes(Z, 1), j in axes(Z, 2)
            _is_valid_answer(Z[i, j]) || continue
            score = Float64(S[i, j])
            0.0 < score < 1.0 || error(
                "transform='logit' requires every valid score in (0, 1); got $score. " *
                "Use transform='linear' for unbounded scores.",
            )
        end
    end
    return _run_score_rule(
        (row, score_row) ->
            _row_logit_weighted(row, score_row, threshold_f, transform),
        Z,
        S,
        single;
        return_index=return_index,
        return_score=return_score,
    )
end

"""Vote after retaining only the top-scoring fraction or count."""
function filtered_vote(
    answers,
    scores;
    keep=0.5,
    weighted::Bool=true,
    return_index::Bool=false,
    return_score::Bool=false,
)
    Z, S, single = _normalize_candidates(answers, scores; require_scores=true)
    _keep_count(keep, size(Z, 2))
    return _run_score_rule(
        (row, score_row) -> _row_filtered(row, score_row, keep, weighted),
        Z,
        S,
        single;
        return_index=return_index,
        return_score=return_score,
    )
end
