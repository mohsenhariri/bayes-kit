# Per-trace confidence signals from chosen-token and top-k log-probabilities.

function _as_logprobs(logprobs)::Vector{Float64}
    lp = _flatten_numeric(logprobs, "logprobs")
    isempty(lp) && error("need at least one token (T >= 1).")
    all(isfinite, lp) || error("logprobs must all be finite.")
    return lp
end

function _topk_rows(topk_logprobs)::Vector{Vector{Float64}}
    rows = Vector{Vector{Float64}}()
    if topk_logprobs isa AbstractArray && ndims(topk_logprobs) >= 2
        size(topk_logprobs, 1) > 0 || error("need at least one token (T >= 1).")
        prod(size(topk_logprobs)[2:end]) > 0 ||
            error("need at least one token and one top-k candidate.")
        # NumPy's ragged fallback treats the first axis as tokens and flattens
        # every remaining axis in C order. Preserve that behavior for arrays
        # with more than the usual two dimensions as well.
        for i in axes(topk_logprobs, 1)
            slice = selectdim(topk_logprobs, 1, i)
            row = try
                _flatten_numeric(slice, "topk_logprobs")
            catch
                error("topk_logprobs must contain numeric per-token rows.")
            end
            push!(rows, row)
        end
    else
        xs = try
            collect(topk_logprobs)
        catch
            error("topk_logprobs must be a 2D matrix or a list of per-token rows.")
        end
        isempty(xs) && error("need at least one token (T >= 1).")
        if all(x -> x isa Real, xs)
            push!(rows, Float64.(xs))
        else
            for x in xs
                row = if x isa Real
                    Float64[Float64(x)]
                else
                    try
                        _flatten_numeric(x, "topk_logprobs")
                    catch
                        error("topk_logprobs must contain numeric per-token rows.")
                    end
                end
                push!(rows, row)
            end
        end
    end
    isempty(rows) && error("need at least one token (T >= 1).")
    for row in rows
        isempty(row) && error("every position needs at least one top-k candidate.")
        all(isfinite, row) || error("topk_logprobs must all be finite.")
    end
    return rows
end

function _reduce_tokens(x::Vector{Float64}, how)::Float64
    if how == "mean"
        return sum(x) / length(x)
    elseif how == "min"
        return minimum(x)
    elseif how == "max"
        return maximum(x)
    end
    error("aggregate must be one of ('mean', 'min', 'max'); got $how.")
end

function _normalized_topk_logprobs(row::Vector{Float64})::Vector{Float64}
    m = maximum(row)
    shifted = row .- m
    return shifted .- log(sum(exp, shifted))
end

function _per_token_entropy(rows::Vector{Vector{Float64}})::Vector{Float64}
    out = Vector{Float64}(undef, length(rows))
    for i in eachindex(rows)
        log_p = _normalized_topk_logprobs(rows[i])
        p = exp.(log_p)
        out[i] = -sum(p .* log_p)
    end
    return out
end

function _per_token_varentropy(rows::Vector{Vector{Float64}})::Vector{Float64}
    out = Vector{Float64}(undef, length(rows))
    for i in eachindex(rows)
        log_p = _normalized_topk_logprobs(rows[i])
        p = exp.(log_p)
        surprisal = .-log_p
        entropy = sum(p .* surprisal)
        out[i] = sum(p .* (surprisal .- entropy) .^ 2)
    end
    return out
end

function _per_token_self_certainty(rows::Vector{Vector{Float64}})::Vector{Float64}
    out = Vector{Float64}(undef, length(rows))
    for i in eachindex(rows)
        log_p = _normalized_topk_logprobs(rows[i])
        out[i] = -log(length(log_p)) - sum(log_p) / length(log_p)
    end
    return out
end

function _per_token_confidence(rows::Vector{Vector{Float64}})::Vector{Float64}
    return Float64[-sum(row) / length(row) for row in rows]
end

function _per_token_max_probability(rows::Vector{Vector{Float64}})::Vector{Float64}
    return Float64[exp(maximum(row)) for row in rows]
end

function _per_token_margin(
    rows::Vector{Vector{Float64}},
    use_prob::Bool,
)::Vector{Float64}
    out = Vector{Float64}(undef, length(rows))
    for i in eachindex(rows)
        row = rows[i]
        if length(row) < 2
            out[i] = 0.0
            continue
        end
        ordered = sort(row; rev=true)
        out[i] = use_prob ? exp(ordered[1]) - exp(ordered[2]) : ordered[1] - ordered[2]
    end
    return out
end

"""Mean chosen-token log-probability (higher is more confident)."""
function mean_logprob(logprobs)::Float64
    lp = _as_logprobs(logprobs)
    return sum(lp) / length(lp)
end

"""Total chosen-token sequence log-likelihood."""
sequence_logprob(logprobs)::Float64 = sum(_as_logprobs(logprobs))

"""Sequence perplexity, `exp(-mean_logprob(logprobs))`."""
perplexity(logprobs)::Float64 = exp(-mean_logprob(logprobs))

"""
    picsar(logprobs; answer_start=nothing, normalize_reasoning=false)

PiCSAR reasoning-plus-answer log-likelihood. `answer_start` is the number of
tokens before the answer span, matching the Python split index.
"""
function picsar(logprobs; answer_start=nothing, normalize_reasoning::Bool=false)::Float64
    lp = _as_logprobs(logprobs)
    answer_start === nothing && return sum(lp)
    answer_start isa Integer || error("answer_start must be an integer or nothing.")
    split = Int(answer_start)
    0 <= split <= length(lp) ||
        error("answer_start must be in [0, $(length(lp))]; got $answer_start.")
    reasoning = split == 0 ? @view(lp[1:0]) : @view(lp[1:split])
    answer = split == length(lp) ? @view(lp[(length(lp) + 1):length(lp)]) :
             @view(lp[(split + 1):length(lp)])
    reasoning_ll = sum(reasoning)
    if normalize_reasoning && !isempty(reasoning)
        reasoning_ll /= length(reasoning)
    end
    return reasoning_ll + sum(answer)
end

"""Top-k KL-from-uniform self-certainty, reduced across tokens."""
function self_certainty(topk_logprobs; aggregate="mean")::Float64
    rows = _topk_rows(topk_logprobs)
    return _reduce_tokens(_per_token_self_certainty(rows), aggregate)
end

"""Top-k Shannon entropy in nats, reduced across tokens."""
function token_entropy(topk_logprobs; aggregate="mean")::Float64
    rows = _topk_rows(topk_logprobs)
    return _reduce_tokens(_per_token_entropy(rows), aggregate)
end

"""Top-k varentropy, reduced across tokens."""
function varentropy(topk_logprobs; aggregate="mean")::Float64
    rows = _topk_rows(topk_logprobs)
    return _reduce_tokens(_per_token_varentropy(rows), aggregate)
end

"""Maximum raw softmax probability, reduced across tokens."""
function max_softmax_probability(topk_logprobs; aggregate="mean")::Float64
    rows = _topk_rows(topk_logprobs)
    return _reduce_tokens(_per_token_max_probability(rows), aggregate)
end

"""Top-one/top-two log-probability (or probability) margin."""
function logprob_margin(topk_logprobs; use_prob::Bool=false, aggregate="mean")::Float64
    rows = _topk_rows(topk_logprobs)
    return _reduce_tokens(_per_token_margin(rows, use_prob), aggregate)
end

"""DeepConf per-token confidence: negative mean raw top-k log-probability."""
token_confidence(topk_logprobs)::Vector{Float64} =
    _per_token_confidence(_topk_rows(topk_logprobs))

function _group_confidences(conf::Vector{Float64}, window)::Vector{Float64}
    w = min(_python_int(window), length(conf))
    w > 0 || error("window must be positive; got $window.")
    if w == length(conf)
        return Float64[sum(conf) / length(conf)]
    end
    csum = Vector{Float64}(undef, length(conf) + 1)
    csum[1] = 0.0
    for i in eachindex(conf)
        csum[i + 1] = csum[i] + conf[i]
    end
    return Float64[(csum[i + w] - csum[i]) / w for i in 1:(length(conf) - w + 1)]
end

# NumPy's default quantile method is linear interpolation at (n - 1)q.
function _quantile_linear(x::Vector{Float64}, q::Real)::Float64
    y = sort(x)
    length(y) == 1 && return y[1]
    h = (length(y) - 1) * Float64(q)
    lo = floor(Int, h)
    hi = ceil(Int, h)
    frac = h - lo
    return (1.0 - frac) * y[lo + 1] + frac * y[hi + 1]
end

"""DeepConf trace confidence using mean, tail, or group reductions."""
function deepconf_confidence(
    topk_logprobs;
    mode="mean",
    window=2048,
    tail_tokens=2048,
    bottom_quantile::Real=0.10,
)::Float64
    conf = token_confidence(topk_logprobs)
    if mode == "mean"
        return sum(conf) / length(conf)
    elseif mode == "tail"
        tau = min(max(_python_int(tail_tokens), 1), length(conf))
        return sum(@view(conf[(end - tau + 1):end])) / tau
    elseif mode == "lowest_group" || mode == "bottom_group"
        groups = _group_confidences(conf, window)
        mode == "lowest_group" && return minimum(groups)
        0.0 < bottom_quantile <= 1.0 ||
            error("bottom_quantile must be in (0, 1]; got $bottom_quantile.")
        threshold = _quantile_linear(groups, bottom_quantile)
        bottom = filter(x -> x <= threshold, groups)
        return sum(bottom) / length(bottom)
    end
    error(
        "mode must be 'mean', 'tail', 'bottom_group', or 'lowest_group'; got $mode.",
    )
end
