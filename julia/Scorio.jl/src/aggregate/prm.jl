"""
    prm_aggregate(step_scores; method="last") -> Float64

Reduce one trace's finite per-step process-reward scores using `"last"`,
`"min"`, `"mean"`, `"prod"`, or `"max"`.
"""
function prm_aggregate(step_scores; method="last")::Float64
    method in ("last", "min", "mean", "prod", "max") ||
        error("method must be one of ('last', 'min', 'mean', 'prod', 'max'); got $method.")
    scores = _flatten_numeric(step_scores, "step_scores")
    isempty(scores) && error("step_scores must be non-empty (L >= 1).")
    all(isfinite, scores) || error("step_scores must all be finite.")
    if method == "last"
        return scores[end]
    elseif method == "min"
        return minimum(scores)
    elseif method == "mean"
        return sum(scores) / length(scores)
    elseif method == "max"
        return maximum(scores)
    end
    return prod(scores)
end
