# Confidence-Guided Early Stopping (CGES) selection and stopping.

"""Identity sentinel for a correct answer that has not appeared yet."""
struct _CGESOther end

const CGES_OTHER = _CGESOther()

Base.show(io::IO, ::_CGESOther) = print(io, "CGES_OTHER")

function _validate_bool(value, name::AbstractString)
    value isa Bool || error("$name must be a bool; got $value.")
    return nothing
end

function _row_cges_details(ans_row, score_row)
    part = _valid_indices(ans_row)
    isempty(part) && return Dict{Any, Float64}(CGES_OTHER => 1.0), Any[]

    concrete = Any[]
    seen = Dict{Any, Nothing}()
    for j in part
        answer = ans_row[j]
        answer === CGES_OTHER &&
            error("CGES_OTHER is reserved and cannot be an observed answer.")
        if !haskey(seen, answer)
            seen[answer] = nothing
            push!(concrete, answer)
        end
    end

    support_size = length(concrete) + 1
    mismatch = Dict{Int, Float64}()
    base = 0.0
    for j in part
        confidence = Float64(score_row[j])
        if !isfinite(confidence) || !(0.0 < confidence < 1.0)
            error(
                "CGES requires every valid candidate score to be finite and " *
                "strictly in (0, 1); got $confidence.",
            )
        end
        value = log1p(-confidence) - log(support_size - 1)
        mismatch[j] = value
        base += value
    end

    log_scores = Dict{Any, Float64}()
    for answer in concrete
        log_scores[answer] = base
    end
    log_scores[CGES_OTHER] = base
    for j in part
        answer = ans_row[j]
        log_scores[answer] += log(Float64(score_row[j])) - mismatch[j]
    end

    ordered_hypotheses = Any[concrete...; CGES_OTHER]
    normalizer = _logsumexp(Float64[log_scores[a] for a in ordered_hypotheses])
    posterior = Dict{Any, Float64}(
        answer => exp(log_scores[answer] - normalizer) for answer in ordered_hypotheses
    )
    return posterior, concrete
end

"""Compute normalized CGES hypothesis probabilities for one question."""
function _row_cges_posterior(ans_row, score_row)::Dict{Any, Float64}
    posterior, _ = _row_cges_details(ans_row, score_row)
    return posterior
end

"""
    cges_vote(answers, scores; allow_other=false, return_index=false, return_score=false)

Select the answer having the largest CGES posterior score. Candidate indices
follow Python's 0-based convention and use `-1` for `CGES_OTHER` or an empty
row.
"""
function cges_vote(
    answers,
    scores;
    allow_other=false,
    return_index::Bool=false,
    return_score::Bool=false,
)
    _validate_bool(allow_other, "allow_other")
    Z, S, single = _normalize_candidates(answers, scores; require_scores=true)

    selected = Any[]
    indices = Int[]
    selected_scores = Float64[]
    for i in axes(Z, 1)
        row = @view Z[i, :]
        score_row = @view S[i, :]
        part = _valid_indices(row)
        if isempty(part)
            push!(selected, nothing)
            push!(indices, 0)
            push!(selected_scores, NaN)
            continue
        end

        posterior, concrete = _row_cges_details(row, score_row)
        hypotheses = allow_other ? Any[concrete...; CGES_OTHER] : concrete
        winner = hypotheses[1]
        for answer in @view hypotheses[2:end]
            posterior[answer] > posterior[winner] && (winner = answer)
        end
        push!(selected, winner)

        if winner === CGES_OTHER
            push!(indices, 0)
            push!(selected_scores, NaN)
            continue
        end

        representative = 0
        for j in part
            row[j] == winner || continue
            if representative == 0 || Float64(score_row[j]) > Float64(score_row[representative])
                representative = j
            end
        end
        push!(indices, representative)
        push!(selected_scores, Float64(score_row[representative]))
    end

    return _pack_selection(
        selected,
        indices,
        selected_scores,
        single;
        return_index=return_index,
        return_score=return_score,
    )
end

"""
    cges_stop(answers, scores; threshold=0.95, include_other=false,
              min_samples=1, return_prob=false)

Stop one sampling stream once the largest checked CGES posterior reaches the
requested threshold.
"""
function cges_stop(
    answers,
    scores;
    threshold::Real=0.95,
    include_other=false,
    min_samples=1,
    return_prob::Bool=false,
)
    threshold_f = Float64(threshold)
    0.0 < threshold_f < 1.0 ||
        error("threshold must be in (0, 1); got $threshold.")
    _validate_bool(include_other, "include_other")
    (min_samples isa Integer && !(min_samples isa Bool) && min_samples >= 1) ||
        error("min_samples must be an integer >= 1; got $min_samples.")

    Z, S, single = _normalize_candidates(answers, scores; require_scores=true)
    single || error("cges_stop expects one 1D sampling stream, not a batch.")
    row = @view Z[1, :]
    score_row = @view S[1, :]
    part = _valid_indices(row)
    if isempty(part)
        return return_prob ? (false, 0.0) : false
    end

    posterior, concrete = _row_cges_details(row, score_row)
    hypotheses = include_other ? Any[concrete...; CGES_OTHER] : concrete
    probability = maximum(posterior[answer] for answer in hypotheses)
    stop = length(part) >= Int(min_samples) && probability >= threshold_f
    return return_prob ? (stop, probability) : stop
end
