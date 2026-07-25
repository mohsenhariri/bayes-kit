# Shared input normalization, validity, tie-breaking, and return packing.

"""Whether an answer label participates in aggregation."""
function _is_valid_answer(a)::Bool
    if a === nothing || a === missing
        return false
    elseif a isa AbstractString
        return !isempty(a)
    elseif a isa AbstractFloat
        return !isnan(a)
    end
    return true
end

# Convert a vector-of-rows (the natural Julia spelling of a nested Python list)
# into an M × N matrix, while leaving ordinary vectors as single-question input.
function _answers_array(answers)
    if answers isa AbstractMatrix
        return Matrix{Any}(answers), false
    elseif answers isa AbstractArray && ndims(answers) != 1
        error("answers must be a 1D (N,) or 2D (M, N) array.")
    end

    xs = try
        collect(answers)
    catch
        error("answers must be a 1D (N,) or 2D (M, N) array.")
    end
    if !isempty(xs) && all(x -> x isa AbstractVector || x isa Tuple, xs)
        rows = [collect(x) for x in xs]
        n = length(rows[1])
        all(length(r) == n for r in rows) ||
            error("answers must be a rectangular 1D (N,) or 2D (M, N) array.")
        Z = Matrix{Any}(undef, length(rows), n)
        for i in eachindex(rows), j in 1:n
            Z[i, j] = rows[i][j]
        end
        return Z, false
    end

    Z = Matrix{Any}(undef, 1, length(xs))
    for j in eachindex(xs)
        Z[1, j] = xs[j]
    end
    return Z, true
end

function _scores_array(scores, target_shape::Tuple{Int, Int})
    if scores isa AbstractMatrix
        S = try
            Float64.(scores)
        catch
            error("scores must be numeric.")
        end
    elseif scores isa AbstractArray && ndims(scores) != 1
        error("answers and scores must have the same shape.")
    else
        xs = try
            collect(scores)
        catch
            error("scores must be numeric and have the same shape as answers.")
        end
        if !isempty(xs) && all(x -> x isa AbstractVector || x isa Tuple, xs)
            rows = [collect(x) for x in xs]
            n = length(rows[1])
            all(length(r) == n for r in rows) ||
                error("answers and scores must have the same shape.")
            S = Matrix{Float64}(undef, length(rows), n)
            try
                for i in eachindex(rows), j in 1:n
                    S[i, j] = Float64(rows[i][j])
                end
            catch
                error("scores must be numeric.")
            end
        else
            vals = try
                Float64.(xs)
            catch
                error("scores must be numeric.")
            end
            S = reshape(vals, 1, :)
        end
    end
    size(S) == target_shape ||
        error("answers and scores must have the same shape; got $target_shape and $(size(S)).")
    return Matrix{Float64}(S)
end

"""Normalize candidate inputs to M × N matrices and flag vector input."""
function _normalize_candidates(answers, scores=nothing; require_scores::Bool=false)
    Z, single = _answers_array(answers)
    size(Z, 2) > 0 || error("need at least one candidate per question (N >= 1).")
    if scores === nothing
        require_scores && error("scores are required for this selection rule.")
        return Z, nothing, single
    end
    return Z, _scores_array(scores, size(Z)), single
end

_valid_indices(row) = Int[j for j in eachindex(row) if _is_valid_answer(row[j])]

_finalize(values::Vector, single::Bool) = single ? values[1] : values

function _pack_selection(
    selected::Vector{Any},
    indices::Vector{Int},
    selected_scores::Vector{Float64},
    single::Bool;
    return_index::Bool,
    return_score::Bool,
)
    selection = _finalize(selected, single)
    # Aggregate intentionally exposes Python-compatible candidate indices:
    # internal kernels use Julia's 1-based indices and `0` sentinel, while the
    # public API returns 0-based indices and `-1` for an all-invalid row.
    public_indices = Int[index - 1 for index in indices]
    index = _finalize(public_indices, single)
    score = _finalize(selected_scores, single)
    if return_index && return_score
        return selection, index, score
    elseif return_index
        return selection, index
    elseif return_score
        return selection, score
    end
    return selection
end

_default_m(n::Integer)::Int = max(1, isqrt(n))

"""Stable log-sum-exp for a nonempty finite vector."""
function _logsumexp(values)::Float64
    maximum_value = maximum(values)
    maximum_value == -Inf && return -Inf
    return Float64(maximum_value + log(sum(exp(value - maximum_value) for value in values)))
end

function _python_int(x)::Int
    if x isa Integer
        return Int(x)
    elseif x isa Real && isfinite(x)
        return trunc(Int, x)
    end
    error("value must be convertible to an integer; got $x.")
end

# NumPy's ``asarray(..., dtype=float).reshape(-1)`` accepts scalars and
# rectangular arbitrary-dimensional inputs, rejects ragged nesting, and
# traverses arrays in C order (last axis fastest). Julia's native ``vec`` is
# column-major, so preserve both the nested shape and explicit traversal order.
function _flatten_numeric_with_shape(x, name::AbstractString)
    x isa Real && return Float64[x], ()

    values, outer_shape = if x isa AbstractArray && ndims(x) > 1
        # Reversing axes before Julia's column-major vec gives C-order traversal.
        (
            vec(permutedims(collect(x), reverse(1:ndims(x)))),
            Tuple(size(x)),
        )
    else
        collected = try
            collect(x)
        catch
            error("$name must be numeric and rectangular.")
        end
        collected, (length(collected),)
    end

    isempty(values) && return Float64[], outer_shape
    if all(value -> value isa Real, values)
        return Float64.(values), outer_shape
    end
    flattened = Vector{Vector{Float64}}(undef, length(values))
    child_shapes = Vector{Tuple}(undef, length(values))
    for i in eachindex(values)
        flattened[i], child_shapes[i] = _flatten_numeric_with_shape(values[i], name)
    end
    child_shape = child_shapes[1]
    all(shape -> shape == child_shape, child_shapes) ||
        error("$name must be numeric and rectangular; ragged inputs are not supported.")
    return reduce(vcat, flattened; init=Float64[]), (outer_shape..., child_shape...)
end

function _flatten_numeric(x, name::AbstractString)::Vector{Float64}
    values, _ = _flatten_numeric_with_shape(x, name)
    return values
end

"""Resolve a fractional or integer filtered-vote cutoff."""
function _keep_count(keep, n::Int)::Int
    keep isa Bool &&
        error("keep must be a float in (0, 1] or an int >= 1; got a bool.")
    if keep isa Integer && keep >= 1
        return min(Int(keep), n)
    elseif keep isa AbstractFloat && 0.0 < Float64(keep) <= 1.0
        return max(1, ceil(Int, Float64(keep) * n - 1e-9))
    end
    error("keep must be a float fraction in (0, 1] or an int count >= 1; got $keep.")
end

# Weighted plurality over a list of candidate indices. Group-weight ties stay
# with the first encountered group; representatives are highest-score members,
# with score ties staying at the lowest index.
function _plurality(ans_row, part::Vector{Int}, weight_of, score_of)
    isempty(part) && return nothing, 0
    totals = Dict{Any, Any}()
    representatives = Dict{Any, Int}()
    first_indices = Dict{Any, Int}()
    labels = Any[]
    for j in part
        a = ans_row[j]
        w = weight_of(j)
        if !haskey(totals, a)
            push!(labels, a)
            totals[a] = zero(w)
            representatives[a] = j
            first_indices[a] = j
        end
        totals[a] += w
        if score_of(j) > score_of(representatives[a])
            representatives[a] = j
        end
    end
    best = labels[1]
    for a in @view labels[2:end]
        if totals[a] > totals[best] ||
           (totals[a] == totals[best] && first_indices[a] < first_indices[best])
            best = a
        end
    end
    return best, representatives[best]
end

function _run_score_rule(row_fn, Z, S, single; return_index::Bool, return_score::Bool)
    selected = Any[]
    indices = Int[]
    selected_scores = Float64[]
    for i in axes(Z, 1)
        a, idx = row_fn(@view(Z[i, :]), @view(S[i, :]))
        push!(selected, a)
        push!(indices, idx)
        push!(selected_scores, idx == 0 ? NaN : Float64(S[i, idx]))
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
