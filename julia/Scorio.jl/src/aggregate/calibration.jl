# KDE-calibrated weighted voting for scalar verifier probabilities.

function _probability_values(values, name::AbstractString)
    flat, shape = try
        _flatten_numeric_with_shape(values, name)
    catch
        error("$name must contain numeric values.")
    end
    all(value -> isfinite(value) && 0.0 < value < 1.0, flat) ||
        error("$name must all be finite and strictly in (0, 1).")
    return flat, shape
end

_logit(value::Real)::Float64 = log(Float64(value)) - log1p(-Float64(value))

function _reshape_c_order(values::Vector{Float64}, shape::Tuple)
    isempty(shape) && return values[1]
    length(shape) == 1 && return copy(values)
    reversed = reshape(copy(values), reverse(shape))
    return permutedims(reversed, reverse(1:length(shape)))
end

function _log_gaussian_kde(
    query_logits::Vector{Float64},
    samples::Vector{Float64},
    bandwidth::Float64,
)::Vector{Float64}
    normalizer = log(length(samples) * bandwidth * sqrt(2.0 * pi))
    output = Vector{Float64}(undef, length(query_logits))
    terms = Vector{Float64}(undef, length(samples))
    for i in eachindex(query_logits)
        value = query_logits[i]
        for j in eachindex(samples)
            standardized = (value - samples[j]) / bandwidth
            terms[j] = -0.5 * standardized * standardized
        end
        output[i] = _logsumexp(terms) - normalizer
    end
    return output
end

"""Immutable fitted state for non-parametric KDE weighted voting."""
struct KDEVoteCalibration
    _correct_logits::Tuple{Vararg{Float64}}
    _incorrect_logits::Tuple{Vararg{Float64}}
    _correct_bandwidth::Float64
    _incorrect_bandwidth::Float64
    _bin_edges::Tuple{Vararg{Float64}}
    _bin_probability::Tuple{Vararg{Float64}}
    _kernel::String
    _binning::String

    function KDEVoteCalibration(
        correct_logits::Tuple{Vararg{Float64}},
        incorrect_logits::Tuple{Vararg{Float64}},
        correct_bandwidth::Float64,
        incorrect_bandwidth::Float64,
        bin_edges::Tuple{Vararg{Float64}},
        bin_probability::Tuple{Vararg{Float64}},
        kernel::String,
        binning::String,
        ::Val{:validated},
    )
        return new(
            correct_logits,
            incorrect_logits,
            correct_bandwidth,
            incorrect_bandwidth,
            bin_edges,
            bin_probability,
            kernel,
            binning,
        )
    end
end

function KDEVoteCalibration(;
    correct_logits,
    incorrect_logits,
    correct_bandwidth,
    incorrect_bandwidth,
    bin_edges,
    bin_probability,
    kernel="gaussian",
    binning="quantile",
)
    correct, correct_shape = try
        _flatten_numeric_with_shape(correct_logits, "correct_logits")
    catch
        error("correct_logits must be one-dimensional.")
    end
    incorrect, incorrect_shape = try
        _flatten_numeric_with_shape(incorrect_logits, "incorrect_logits")
    catch
        error("incorrect_logits must be one-dimensional.")
    end
    edges, edges_shape = try
        _flatten_numeric_with_shape(bin_edges, "bin_edges")
    catch
        error("bin_edges must be one-dimensional.")
    end
    probabilities, probabilities_shape = try
        _flatten_numeric_with_shape(bin_probability, "bin_probability")
    catch
        error("bin_probability must be one-dimensional.")
    end
    length(correct_shape) == 1 || error("correct_logits must be one-dimensional.")
    length(incorrect_shape) == 1 || error("incorrect_logits must be one-dimensional.")
    length(edges_shape) == 1 || error("bin_edges must be one-dimensional.")
    length(probabilities_shape) == 1 ||
        error("bin_probability must be one-dimensional.")

    if isempty(correct) || isempty(incorrect)
        error("KDE calibration needs correct and incorrect samples.")
    end
    all(isfinite, correct) && all(isfinite, incorrect) ||
        error("KDE logit samples must all be finite.")

    correct_bw = try
        Float64(correct_bandwidth)
    catch
        NaN
    end
    incorrect_bw = try
        Float64(incorrect_bandwidth)
    catch
        NaN
    end
    isfinite(correct_bw) && correct_bw > 0.0 &&
        isfinite(incorrect_bw) && incorrect_bw > 0.0 ||
        error("KDE bandwidths must be finite and > 0.")

    length(edges) == length(probabilities) + 1 && length(edges) >= 2 ||
        error("bin_edges must contain exactly one more value than bins.")
    isinf(edges[1]) && edges[1] < 0.0 ||
        error("bin_edges must start at -inf.")
    isinf(edges[end]) && edges[end] > 0.0 ||
        error("bin_edges must end at +inf.")
    all(edges[i] > edges[i - 1] for i in 2:length(edges)) ||
        error("bin_edges must be strictly increasing.")
    all(value -> isfinite(value) && 0.0 <= value <= 1.0, probabilities) ||
        error("bin_probability values must be finite and in [0, 1].")
    kernel == "gaussian" ||
        error("only the implemented 'gaussian' kernel is valid.")
    binning == "quantile" ||
        error("only the implemented 'quantile' binning is valid.")

    return KDEVoteCalibration(
        Tuple(correct),
        Tuple(incorrect),
        correct_bw,
        incorrect_bw,
        Tuple(edges),
        Tuple(probabilities),
        String(kernel),
        String(binning),
        Val(:validated),
    )
end

# Python's dataclass constructor accepts the state positionally as well as by
# keyword. Route the positional spelling through the same validating path.
function KDEVoteCalibration(
    correct_logits,
    incorrect_logits,
    correct_bandwidth,
    incorrect_bandwidth,
    bin_edges,
    bin_probability,
    kernel="gaussian",
    binning="quantile",
)
    return KDEVoteCalibration(
        correct_logits=correct_logits,
        incorrect_logits=incorrect_logits,
        correct_bandwidth=correct_bandwidth,
        incorrect_bandwidth=incorrect_bandwidth,
        bin_edges=bin_edges,
        bin_probability=bin_probability,
        kernel=kernel,
        binning=binning,
    )
end

function Base.getproperty(calibration::KDEVoteCalibration, name::Symbol)
    if name === :correct_logits
        return collect(getfield(calibration, :_correct_logits))
    elseif name === :incorrect_logits
        return collect(getfield(calibration, :_incorrect_logits))
    elseif name === :correct_bandwidth
        return getfield(calibration, :_correct_bandwidth)
    elseif name === :incorrect_bandwidth
        return getfield(calibration, :_incorrect_bandwidth)
    elseif name === :bin_edges
        return collect(getfield(calibration, :_bin_edges))
    elseif name === :bin_probability
        return collect(getfield(calibration, :_bin_probability))
    elseif name === :kernel
        return getfield(calibration, :_kernel)
    elseif name === :binning
        return getfield(calibration, :_binning)
    elseif name === :n_bins
        return length(getfield(calibration, :_bin_probability))
    elseif name === :calibrated_probability
        return scores -> calibrated_probability(calibration, scores)
    elseif name === :log_density_ratio
        return scores -> log_density_ratio(calibration, scores)
    elseif name === :weights
        return (scores; n_answers) -> weights(calibration, scores; n_answers=n_answers)
    end
    return getfield(calibration, name)
end

function Base.propertynames(::KDEVoteCalibration, private::Bool=false)
    public = (
        :correct_logits,
        :incorrect_logits,
        :correct_bandwidth,
        :incorrect_bandwidth,
        :bin_edges,
        :bin_probability,
        :kernel,
        :binning,
        :n_bins,
        :calibrated_probability,
        :log_density_ratio,
        :weights,
    )
    private || return public
    return (public..., fieldnames(KDEVoteCalibration)...)
end

function calibrated_probability(calibration::KDEVoteCalibration, scores)
    values, shape = _probability_values(scores, "scores")
    edges = collect(getfield(calibration, :_bin_edges))
    probabilities = getfield(calibration, :_bin_probability)
    internal = @view edges[2:(end - 1)]
    output = Float64[
        probabilities[searchsortedlast(internal, value) + 1] for value in values
    ]
    return _reshape_c_order(output, shape)
end

function log_density_ratio(calibration::KDEVoteCalibration, scores)
    values, shape = _probability_values(scores, "scores")
    logits = _logit.(values)
    correct = collect(getfield(calibration, :_correct_logits))
    incorrect = collect(getfield(calibration, :_incorrect_logits))
    log_correct = _log_gaussian_kde(
        logits,
        correct,
        getfield(calibration, :_correct_bandwidth),
    )
    log_incorrect = _log_gaussian_kde(
        logits,
        incorrect,
        getfield(calibration, :_incorrect_bandwidth),
    )
    return _reshape_c_order(log_correct .- log_incorrect, shape)
end

function weights(calibration::KDEVoteCalibration, scores; n_answers)
    (n_answers isa Integer && !(n_answers isa Bool)) ||
        error("n_answers must be an integer >= 2; got $n_answers.")
    n_answers >= 2 ||
        error("n_answers must be >= 2 for the KDE weight formula.")
    values, shape = _probability_values(scores, "scores")
    length(shape) == 1 && !isempty(values) ||
        error("scores must be a nonempty 1D response pool.")
    probabilities = calibrated_probability(calibration, values)
    q_hat = sum(probabilities) / length(probabilities)
    offset = if q_hat == 0.0
        -Inf
    elseif q_hat == 1.0
        Inf
    else
        log(q_hat) + log(Int(n_answers) - 1) - log1p(-q_hat)
    end
    return log_density_ratio(calibration, values) .+ offset
end

function _resolve_bandwidth(samples::Vector{Float64}, specification, label::AbstractString)
    if specification isa AbstractString
        specification == "scott" ||
            error("bandwidth must be a positive number, pair, or 'scott'.")
        length(samples) >= 2 || error(
            "Scott bandwidth for $label needs at least two samples; " *
            "supply an explicit bandwidth instead.",
        )
        sample_mean = sum(samples) / length(samples)
        standard_deviation = sqrt(
            sum((value - sample_mean)^2 for value in samples) / (length(samples) - 1),
        )
        isfinite(standard_deviation) && standard_deviation > 0.0 || error(
            "Scott bandwidth is undefined for constant $label logits; " *
            "supply an explicit positive bandwidth instead.",
        )
        return standard_deviation * length(samples)^(-1.0 / 5.0)
    end

    value = try
        Float64(specification)
    catch
        error("bandwidth must be a positive number, pair, or 'scott'.")
    end
    isfinite(value) && value > 0.0 ||
        error("bandwidth must be finite and > 0; got $specification.")
    return value
end

function _bandwidth_specifications(bandwidth)
    if bandwidth isa AbstractArray && ndims(bandwidth) == 0
        scalar = bandwidth[]
        return scalar, scalar
    elseif bandwidth isa Tuple || bandwidth isa AbstractVector
        length(bandwidth) == 2 ||
            error("a bandwidth sequence must be (correct, incorrect).")
        return bandwidth[1], bandwidth[2]
    end
    return bandwidth, bandwidth
end

"""Fit class-conditional Gaussian KDEs and a quantile-binned calibrator."""
function fit_kde_vote_calibration(
    scores,
    correct;
    n_bins=10,
    bandwidth="scott",
)::KDEVoteCalibration
    (n_bins isa Integer && !(n_bins isa Bool) && n_bins >= 1) ||
        error("n_bins must be an integer >= 1; got $n_bins.")

    score_values, score_shape = _probability_values(scores, "scores")
    correct_values, correct_shape = try
        _flatten_numeric_with_shape(correct, "correct")
    catch
        error("correct must contain only boolean or 0/1 values.")
    end
    score_shape == correct_shape || error(
        "scores and correct must have the same shape; got $score_shape and $correct_shape.",
    )
    all(value -> isfinite(value) && (value == 0.0 || value == 1.0), correct_values) ||
        error("correct must contain only boolean or 0/1 values.")
    isempty(score_values) && error("need at least one calibration response.")

    correct_mask = Bool[value == 1.0 for value in correct_values]
    any(correct_mask) && !all(correct_mask) ||
        error("KDE calibration needs correct and incorrect responses.")
    logits = _logit.(score_values)
    correct_logits = logits[correct_mask]
    incorrect_logits = logits[.!correct_mask]

    correct_spec, incorrect_spec = _bandwidth_specifications(bandwidth)
    correct_bandwidth = _resolve_bandwidth(
        correct_logits,
        correct_spec,
        "correct-class",
    )
    incorrect_bandwidth = _resolve_bandwidth(
        incorrect_logits,
        incorrect_spec,
        "incorrect-class",
    )

    sorted_scores = sort(score_values)
    quantiles = Vector{Float64}(undef, Int(n_bins) + 1)
    for i in 0:Int(n_bins)
        q = i / Int(n_bins)
        # NumPy's method="nearest" selects the nearest observation at
        # `(n - 1)q`, using ties-to-even rounding.
        observation = round(Int, (length(sorted_scores) - 1) * q) + 1
        quantiles[i + 1] = sorted_scores[observation]
    end
    minimum_score = sorted_scores[1]
    maximum_score = sorted_scores[end]
    internal = unique(quantiles[2:(end - 1)])
    filter!(value -> minimum_score < value < maximum_score, internal)
    edges = Float64[-Inf; internal; Inf]

    bin_indices = Int[searchsortedlast(internal, value) + 1 for value in score_values]
    probabilities = Vector{Float64}(undef, length(edges) - 1)
    for bin_index in eachindex(probabilities)
        members = findall(==(bin_index), bin_indices)
        isempty(members) &&
            error("internal quantile construction produced an empty bin.")
        probabilities[bin_index] =
            sum(correct_mask[index] for index in members) / length(members)
    end

    return KDEVoteCalibration(
        correct_logits=correct_logits,
        incorrect_logits=incorrect_logits,
        correct_bandwidth=correct_bandwidth,
        incorrect_bandwidth=incorrect_bandwidth,
        bin_edges=edges,
        bin_probability=probabilities,
    )
end

function _row_kde_vote(ans_row, score_row, calibration::KDEVoteCalibration)
    part = _valid_indices(ans_row)
    isempty(part) && return nothing, 0
    values, _ = _probability_values(
        Float64[score_row[j] for j in part],
        "valid scores",
    )

    local_members = Dict{Any, Vector{Int}}()
    representatives = Dict{Any, Int}()
    first_indices = Dict{Any, Int}()
    labels = Any[]
    for (local_index, j) in enumerate(part)
        answer = ans_row[j]
        if !haskey(local_members, answer)
            push!(labels, answer)
            local_members[answer] = Int[]
            representatives[answer] = j
            first_indices[answer] = j
        end
        push!(local_members[answer], local_index)
        if Float64(score_row[j]) > Float64(score_row[representatives[answer]])
            representatives[answer] = j
        end
    end

    length(labels) == 1 && return labels[1], representatives[labels[1]]

    density_ratio = log_density_ratio(calibration, values)
    calibrated = calibrated_probability(calibration, values)
    q_hat = sum(calibrated) / length(calibrated)
    n_answers = length(labels)

    function key(answer)
        members = local_members[answer]
        count = length(members)
        ratio_sum = sum(density_ratio[index] for index in members)
        if q_hat == 1.0
            return Float64(count), Float64(ratio_sum)
        elseif q_hat == 0.0
            return Float64(-count), Float64(ratio_sum)
        end
        offset = log(q_hat) + log(n_answers - 1) - log1p(-q_hat)
        return Float64(ratio_sum + count * offset), 0.0
    end

    winner = labels[1]
    winner_key = key(winner)
    for answer in @view labels[2:end]
        answer_key = key(answer)
        if answer_key[1] > winner_key[1] ||
           (answer_key[1] == winner_key[1] && answer_key[2] > winner_key[2])
            winner = answer
            winner_key = answer_key
        end
    end
    return winner, representatives[winner]
end

"""Select answers using a fitted non-parametric KDE vote calibration."""
function kde_weighted_vote(
    answers,
    scores,
    calibration;
    return_index::Bool=false,
    return_score::Bool=false,
)
    calibration isa KDEVoteCalibration ||
        throw(ArgumentError("calibration must be a KDEVoteCalibration."))
    Z, S, single = _normalize_candidates(answers, scores; require_scores=true)
    return _run_score_rule(
        (row, score_row) -> _row_kde_vote(row, score_row, calibration),
        Z,
        S,
        single;
        return_index=return_index,
        return_score=return_score,
    )
end
