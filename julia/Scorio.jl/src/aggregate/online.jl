# Online candidate-sampling and within-trace stopping rules.

using SpecialFunctions: beta_inc, gamma_inc, gamma_inc_inv

function _ordered_answer_counts(answers)::Vector{Int}
    tally = Dict{Any, Int}()
    order = Any[]
    for answer in answers
        _is_valid_answer(answer) || continue
        if !haskey(tally, answer)
            push!(order, answer)
            tally[answer] = 0
        end
        tally[answer] += 1
    end
    return Int[tally[answer] for answer in order]
end

function _top_two_answer_counts(answers)::Tuple{Int, Int}
    counts = sort(_ordered_answer_counts(answers); rev=true)
    first = isempty(counts) ? 0 : counts[1]
    second = length(counts) < 2 ? 0 : counts[2]
    return first, second
end

function _adaptive_simpson(
    fn,
    left::Float64,
    right::Float64,
    tolerance::Float64,
    whole::Float64,
    f_left::Float64,
    f_middle::Float64,
    f_right::Float64,
    depth::Int,
)::Float64
    middle = (left + right) / 2.0
    left_middle = (left + middle) / 2.0
    right_middle = (middle + right) / 2.0
    f_left_middle = Float64(fn(left_middle))
    f_right_middle = Float64(fn(right_middle))
    left_area = (middle - left) * (f_left + 4.0 * f_left_middle + f_middle) / 6.0
    right_area =
        (right - middle) * (f_middle + 4.0 * f_right_middle + f_right) / 6.0
    delta = left_area + right_area - whole
    if depth <= 0 || abs(delta) <= 15.0 * tolerance
        return left_area + right_area + delta / 15.0
    end
    return _adaptive_simpson(
        fn,
        left,
        middle,
        tolerance / 2.0,
        left_area,
        f_left,
        f_left_middle,
        f_middle,
        depth - 1,
    ) + _adaptive_simpson(
        fn,
        middle,
        right,
        tolerance / 2.0,
        right_area,
        f_middle,
        f_right_middle,
        f_right,
        depth - 1,
    )
end

"""Probability that the fixed-tie-broken count leader is Dirichlet-largest."""
function _dirichlet_leader_probability(counts)::Float64
    isempty(counts) && return 0.0
    values = Int[count for count in counts]
    leader = 1
    for i in 2:length(values)
        values[i] > values[leader] && (leader = i)
    end
    alpha_leader = Float64(values[leader] + 1)
    other = Float64[values[i] + 1 for i in eachindex(values) if i != leader]
    all(==(alpha_leader), other) && return 1.0 / length(values)

    cache = Dict{Float64, Float64}()
    function integrand(quantile::Float64)::Float64
        quantile <= 0.0 && return 0.0
        quantile >= 1.0 && return 1.0
        haskey(cache, quantile) && return cache[quantile]
        value = gamma_inc_inv(alpha_leader, quantile, 1.0 - quantile)
        log_product = 0.0
        for shape in other
            cdf = gamma_inc(shape, value)[1]
            if cdf <= 0.0
                cache[quantile] = 0.0
                return 0.0
            end
            log_product += log(cdf)
        end
        result = exp(log_product)
        cache[quantile] = result
        return result
    end

    f_left = 0.0
    f_middle = integrand(0.5)
    f_right = 1.0
    whole = (f_left + 4.0 * f_middle + f_right) / 6.0
    probability = _adaptive_simpson(
        integrand,
        0.0,
        1.0,
        1e-10,
        whole,
        f_left,
        f_middle,
        f_right,
        # The probability-integral transform can leave a very narrow boundary
        # layer next to zero when the observed leader is overwhelming.  A
        # depth of 22 only resolves intervals down to about 2^-22 and therefore
        # biased probabilities near one downward by roughly 2e-8.  Permit
        # bisection down to Float64's useful resolution, matching SciPy quad's
        # ability to resolve that endpoint while retaining the same 1e-10
        # absolute tolerance.
        50,
    )
    isfinite(probability) ||
        error("Dirichlet leader-probability integration did not converge.")
    return clamp(probability, 0.0, 1.0)
end

"""Full observed-support Dirichlet Adaptive-Consistency stopping rule."""
function adaptive_consistency_dirichlet_stop(
    answers;
    threshold::Real=0.95,
    return_prob::Bool=false,
)
    threshold_f = Float64(threshold)
    0.0 < threshold_f < 1.0 ||
        error("threshold must be in (0, 1); got $threshold.")
    counts = _ordered_answer_counts(answers)
    probability = if isempty(counts)
        0.0
    elseif length(counts) < 3
        ordered = sort(counts; rev=true)
        v1 = ordered[1]
        v2 = length(ordered) > 1 ? ordered[2] : 0
        1.0 - beta_inc(v1 + 1.0, v2 + 1.0, 0.5)[1]
    else
        _dirichlet_leader_probability(counts)
    end
    stop = !isempty(counts) && probability >= threshold_f
    return return_prob ? (stop, Float64(probability)) : stop
end

const _CRP_MAX_CHUNK_SIZE = 8192
const _CRP_TARGET_CELLS = 500_000

function _crp_leader_probability(
    counts::Vector{Int};
    horizon::Int,
    n_alpha::Int,
    n_simulations::Int,
    seed,
)::Float64
    observed = sum(counts)
    remaining = horizon - observed
    leader = 1
    for i in 2:length(counts)
        counts[i] > counts[leader] && (leader = i)
    end
    rng = _NumpyRNG(seed)
    rate = 1.0 + Base.MathConstants.eulergamma + _numpy_log(Float64(observed))
    scale = 1.0 / rate
    alpha_draws = Vector{Float64}(undef, n_alpha)
    for draw in eachindex(alpha_draws)
        alpha_draws[draw] =
            _numpy_standard_gamma!(rng, Float64(length(counts))) * scale
    end

    n_runs = n_alpha * n_simulations
    row_width = 3 * (length(counts) + 1) + 2 * remaining
    chunk_size = min(
        _CRP_MAX_CHUNK_SIZE,
        max(1, _CRP_TARGET_CELLS ÷ max(1, row_width)),
    )

    successes = 0
    start = 0
    while start < n_runs
        stop = min(start + chunk_size, n_runs)
        batch_size = stop - start
        alpha = Vector{Float64}(undef, batch_size)
        for row in 1:batch_size
            alpha_index = (start + row - 1) ÷ n_simulations + 1
            alpha[row] = alpha_draws[alpha_index]
        end

        categories = length(counts) + 1
        masses = Matrix{Float64}(undef, batch_size, categories)
        for row in 1:batch_size
            for category in eachindex(counts)
                masses[row, category] =
                    _numpy_standard_gamma!(rng, Float64(counts[category]))
            end
        end
        for row in 1:batch_size
            masses[row, categories] = _numpy_standard_gamma!(rng, alpha[row])
        end
        for row in 1:batch_size
            total = 0.0
            for category in 1:categories
                total += masses[row, category]
            end
            for category in 1:categories
                masses[row, category] /= total
            end
        end

        allocations = zeros(Int64, batch_size, categories)
        for row in 1:batch_size
            draw = _numpy_multinomial!(rng, remaining, @view(masses[row, :]))
            for category in 1:categories
                allocations[row, category] = draw[category]
            end
        end

        labels = Matrix{Int32}(undef, batch_size, remaining)
        new_counts = zeros(Int32, batch_size, remaining)
        active_clusters = zeros(Int32, batch_size)
        for customer in 0:(remaining - 1)
            any_active = false
            for row in 1:batch_size
                allocations[row, categories] > customer || continue
                any_active = true
                draw = _numpy_uniform!(rng) * (customer + alpha[row])
                choice = if draw < customer
                    labels[row, trunc(Int, draw) + 1]
                else
                    cluster = active_clusters[row]
                    active_clusters[row] += Int32(1)
                    cluster
                end
                labels[row, customer + 1] = choice
                new_counts[row, Int(choice) + 1] += Int32(1)
            end
            any_active || break
        end

        for row in 1:batch_size
            final_leader = 1
            for category in 2:length(counts)
                candidate = allocations[row, category] + counts[category]
                incumbent = allocations[row, final_leader] + counts[final_leader]
                candidate > incumbent && (final_leader = category)
            end
            final_leader == leader || continue

            largest_unseen = Int32(0)
            for cluster in 1:remaining
                new_counts[row, cluster] > largest_unseen &&
                    (largest_unseen = new_counts[row, cluster])
            end
            leader_count = allocations[row, leader] + counts[leader]
            leader_count >= largest_unseen && (successes += 1)
        end

        start = stop
    end
    return successes / n_runs
end

"""Finite-horizon CRP Adaptive-Consistency Monte Carlo stopping rule."""
function adaptive_consistency_crp_stop(
    answers;
    threshold::Real=0.95,
    horizon=40,
    n_alpha=100,
    n_simulations=1000,
    seed=0,
    return_prob::Bool=false,
)
    threshold_f = Float64(threshold)
    0.0 < threshold_f < 1.0 ||
        error("threshold must be in (0, 1); got $threshold.")
    for (value, name) in (
        (horizon, "horizon"),
        (n_alpha, "n_alpha"),
        (n_simulations, "n_simulations"),
    )
        (value isa Integer && !(value isa Bool)) ||
            error("$name must be an integer >= 1; got $value.")
        value >= 1 || error("$name must be >= 1; got $value.")
    end
    if seed !== nothing
        (seed isa Integer && !(seed isa Bool) && seed >= 0) ||
            error("seed must be a non-negative integer or nothing; got $seed.")
    end

    counts = _ordered_answer_counts(answers)
    if isempty(counts)
        return return_prob ? (false, 0.0) : false
    end
    observed = sum(counts)
    if observed >= Int(horizon)
        return return_prob ? (true, 1.0) : true
    end
    probability = _crp_leader_probability(
        counts;
        horizon=Int(horizon),
        n_alpha=Int(n_alpha),
        n_simulations=Int(n_simulations),
        seed=seed,
    )
    stop = probability >= threshold_f
    return return_prob ? (stop, probability) : stop
end

"""
    adaptive_consistency_stop(answers; threshold=0.95, return_prob=false)

Stop candidate sampling when the Beta posterior probability that the leading
answer beats the runner-up reaches `threshold`.
"""
function adaptive_consistency_stop(
    answers;
    threshold::Real=0.95,
    return_prob::Bool=false,
)
    threshold_f = Float64(threshold)
    0.0 < threshold_f < 1.0 ||
        error("threshold must be in (0, 1); got $threshold.")
    v1, v2 = _top_two_answer_counts(answers)
    probability =
        v1 == 0 ? 0.0 : 1.0 - beta_inc(v1 + 1.0, v2 + 1.0, 0.5)[1]
    stop = probability >= threshold_f
    return return_prob ? (stop, probability) : stop
end

"""Early-Stopping Self-Consistency: true for a nonempty unanimous valid window."""
function esc_stop(window_answers)::Bool
    answers = collect(window_answers)
    isempty(answers) && return false
    first = answers[1]
    _is_valid_answer(first) || return false
    return all(a -> _is_valid_answer(a) && a == first, @view answers[2:end])
end

"""DeepConf warmup threshold: the `(1 - keep)` linear quantile."""
function deepconf_stop_threshold(warmup_confidences; keep::Real=0.1)::Float64
    keep_f = Float64(keep)
    0.0 < keep_f <= 1.0 || error("keep must be in (0, 1]; got $keep.")
    confidences = _flatten_numeric(warmup_confidences, "warmup_confidences")
    isempty(confidences) && error("need at least one warmup confidence.")
    all(isfinite, confidences) || error("warmup_confidences must all be finite.")
    return _quantile_linear(confidences, 1.0 - keep_f)
end

"""
    deepconf_online_stop(topk_logprobs, threshold; window=2048)

Return Python's 0-based end-token index for the first below-threshold confidence
window, or `nothing` if generation runs to completion.
"""
function deepconf_online_stop(topk_logprobs, threshold::Real; window=2048)
    confidence = token_confidence(topk_logprobs)
    w = min(_python_int(window), length(confidence))
    groups = _group_confidences(confidence, w)
    first_below = findfirst(x -> x < Float64(threshold), groups)
    first_below === nothing && return nothing
    return first_below + w - 2
end
