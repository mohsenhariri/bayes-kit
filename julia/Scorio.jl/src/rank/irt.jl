"""Item Response Theory (IRT) ranking methods."""

using LinearAlgebra

function _to_binomial_counts(R)
    Rv = validate_input(R)
    k_correct = Float64.(dropdims(sum(Rv; dims=3); dims=3))
    n_trials = Int(size(Rv, 3))
    return k_correct, n_trials
end

function _validate_positive_int(name::AbstractString, value; min_value::Int=1)
    if value isa Bool || !(value isa Integer)
        error("$name must be an integer, got $(typeof(value))")
    end
    ivalue = Int(value)
    if ivalue < min_value
        error("$name must be >= $min_value, got $ivalue")
    end
    return ivalue
end

function _coerce_ability_prior(prior)
    if prior isa Real
        prior_var = Float64(prior)
        if !isfinite(prior_var) || prior_var <= 0.0
            error("prior variance must be a positive finite scalar.")
        end
        return GaussianPrior(0.0, prior_var)
    end
    if prior isa Prior
        return prior
    end
    error("prior must be a Prior object or float, got $(typeof(prior))")
end

function _validate_nonnegative_float(name::AbstractString, value)
    fvalue = Float64(value)
    if !isfinite(fvalue) || fvalue < 0.0
        error("$name must be a finite scalar >= 0.0, got $(repr(value))")
    end
    return fvalue
end

function _validate_guessing_upper(guessing_upper)
    value = Float64(guessing_upper)
    if !isfinite(value) || !(0.0 < value < 1.0)
        error("guessing_upper must be in (0, 1) and finite.")
    end
    return value
end

function _validate_fix_guessing(fix_guessing, guessing_upper::Float64)
    if isnothing(fix_guessing)
        return nothing
    end
    value = Float64(fix_guessing)
    if !isfinite(value) || !(0.0 <= value <= guessing_upper)
        error("fix_guessing must be in [0, guessing_upper=$guessing_upper] and finite.")
    end
    return value
end

function _estimate_rasch_abilities(k_correct, n_trials::Int; max_iter::Int=500)
    L, M = size(k_correct)
    n_trials_f = Float64(n_trials)

    function negative_log_likelihood(params::Vector{Float64})
        theta = @view params[1:L]
        beta_raw = @view params[(L + 1):(L + M)]
        beta_mean = sum(beta_raw) / Float64(length(beta_raw))
        beta = beta_raw .- beta_mean

        diff = theta .- transpose(beta)
        prob = clamp.(sigmoid(diff), 1e-10, 1.0 - 1e-10)
        nll = -sum(k_correct .* log.(prob) .+ (n_trials_f .- k_correct) .* log.(1.0 .- prob))
        return Float64(nll)
    end

    p_lm = clamp.((k_correct .+ 0.5) ./ (n_trials_f + 1.0), 1e-6, 1.0 - 1e-6)
    model_scores = vec(sum(p_lm; dims=2)) ./ Float64(M)
    question_difficulty = vec(sum(p_lm; dims=1)) ./ Float64(L)

    theta_init = log.(model_scores ./ (1.0 .- model_scores))
    beta_init = -log.(question_difficulty ./ (1.0 .- question_difficulty))
    params_init = vcat(theta_init, beta_init)

    x = _minimize_objective(negative_log_likelihood, params_init; max_iter=max_iter)

    theta = copy(@view x[1:L])
    beta = copy(@view x[(L + 1):(L + M)])
    beta .-= (sum(beta) / Float64(length(beta)))
    return theta, beta
end

function _estimate_rasch_abilities_map(k_correct, n_trials::Int, prior::Prior; max_iter::Int=500)
    L, M = size(k_correct)
    n_trials_f = Float64(n_trials)

    function negative_log_posterior(params::Vector{Float64})
        theta = @view params[1:L]
        beta_raw = @view params[(L + 1):(L + M)]
        beta_mean = sum(beta_raw) / Float64(length(beta_raw))
        beta = beta_raw .- beta_mean

        diff = theta .- transpose(beta)
        prob = clamp.(sigmoid(diff), 1e-10, 1.0 - 1e-10)

        nll = -sum(k_correct .* log.(prob) .+ (n_trials_f .- k_correct) .* log.(1.0 .- prob))
        prior_penalty = penalty(prior, theta)
        return Float64(nll + prior_penalty)
    end

    p_lm = clamp.((k_correct .+ 0.5) ./ (n_trials_f + 1.0), 1e-6, 1.0 - 1e-6)
    model_scores = vec(sum(p_lm; dims=2)) ./ Float64(M)
    question_difficulty = vec(sum(p_lm; dims=1)) ./ Float64(L)

    theta_init = log.(model_scores ./ (1.0 .- model_scores))
    beta_init = -log.(question_difficulty ./ (1.0 .- question_difficulty))
    params_init = vcat(theta_init, beta_init)

    x = _minimize_objective(negative_log_posterior, params_init; max_iter=max_iter)

    theta = copy(@view x[1:L])
    beta = copy(@view x[(L + 1):(L + M)])
    beta .-= (sum(beta) / Float64(length(beta)))
    return theta, beta
end

function _estimate_2pl_abilities(
    k_correct,
    n_trials::Int;
    max_iter::Int=500,
    reg_discrimination::Float64=0.01,
)
    L, M = size(k_correct)
    n_trials_f = Float64(n_trials)

    function negative_log_likelihood(params::Vector{Float64})
        theta = @view params[1:L]
        beta_raw = @view params[(L + 1):(L + M)]
        log_a = @view params[(L + M + 1):(L + 2 * M)]

        beta_mean = sum(beta_raw) / Float64(length(beta_raw))
        beta = beta_raw .- beta_mean
        a = exp.(clamp.(log_a, -3.0, 3.0))

        diff = theta .- transpose(beta)
        logit = diff .* transpose(a)
        prob = clamp.(sigmoid(logit), 1e-10, 1.0 - 1e-10)

        nll = -sum(k_correct .* log.(prob) .+ (n_trials_f .- k_correct) .* log.(1.0 .- prob))
        nll += reg_discrimination * sum(log_a .^ 2)
        return Float64(nll)
    end

    p_lm = clamp.((k_correct .+ 0.5) ./ (n_trials_f + 1.0), 1e-6, 1.0 - 1e-6)
    model_scores = vec(sum(p_lm; dims=2)) ./ Float64(M)
    question_difficulty = vec(sum(p_lm; dims=1)) ./ Float64(L)

    theta_init = log.(model_scores ./ (1.0 .- model_scores))
    beta_init = -log.(question_difficulty ./ (1.0 .- question_difficulty))
    log_a_init = zeros(Float64, M)
    params_init = vcat(theta_init, beta_init, log_a_init)

    x = _minimize_objective(negative_log_likelihood, params_init; max_iter=max_iter)

    theta = copy(@view x[1:L])
    beta = copy(@view x[(L + 1):(L + M)])
    beta .-= (sum(beta) / Float64(length(beta)))
    a = exp.(clamp.(copy(@view x[(L + M + 1):(L + 2 * M)]), -3.0, 3.0))
    return theta, beta, a
end

function _estimate_2pl_abilities_map(
    k_correct,
    n_trials::Int,
    prior::Prior;
    max_iter::Int=500,
    reg_discrimination::Float64=0.01,
)
    L, M = size(k_correct)
    n_trials_f = Float64(n_trials)

    function negative_log_posterior(params::Vector{Float64})
        theta = @view params[1:L]
        beta_raw = @view params[(L + 1):(L + M)]
        log_a = @view params[(L + M + 1):(L + 2 * M)]

        beta_mean = sum(beta_raw) / Float64(length(beta_raw))
        beta = beta_raw .- beta_mean
        a = exp.(clamp.(log_a, -3.0, 3.0))

        diff = theta .- transpose(beta)
        logit = diff .* transpose(a)
        prob = clamp.(sigmoid(logit), 1e-10, 1.0 - 1e-10)

        nll = -sum(k_correct .* log.(prob) .+ (n_trials_f .- k_correct) .* log.(1.0 .- prob))
        nll += reg_discrimination * sum(log_a .^ 2)
        nll += penalty(prior, theta)
        return Float64(nll)
    end

    p_lm = clamp.((k_correct .+ 0.5) ./ (n_trials_f + 1.0), 1e-6, 1.0 - 1e-6)
    model_scores = vec(sum(p_lm; dims=2)) ./ Float64(M)
    question_difficulty = vec(sum(p_lm; dims=1)) ./ Float64(L)

    theta_init = log.(model_scores ./ (1.0 .- model_scores))
    beta_init = -log.(question_difficulty ./ (1.0 .- question_difficulty))
    log_a_init = zeros(Float64, M)
    params_init = vcat(theta_init, beta_init, log_a_init)

    x = _minimize_objective(negative_log_posterior, params_init; max_iter=max_iter)

    theta = copy(@view x[1:L])
    beta = copy(@view x[(L + 1):(L + M)])
    beta .-= (sum(beta) / Float64(length(beta)))
    a = exp.(clamp.(copy(@view x[(L + M + 1):(L + 2 * M)]), -3.0, 3.0))
    return theta, beta, a
end

function _estimate_3pl_abilities(
    k_correct,
    n_trials::Int;
    max_iter::Int=500,
    fix_guessing=nothing,
    reg_discrimination::Float64=0.01,
    reg_guessing::Float64=0.1,
    guessing_upper::Float64=0.5,
)
    L, M = size(k_correct)
    n_trials_f = Float64(n_trials)

    function negative_log_likelihood(params::Vector{Float64})
        theta = @view params[1:L]
        beta_raw = @view params[(L + 1):(L + M)]
        log_a = @view params[(L + M + 1):(L + 2 * M)]

        local c::Vector{Float64}
        if isnothing(fix_guessing)
            logit_c = @view params[(L + 2 * M + 1):(L + 3 * M)]
            c = guessing_upper .* sigmoid(logit_c)
        else
            c = fill(Float64(fix_guessing), M)
        end

        beta_mean = sum(beta_raw) / Float64(length(beta_raw))
        beta = beta_raw .- beta_mean
        a = exp.(clamp.(log_a, -3.0, 3.0))

        diff = theta .- transpose(beta)
        logit = diff .* transpose(a)
        base_prob = sigmoid(logit)
        c_row = reshape(c, 1, :)
        prob = c_row .+ (1.0 .- c_row) .* base_prob
        prob = clamp.(prob, 1e-10, 1.0 - 1e-10)

        nll = -sum(k_correct .* log.(prob) .+ (n_trials_f .- k_correct) .* log.(1.0 .- prob))
        nll += reg_discrimination * sum(log_a .^ 2)
        if isnothing(fix_guessing)
            logit_c = @view params[(L + 2 * M + 1):(L + 3 * M)]
            nll += reg_guessing * sum(logit_c .^ 2)
        end
        return Float64(nll)
    end

    p_lm = clamp.((k_correct .+ 0.5) ./ (n_trials_f + 1.0), 1e-6, 1.0 - 1e-6)
    model_scores = vec(sum(p_lm; dims=2)) ./ Float64(M)
    question_difficulty = vec(sum(p_lm; dims=1)) ./ Float64(L)

    theta_init = log.(model_scores ./ (1.0 .- model_scores))
    beta_init = -log.(question_difficulty ./ (1.0 .- question_difficulty))
    log_a_init = zeros(Float64, M)

    params_init = if isnothing(fix_guessing)
        logit_c_init = zeros(Float64, M)
        vcat(theta_init, beta_init, log_a_init, logit_c_init)
    else
        vcat(theta_init, beta_init, log_a_init)
    end

    x = _minimize_objective(negative_log_likelihood, params_init; max_iter=max_iter)

    theta = copy(@view x[1:L])
    beta = copy(@view x[(L + 1):(L + M)])
    beta .-= (sum(beta) / Float64(length(beta)))
    log_a = copy(@view x[(L + M + 1):(L + 2 * M)])
    a = exp.(clamp.(log_a, -3.0, 3.0))

    c = if isnothing(fix_guessing)
        logit_c = @view x[(L + 2 * M + 1):(L + 3 * M)]
        guessing_upper .* sigmoid(logit_c)
    else
        fill(Float64(fix_guessing), M)
    end

    return theta, beta, a, c
end

function _estimate_3pl_abilities_map(
    k_correct,
    n_trials::Int,
    prior::Prior;
    max_iter::Int=500,
    fix_guessing=nothing,
    reg_discrimination::Float64=0.01,
    reg_guessing::Float64=0.1,
    guessing_upper::Float64=0.5,
)
    L, M = size(k_correct)
    n_trials_f = Float64(n_trials)

    function negative_log_posterior(params::Vector{Float64})
        theta = @view params[1:L]
        beta_raw = @view params[(L + 1):(L + M)]
        log_a = @view params[(L + M + 1):(L + 2 * M)]

        local c::Vector{Float64}
        if isnothing(fix_guessing)
            logit_c = @view params[(L + 2 * M + 1):(L + 3 * M)]
            c = guessing_upper .* sigmoid(logit_c)
        else
            c = fill(Float64(fix_guessing), M)
        end

        beta_mean = sum(beta_raw) / Float64(length(beta_raw))
        beta = beta_raw .- beta_mean
        a = exp.(clamp.(log_a, -3.0, 3.0))

        diff = theta .- transpose(beta)
        logit = diff .* transpose(a)
        base_prob = sigmoid(logit)
        c_row = reshape(c, 1, :)
        prob = c_row .+ (1.0 .- c_row) .* base_prob
        prob = clamp.(prob, 1e-10, 1.0 - 1e-10)

        nll = -sum(k_correct .* log.(prob) .+ (n_trials_f .- k_correct) .* log.(1.0 .- prob))
        nll += penalty(prior, theta)
        nll += reg_discrimination * sum(log_a .^ 2)
        if isnothing(fix_guessing)
            logit_c = @view params[(L + 2 * M + 1):(L + 3 * M)]
            nll += reg_guessing * sum(logit_c .^ 2)
        end

        return Float64(nll)
    end

    p_lm = clamp.((k_correct .+ 0.5) ./ (n_trials_f + 1.0), 1e-6, 1.0 - 1e-6)
    model_scores = vec(sum(p_lm; dims=2)) ./ Float64(M)
    question_difficulty = vec(sum(p_lm; dims=1)) ./ Float64(L)

    theta_init = log.(model_scores ./ (1.0 .- model_scores))
    beta_init = -log.(question_difficulty ./ (1.0 .- question_difficulty))
    log_a_init = zeros(Float64, M)

    params_init = if isnothing(fix_guessing)
        logit_c_init = zeros(Float64, M)
        vcat(theta_init, beta_init, log_a_init, logit_c_init)
    else
        vcat(theta_init, beta_init, log_a_init)
    end

    x = _minimize_objective(negative_log_posterior, params_init; max_iter=max_iter)

    theta = copy(@view x[1:L])
    beta = copy(@view x[(L + 1):(L + M)])
    beta .-= (sum(beta) / Float64(length(beta)))
    log_a = copy(@view x[(L + M + 1):(L + 2 * M)])
    a = exp.(clamp.(log_a, -3.0, 3.0))

    c = if isnothing(fix_guessing)
        logit_c = @view x[(L + 2 * M + 1):(L + 3 * M)]
        guessing_upper .* sigmoid(logit_c)
    else
        fill(Float64(fix_guessing), M)
    end

    return theta, beta, a, c
end

function _validate_time_points(time_points, n_time::Int)
    raw_time = if isnothing(time_points)
        collect(range(0.0, 1.0, length=n_time))
    else
        if !(time_points isa AbstractVector)
            error("time_points must be a 1D array with length equal to R.shape[2].")
        end
        raw = Float64.(collect(time_points))
        if length(raw) != n_time
            error("time_points must be a 1D array with length equal to R.shape[2].")
        end
        if any(x -> !isfinite(x), raw)
            error("time_points must contain only finite values.")
        end
        if n_time >= 2
            for i in 2:n_time
                if raw[i] <= raw[i - 1]
                    error("time_points must be strictly increasing.")
                end
            end
        end
        raw
    end

    if n_time < 2
        return raw_time, zeros(Float64, n_time)
    end

    span = raw_time[end] - raw_time[1]
    if !isfinite(span) || span <= 0.0
        error("time_points must span a positive interval.")
    end

    time_unit = (raw_time .- raw_time[1]) ./ span
    return raw_time, time_unit
end

function _validate_dynamic_score_target(score_target)
    target = lowercase(strip(string(score_target)))
    aliases = Dict(
        "baseline" => "initial",
        "start" => "initial",
        "end" => "final",
        "average" => "mean",
        "delta" => "gain",
        "trend" => "gain",
    )

    target = get(aliases, target, target)
    if target ∉ ("initial", "final", "mean", "gain")
        error(
            "score_target must be one of {'initial', 'final', 'mean', 'gain'} (aliases: baseline, start, end, average, delta, trend).",
        )
    end
    return target
end

function _score_dynamic_path(theta_path, score_target)
    target = _validate_dynamic_score_target(score_target)

    if target == "initial"
        return vec(theta_path[:, 1])
    elseif target == "final"
        return vec(theta_path[:, end])
    elseif target == "mean"
        return vec(sum(theta_path; dims=2)) ./ Float64(size(theta_path, 2))
    end
    return vec(theta_path[:, end] .- theta_path[:, 1])
end

function _estimate_growth_model_abilities(
    R,
    time_unit;
    max_iter::Int=500,
    slope_reg::Float64=0.01,
)
    Rv = validate_input(R)
    L, M, N = size(Rv)

    if !(time_unit isa AbstractVector) || length(time_unit) != N
        error("time_unit must have shape (N,) where N = R.shape[2].")
    end
    time_unit_f = Float64.(collect(time_unit))

    if N < 2
        k_correct = Float64.(dropdims(sum(Rv; dims=3); dims=3))
        theta0, beta = _estimate_rasch_abilities(k_correct, Int(N); max_iter=max_iter)
        theta1 = zeros(Float64, L)
        return theta0, theta1, beta
    end

    p0 = vec(sum(Float64.(Rv[:, :, 1]); dims=2)) ./ Float64(M)
    p0 = clamp.(p0, 1e-6, 1.0 - 1e-6)
    theta0_init = log.(p0 ./ (1.0 .- p0))
    theta1_init = zeros(Float64, L)

    p_m = vec(sum(Float64.(Rv); dims=(1, 3))) ./ Float64(L * N)
    p_m = clamp.(p_m, 1e-6, 1.0 - 1e-6)
    beta_init = -log.(p_m ./ (1.0 .- p_m))

    params_init = vcat(theta0_init, theta1_init, beta_init)
    Rf = Float64.(Rv)

    function negative_log_likelihood(params::Vector{Float64})
        theta0 = @view params[1:L]
        theta1 = @view params[(L + 1):(2 * L)]
        beta_raw = @view params[(2 * L + 1):(2 * L + M)]

        beta_mean = sum(beta_raw) / Float64(length(beta_raw))
        beta = beta_raw .- beta_mean

        nll = 0.0
        for l in 1:L
            for m in 1:M
                for n in 1:N
                    diff = theta0[l] + theta1[l] * time_unit_f[n] - beta[m]
                    p = clamp(sigmoid(diff), 1e-10, 1.0 - 1e-10)
                    r = Rf[l, m, n]
                    nll -= r * log(p) + (1.0 - r) * log(1.0 - p)
                end
            end
        end

        nll += slope_reg * sum(theta1 .^ 2)
        return Float64(nll)
    end

    x = _minimize_objective(negative_log_likelihood, params_init; max_iter=max_iter)

    theta0 = copy(@view x[1:L])
    theta1 = copy(@view x[(L + 1):(2 * L)])
    beta = copy(@view x[(2 * L + 1):(2 * L + M)])
    beta .-= (sum(beta) / Float64(length(beta)))
    return theta0, theta1, beta
end

function _estimate_state_space_abilities(
    R,
    time_unit;
    max_iter::Int=500,
    state_reg::Float64=1.0,
)
    Rv = validate_input(R)
    L, M, N = size(Rv)

    if !(time_unit isa AbstractVector) || length(time_unit) != N
        error("time_unit must have shape (N,) where N = R.shape[2].")
    end
    time_unit_f = Float64.(collect(time_unit))

    if N < 2
        k_correct = Float64.(dropdims(sum(Rv; dims=3); dims=3))
        theta, beta = _estimate_rasch_abilities(k_correct, Int(N); max_iter=max_iter)
        return reshape(theta, L, 1), beta
    end

    p_ln = zeros(Float64, L, N)
    for l in 1:L
        for n in 1:N
            p_ln[l, n] = sum(Float64.(Rv[l, :, n])) / Float64(M)
        end
    end
    p_ln = clamp.(p_ln, 1e-6, 1.0 - 1e-6)
    theta_init = log.(p_ln ./ (1.0 .- p_ln))

    p_m = vec(sum(Float64.(Rv); dims=(1, 3))) ./ Float64(L * N)
    p_m = clamp.(p_m, 1e-6, 1.0 - 1e-6)
    beta_init = -log.(p_m ./ (1.0 .- p_m))

    params_init = vcat(vec(theta_init), beta_init)
    Rf = Float64.(Rv)
    dt = diff(time_unit_f)

    function negative_log_posterior(params::Vector{Float64})
        theta = reshape(@view(params[1:(L * N)]), L, N)
        beta_raw = @view params[(L * N + 1):(L * N + M)]
        beta_mean = sum(beta_raw) / Float64(length(beta_raw))
        beta = beta_raw .- beta_mean

        nll = 0.0
        for l in 1:L
            for m in 1:M
                for n in 1:N
                    diff = theta[l, n] - beta[m]
                    p = clamp(sigmoid(diff), 1e-10, 1.0 - 1e-10)
                    r = Rf[l, m, n]
                    nll -= r * log(p) + (1.0 - r) * log(1.0 - p)
                end
            end
        end

        for l in 1:L
            for n in 1:(N - 1)
                step = (theta[l, n + 1] - theta[l, n]) / sqrt(dt[n])
                nll += state_reg * step^2
            end
        end

        nll += 1e-3 * sum(theta[:, 1] .^ 2)
        return Float64(nll)
    end

    x = _minimize_objective(negative_log_posterior, params_init; max_iter=max_iter)

    theta_path = reshape(copy(@view x[1:(L * N)]), L, N)
    beta = copy(@view x[(L * N + 1):(L * N + M)])
    beta .-= (sum(beta) / Float64(length(beta)))
    return theta_path, beta
end

function _posterior_sd(posterior, theta_q)
    posterior_f = Float64.(posterior)
    theta_q_f = Float64.(theta_q)

    mean_post = posterior_f * theta_q_f
    second = posterior_f * (theta_q_f .^ 2)
    var_post = max.(second .- (mean_post .^ 2), 0.0)
    return sqrt.(var_post)
end

function _posterior_quantile(posterior, theta_q, q)
    qf = Float64(q)
    if !(0.0 < qf < 1.0)
        error("q must be in (0, 1)")
    end

    posterior_f = Float64.(posterior)
    theta_q_f = Float64.(theta_q)
    order = sortperm(theta_q_f)
    theta_sorted = theta_q_f[order]
    post_sorted = posterior_f[:, order]

    L = size(post_sorted, 1)
    Q = size(post_sorted, 2)
    out = zeros(Float64, L)

    for i in 1:L
        c = 0.0
        idx = Q
        for j in 1:Q
            c += post_sorted[i, j]
            if c >= qf
                idx = j
                break
            end
        end
        out[i] = theta_sorted[idx]
    end
    return out
end

function _hermgauss(n::Int)
    d = zeros(Float64, n)
    e = sqrt.(collect(1:(n - 1)) ./ 2.0)
    eig = eigen(SymTridiagonal(d, e))
    x = eig.values
    w = sqrt(pi) .* (eig.vectors[1, :] .^ 2)
    return x, w
end

function _estimate_rasch_mml(
    k_correct,
    n_trials::Int;
    max_iter::Int=100,
    em_iter::Int=20,
    n_quadrature::Int=21,
)
    L, M = size(k_correct)
    n_trials_f = Float64(n_trials)

    x_gh, w_gh = _hermgauss(n_quadrature)
    theta_q = sqrt(2.0) .* x_gh
    w_q = w_gh ./ sqrt(pi)

    p_lm = clamp.((k_correct .+ 0.5) ./ (n_trials_f + 1.0), 1e-6, 1.0 - 1e-6)
    question_difficulty = vec(sum(p_lm; dims=1)) ./ Float64(L)
    beta = -log.((question_difficulty .+ 0.01) ./ (1.0 .- question_difficulty .+ 0.01))
    beta .-= (sum(beta) / Float64(length(beta)))

    Q = n_quadrature
    posterior = zeros(Float64, L, Q)
    log_lik = zeros(Float64, L, Q)

    function e_step!(posterior_out, log_lik_out, beta_local)
        for q in 1:Q
            diff = theta_q[q] .- beta_local
            prob = clamp.(sigmoid(diff), 1e-10, 1.0 - 1e-10)
            log_prob = log.(prob)
            log_one_minus = log.(1.0 .- prob)
            for l in 1:L
                s = 0.0
                for m in 1:M
                    kc = k_correct[l, m]
                    s += kc * log_prob[m] + (n_trials_f - kc) * log_one_minus[m]
                end
                log_lik_out[l, q] = s
            end
        end

        for l in 1:L
            mmax = maximum(@view log_lik_out[l, :])
            denom = 0.0
            for q in 1:Q
                v = exp(log_lik_out[l, q] - mmax) * w_q[q]
                posterior_out[l, q] = v
                denom += v
            end
            posterior_out[l, :] ./= denom
        end
    end

    for _ in 1:em_iter
        e_step!(posterior, log_lik, beta)

        for m in 1:M
            k_m = @view k_correct[:, m]
            function item_nll(x::Vector{Float64})
                b = x[1]
                nll = 0.0
                for q in 1:Q
                    p = clamp(sigmoid(theta_q[q] - b), 1e-10, 1.0 - 1e-10)
                    lp = log(p)
                    lq = log(1.0 - p)
                    acc = 0.0
                    for l in 1:L
                        log_p = k_m[l] * lp + (n_trials_f - k_m[l]) * lq
                        acc += posterior[l, q] * log_p
                    end
                    nll -= acc
                end
                return Float64(nll)
            end

            xopt = _minimize_objective(item_nll, [beta[m]]; max_iter=max_iter)
            beta[m] = xopt[1]
        end

        beta .-= (sum(beta) / Float64(length(beta)))
    end

    e_step!(posterior, log_lik, beta)
    abilities = posterior * theta_q
    return abilities, beta, posterior, theta_q
end

"""
    rasch(
        R;
        method="competition",
        return_scores=false,
        max_iter=500,
        return_item_params=false,
    )

Rank models with Rasch (1PL) maximum-likelihood estimation.

Returns rankings from estimated abilities `theta`. When
`return_item_params=true`, also returns item difficulties.

For counts ``k_{lm}=\\sum_n R_{lmn}``:

```math
k_{lm} \\sim \\mathrm{Binomial}\\!\\left(N,\\sigma(\\theta_l-b_m)\\right)
```

Item difficulties are mean-centered for identifiability:

```math
b \\leftarrow b - \\frac{1}{M}\\sum_m b_m
```

# Reference
Rasch, G. (1960). *Probabilistic Models for Some Intelligence and Attainment Tests*.
"""
function rasch(
    R;
    method="competition",
    return_scores=false,
    max_iter=500,
    return_item_params=false,
)
    max_iter_i = _validate_positive_int("max_iter", max_iter)
    k_correct, n_trials = _to_binomial_counts(R)

    theta, beta = _estimate_rasch_abilities(k_correct, n_trials; max_iter=max_iter_i)
    scores = theta
    ranking = rank_scores(scores)[string(method)]

    if return_item_params
        return ranking, scores, Dict("difficulty" => beta)
    end
    return return_scores ? (ranking, scores) : ranking
end

"""
    rasch_map(
        R;
        prior=1.0,
        method="competition",
        return_scores=false,
        max_iter=500,
        return_item_params=false,
    )

Rank models with Rasch (1PL) MAP estimation using an ability prior.

```math
(\\hat\\theta,\\hat b)
= \\arg\\min_{\\theta,b}
\\left[
-\\sum_{l,m}\\log p(k_{lm}\\mid \\theta_l,b_m)
+ \\operatorname{penalty}(\\theta)
\\right]
```

# Reference
Mislevy, R. J. (1986). Bayes modal estimation in item response models.
*Psychometrika*.
"""
function rasch_map(
    R;
    prior=1.0,
    method="competition",
    return_scores=false,
    max_iter=500,
    return_item_params=false,
)
    max_iter_i = _validate_positive_int("max_iter", max_iter)
    k_correct, n_trials = _to_binomial_counts(R)
    prior_obj = _coerce_ability_prior(prior)

    theta, beta =
        _estimate_rasch_abilities_map(k_correct, n_trials, prior_obj; max_iter=max_iter_i)
    scores = theta
    ranking = rank_scores(scores)[string(method)]

    if return_item_params
        return ranking, scores, Dict("difficulty" => beta)
    end
    return return_scores ? (ranking, scores) : ranking
end

"""
    rasch_2pl(
        R;
        method="competition",
        return_scores=false,
        max_iter=500,
        return_item_params=false,
        reg_discrimination=0.01,
    )

Rank models with 2PL IRT maximum likelihood (ability + item discrimination).

```math
k_{lm} \\sim \\mathrm{Binomial}\\!\\left(
N,\\sigma\\!\\left(a_m(\\theta_l-b_m)\\right)\\right)
```
"""
function rasch_2pl(
    R;
    method="competition",
    return_scores=false,
    max_iter=500,
    return_item_params=false,
    reg_discrimination=0.01,
)
    max_iter_i = _validate_positive_int("max_iter", max_iter)
    reg_discrimination_f = _validate_nonnegative_float("reg_discrimination", reg_discrimination)
    k_correct, n_trials = _to_binomial_counts(R)

    theta, beta, a = _estimate_2pl_abilities(
        k_correct,
        n_trials;
        max_iter=max_iter_i,
        reg_discrimination=reg_discrimination_f,
    )
    scores = theta
    ranking = rank_scores(scores)[string(method)]

    if return_item_params
        return ranking, scores, Dict("difficulty" => beta, "discrimination" => a)
    end
    return return_scores ? (ranking, scores) : ranking
end

"""
    rasch_2pl_map(
        R;
        prior=1.0,
        method="competition",
        return_scores=false,
        max_iter=500,
        return_item_params=false,
        reg_discrimination=0.01,
    )

Rank models with 2PL IRT MAP estimation.

Same 2PL likelihood as [`rasch_2pl`](@ref), plus prior regularization on
abilities:

```math
\\hat\\theta \\in \\arg\\min_{\\theta,\\cdots}
\\left[-\\log p(k\\mid \\theta,\\cdots)+\\operatorname{penalty}(\\theta)\\right]
```
"""
function rasch_2pl_map(
    R;
    prior=1.0,
    method="competition",
    return_scores=false,
    max_iter=500,
    return_item_params=false,
    reg_discrimination=0.01,
)
    max_iter_i = _validate_positive_int("max_iter", max_iter)
    reg_discrimination_f = _validate_nonnegative_float("reg_discrimination", reg_discrimination)
    k_correct, n_trials = _to_binomial_counts(R)
    prior_obj = _coerce_ability_prior(prior)

    theta, beta, a = _estimate_2pl_abilities_map(
        k_correct,
        n_trials,
        prior_obj;
        max_iter=max_iter_i,
        reg_discrimination=reg_discrimination_f,
    )
    scores = theta
    ranking = rank_scores(scores)[string(method)]

    if return_item_params
        return ranking, scores, Dict("difficulty" => beta, "discrimination" => a)
    end
    return return_scores ? (ranking, scores) : ranking
end

"""
    dynamic_irt(
        R;
        variant="linear",
        method="competition",
        return_scores=false,
        max_iter=500,
        return_item_params=false,
        time_points=nothing,
        score_target="final",
        slope_reg=0.01,
        state_reg=1.0,
        assume_time_axis=false,
    )

Rank models with dynamic IRT variants:
- `"linear"`: static Rasch baseline
- `"growth"`: linear growth path
- `"state_space"`: smoothed latent trajectory

Growth variant:

```math
\\theta_{ln} = \\theta_{0,l} + \\theta_{1,l} t_n,\\qquad
P(R_{lmn}=1)=\\sigma(\\theta_{ln}-b_m)
```

State-space variant:

```math
P(R_{lmn}=1)=\\sigma(\\theta_{ln}-b_m)
```

with smoothness penalty

```math
\\lambda \\sum_{l,n>1}
\\frac{(\\theta_{ln}-\\theta_{l,n-1})^2}{t_n-t_{n-1}}
```

# References
Verhelst, N. D., & Glas, C. A. (1993). A dynamic generalization of the Rasch model.
*Psychometrika*.
"""
function dynamic_irt(
    R;
    variant="linear",
    method="competition",
    return_scores=false,
    max_iter=500,
    return_item_params=false,
    time_points=nothing,
    score_target="final",
    slope_reg=0.01,
    state_reg=1.0,
    assume_time_axis=false,
)
    max_iter_i = _validate_positive_int("max_iter", max_iter)
    variant_s = lowercase(strip(string(variant)))
    Rv = validate_input(R)
    k_correct = Float64.(dropdims(sum(Rv; dims=3); dims=3))
    n_trials = Int(size(Rv, 3))
    score_target_s = _validate_dynamic_score_target(score_target)
    slope_reg_f = _validate_nonnegative_float("slope_reg", slope_reg)
    state_reg_f = _validate_nonnegative_float("state_reg", state_reg)

    local scores::Vector{Float64}
    local beta::Vector{Float64}
    local theta0::Vector{Float64}
    local theta1::Vector{Float64}
    local theta_path::Matrix{Float64}
    local raw_time::Vector{Float64}

    if variant_s == "linear"
        if score_target_s != "final"
            error(
                "score_target is only used for longitudinal variants ('growth' and 'state_space').",
            )
        end
        theta, beta_est = _estimate_rasch_abilities(k_correct, n_trials; max_iter=max_iter_i)
        scores = theta
        beta = beta_est
    elseif variant_s == "growth"
        if !assume_time_axis
            error(
                "variant='growth' interprets axis-2 as ordered longitudinal time. Set assume_time_axis=True to proceed.",
            )
        end
        raw_time, time_unit = _validate_time_points(time_points, n_trials)
        theta0_est, theta1_est, beta_est = _estimate_growth_model_abilities(
            Rv,
            time_unit;
            max_iter=max_iter_i,
            slope_reg=slope_reg_f,
        )
        theta0 = theta0_est
        theta1 = theta1_est
        beta = beta_est
        theta_path = zeros(Float64, length(theta0), length(time_unit))
        for l in 1:length(theta0), n in 1:length(time_unit)
            theta_path[l, n] = theta0[l] + theta1[l] * time_unit[n]
        end
        scores = _score_dynamic_path(theta_path, score_target_s)
    elseif variant_s == "state_space"
        if !assume_time_axis
            error(
                "variant='state_space' interprets axis-2 as ordered longitudinal time. Set assume_time_axis=True to proceed.",
            )
        end
        raw_time, time_unit = _validate_time_points(time_points, n_trials)
        theta_path_est, beta_est = _estimate_state_space_abilities(
            Rv,
            time_unit;
            max_iter=max_iter_i,
            state_reg=state_reg_f,
        )
        theta_path = theta_path_est
        beta = beta_est
        scores = _score_dynamic_path(theta_path, score_target_s)
    else
        error("Unknown variant: $variant_s. Use 'linear', 'growth', or 'state_space'.")
    end

    ranking = rank_scores(scores)[string(method)]
    if return_item_params
        if variant_s == "linear"
            return ranking, scores, Dict("difficulty" => beta)
        elseif variant_s == "growth"
            return ranking, scores, Dict(
                "difficulty" => beta,
                "baseline" => theta0,
                "slope" => theta1,
                "ability_path" => theta_path,
                "time_points" => raw_time,
            )
        end
        return ranking, scores, Dict(
            "difficulty" => beta,
            "ability_path" => theta_path,
            "time_points" => raw_time,
            "gain" => vec(theta_path[:, end] .- theta_path[:, 1]),
        )
    end

    return return_scores ? (ranking, scores) : ranking
end

"""
    rasch_3pl(
        R;
        method="competition",
        return_scores=false,
        max_iter=500,
        fix_guessing=nothing,
        return_item_params=false,
        reg_discrimination=0.01,
        reg_guessing=0.1,
        guessing_upper=0.5,
    )

Rank models with 3PL IRT maximum likelihood (ability, discrimination, guessing).

```math
p_{lm} = c_m + (1-c_m)\\sigma\\!\\left(a_m(\\theta_l-b_m)\\right)
```

with ``c_m \\in [0, \\text{guessing_upper}]``.
"""
function rasch_3pl(
    R;
    method="competition",
    return_scores=false,
    max_iter=500,
    fix_guessing=nothing,
    return_item_params=false,
    reg_discrimination=0.01,
    reg_guessing=0.1,
    guessing_upper=0.5,
)
    max_iter_i = _validate_positive_int("max_iter", max_iter)
    reg_discrimination_f = _validate_nonnegative_float("reg_discrimination", reg_discrimination)
    reg_guessing_f = _validate_nonnegative_float("reg_guessing", reg_guessing)
    guessing_upper_f = _validate_guessing_upper(guessing_upper)
    fix_guessing_v = _validate_fix_guessing(fix_guessing, guessing_upper_f)
    k_correct, n_trials = _to_binomial_counts(R)

    theta, beta, a, c = _estimate_3pl_abilities(
        k_correct,
        n_trials;
        max_iter=max_iter_i,
        fix_guessing=fix_guessing_v,
        reg_discrimination=reg_discrimination_f,
        reg_guessing=reg_guessing_f,
        guessing_upper=guessing_upper_f,
    )
    scores = theta
    ranking = rank_scores(scores)[string(method)]

    if return_item_params
        return ranking, scores, Dict("difficulty" => beta, "discrimination" => a, "guessing" => c)
    end
    return return_scores ? (ranking, scores) : ranking
end

"""
    rasch_3pl_map(
        R;
        prior=1.0,
        method="competition",
        return_scores=false,
        max_iter=500,
        fix_guessing=nothing,
        return_item_params=false,
        reg_discrimination=0.01,
        reg_guessing=0.1,
        guessing_upper=0.5,
    )

Rank models with 3PL IRT MAP estimation.

Same 3PL likelihood as [`rasch_3pl`](@ref), with prior penalty on abilities:

```math
\\hat\\theta \\in \\arg\\min_{\\theta,\\cdots}
\\left[-\\log p(k\\mid \\theta,\\cdots)+\\operatorname{penalty}(\\theta)\\right]
```
"""
function rasch_3pl_map(
    R;
    prior=1.0,
    method="competition",
    return_scores=false,
    max_iter=500,
    fix_guessing=nothing,
    return_item_params=false,
    reg_discrimination=0.01,
    reg_guessing=0.1,
    guessing_upper=0.5,
)
    max_iter_i = _validate_positive_int("max_iter", max_iter)
    reg_discrimination_f = _validate_nonnegative_float("reg_discrimination", reg_discrimination)
    reg_guessing_f = _validate_nonnegative_float("reg_guessing", reg_guessing)
    guessing_upper_f = _validate_guessing_upper(guessing_upper)
    fix_guessing_v = _validate_fix_guessing(fix_guessing, guessing_upper_f)
    k_correct, n_trials = _to_binomial_counts(R)
    prior_obj = _coerce_ability_prior(prior)

    theta, beta, a, c = _estimate_3pl_abilities_map(
        k_correct,
        n_trials,
        prior_obj;
        max_iter=max_iter_i,
        fix_guessing=fix_guessing_v,
        reg_discrimination=reg_discrimination_f,
        reg_guessing=reg_guessing_f,
        guessing_upper=guessing_upper_f,
    )
    scores = theta
    ranking = rank_scores(scores)[string(method)]

    if return_item_params
        return ranking, scores, Dict("difficulty" => beta, "discrimination" => a, "guessing" => c)
    end
    return return_scores ? (ranking, scores) : ranking
end

"""
    rasch_mml(
        R;
        method="competition",
        return_scores=false,
        max_iter=100,
        em_iter=20,
        n_quadrature=21,
        return_item_params=false,
    )

Rank models with Rasch marginal maximum likelihood using EM + quadrature.

Using quadrature nodes ``\\theta_q`` and weights `w_q`, posterior mass for model
`l` is:

```math
w_{lq} \\propto p(k_l\\mid \\theta_q,b)\\,w_q
```

EAP ability score:

```math
\\hat\\theta_l^{\\mathrm{EAP}} = \\sum_q w_{lq}\\theta_q
```

# References
Bock, R. D., & Aitkin, M. (1981). Marginal maximum likelihood estimation
of item parameters: Application of an EM algorithm. *Psychometrika*.
"""
function rasch_mml(
    R;
    method="competition",
    return_scores=false,
    max_iter=100,
    em_iter=20,
    n_quadrature=21,
    return_item_params=false,
)
    max_iter_i = _validate_positive_int("max_iter", max_iter)
    em_iter_i = _validate_positive_int("em_iter", em_iter)
    n_quadrature_i = _validate_positive_int("n_quadrature", n_quadrature; min_value=2)
    k_correct, n_trials = _to_binomial_counts(R)

    theta, beta, posterior, theta_q = _estimate_rasch_mml(
        k_correct,
        n_trials;
        max_iter=max_iter_i,
        em_iter=em_iter_i,
        n_quadrature=n_quadrature_i,
    )
    scores = theta

    ranking = rank_scores(scores)[string(method)]
    if return_item_params
        theta_sd = _posterior_sd(posterior, theta_q)
        return ranking, scores, Dict("difficulty" => beta, "ability_sd" => theta_sd)
    end
    return return_scores ? (ranking, scores) : ranking
end

"""
    rasch_mml_credible(
        R;
        quantile=0.05,
        method="competition",
        return_scores=false,
        max_iter=100,
        em_iter=20,
        n_quadrature=21,
    )

Rank models by posterior ability quantiles from Rasch MML posterior mass.

```math
s_l = Q_q(\\theta_l \\mid R)
```

Lower `q` (for example `0.05`) yields a more conservative ranking.
"""
function rasch_mml_credible(
    R;
    quantile=0.05,
    method="competition",
    return_scores=false,
    max_iter=100,
    em_iter=20,
    n_quadrature=21,
)
    quantile_f = Float64(quantile)
    if !(0.0 < quantile_f < 1.0)
        error("quantile must be in (0, 1)")
    end

    max_iter_i = _validate_positive_int("max_iter", max_iter)
    em_iter_i = _validate_positive_int("em_iter", em_iter)
    n_quadrature_i = _validate_positive_int("n_quadrature", n_quadrature; min_value=2)

    k_correct, n_trials = _to_binomial_counts(R)
    _, _, posterior, theta_q = _estimate_rasch_mml(
        k_correct,
        n_trials;
        max_iter=max_iter_i,
        em_iter=em_iter_i,
        n_quadrature=n_quadrature_i,
    )

    scores = _posterior_quantile(posterior, theta_q, quantile_f)
    ranking = rank_scores(scores)[string(method)]
    return return_scores ? (ranking, scores) : ranking
end

function _build_product_quadrature(n_factors::Int, n_quadrature::Int)
    x_gh, w_gh = _hermgauss(n_quadrature)
    nodes_1d = sqrt(2.0) .* x_gh
    logw_1d = log.(w_gh) .- 0.5 * log(pi)

    G = n_quadrature^n_factors
    grid = zeros(Float64, G, n_factors)
    logw = zeros(Float64, G)
    idx = ones(Int, n_factors)
    for g in 1:G
        acc = 0.0
        for dcol in 1:n_factors
            grid[g, dcol] = nodes_1d[idx[dcol]]
            acc += logw_1d[idx[dcol]]
        end
        logw[g] = acc
        for dcol in 1:n_factors
            idx[dcol] += 1
            if idx[dcol] <= n_quadrature
                break
            else
                idx[dcol] = 1
            end
        end
    end
    return grid, logw
end

function _estimate_mirt(
    k_correct,
    n_trials::Int;
    n_factors::Int,
    model::AbstractString,
    max_iter::Int,
    em_iter::Int,
    n_quadrature::Int,
    fix_guessing,
    reg_discrimination::Float64,
    reg_guessing::Float64,
    guessing_upper::Float64,
    tol::Float64,
)
    L, M = size(k_correct)
    D = n_factors
    n_trials_f = Float64(n_trials)
    estimate_c = model == "3pl" && isnothing(fix_guessing)
    c_fixed =
        (model == "3pl" && !isnothing(fix_guessing)) ? fill(Float64(fix_guessing), M) :
        nothing

    grid, logw = _build_product_quadrature(D, n_quadrature)
    G = size(grid, 1)
    n_incorrect = n_trials_f .- k_correct

    # Initialization: intercepts from item easiness, slopes from the leading
    # singular directions of the centered logit matrix.
    p_lm = clamp.((k_correct .+ 0.5) ./ (n_trials_f + 1.0), 1e-6, 1.0 - 1e-6)
    z = log.(p_lm ./ (1.0 .- p_lm))
    d0 = vec(sum(z; dims=1)) ./ Float64(L)
    F = svd(z .- transpose(d0))
    a = zeros(Float64, M, D)
    for dd in 1:min(D, length(F.S))
        a[:, dd] = F.V[:, dd] .* sqrt(max(F.S[dd], 0.0))
    end
    a = clamp.(a, -3.0, 3.0)
    d = copy(d0)
    gamma = zeros(Float64, M)

    current_c = g -> estimate_c ? (guessing_upper .* sigmoid(g)) : c_fixed

    function probs(a_, d_, c_)
        lin = grid * transpose(a_) .+ transpose(d_)
        s = sigmoid(lin)
        p = isnothing(c_) ? s : transpose(c_) .+ (1.0 .- transpose(c_)) .* s
        return clamp.(p, 1e-10, 1.0 - 1e-10)
    end

    function posterior(a_, d_, c_)
        p = probs(a_, d_, c_)
        loglik = k_correct * transpose(log.(p)) .+ n_incorrect * transpose(log.(1.0 .- p))
        logpost = loglik .+ transpose(logw)
        post = similar(logpost)
        for l in 1:L
            mx = maximum(@view logpost[l, :])
            denom = 0.0
            for g in 1:G
                e = exp(logpost[l, g] - mx)
                post[l, g] = e
                denom += e
            end
            post[l, :] ./= denom
        end
        return post
    end

    for _ in 1:em_iter
        # E-step: posterior over the latent grid for each model.
        post = posterior(a, d, current_c(gamma))
        f = n_trials_f .* vec(sum(post; dims=1))
        r = transpose(post) * k_correct

        # M-step: maximize the expected complete-data likelihood for the
        # separable item parameters jointly.
        function mstep(params::Vector{Float64})
            a_ = reshape(@view(params[1:(M * D)]), M, D)
            d_ = @view params[(M * D + 1):(M * D + M)]
            local c_
            if estimate_c
                gamma_ = @view params[(M * D + M + 1):(M * D + 2 * M)]
                c_ = guessing_upper .* sigmoid(gamma_)
            else
                c_ = c_fixed
            end
            lin = grid * transpose(a_) .+ transpose(d_)
            s = sigmoid(lin)
            p = isnothing(c_) ? s : transpose(c_) .+ (1.0 .- transpose(c_)) .* s
            p = clamp.(p, 1e-10, 1.0 - 1e-10)
            nll = -sum(r .* log.(p) .+ (f .- r) .* log.(1.0 .- p))
            nll += reg_discrimination * sum(a_ .^ 2)
            if estimate_c
                gamma_ = @view params[(M * D + M + 1):(M * D + 2 * M)]
                nll += reg_guessing * sum(gamma_ .^ 2)
            end
            return Float64(nll)
        end

        x0 = estimate_c ? vcat(vec(a), d, gamma) : vcat(vec(a), d)
        x = _minimize_objective(mstep, x0; max_iter=max_iter)
        a_new = reshape(copy(@view x[1:(M * D)]), M, D)
        d_new = copy(@view x[(M * D + 1):(M * D + M)])
        gamma_new = estimate_c ? copy(@view x[(M * D + M + 1):(M * D + 2 * M)]) : gamma

        delta = max(
            maximum(abs.(a_new .- a)),
            maximum(abs.(d_new .- d)),
            estimate_c ? maximum(abs.(gamma_new .- gamma)) : 0.0,
        )
        a = a_new
        d = d_new
        gamma = gamma_new
        if delta < tol
            break
        end
    end

    # Final E-step: EAP abilities and posterior SD per dimension.
    c_final = current_c(gamma)
    post = posterior(a, d, c_final)
    theta = post * grid
    second = post * (grid .^ 2)
    theta_sd = sqrt.(max.(second .- theta .^ 2, 0.0))

    # Orient each latent axis so its mean slope is non-negative.
    for dd in 1:D
        if sum(@view a[:, dd]) < 0.0
            a[:, dd] .*= -1.0
            theta[:, dd] .*= -1.0
        end
    end

    c_out = isnothing(c_final) ? zeros(Float64, M) : c_final
    mdisc = sqrt.(vec(sum(a .^ 2; dims=2)))
    mdiff = -d ./ max.(mdisc, 1e-12)

    # Rotation-invariant reference composite for ranking.
    a_bar = vec(sum(a; dims=1)) ./ Float64(M)
    scores = theta * a_bar
    return theta, a, d, c_out, mdisc, mdiff, theta_sd, scores
end

"""
    mirt(
        R;
        n_factors=2,
        model="2pl",
        method="competition",
        return_scores=false,
        max_iter=50,
        em_iter=100,
        n_quadrature=15,
        fix_guessing=nothing,
        reg_discrimination=0.01,
        reg_guessing=0.1,
        guessing_upper=0.5,
        tol=1e-4,
        return_item_params=false,
    )

Rank models with compensatory multidimensional IRT (MIRT) via marginal-MLE EM.

Each model `l` has a `D`-dimensional latent ability vector ``\\theta_l``
(`D = n_factors`) and each question `m` a slope vector ``a_m`` and intercept
``d_m``. The compensatory dichotomous model is

```math
P(R_{lmn}=1\\mid \\theta_l)
= c_m + (1-c_m)\\,\\sigma\\!\\left(a_m^{\\top}\\theta_l + d_m\\right)
```

with ``c_m=0`` for `model="2pl"` and item pseudo-guessing for `model="3pl"`.
Item parameters are estimated by a Bock-Aitkin EM algorithm integrating
abilities over a standard multivariate-normal prior on a product
Gauss-Hermite quadrature grid; abilities are summarized by their EAP values.

Multidimensional abilities are collapsed to a scalar ranking score via the
rotation-invariant reference composite, the projection of each ability vector
onto the mean item-slope direction:

```math
s_l = \\bar a^{\\top}\\theta_l,\\qquad \\bar a = \\frac{1}{M}\\sum_m a_m
```

When `return_item_params=true`, also returns a dictionary with multidimensional
difficulty (`"difficulty"`, ``\\mathrm{MDIFF}_m = -d_m/\\lVert a_m\\rVert``),
multidimensional discrimination (`"discrimination"`,
``\\mathrm{MDISC}_m = \\lVert a_m\\rVert``), slopes `"slopes"`, intercepts
`"intercept"`, EAP abilities `"abilities"`, posterior SD `"ability_sd"`, and,
for `model="3pl"`, guessing `"guessing"`.

# References
Chalmers, R. P. (2012). mirt: A Multidimensional Item Response Theory Package
for the R Environment. *Journal of Statistical Software*, 48(6), 1-29.

Bock, R. D., & Aitkin, M. (1981). Marginal maximum likelihood estimation of
item parameters: Application of an EM algorithm. *Psychometrika*.
"""
function mirt(
    R;
    n_factors=2,
    model="2pl",
    method="competition",
    return_scores=false,
    max_iter=50,
    em_iter=100,
    n_quadrature=15,
    fix_guessing=nothing,
    reg_discrimination=0.01,
    reg_guessing=0.1,
    guessing_upper=0.5,
    tol=1e-4,
    return_item_params=false,
)
    n_factors_i = _validate_positive_int("n_factors", n_factors; min_value=1)
    max_iter_i = _validate_positive_int("max_iter", max_iter)
    em_iter_i = _validate_positive_int("em_iter", em_iter)
    n_quadrature_i = _validate_positive_int("n_quadrature", n_quadrature; min_value=2)
    reg_discrimination_f = _validate_nonnegative_float("reg_discrimination", reg_discrimination)
    reg_guessing_f = _validate_nonnegative_float("reg_guessing", reg_guessing)
    tol_f = _validate_nonnegative_float("tol", tol)
    guessing_upper_f = _validate_guessing_upper(guessing_upper)

    model_s = lowercase(strip(string(model)))
    if model_s ∉ ("2pl", "3pl")
        error("model must be '2pl' or '3pl'.")
    end
    if model_s == "2pl" && !isnothing(fix_guessing)
        error("fix_guessing is only valid for model='3pl'.")
    end
    fix_guessing_v = _validate_fix_guessing(fix_guessing, guessing_upper_f)

    grid_size = big(n_quadrature_i)^n_factors_i
    if grid_size > 200_000
        error(
            "Product quadrature grid would have $grid_size nodes (n_quadrature=$n_quadrature_i ^ n_factors=$n_factors_i). Reduce n_factors or n_quadrature; compensatory MML-EM is intended for a small number of factors.",
        )
    end

    k_correct, n_trials = _to_binomial_counts(R)
    M = size(k_correct, 2)
    if n_factors_i > M
        error("n_factors=$n_factors_i cannot exceed number of questions M=$M.")
    end

    theta, a, d, c, mdisc, mdiff, theta_sd, scores = _estimate_mirt(
        k_correct,
        n_trials;
        n_factors=n_factors_i,
        model=model_s,
        max_iter=max_iter_i,
        em_iter=em_iter_i,
        n_quadrature=n_quadrature_i,
        fix_guessing=fix_guessing_v,
        reg_discrimination=reg_discrimination_f,
        reg_guessing=reg_guessing_f,
        guessing_upper=guessing_upper_f,
        tol=tol_f,
    )

    ranking = rank_scores(scores)[string(method)]
    if return_item_params
        params = Dict{String,Any}(
            "difficulty" => mdiff,
            "discrimination" => mdisc,
            "slopes" => a,
            "intercept" => d,
            "abilities" => theta,
            "ability_sd" => theta_sd,
        )
        if model_s == "3pl"
            params["guessing"] = c
        end
        return ranking, scores, params
    end
    return return_scores ? (ranking, scores) : ranking
end
