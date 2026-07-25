"""Bayesian ranking methods."""

function _allclose_to_first(v::AbstractVector{<:Real}; rtol::Real=1e-5, atol::Real=1e-8)
    if isempty(v)
        return true
    end
    ref = Float64(v[1])
    for x in v
        if !isapprox(Float64(x), ref; rtol=rtol, atol=atol)
            return false
        end
    end
    return true
end


"""
    thompson(
        R;
        n_samples=10000,
        prior_alpha=1.0,
        prior_beta=1.0,
        seed=42,
        method="competition",
        return_scores=false,
    )

Rank models by Thompson sampling over Beta posteriors of model success rates.

The returned score for each model is negative average sampled rank (higher is
better).

Let ``S_l = \\sum_{m,n} R_{lmn}`` and `T = M N`. Posterior per model:

```math
p_l \\mid R \\sim \\mathrm{Beta}(\\alpha + S_l,\\ \\beta + T - S_l)
```

With posterior draws ``t=1,\\dots,T_s`` and sampled rank ``r_l^{(t)}``:

```math
s_l^{\\mathrm{TS}} = -\\frac{1}{T_s}\\sum_{t=1}^{T_s} r_l^{(t)}
```

# References
Thompson, W. R. (1933). On the Likelihood that One Unknown Probability
Exceeds Another in View of the Evidence of Two Samples. *Biometrika*.
https://doi.org/10.1093/biomet/25.3-4.285

Russo, D. J., et al. (2018). A Tutorial on Thompson Sampling.
https://doi.org/10.1561/2200000070
"""
function thompson(
    R;
    n_samples=10000,
    prior_alpha=1.0,
    prior_beta=1.0,
    seed=42,
    method="competition",
    return_scores=false,
)
    Rv = validate_input(R)

    if n_samples isa Bool || !(n_samples isa Integer)
        error("n_samples must be an integer, got $(typeof(n_samples))")
    end
    n_samples_i = Int(n_samples)
    if n_samples_i < 1
        error("n_samples must be >= 1, got $n_samples_i")
    end

    prior_alpha_f = Float64(prior_alpha)
    if !isfinite(prior_alpha_f) || prior_alpha_f <= 0.0
        error("prior_alpha must be > 0 and finite.")
    end

    prior_beta_f = Float64(prior_beta)
    if !isfinite(prior_beta_f) || prior_beta_f <= 0.0
        error("prior_beta must be > 0 and finite.")
    end

    L, M, N = size(Rv)
    rng = _NumpyRNG(seed)

    successes = vec(sum(reshape(Rv, L, :), dims=2))
    total = Float64(M * N)
    post_alphas = prior_alpha_f .+ successes
    post_betas = prior_beta_f .+ (total .- successes)

    if _allclose_to_first(post_alphas) && _allclose_to_first(post_betas)
        scores = fill(-((L + 1) / 2.0), L)
        ranking = rank_scores(scores)[string(method)]
        return return_scores ? (ranking, scores) : ranking
    end

    rank_sums = zeros(Float64, L)
    samples = zeros(Float64, L)
    ranks = zeros(Float64, L)
    for _ in 1:n_samples_i
        for l in 1:L
            samples[l] = _numpy_beta!(rng, post_alphas[l], post_betas[l])
        end

        order = sortperm(samples; rev=true)
        for k in 1:L
            ranks[order[k]] = k
        end
        rank_sums .+= ranks
    end

    avg_ranks = rank_sums ./ n_samples_i
    scores = average_equivalent_scores(
        -avg_ranks,
        hcat(post_alphas, post_betas),
    )
    ranking = rank_scores(scores)[string(method)]
    return return_scores ? (ranking, scores) : ranking
end

"""
    bayesian_mcmc(
        R;
        n_samples=5000,
        burnin=1000,
        prior_var=1.0,
        seed=42,
        method="competition",
        return_scores=false,
    )

Rank models via random-walk Metropolis MCMC under a Bradley-Terry-style
pairwise likelihood with Gaussian prior on latent abilities.

Scores are posterior means of sampled latent abilities.

Let ``W_{ij}`` be decisive wins of model `i` over `j`, and latent log-strengths
`theta_i` with Gaussian prior variance `sigma^2 = prior_var`.

```math
\\Pr(i \\succ j \\mid \\theta)
= \\frac{\\exp(\\theta_i)}{\\exp(\\theta_i)+\\exp(\\theta_j)},
\\qquad
\\theta_i \\sim \\mathcal{N}(0,\\sigma^2)
```

The returned score is the posterior mean:

```math
s_i^{\\mathrm{MCMC}} = \\mathbb{E}[\\theta_i \\mid W]
```

# References
Bradley, R. A., & Terry, M. E. (1952). Rank Analysis of Incomplete
Block Designs: I. The Method of Paired Comparisons. *Biometrika*.
https://doi.org/10.1093/biomet/39.3-4.324

Metropolis, N., et al. (1953). Equation of State Calculations by Fast
Computing Machines. *The Journal of Chemical Physics*.
https://doi.org/10.1063/1.1699114
"""
function bayesian_mcmc(
    R;
    n_samples=5000,
    burnin=1000,
    prior_var=1.0,
    seed=42,
    method="competition",
    return_scores=false,
)
    Rv = validate_input(R)

    if n_samples isa Bool || !(n_samples isa Integer)
        error("n_samples must be an integer, got $(typeof(n_samples))")
    end
    n_samples_i = Int(n_samples)
    if n_samples_i < 1
        error("n_samples must be >= 1, got $n_samples_i")
    end

    if burnin isa Bool || !(burnin isa Integer)
        error("burnin must be an integer, got $(typeof(burnin))")
    end
    burnin_i = Int(burnin)
    if burnin_i < 0
        error("burnin must be >= 0, got $burnin_i")
    end

    prior_var_f = Float64(prior_var)
    if !isfinite(prior_var_f) || prior_var_f <= 0.0
        error("prior_var must be > 0 and finite.")
    end

    L = size(Rv, 1)
    rng = _NumpyRNG(seed)

    wins = build_pairwise_wins(Rv)
    if sum(wins) <= 0.0
        scores = zeros(Float64, L)
        ranking = rank_scores(scores)[string(method)]
        return return_scores ? (ranking, scores) : ranking
    end

    function log_likelihood(theta::Vector{Float64})
        ll = 0.0
        for i in 1:L
            for j in 1:L
                if i == j || wins[i, j] == 0.0
                    continue
                end
                diff = theta[j] - theta[i]
                log_p = if diff > 20.0
                    -diff
                elseif diff < -20.0
                    0.0
                else
                    -log(1.0 + exp(diff))
                end
                ll += wins[i, j] * log_p
            end
        end
        return ll
    end

    log_prior(theta::Vector{Float64}) = -0.5 * sum(theta .^ 2) / prior_var_f
    log_posterior(theta::Vector{Float64}) = log_likelihood(theta) + log_prior(theta)

    theta_current = zeros(Float64, L)
    log_post_current = log_posterior(theta_current)

    proposal_std = 0.1
    accepted = 0
    score_sum = zeros(Float64, L)

    total_iters = n_samples_i + burnin_i
    for iteration in 0:(total_iters - 1)
        theta_proposed = theta_current .+
                         [_numpy_standard_normal!(rng) for _ in 1:L] .* proposal_std
        log_post_proposed = log_posterior(theta_proposed)
        log_accept_prob = log_post_proposed - log_post_current

        if log(_numpy_uniform!(rng)) < min(log_accept_prob, 0.0)
            theta_current = theta_proposed
            log_post_current = log_post_proposed
            accepted += 1
        end

        if iteration >= burnin_i
            score_sum .+= theta_current
        end

        if iteration > 0 && (iteration % 500 == 0) && iteration < burnin_i
            accept_rate = accepted / iteration
            if accept_rate < 0.2
                proposal_std *= 0.8
            elseif accept_rate > 0.5
                proposal_std *= 1.2
            end
        end
    end

    scores = average_equivalent_scores(score_sum ./ n_samples_i, Rv)
    ranking = rank_scores(scores)[string(method)]
    return return_scores ? (ranking, scores) : ranking
end
