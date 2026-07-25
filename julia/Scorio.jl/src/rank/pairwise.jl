"""Sequential pairwise rating methods."""

using SpecialFunctions: erfc, erfcx, logerfc

function elo()
    error("elo requires input tensor R")
end

"""
    elo(
        R;
        K=32.0,
        initial_rating=1500.0,
        tie_handling="correct_draw_only",
        method="competition",
        return_scores=false,
    )

Sequential Elo rating over pairwise outcomes induced by `R`.

For each `(question, trial)`, all model pairs are compared and Elo updates are
applied in fixed iteration order. Pair outcomes are:
- decisive (`1` vs `0`): win/loss update
- tie (`1` vs `1` or `0` vs `0`): handled by `tie_handling`

# Arguments
- `R`: binary response tensor of shape `(L, M, N)` or matrix `(L, M)` promoted
  to `(L, M, 1)`.
- `K`: positive Elo step size.
- `initial_rating`: finite initial rating for all models.
- `tie_handling`: one of `"skip"`, `"draw"`, `"correct_draw_only"`.
- `method`: rank tie-handling method passed to `rank_scores`.
- `return_scores`: if `true`, return `(ranking, ratings)`.

# Returns
- `ranking` by default.
- `(ranking, ratings)` when `return_scores=true`.

# Formula
For each induced pairwise match `(i,j)` with observed score ``S_{ij} \\in \\{0, 0.5, 1\\}``:

```math
E_{ij} = \\frac{1}{1 + 10^{(r_j-r_i)/400}}
```

```math
r_i \\leftarrow r_i + K(S_{ij} - E_{ij}), \\quad
r_j \\leftarrow r_j + K((1-S_{ij}) - (1-E_{ij}))
```

# Reference
Elo, A. E. (1978). *The Rating of Chessplayers, Past and Present*.
"""
function elo(
    R;
    K=32.0,
    initial_rating=1500.0,
    tie_handling="correct_draw_only",
    method="competition",
    return_scores=false,
)
    Rv = validate_input(R)
    L, M, N = size(Rv)

    Kf = Float64(K)
    if !isfinite(Kf) || Kf <= 0.0
        error("K must be a positive finite scalar; got $Kf")
    end

    initial_rating_f = Float64(initial_rating)
    if !isfinite(initial_rating_f)
        error("initial_rating must be finite.")
    end

    tie_mode = string(tie_handling)
    if tie_mode ∉ ("skip", "draw", "correct_draw_only")
        error("tie_handling must be one of: \"skip\", \"draw\", \"correct_draw_only\"")
    end

    ratings = fill(initial_rating_f, L)

    for t in 1:N
        for q in 1:M
            outcomes = @view Rv[:, q, t]
            event_ratings = copy(ratings)
            rating_delta = zeros(Float64, L)

            for i in 1:L
                for j in (i + 1):L
                    r_i = Int(outcomes[i])
                    r_j = Int(outcomes[j])

                    local S_i::Float64
                    local S_j::Float64
                    if r_i == r_j
                        if tie_mode == "skip"
                            continue
                        elseif tie_mode == "draw"
                            S_i, S_j = 0.5, 0.5
                        else
                            if r_i == 1
                                S_i, S_j = 0.5, 0.5
                            else
                                continue
                            end
                        end
                    else
                        S_i, S_j = r_i > r_j ? (1.0, 0.0) : (0.0, 1.0)
                    end

                    R_i = event_ratings[i]
                    R_j = event_ratings[j]
                    E_i = 1.0 / (1.0 + 10.0^((R_j - R_i) / 400.0))
                    E_j = 1.0 - E_i

                    rating_delta[i] += Kf * (S_i - E_i)
                    rating_delta[j] += Kf * (S_j - E_j)
                end
            end
            ratings .+= rating_delta
        end
    end

    ranking = rank_scores(ratings)[string(method)]
    return return_scores ? (ranking, ratings) : ranking
end

"""
    trueskill(
        R;
        mu_initial=25.0,
        sigma_initial=25.0 / 3,
        beta=25.0 / 6,
        tau=25.0 / 300,
        method="competition",
        return_scores=false,
        tie_handling="skip",
        draw_margin=0.0,
    )

Rank models with a sequential two-player TrueSkill-style update over induced
pairwise comparisons.

Returns rankings from posterior means `mu`.

# Formula
For one match between models `i` and `j`:

```math
c = \\sqrt{2\\beta^2 + \\sigma_i^2 + \\sigma_j^2}, \\quad
t = (\\mu_i-\\mu_j)/c, \\quad
\\epsilon = \\text{draw\\_margin}/c
```

For decisive outcomes, the update uses
``v_{win}(t,\\epsilon)`` and ``w_{win}(t,\\epsilon)``:

```math
\\mu_i' = \\mu_i + \\frac{\\sigma_i^2}{c} v_{win}(t,\\epsilon), \\quad
\\sigma_i'^2 = \\sigma_i^2\\!\\left(1 - \\frac{\\sigma_i^2}{c^2}w_{win}(t,\\epsilon)\\right)
```

Draw updates use the analogous ``v_{draw}`` and ``w_{draw}`` corrections.

# Reference
Herbrich, R., Minka, T., & Graepel, T. (2006).
TrueSkill(TM): A Bayesian Skill Rating System. *NeurIPS 19*.
"""
function trueskill(
    R;
    mu_initial=25.0,
    sigma_initial=25.0 / 3,
    beta=25.0 / 6,
    tau=25.0 / 300,
    method="competition",
    return_scores=false,
    tie_handling="skip",
    draw_margin=0.0,
)
    Rv = validate_input(R)
    L, M, N = size(Rv)

    mu_initial_f = Float64(mu_initial)
    sigma_initial_f = Float64(sigma_initial)
    beta_f = Float64(beta)
    tau_f = Float64(tau)
    draw_margin_f = Float64(draw_margin)

    if !isfinite(mu_initial_f)
        error("mu_initial must be finite.")
    end
    if !isfinite(sigma_initial_f) || sigma_initial_f <= 0.0
        error("sigma_initial must be a positive finite scalar.")
    end
    if !isfinite(beta_f) || beta_f <= 0.0
        error("beta must be a positive finite scalar.")
    end
    if !isfinite(tau_f) || tau_f < 0.0
        error("tau must be a nonnegative finite scalar.")
    end
    if !isfinite(draw_margin_f) || draw_margin_f < 0.0
        error("draw_margin must be a nonnegative finite scalar.")
    end

    tie_mode = string(tie_handling)
    if tie_mode ∉ ("skip", "draw", "correct_draw_only")
        error("tie_handling must be one of: \"skip\", \"draw\", \"correct_draw_only\"")
    end

    mu = fill(mu_initial_f, L)
    sigma = fill(sigma_initial_f, L)

    norm_pdf(x::Float64) = exp(-0.5 * x^2) / sqrt(2.0 * π)
    norm_cdf(x::Float64) = 0.5 * erfc(-x / sqrt(2.0))
    log_norm_cdf(x::Float64) = x < 0.0 ?
        (-log(2.0) + logerfc(-x / sqrt(2.0))) : log(norm_cdf(x))

    function win_corrections(t::Float64, epsilon::Float64)
        x = t - epsilon
        local v::Float64
        local w::Float64
        if x < -10.0
            y = -x
            inverse_y = 1.0 / y
            inverse_y2 = inverse_y^2
            v = y + inverse_y * (
                1.0 + inverse_y2 * (-2.0 + inverse_y2 * (10.0 + inverse_y2 * -74.0))
            )
            w = 1.0 + inverse_y2 * (-1.0 + inverse_y2 * (6.0 + inverse_y2 * -50.0))
        elseif x < 0.0
            v = sqrt(2.0 / π) / erfcx(-x / sqrt(2.0))
            w = v * (v + x)
        else
            log_v = -0.5 * x^2 - 0.5 * log(2.0 * π) - log_norm_cdf(x)
            v = exp(log_v)
            w = v * (v + x)
        end
        return v, clamp(w, 0.0, 1.0)
    end

    function draw_corrections(t::Float64, epsilon::Float64)
        if epsilon <= sqrt(eps(Float64)) * (1.0 + abs(t))
            return -t, 1.0
        end

        a = -epsilon - t
        b = epsilon - t
        if a < -10.0 && b > 10.0
            return 0.0, 0.0
        end

        cdf_a = norm_cdf(a)
        cdf_b = norm_cdf(b)
        denom = cdf_b - cdf_a
        v = (norm_pdf(a) - norm_pdf(b)) / denom
        variance = 1.0 + (a * norm_pdf(a) - b * norm_pdf(b)) / denom - v^2

        if !isfinite(v) || !isfinite(variance) || variance < 0.0
            reflected = b < 0.0
            lower = reflected ? -b : a
            upper = reflected ? -a : b
            lower <= 0.0 && return 0.0, 0.0

            width = upper - lower
            scaled_width = lower * width
            if scaled_width > 50.0
                tail_v, tail_w = win_corrections(-lower, 0.0)
                v = tail_v
                variance = 1.0 - tail_w
            elseif scaled_width < 1e-4
                mean_offset = width * (0.5 - scaled_width / 12.0)
                variance = width^2 / 12.0
                v = lower + mean_offset
            else
                denominator = expm1(scaled_width)
                mean_offset = 1.0 / lower - width / denominator
                variance = 1.0 / lower^2 -
                           width^2 * exp(scaled_width) / denominator^2
                v = lower + mean_offset
            end
            reflected && (v = -v)
        end
        return v, clamp(1.0 - variance, 0.0, 1.0)
    end

    function update_decisive(
        mu1::Float64,
        sigma1::Float64,
        mu2::Float64,
        sigma2::Float64,
        player1_wins::Bool,
    )
        c_local = sqrt(2.0 * beta_f^2 + sigma1^2 + sigma2^2)
        epsilon = draw_margin_f / c_local

        local mu1_new::Float64
        local mu2_new::Float64
        local w::Float64
        if player1_wins
            t_local = (mu1 - mu2) / c_local
            v, w = win_corrections(t_local, epsilon)

            mu1_new = mu1 + (sigma1^2 / c_local) * v
            mu2_new = mu2 - (sigma2^2 / c_local) * v
        else
            t_local = (mu2 - mu1) / c_local
            v, w = win_corrections(t_local, epsilon)

            mu2_new = mu2 + (sigma2^2 / c_local) * v
            mu1_new = mu1 - (sigma1^2 / c_local) * v
        end

        sigma1_new = sigma1 * sqrt(max(1.0 - (sigma1^2 / c_local^2) * w, 1e-12))
        sigma2_new = sigma2 * sqrt(max(1.0 - (sigma2^2 / c_local^2) * w, 1e-12))

        return mu1_new, sigma1_new, mu2_new, sigma2_new
    end

    function update_draw(mu1::Float64, sigma1::Float64, mu2::Float64, sigma2::Float64)
        c_local = sqrt(2.0 * beta_f^2 + sigma1^2 + sigma2^2)
        epsilon = draw_margin_f / c_local
        t_local = (mu1 - mu2) / c_local
        v, w = draw_corrections(t_local, epsilon)

        mu1_new = mu1 + (sigma1^2 / c_local) * v
        mu2_new = mu2 - (sigma2^2 / c_local) * v
        sigma1_new = sigma1 * sqrt(max(1.0 - (sigma1^2 / c_local^2) * w, 1e-12))
        sigma2_new = sigma2 * sqrt(max(1.0 - (sigma2^2 / c_local^2) * w, 1e-12))

        return mu1_new, sigma1_new, mu2_new, sigma2_new
    end

    for t in 1:N
        for q in 1:M
            outcomes = @view Rv[:, q, t]
            event_mu = copy(mu)
            event_sigma = copy(sigma)
            prior_variance = event_sigma .^ 2
            precision_increment = zeros(Float64, L)
            natural_increment = zeros(Float64, L)

            for i in 1:L
                for j in (i + 1):L
                    r_i = outcomes[i]
                    r_j = outcomes[j]

                    if r_i == r_j
                        if tie_mode == "skip"
                            continue
                        end
                        if tie_mode == "correct_draw_only" && Int(r_i) == 0
                            continue
                        end

                        pair_mu_i, pair_sigma_i, pair_mu_j, pair_sigma_j = update_draw(
                            event_mu[i],
                            event_sigma[i],
                            event_mu[j],
                            event_sigma[j],
                        )
                    else
                        pair_mu_i, pair_sigma_i, pair_mu_j, pair_sigma_j = update_decisive(
                            event_mu[i],
                            event_sigma[i],
                            event_mu[j],
                            event_sigma[j],
                            r_i > r_j,
                        )
                    end

                    for (player, pair_mu, pair_sigma) in (
                        (i, pair_mu_i, pair_sigma_i),
                        (j, pair_mu_j, pair_sigma_j),
                    )
                        pair_variance = pair_sigma^2
                        precision_increment[player] +=
                            1.0 / pair_variance - 1.0 / prior_variance[player]
                        natural_increment[player] += pair_mu / pair_variance -
                                                     event_mu[player] / prior_variance[player]
                    end
                end
            end

            posterior_precision = 1.0 ./ prior_variance .+ precision_increment
            posterior_natural = event_mu ./ prior_variance .+ natural_increment
            sigma = sqrt.(1.0 ./ posterior_precision)
            mu = posterior_natural ./ posterior_precision
        end

        sigma .= sqrt.(sigma .^ 2 .+ tau_f^2)
    end

    ranking = rank_scores(mu)[string(method)]
    return return_scores ? (ranking, mu) : ranking
end

"""
    glicko(
        R;
        initial_rating=1500.0,
        initial_rd=350.0,
        c=0.0,
        rd_max=350.0,
        tie_handling="correct_draw_only",
        return_deviation=false,
        method="competition",
        return_scores=false,
    )

Rank models with sequential Glicko updates over induced pairwise comparisons.

If `return_deviation=true`, returns `(ranking, rating, rd)`; otherwise returns
`ranking` or `(ranking, rating)` when `return_scores=true`.

# Formula
Let ``q = \\ln(10)/400`` and
``g(RD) = 1/\\sqrt{1 + 3q^2 RD^2/\\pi^2}``.
For model `i` in one period:

```math
E_{ij} = \\frac{1}{1 + 10^{-g(RD_j)(r_i-r_j)/400}}
```

```math
d_i^2 =
\\left(q^2\\sum_j g(RD_j)^2 E_{ij}(1-E_{ij})\\right)^{-1}
```

```math
RD_i' = \\left(\\frac{1}{RD_i^2} + \\frac{1}{d_i^2}\\right)^{-1/2}, \\quad
r_i' = r_i +
\\frac{q}{\\frac{1}{RD_i^2}+\\frac{1}{d_i^2}}
\\sum_j g(RD_j)(S_{ij}-E_{ij})
```

# References
Glickman, M. E. (1999). Parameter Estimation in Large Dynamic Paired
Comparison Experiments. *JRSS C*, 48(3), 377-394.
https://doi.org/10.1111/1467-9876.00159
"""
function glicko(
    R;
    initial_rating=1500.0,
    initial_rd=350.0,
    c=0.0,
    rd_max=350.0,
    tie_handling="correct_draw_only",
    return_deviation=false,
    method="competition",
    return_scores=false,
)
    Rv = validate_input(R)
    L, M, N = size(Rv)

    initial_rating_f = Float64(initial_rating)
    initial_rd_f = Float64(initial_rd)
    if !isfinite(initial_rating_f)
        error("initial_rating must be finite.")
    end
    if !isfinite(initial_rd_f) || initial_rd_f <= 0.0
        error("initial_rd must be > 0 and finite.")
    end

    rd_max_f = Float64(rd_max)
    if !isfinite(rd_max_f) || rd_max_f <= 0.0
        error("rd_max must be > 0 and finite.")
    end

    c_f = Float64(c)
    if !isfinite(c_f) || c_f < 0.0
        error("c must be >= 0 and finite.")
    end

    tie_mode = string(tie_handling)
    if tie_mode ∉ ("skip", "draw", "correct_draw_only")
        error("tie_handling must be one of: \"skip\", \"draw\", \"correct_draw_only\"")
    end

    rating = fill(initial_rating_f, L)
    rd = fill(min(initial_rd_f, rd_max_f), L)

    q_const = log(10.0) / 400.0

    gfun(rd_opponent::AbstractVector{<:Real}) =
        1.0 ./ sqrt.(1.0 .+ (3.0 * q_const^2 .* (Float64.(rd_opponent) .^ 2)) ./ (π^2))

    expected_score(r_i::Float64, r_j::AbstractVector{<:Real}, g_j::AbstractVector{<:Real}) =
        1.0 ./ (1.0 .+ 10.0 .^ (-(Float64.(g_j) .* (r_i .- Float64.(r_j)) ./ 400.0)))

    for t in 1:N
        for m in 1:M
            if c_f > 0.0
                rd .= min.(sqrt.(rd .^ 2 .+ c_f^2), rd_max_f)
            end

            outcomes = @view Rv[:, m, t]
            opponents = [Int[] for _ in 1:L]
            results = [Float64[] for _ in 1:L]

            for i in 1:L
                for j in (i + 1):L
                    r_i = Int(outcomes[i])
                    r_j = Int(outcomes[j])

                    local s_i::Float64
                    local s_j::Float64
                    if r_i == r_j
                        if tie_mode == "skip"
                            continue
                        elseif tie_mode == "draw"
                            s_i = 0.5
                            s_j = 0.5
                        else
                            if r_i == 1
                                s_i = 0.5
                                s_j = 0.5
                            else
                                continue
                            end
                        end
                    else
                        if r_i > r_j
                            s_i, s_j = 1.0, 0.0
                        else
                            s_i, s_j = 0.0, 1.0
                        end
                    end

                    push!(opponents[i], j)
                    push!(results[i], s_i)
                    push!(opponents[j], i)
                    push!(results[j], s_j)
                end
            end

            new_rating = copy(rating)
            new_rd = copy(rd)

            for i in 1:L
                if isempty(opponents[i])
                    continue
                end

                opp = opponents[i]
                s = results[i]
                rd_opp = rd[opp]
                rating_opp = rating[opp]

                g_opp = gfun(rd_opp)
                E = expected_score(rating[i], rating_opp, g_opp)

                denom = sum((g_opp .^ 2) .* E .* (1.0 .- E))
                if denom <= 0.0 || !isfinite(denom)
                    continue
                end

                d2 = 1.0 / ((q_const^2) * denom)

                inv_var = (1.0 / (rd[i]^2)) + (1.0 / d2)
                if inv_var <= 0.0 || !isfinite(inv_var)
                    continue
                end

                rd_new = sqrt(1.0 / inv_var)
                rd_new = clamp(rd_new, 1e-12, rd_max_f)

                delta = sum(g_opp .* (s .- E))
                rating_new = rating[i] + (q_const / inv_var) * delta

                new_rating[i] = rating_new
                new_rd[i] = rd_new
            end

            rating = new_rating
            rd = new_rd
        end
    end

    ranking = rank_scores(rating)[string(method)]

    if return_deviation
        return ranking, rating, rd
    end

    return return_scores ? (ranking, rating) : ranking
end
