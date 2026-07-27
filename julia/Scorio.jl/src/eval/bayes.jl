"""Bayes-family eval metrics for categorical outcomes."""

"""
    bayes(R, w=nothing, R0=nothing) -> (mu, sigma)

Performance evaluation using the Bayes@N framework.

# References
Hariri, M., Samandar, A., Hinczewski, M., & Chaudhary, V. (2026).
Don't Pass@k: A Bayesian Framework for Large Language Model Evaluation.
*The Fourteenth International Conference on Learning Representations*.
https://openreview.net/forum?id=PTXi3Ef4sT

# Arguments
- `R::Union{AbstractVector, AbstractMatrix}`:
  integer outcomes. A 1D input with length `N` is reshaped to ``1 \\times N``.
  After coercion, ``R`` is ``M \\times N`` with entries in ``\\{0,\\ldots,C\\}``.
- `w`:
  optional length-``(C+1)`` weight vector ``(w_0,\\ldots,w_C)`` mapping class
  ``k`` to score ``w_k``.
  If omitted and `R` is binary, defaults to `[0.0, 1.0]`.
  For non-binary `R`, `w` is required.
- `R0`:
  optional prior outcomes. Accepts 1D/2D integer input; after coercion it must
  be an ``M \\times D`` matrix with entries in ``\\{0,\\ldots,C\\}``.
  If omitted, ``D=0``.

# Returns
- `Tuple{Float64, Float64}`: ``(\\mu, \\sigma)`` posterior mean and posterior
  standard deviation.

# Notation
``\\delta_{a,b}`` is the Kronecker delta. For each question ``\\alpha`` and class
``k \\in \\{0,\\ldots,C\\}``:

```math
n_{\\alpha k} = \\sum_{i=1}^{N} \\delta_{k, R_{\\alpha i}}
```

```math
n^0_{\\alpha k} = 1 + \\sum_{i=1}^{D} \\delta_{k, R^0_{\\alpha i}}
```

```math
\\nu_{\\alpha k} = n_{\\alpha k} + n^0_{\\alpha k}
```

The effective sample size is:

```math
T = 1 + C + D + N
```

# Formula

```math
\\mu = w_0 + \\frac{1}{M \\cdot T}
\\sum_{\\alpha=1}^{M}\\sum_{j=0}^{C}\\nu_{\\alpha j}(w_j - w_0)
```

```math
\\sigma = \\sqrt{
\\frac{1}{M^2 (T+1)} \\sum_{\\alpha=1}^{M} \\left[
\\sum_j \\frac{\\nu_{\\alpha j}}{T}(w_j-w_0)^2 -
\\left(\\sum_j \\frac{\\nu_{\\alpha j}}{T}(w_j-w_0)\\right)^2
\\right]
}
```

# Examples
```julia
R = [0 1 2 2 1;
     1 1 0 2 2]
w = [0.0, 0.5, 1.0]
R0 = [0 2;
      1 2]

mu, sigma = bayes(R, w, R0)
```
"""
function bayes(
    R,
    w,
    R0,
)::Tuple{Float64, Float64}
    Rm = _as_2d_int_matrix(R)

    if isnothing(w)
        unique_vals = unique(Rm)
        is_binary = length(unique_vals) <= 2 && all(v -> v == 0 || v == 1, unique_vals)

        if is_binary
            wv = [0.0, 1.0]
        else
            unique_str = join(sort(unique_vals), ", ")
            error(
                "R contains more than 2 unique values ($unique_str), so weight vector 'w' must be provided. " *
                "Please specify a weight vector of length $(length(unique_vals)) to map each category to a score.",
            )
        end
    else
        wv = Float64.(collect(w))
    end

    M, N = size(Rm)
    C = length(wv) - 1

    if isnothing(R0)
        D = 0
        R0m = zeros(Int, M, 0)
    else
        R0m = _as_eval_int_array(R0, "R0")
        if ndims(R0m) == 1
            try
                # NumPy's `reshape(M, -1)` consumes a flat prior row by row.
                # Julia arrays are column-major, so first form the transposed
                # `D × M` layout and then materialize its transpose.
                R0m = permutedims(reshape(R0m, :, M))
            catch
                error("R0 must have the same number of rows (M) as R.")
            end
        elseif ndims(R0m) != 2
            error("R0 must be a 1D or 2D array.")
        end

        if size(R0m, 1) != M
            error("R0 must have the same number of rows (M) as R.")
        end
        D = size(R0m, 2)
    end

    _validate_matrix_range(Rm, 0, C, "R")
    _validate_matrix_range(R0m, 0, C, "R0")

    T = 1 + C + D + N

    function _row_bincount(A::AbstractMatrix{<:Integer}, length::Int)::Matrix{Int}
        if size(A, 2) == 0
            return zeros(Int, size(A, 1), length)
        end

        out = zeros(Int, size(A, 1), length)
        @inbounds for i in 1:size(A, 1)
            for j in 1:size(A, 2)
                out[i, A[i, j] + 1] += 1
            end
        end
        return out
    end

    n_counts = _row_bincount(Rm, C + 1)
    n0_counts = _row_bincount(R0m, C + 1) .+ 1
    nu = n_counts .+ n0_counts

    delta_w = wv .- wv[1]
    mu = wv[1] + sum(nu * delta_w) / (M * T)

    nu_over_T = nu ./ T
    termA = vec(sum(nu_over_T .* reshape(delta_w .^ 2, 1, :), dims=2))
    termB = (nu_over_T * delta_w) .^ 2
    sigma = sqrt(sum(termA .- termB) / (M^2 * (T + 1)))

    return Float64(mu), Float64(sigma)
end

"""
    bayes_ci(R, w=nothing, R0=nothing, confidence=0.95, bounds=nothing)
        -> (mu, sigma, lo, hi)

Bayes@N posterior summary with a normal-approximation credible interval.

# References
Hariri, M., Samandar, A., Hinczewski, M., & Chaudhary, V. (2026).
Don't Pass@k: A Bayesian Framework for Large Language Model Evaluation.
*The Fourteenth International Conference on Learning Representations*.
https://openreview.net/forum?id=PTXi3Ef4sT

# Arguments
- `R::Union{AbstractVector, AbstractMatrix}`:
  same contract as [`bayes`](@ref): coerced to an ``M \\times N`` integer matrix.
- `w`:
  same contract as [`bayes`](@ref): optional class weights
  ``(w_0,\\ldots,w_C)``.
- `R0`:
  same contract as [`bayes`](@ref): optional prior outcomes as ``M \\times D``.
- `confidence::Real`:
  credibility level ``\\gamma \\in (0,1)`` (for example, `0.95`).
- `bounds::Union{Nothing, Tuple{<:Real, <:Real}}`:
  optional clipping interval ``(\\ell, u)`` applied to the returned bounds.

# Returns
- `Tuple{Float64, Float64, Float64, Float64}`:
  ``(\\mu, \\sigma, \\mathrm{lo}, \\mathrm{hi})``.

# Notation
Let ``(\\mu, \\sigma)`` be the Bayes@N posterior summary returned by [`bayes`](@ref)
on the same inputs.
Let ``\\gamma = \\texttt{confidence}`` and
``z_{(1+\\gamma)/2}`` be the standard normal quantile.

# Formula
The interval is:

```math
(\\mathrm{lo}, \\mathrm{hi})
= \\mu \\pm z_{(1+\\gamma)/2}\\,\\sigma
```

and then clipped to `bounds` when provided.

# Examples
```julia
R = [0 1 2 2 1;
     1 1 0 2 2]
w = [0.0, 0.5, 1.0]
R0 = [0 2;
      1 2]

mu, sigma, lo, hi = bayes_ci(R, w, R0, 0.95, (0.0, 1.0))
```
"""
function bayes_ci(
    R,
    w,
    R0,
    confidence::Real,
    bounds,
)::Tuple{Float64, Float64, Float64, Float64}
    mu, sigma = bayes(R, w, R0)
    lo, hi = normal_credible_interval(
        mu,
        sigma;
        credibility=confidence,
        two_sided=true,
        bounds=bounds,
    )
    return Float64(mu), Float64(sigma), Float64(lo), Float64(hi)
end
