"""Average-family eval metrics with Bayesian uncertainty scaling."""

function _avg(
    R,
    w=nothing,
)::Float64
    Rm = _as_2d_int_matrix(R)

    if isnothing(w)
        _validate_binary(Rm)
        return Float64(sum(Rm) / length(Rm))
    end

    wv = Float64.(collect(w))
    C = length(wv) - 1
    _validate_matrix_range(Rm, 0, C, "R")

    return Float64(sum(wv[Rm .+ 1]) / length(Rm))
end

"""
    avg(R, w=nothing) -> (a, sigma_a)

Average score with Bayes-scaled uncertainty (Avg@N).

# References
Hariri, M., Samandar, A., Hinczewski, M., & Chaudhary, V. (2026).
Don't Pass@k: A Bayesian Framework for Large Language Model Evaluation.
*The Fourteenth International Conference on Learning Representations*.
https://openreview.net/forum?id=PTXi3Ef4sT

# Arguments
- `R::Union{AbstractVector, AbstractMatrix}`:
  integer outcomes. A 1D input with length `N` is reshaped to ``1 \\times N``.
  If `w` is omitted, entries must be binary in ``\\{0,1\\}``.
- `w`:
  optional length-``(C+1)`` weight vector ``(w_0,\\ldots,w_C)``.
  When omitted, binary mode is used with `w = [0.0, 1.0]`.

# Returns
- `Tuple{Float64, Float64}`:
  ``(a, \\sigma_a)`` where ``a`` is the weighted average and ``\\sigma_a`` is
  the uncertainty on the same scale.

# Notation
After coercion, let outcomes be ``R \\in \\{0,\\ldots,C\\}^{M \\times N}``.
For question ``\\alpha`` and trial ``i``, the score contribution is ``w_{R_{\\alpha i}}``.

# Formula
Point estimate:

```math
a = \\frac{1}{M \\cdot N}\\sum_{\\alpha=1}^{M}\\sum_{i=1}^{N} w_{R_{\\alpha i}}
```

Uncertainty rescales Bayes@N uncertainty with ``D=0`` and
``T = 1 + C + N``:

```math
\\sigma_a = \\frac{T}{N} \\cdot \\sigma_{\\text{Bayes}}
```

# Examples
```julia
R = [0 1 1 0 1;
     1 1 0 1 1]
a, sigma = avg(R)
```
"""
function avg(
    R,
    w,
)::Tuple{Float64, Float64}
    Rm = _as_2d_int_matrix(R)

    if isnothing(w)
        _validate_binary(Rm)
        wv = [0.0, 1.0]
    else
        wv = Float64.(collect(w))
    end

    _, N = size(Rm)
    if N <= 0
        error("R must have at least one column (N>=1)")
    end

    C = length(wv) - 1
    _, sigma_bayes = bayes(Rm, wv, nothing)
    T = 1 + C + N
    sigma_avg = (T / N) * sigma_bayes

    return _avg(Rm, wv), Float64(sigma_avg)
end

"""
    avg_ci(R, w=nothing, confidence=0.95, bounds=nothing)
        -> (a, sigma_a, lo, hi)

Avg@N plus Bayesian uncertainty and normal-approximation credible interval.

# References
Hariri, M., Samandar, A., Hinczewski, M., & Chaudhary, V. (2026).
Don't Pass@k: A Bayesian Framework for Large Language Model Evaluation.
*The Fourteenth International Conference on Learning Representations*.
https://openreview.net/forum?id=PTXi3Ef4sT

# Arguments
- `R::Union{AbstractVector, AbstractMatrix}`:
  same contract as [`avg`](@ref): coerced to an ``M \\times N`` outcome matrix.
- `w`:
  same contract as [`avg`](@ref): optional weights ``(w_0,\\ldots,w_C)``.
- `confidence::Real`:
  credibility level ``\\gamma \\in (0,1)``.
- `bounds::Union{Nothing, Tuple{<:Real, <:Real}}`:
  optional clipping interval ``(\\ell, u)`` applied to the returned bounds.

# Returns
- `Tuple{Float64, Float64, Float64, Float64}`:
  ``(a, \\sigma_a, \\mathrm{lo}, \\mathrm{hi})``.

# Notation
Let ``(a, \\sigma_a)`` be the Avg@N summary from [`avg`](@ref)
on the same inputs.
Let ``\\gamma = \\texttt{confidence}`` and
``z_{(1+\\gamma)/2}`` be the standard normal quantile.

# Formula

```math
(\\mathrm{lo}, \\mathrm{hi}) = a \\pm z_{(1+\\gamma)/2}\\,\\sigma_a
```

with ``\\gamma = \\texttt{confidence}``, then clipped to `bounds` if provided.

# Examples
```julia
R = [0 1 1 0 1;
     1 1 0 1 1]

a, sigma, lo, hi = avg_ci(R, nothing, 0.95, (0.0, 1.0))
```
"""
function avg_ci(
    R,
    w,
    confidence::Real,
    bounds,
)::Tuple{Float64, Float64, Float64, Float64}
    a, sigma = avg(R, w)
    lo, hi = normal_credible_interval(
        a,
        sigma;
        credibility=confidence,
        two_sided=true,
        bounds=bounds,
    )
    return Float64(a), Float64(sigma), Float64(lo), Float64(hi)
end
