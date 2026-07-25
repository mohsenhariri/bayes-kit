"""Shared helpers for eval metrics and credible intervals."""

using SpecialFunctions: erfcinv

_is_eval_sequence(value) = value isa AbstractVector || value isa Tuple

function _eval_int_cast(value)::Int
    if value isa Complex
        return trunc(Int, real(value))
    elseif value isa Real
        return trunc(Int, value)
    elseif value isa AbstractString
        return parse(Int, value)
    end
    return Int(value)
end

function _as_eval_int_array(R, name::AbstractString="R")
    raw = if R isa AbstractMatrix
        Array(R)
    elseif R isa AbstractVector || R isa Tuple
        values = collect(R)
        if !isempty(values) && all(_is_eval_sequence, values)
            rows = collect.(values)
            width = length(rows[1])
            all(length(row) == width for row in rows) ||
                error("$name must be a rectangular 1D or 2D array.")
            matrix = Matrix{Any}(undef, length(rows), width)
            for row in eachindex(rows), column in 1:width
                matrix[row, column] = rows[row][column]
            end
            matrix
        else
            values
        end
    elseif R isa AbstractArray
        Array(R)
    else
        error("$name must be a 1D or 2D array.")
    end

    ndims(raw) in (1, 2) || error("$name must be a 1D or 2D array.")
    return _eval_int_cast.(raw)
end

function _as_2d_int_matrix(R)::Matrix{Int}
    converted = _as_eval_int_array(R)
    return ndims(converted) == 1 ? reshape(converted, 1, :) : converted
end

function _validate_matrix_range(
    R::AbstractMatrix{<:Integer},
    low::Integer,
    high::Integer,
    name::AbstractString,
)::Nothing
    if isempty(R)
        return nothing
    end
    if minimum(R) < low || maximum(R) > high
        error("Entries of $name must be integers in [$low, $high].")
    end
    return nothing
end

function _validate_binary(R::AbstractMatrix{<:Integer}, name::AbstractString="R")::Nothing
    _validate_matrix_range(R, 0, 1, name)
    return nothing
end

# Inverse standard normal CDF, matching scipy.special.ndtri to floating-point
# precision through the inverse complementary error function.
function _normal_ppf(p::Float64)::Float64
    if p == 0.0
        return -Inf
    elseif p == 1.0
        return Inf
    end

    return Float64(-sqrt(2.0) * erfcinv(2.0 * p))
end

# Abramowitz-Stegun normal CDF approximation.
function _normal_cdf(x::Float64)::Float64
    z = abs(x)
    t = 1.0 / (1.0 + 0.2316419 * z)
    poly = t * (
        0.319381530 +
        t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + 1.330274429 * t)))
    )
    pdf = 0.3989422804014327 * exp(-0.5 * z * z)
    cdf = 1.0 - pdf * poly
    return x >= 0.0 ? cdf : 1.0 - cdf
end

function _z_value(confidence::Real; two_sided::Bool=true)::Float64
    conf = Float64(confidence)
    if !(0.0 < conf < 1.0)
        error("confidence must be in (0,1); got $confidence")
    end
    if two_sided
        return _normal_ppf(0.5 + 0.5 * conf)
    end
    return _normal_ppf(conf)
end

function normal_credible_interval(
    mu::Real,
    sigma::Real;
    credibility::Real=0.95,
    two_sided::Bool=true,
    bounds=nothing,
)::Tuple{Float64, Float64}
    mu_f = Float64(mu)
    sigma_f = Float64(sigma)

    if sigma_f < 0.0
        error("sigma must be >= 0; got $sigma")
    end

    z = _z_value(credibility; two_sided=two_sided)
    if two_sided
        lo = mu_f - z * sigma_f
        hi = mu_f + z * sigma_f
    else
        lo = -Inf
        hi = mu_f + z * sigma_f
    end

    if !isnothing(bounds)
        length(bounds) == 2 || error("bounds must contain exactly two values")
        b_lo = Float64(bounds[1])
        b_hi = Float64(bounds[2])
        if b_lo > b_hi
            error("bounds must satisfy bounds[1] <= bounds[2]")
        end
        lo = max(lo, b_lo)
        hi = min(hi, b_hi)
    end

    return lo, hi
end
