"""NumPy-compatible SeedSequence, PCG64, and distribution primitives."""

# Derived from NumPy v2.4.0, commit c5ab79c14c98bfda1e60770ffa23a6130f8267b7:
#   numpy/random/bit_generator.pyx
#   numpy/random/src/pcg64/{pcg64.c,pcg64.h}
#   numpy/random/src/distributions/{distributions.c,ziggurat_constants.h}
# Copyright (c) 2005-2025, NumPy Developers. NumPy is distributed under the
# BSD-3-Clause license; see ../THIRDPARTY.md.
# The PCG code is Copyright (c) 2014 Melissa O'Neill and (c) 2015 Robert Kern,
# distributed under the MIT license recorded in that notices file.

import Base64
import Random

# NumPy's distribution kernels call the platform C math library directly.
# Julia normally routes these functions through libopenlibm, whose last-bit
# results can differ.  Use the same platform entry points so seeded draws are
# bit-for-bit reproducible, including uncommon rejection branches.
@static if Sys.islinux()
    const _NP_PLATFORM_LIBM = "libm.so.6"
elseif Sys.isapple()
    const _NP_PLATFORM_LIBM = "/usr/lib/libSystem.B.dylib"
elseif Sys.iswindows()
    const _NP_PLATFORM_LIBM = "api-ms-win-crt-math-l1-1-0.dll"
elseif Sys.isfreebsd()
    const _NP_PLATFORM_LIBM = "libm.so.5"
else
    const _NP_PLATFORM_LIBM = Base.libm_name
end

@inline _numpy_exp(value::Float64) =
    ccall((:exp, _NP_PLATFORM_LIBM), Cdouble, (Cdouble,), value)
@inline _numpy_log(value::Float64) =
    ccall((:log, _NP_PLATFORM_LIBM), Cdouble, (Cdouble,), value)
@inline _numpy_log1p(value::Float64) =
    ccall((:log1p, _NP_PLATFORM_LIBM), Cdouble, (Cdouble,), value)
@inline _numpy_pow(base::Float64, exponent::Float64) =
    ccall((:pow, _NP_PLATFORM_LIBM), Cdouble, (Cdouble, Cdouble), base, exponent)
@inline _numpy_sqrt(value::Float64) =
    ccall((:sqrt, _NP_PLATFORM_LIBM), Cdouble, (Cdouble,), value)

const _NP_SS_INIT_A = UInt32(0x43b0d7e5)
const _NP_SS_MULT_A = UInt32(0x931e8875)
const _NP_SS_INIT_B = UInt32(0x8b51f9dd)
const _NP_SS_MULT_B = UInt32(0x58f38ded)
const _NP_SS_MIX_MULT_L = UInt32(0xca01f9dd)
const _NP_SS_MIX_MULT_R = UInt32(0x4973f715)

function _numpy_entropy_words!(out::Vector{UInt32}, value)
    if value isa Integer
        value < 0 && error("expected non-negative integer")
        remaining = BigInt(value)
        if iszero(remaining)
            push!(out, UInt32(0))
        else
            mask = BigInt(0xffffffff)
            while remaining > 0
                push!(out, UInt32(remaining & mask))
                remaining >>= 32
            end
        end
    elseif value isa AbstractArray || value isa Tuple || value isa AbstractRange
        for element in value
            _numpy_entropy_words!(out, element)
        end
    elseif value isa AbstractFloat
        error("seed must be integer")
    else
        error(
            "SeedSequence expects int or sequence of ints for entropy, got " *
            repr(value),
        )
    end
    return out
end

@inline function _numpy_hashmix(value::UInt32, hash_const::UInt32)
    value = xor(value, hash_const)
    hash_const *= _NP_SS_MULT_A
    value *= hash_const
    value = xor(value, value >> 16)
    return value, hash_const
end

@inline function _numpy_mix(x::UInt32, y::UInt32)
    result = _NP_SS_MIX_MULT_L * x - _NP_SS_MIX_MULT_R * y
    return xor(result, result >> 16)
end

function _numpy_seed_sequence_pool(seed)
    entropy = UInt32[]
    if isnothing(seed)
        device = Random.RandomDevice()
        for _ in 1:4
            push!(entropy, rand(device, UInt32))
        end
    else
        _numpy_entropy_words!(entropy, seed)
    end

    pool = zeros(UInt32, 4)
    hash_const = _NP_SS_INIT_A
    for i in eachindex(pool)
        word = i <= length(entropy) ? entropy[i] : UInt32(0)
        pool[i], hash_const = _numpy_hashmix(word, hash_const)
    end
    for src in eachindex(pool)
        for dst in eachindex(pool)
            if src != dst
                hashed, hash_const = _numpy_hashmix(pool[src], hash_const)
                pool[dst] = _numpy_mix(pool[dst], hashed)
            end
        end
    end
    for src in 5:length(entropy)
        for dst in eachindex(pool)
            hashed, hash_const = _numpy_hashmix(entropy[src], hash_const)
            pool[dst] = _numpy_mix(pool[dst], hashed)
        end
    end
    return pool
end

function _numpy_seed_sequence_state(seed, n_words::Int=8)
    pool = _numpy_seed_sequence_pool(seed)
    state = Vector{UInt32}(undef, n_words)
    hash_const = _NP_SS_INIT_B
    for i in eachindex(state)
        value = pool[mod1(i, length(pool))]
        value = xor(value, hash_const)
        hash_const *= _NP_SS_MULT_B
        value *= hash_const
        state[i] = xor(value, value >> 16)
    end
    return state
end

const _NP_PCG64_MULTIPLIER =
    (UInt128(2549297995355413924) << 64) | UInt128(4865540595714422341)

mutable struct _NumpyBinomialCache
    has_binomial::Bool
    psave::Float64
    nsave::Int64
    r::Float64
    q::Float64
    fm::Float64
    m::Int64
    p1::Float64
    xm::Float64
    xl::Float64
    xr::Float64
    c::Float64
    laml::Float64
    lamr::Float64
    p2::Float64
    p3::Float64
    p4::Float64
end

_NumpyBinomialCache() = _NumpyBinomialCache(
    false,
    0.0,
    0,
    0.0,
    0.0,
    0.0,
    0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
)

mutable struct _NumpyRNG
    state::UInt128
    inc::UInt128
    has_uint32::Bool
    uinteger::UInt32
    binomial::_NumpyBinomialCache
end

@inline function _numpy_pcg_step!(rng::_NumpyRNG)
    rng.state = rng.state * _NP_PCG64_MULTIPLIER + rng.inc
    return nothing
end

function _NumpyRNG(seed)
    words = _numpy_seed_sequence_state(seed, 8)
    values = Vector{UInt64}(undef, 4)
    for i in eachindex(values)
        values[i] = UInt64(words[2i - 1]) | (UInt64(words[2i]) << 32)
    end
    initstate = (UInt128(values[1]) << 64) | UInt128(values[2])
    initseq = (UInt128(values[3]) << 64) | UInt128(values[4])

    rng = _NumpyRNG(
        UInt128(0),
        (initseq << 1) | UInt128(1),
        false,
        UInt32(0),
        _NumpyBinomialCache(),
    )
    _numpy_pcg_step!(rng)
    rng.state += initstate
    _numpy_pcg_step!(rng)
    return rng
end

@inline function _numpy_next_u64!(rng::_NumpyRNG)
    _numpy_pcg_step!(rng)
    high = UInt64(rng.state >> 64)
    low = UInt64(rng.state & UInt128(typemax(UInt64)))
    x = xor(high, low)
    rotation = Int(rng.state >> 122)
    return rotation == 0 ? x : (x >> rotation) | (x << (64 - rotation))
end

@inline function _numpy_next_u32!(rng::_NumpyRNG)
    if rng.has_uint32
        rng.has_uint32 = false
        return rng.uinteger
    end
    value = _numpy_next_u64!(rng)
    rng.has_uint32 = true
    rng.uinteger = UInt32(value >> 32)
    return UInt32(value)
end

@inline _numpy_uniform!(rng::_NumpyRNG) =
    Float64(_numpy_next_u64!(rng) >> 11) * (1.0 / 9007199254740992.0)

const _NP_ZIGGURAT_NORMAL_R = 3.6541528853610087963519472518
const _NP_ZIGGURAT_NORMAL_INV_R = 0.27366123732975827203338247596
const _NP_ZIGGURAT_EXP_R = 7.6971174701310497140446280481

function _numpy_standard_normal!(rng::_NumpyRNG)
    while true
        bits = _numpy_next_u64!(rng)
        index0 = Int(bits & UInt64(0xff))
        shifted = bits >> 8
        negative = isodd(shifted)
        magnitude = (shifted >> 1) & UInt64(0x000fffffffffffff)
        x = Float64(magnitude) * _NP_WI_DOUBLE[index0 + 1]
        negative && (x = -x)

        magnitude < _NP_KI_DOUBLE[index0 + 1] && return x
        if index0 == 0
            while true
                xx =
                    -_NP_ZIGGURAT_NORMAL_INV_R * _numpy_log1p(-_numpy_uniform!(rng))
                yy = -_numpy_log1p(-_numpy_uniform!(rng))
                if yy + yy > xx * xx
                    tail = _NP_ZIGGURAT_NORMAL_R + xx
                    return isodd(magnitude >> 8) ? -tail : tail
                end
            end
        else
            lower = _NP_FI_DOUBLE[index0]
            upper = _NP_FI_DOUBLE[index0 + 1]
            if (lower - upper) * _numpy_uniform!(rng) + upper <
               _numpy_exp(-0.5 * x * x)
                return x
            end
        end
    end
end

function _numpy_standard_exponential!(rng::_NumpyRNG)
    while true
        bits = _numpy_next_u64!(rng) >> 3
        index0 = Int(bits & UInt64(0xff))
        magnitude = bits >> 8
        x = Float64(magnitude) * _NP_WE_DOUBLE[index0 + 1]
        magnitude < _NP_KE_DOUBLE[index0 + 1] && return x

        if index0 == 0
            return _NP_ZIGGURAT_EXP_R - _numpy_log1p(-_numpy_uniform!(rng))
        end
        lower = _NP_FE_DOUBLE[index0]
        upper = _NP_FE_DOUBLE[index0 + 1]
        if (lower - upper) * _numpy_uniform!(rng) + upper < _numpy_exp(-x)
            return x
        end
    end
end

function _numpy_standard_gamma!(rng::_NumpyRNG, shape::Float64)
    if shape == 1.0
        return _numpy_standard_exponential!(rng)
    elseif shape == 0.0
        return 0.0
    elseif shape < 1.0
        while true
            uniform = _numpy_uniform!(rng)
            exponential = _numpy_standard_exponential!(rng)
            if uniform <= 1.0 - shape
                candidate = _numpy_pow(uniform, 1.0 / shape)
                candidate <= exponential && return candidate
            else
                y = -_numpy_log((1.0 - uniform) / shape)
                candidate = _numpy_pow(1.0 - shape + shape * y, 1.0 / shape)
                candidate <= exponential + y && return candidate
            end
        end
    end

    b = shape - 1.0 / 3.0
    c = 1.0 / _numpy_sqrt(9.0 * b)
    while true
        x = _numpy_standard_normal!(rng)
        v = 1.0 + c * x
        while v <= 0.0
            x = _numpy_standard_normal!(rng)
            v = 1.0 + c * x
        end
        v = v * v * v
        uniform = _numpy_uniform!(rng)
        uniform < 1.0 - 0.0331 * (x * x) * (x * x) && return b * v
        _numpy_log(uniform) <
        0.5 * x * x + b * (1.0 - v + _numpy_log(v)) && return b * v
    end
end

function _numpy_beta!(rng::_NumpyRNG, alpha::Float64, beta::Float64)
    if alpha <= 1.0 && beta <= 1.0
        if alpha < 3e-103 && beta < 3e-103
            return (alpha + beta) * _numpy_uniform!(rng) < alpha ? 1.0 : 0.0
        end
        while true
            uniform_a = _numpy_uniform!(rng)
            uniform_b = _numpy_uniform!(rng)
            x = _numpy_pow(uniform_a, 1.0 / alpha)
            y = _numpy_pow(uniform_b, 1.0 / beta)
            total = x + y
            if total <= 1.0 && uniform_a + uniform_b > 0.0
                if x > 0.0 && y > 0.0
                    return x / total
                end
                log_x = _numpy_log(uniform_a) / alpha
                log_y = _numpy_log(uniform_b) / beta
                delta = log_x - log_y
                return delta > 0.0 ?
                       _numpy_exp(-_numpy_log1p(_numpy_exp(-delta))) :
                       _numpy_exp(delta - _numpy_log1p(_numpy_exp(delta)))
            end
        end
    end

    gamma_a = _numpy_standard_gamma!(rng, alpha)
    gamma_b = _numpy_standard_gamma!(rng, beta)
    return gamma_a / (gamma_a + gamma_b)
end

@inline _numpy_normal!(rng::_NumpyRNG, location::Float64, scale::Float64) =
    location + scale * _numpy_standard_normal!(rng)

function _numpy_binomial_inversion!(rng::_NumpyRNG, n::Int64, probability::Float64)
    cache = rng.binomial
    if !cache.has_binomial || cache.nsave != n || cache.psave != probability
        cache.nsave = n
        cache.psave = probability
        cache.has_binomial = true
        cache.q = 1.0 - probability
        cache.r = _numpy_exp(n * _numpy_log1p(-probability))
        cache.c = n * probability
        cache.m = floor(
            Int64,
            min(
                Float64(n),
                cache.c + 10.0 * _numpy_sqrt(cache.c * cache.q + 1.0),
            ),
        )
    end

    q = cache.q
    probability_zero = cache.r
    bound = cache.m
    successes = Int64(0)
    mass = probability_zero
    uniform = _numpy_uniform!(rng)
    while uniform > mass
        successes += 1
        if successes > bound
            successes = 0
            mass = probability_zero
            uniform = _numpy_uniform!(rng)
        else
            uniform -= mass
            mass = ((n - successes + 1) * probability * mass) /
                   (successes * q)
        end
    end
    return successes
end

@inline function _numpy_stirling_tail(value::Float64, squared::Float64)
    numerator = 13680.0 -
                (462.0 -
                 (132.0 - (99.0 - 140.0 / squared) / squared) / squared) /
                squared
    return numerator / value / 166320.0
end

function _numpy_binomial_btpe!(rng::_NumpyRNG, n::Int64, probability::Float64)
    cache = rng.binomial
    if !cache.has_binomial || cache.nsave != n || cache.psave != probability
        cache.nsave = n
        cache.psave = probability
        cache.has_binomial = true
        cache.r = min(probability, 1.0 - probability)
        cache.q = 1.0 - cache.r
        cache.fm = n * cache.r + cache.r
        cache.m = floor(Int64, cache.fm)
        cache.p1 =
            floor(2.195 * _numpy_sqrt(n * cache.r * cache.q) - 4.6 * cache.q) +
            0.5
        cache.xm = cache.m + 0.5
        cache.xl = cache.xm - cache.p1
        cache.xr = cache.xm + cache.p1
        cache.c = 0.134 + 20.5 / (15.3 + cache.m)
        a = (cache.fm - cache.xl) / (cache.fm - cache.xl * cache.r)
        cache.laml = a * (1.0 + a / 2.0)
        a = (cache.xr - cache.fm) / (cache.xr * cache.q)
        cache.lamr = a * (1.0 + a / 2.0)
        cache.p2 = cache.p1 * (1.0 + 2.0 * cache.c)
        cache.p3 = cache.p2 + cache.c / cache.laml
        cache.p4 = cache.p3 + cache.c / cache.lamr
    end

    r = cache.r
    q = cache.q
    m = cache.m
    nrq = n * r * q
    while true
        uniform_region = _numpy_uniform!(rng) * cache.p4
        acceptance = _numpy_uniform!(rng)
        candidate = Int64(0)

        if uniform_region <= cache.p1
            candidate = floor(Int64, cache.xm - cache.p1 * acceptance + uniform_region)
            return probability > 0.5 ? n - candidate : candidate
        elseif uniform_region <= cache.p2
            x = cache.xl + (uniform_region - cache.p1) / cache.c
            acceptance = acceptance * cache.c + 1.0 -
                         abs(m - x + 0.5) / cache.p1
            acceptance > 1.0 && continue
            candidate = floor(Int64, x)
        elseif uniform_region <= cache.p3
            acceptance == 0.0 && continue
            candidate = floor(
                Int64,
                cache.xl + _numpy_log(acceptance) / cache.laml,
            )
            candidate < 0 && continue
            acceptance *= (uniform_region - cache.p2) * cache.laml
        else
            acceptance == 0.0 && continue
            candidate = floor(
                Int64,
                cache.xr - _numpy_log(acceptance) / cache.lamr,
            )
            candidate > n && continue
            acceptance *= (uniform_region - cache.p3) * cache.lamr
        end

        distance = abs(candidate - m)
        if distance > 20 && distance < nrq / 2.0 - 1.0
            rho = (distance / nrq) *
                  (
                (distance * (distance / 3.0 + 0.625) +
                 0.16666666666666666) / nrq + 0.5
            )
            center = -distance * distance / (2.0 * nrq)
            log_acceptance = _numpy_log(acceptance)
            if log_acceptance < center - rho
                return probability > 0.5 ? n - candidate : candidate
            elseif log_acceptance > center + rho
                continue
            end

            x1 = Float64(candidate) + 1.0
            f1 = Float64(m) + 1.0
            z = Float64(n) + 1.0 - Float64(m)
            w = Float64(n) - Float64(candidate) + 1.0
            x2 = x1 * x1
            f2 = f1 * f1
            z2 = z * z
            w2 = w * w
            threshold = cache.xm * _numpy_log(f1 / x1) +
                        (n - m + 0.5) * _numpy_log(z / w) +
                        (candidate - m) * _numpy_log(w * r / (x1 * q)) +
                        _numpy_stirling_tail(f1, f2) +
                        _numpy_stirling_tail(z, z2) +
                        _numpy_stirling_tail(x1, x2) +
                        _numpy_stirling_tail(w, w2)
            log_acceptance > threshold && continue
            return probability > 0.5 ? n - candidate : candidate
        end

        ratio = r / q
        scaled = ratio * (n + 1)
        likelihood_ratio = 1.0
        if m < candidate
            for value in (m + 1):candidate
                likelihood_ratio *= scaled / value - ratio
            end
        elseif m > candidate
            for value in (candidate + 1):m
                likelihood_ratio /= scaled / value - ratio
            end
        end
        acceptance > likelihood_ratio && continue
        return probability > 0.5 ? n - candidate : candidate
    end
end

function _numpy_binomial!(rng::_NumpyRNG, n::Integer, probability::Float64)
    trials = Int64(n)
    (trials == 0 || probability == 0.0) && return Int64(0)
    if probability <= 0.5
        return probability * trials <= 30.0 ?
               _numpy_binomial_inversion!(rng, trials, probability) :
               _numpy_binomial_btpe!(rng, trials, probability)
    end

    complement = 1.0 - probability
    failures = complement * trials <= 30.0 ?
               _numpy_binomial_inversion!(rng, trials, complement) :
               _numpy_binomial_btpe!(rng, trials, complement)
    return trials - failures
end

function _numpy_multinomial!(
    rng::_NumpyRNG,
    n::Integer,
    probabilities::AbstractVector{<:Real},
)
    categories = length(probabilities)
    categories >= 1 || error("multinomial requires at least one category")
    allocations = zeros(Int64, categories)
    remaining_probability = 1.0
    remaining_trials = Int64(n)
    if categories > 1
        for category in 1:(categories - 1)
            probability = Float64(probabilities[category]) / remaining_probability
            draw = _numpy_binomial!(rng, remaining_trials, probability)
            allocations[category] = draw
            remaining_trials -= draw
            remaining_trials <= 0 && break
            remaining_probability -= Float64(probabilities[category])
        end
    end
    remaining_trials > 0 && (allocations[end] = remaining_trials)
    return allocations
end

function _numpy_decode_u64_table(encoded::String)
    bytes = Base64.base64decode(encoded)
    length(bytes) % 8 == 0 || error("invalid NumPy table encoding")
    out = Vector{UInt64}(undef, length(bytes) ÷ 8)
    for i in eachindex(out)
        value = UInt64(0)
        for byte in 0:7
            value |= UInt64(bytes[8(i - 1) + byte + 1]) << (8 * byte)
        end
        out[i] = value
    end
    return out
end

_numpy_decode_f64_table(encoded::String) =
    reinterpret.(Float64, _numpy_decode_u64_table(encoded))

const _NP_KI_DOUBLE = _numpy_decode_u64_table("au8lgD3zDgAAAAAAAAAAAKjG+5i+CAwAQoG9+lSjDQDq7sF+9lEOAH730+lVsg4Aucp+gUvvDgCqRPoKRxkPABjL/2HtNw8AXCVhlUZPDwCWoxvkpWEPAKSWU3V6cA8AmkQo7LJ8DwDTV2MM8YYPAN4lg1emjw8A2tBNxySXDwAJ9dsHqZ0PAHT6gfVgow8A+Etb3m+oDwDcVNNg8awPAA+5GGf7sA8AxnRTjZ+0DwB3/mYj7LcPAA7loensug8A7QsEnau9DwBXbP9gMMAPAEiiNxCCwg8A0VvieqbEDwAx7nqXosYPAKSWKKl6yA8Ahd5LXjLKDwAaIwLpzMsPAMQ5+BJNzQ8AmeyPTbXODwAwyR2/B9APAObE1k1G0Q8AUPTiqHLSDwAeyfBPjtMPAHi0kJma1A8AUw+SuJjVDwDsmY7AidYPADLoyKlu1w8A6Ah7VEjYDwCMLK2LF9kPANKtpwfd2Q8AjF4QcJnaDwAgLsBdTdsPAND8W1z52w8AfZq5653cDwCdchiBO90PAJAvNIjS3Q8AZJ82ZGPeDwBOUY1w7t4PAC60pgF03w8AQO2ZZfTfDwDyJLzkb+APAFiiJcLm4A8ATLgoPFnhDwCZP7yMx+EPAKoc2+kx4g8AkRvahZjiDwCGQbWP++IPAEqNVTNb4w8AKgDQmbfjDwB/rZ7pEOQPADR31EZn5A8AXAlM07rkDwAkldKuC+UPAHi8TvdZ5Q8AEhLkyKXlDwCJhhM+7+UPAHgQ2W825g8AeNXGdXvmDwCqER5mvuYPAPL05VX/5g8AAqcAWT7nDwA5nj6Ce+cPAKJwcOO25w8AQ0J3jfDnDwCM8FOQKOgPADoXNfte6A8AZAiE3JPoDwC8zvBBx+gPAPZOfTj56A8AHZuHzCnpDwDqiNMJWekPAKKak/uG6Q8AZkhxrLPpDwDVtpQm3+kPAHzmq3MJ6g8ApGbxnDLqDwAslTKrWuoPABp01aaB6g8A8Bzel6fqDwAg2fOFzOoPADzmZXjw6g8AE+wvdhPrDwBKKv6FNesPALRiMa5W6w8A+oTi9HbrDwAUIOZflusPAHydz/S06w8A0En0uNLrDwA+Lm6x7+sPAOi9HuML7A8AFVqxUifsDwDTr50EQuwPAJbxKf1b7A8A9O5sQHXsDwC0DFDSjewPABIfkbal7A8A/ifE8LzsDwAV+1SE0+wPALPIiHTp7A8At5F/xP7sDwAohTV3E+0PAANJhI8n7Q8ATC8kEDvtDwBuWK37Te0PAN3DmFRg7Q8A6E9BHXLtDwCCqeRXg+0PAMgspAaU7Q8ABLeFK6TtDwC0anTIs+0PAFJmQd/C7Q8AUm6kcdHtDwDTijyB3+0PAICZkA/t7Q8AFNQPHvrtDwDESxKuBu4PAAZa2cAS7g8A4AaQVx7uDwAkZUtzKe4PALzkChU07g8APJu4PT7uDwD0ginuR+4PAIawHSdR7g8AQX9A6VnuDwAutCg1Yu4PAPGXWAtq7g8Aegc+bHHuDwCCezJYeO4PALoGe89+7g8AskpI0oTuDwBDY7Zgiu4PAFHIzHqP7g8A2iV+IJTuDwDqKahRmO4PAFxIEw6c7g8A9HNyVZ/uDwCuzGInou4PAKxCa4Ok7g8AcS38aKbuDwD61m7Xp+4PAAr6BM6o7g8AOzPoS6nuDwAQZClQqe4PAF4HwNmo7g8AVHaJ56fuDwAkHUh4pu4PAIOeooqk7g8A2uQiHaLuDwAkIDUun+4PAC6vJryb7g8A5PIkxZfuDwA6CjxHk+4PABZ1VUCO7g8Aepw2rojuDwD9PX+Ogu4PAIi4p9577g8A/zf/m3TuDwBevanDbO4PAH4AnlJk7g8AiCijRVvuDwC2V06ZUe4PAM8GAEpH7g8AUCzhUzzuDwDYKuCyMO4PAAWCrWIk7g8AWjy4XhfuDwBHFCqiCe4PAMxJ4yf77Q8AbCF26uvtDwB+BCLk2+0PANM5zg7L7Q8A9CwEZLntDwDJOOncpu0PAI3pN3KT7Q8ANqg4HH/tDwArwLnSae0PAACuBo1T7Q8AIqTeQTztDwDYL2rnI+0PAETmL3MK7Q8ANP4H2u/sDwC4tw4Q1OwPALRulQi37A8AwTAStpjsDwB4qQ0KeewPAP4xD/VX7A8AYsmGZjXsDwA1s7RMEewPANBvjpTr6w8AkragKcTrDwDcDO71musPAEKFyeFv6w8Anh+t00LrDwBLLQuwE+sPAOkCGlni6g8AVyKZrq7qDwAm446NeOoPAOVz/c8/6g8A9tmNTATqDwA7Vi/WxekPAKRHqTuE6Q8AKEcdRz/pDwDWxXa99ugPAOboxF2q6A8A6rF64FnoDwBAqZD2BOgPAMAzgkir5w8ApWofdUznDwACoioQ6OYPANirtqB95g8AfjA4nwzmDwBC9zhzlOUPAIByl3AU5Q8AWPQ21IvkDwA3Hv2/+eMPAJyx7jVd4w8A/uQvErXiDwBXVZkDAOIPABSDeII84Q8AsGfuxGjgDwCqcSuwgt8PAKr+fsWH3g8A/TvGCXXdDwATvynlRtwPAIICLvj42g8Adbqy4YXZDwAEz0jv5tcPAAtlva0T1g8AEvDiSQHUDwCsx7SnodEPAJ4fdgTizg8AshFe2KjLDwAiLc1u0scPAO0iHi8rww8AOrjAgWW9DwA0VADEBrYPAHQoKlhArA8AmEUBHpeeDwD8HaRI+okPACww8PfFZg8AShwzS1oaDwA=")
const _NP_WI_DOUBLE = _numpy_decode_f64_table("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")
const _NP_FI_DOUBLE = _numpy_decode_f64_table("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")
const _NP_KE_DOUBLE = _numpy_decode_u64_table("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")
const _NP_WE_DOUBLE = _numpy_decode_f64_table("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")
const _NP_FE_DOUBLE = _numpy_decode_f64_table("AAAAAAAA8D83EYjlRQXuP/H/gVCm0Ow/J3vrewDl6z8qf+YODyHrP+f6YqW6duo/m21VFZfe6T85qlXEMVTpPy/S03aj1Og/uMUGeOhd6D8mMSQtiu7nP37UCZtuhec/Y0upW7sh5z/GGIRJw8LmPwZcT236Z+Y/Zq+nwe0Q5j91rExpPb3lP3OH2oKYbOU/mol4Fboe5T+v+FHBZtPkP2ngjvtqiuQ/JeGor5lD5D+Ai7Ery/7jPxTR4UTcu+M/2d0Ip6164z8YYw5FIzvjP17aReMj/eI/JE8ftpjA4j+9MhERbYXiP6NQjCKOS+I/yD6BuuoS4j+Je4cZc9vhPyU7HscYpeE/7m/Obc5v4T+cFjO8hzvhP43DHEo5COE/Kx4rgdjV4D8q0FSIW6TgP3077jG5c+A/SGXS6+hD4D8k82Cx4hTgP3ZFIf49zd8/+sW/ji1y3z9NQuvRhhjfP5Cdlks9wN4/UdN9NkVp3j/8N+F1kxPePwwhp4gdv90/eu25fdlr3T8LGn7pvRndP5LgQNzByNw/YPuD2dx43D+DpQ7QBircP7XurhI43Ns/iAuZUWmP2z9vgFSUk0PbP1/vKDSw+No/5fb91riu2j9AAaNqp2XaP/QhdSB2Hdo/kjdaaR/W2T+oewnynY/ZPxCBmp/sSdk/BF1UjAYF2T85XbcE58DYP4w/vISJfdg/OGFEtek62D9ZzrZpA/nXPx6Axp3St9c/43Jec1N31z/qjbAwgjfXP52eZD5b+NY/nOnkJdu51j+fDcaP/nvWP+QnSELCPtY/dljvHyMC1j9s7jEmHsbVP++pOmywitU/56O9IddP1T/1id6NjxXVPx35Jg7X29Q/09qLFaui1D/vvoArCWrUP+JBGOvuMdQ/TqEwAlr60z+FsqswSMPTP+99sUe3jNM/3dD8KKVW0z81JDHGDyHTP3BCOSD169I/YiKuRlO30j8pdkVXKIPSP/12R31yT9I//34L8S8c0j/bCXv3XunRP1q8muH9ttE/ghkZDAuF0T/vkeLehFPRP7qfusxpItE/bKbZUrjx0D8zU4/4bsHQPxM+6U6MkdA/0pBd8A5i0D8sfHmA9TLQP2pHk6s+BNA/VJP/TNKrzz9+PpZc50/PP5vg6A+69M4/8kBZAEiazj+ngy/WjkDOPzlPIkiM580/uO7jGj6PzT/9MbQgojfNP5/Q9ji24Mw/AhjOT3iKzD/ur7ld5jTMPzVEOWf+38s/peRyfL6Lyz8+79y4JDjLPwtb60Iv5co/STzAS9ySyj+8XN8OKkHKPxLF5NEW8Mk/IxY+5KCfyT+hkuaexk/JP3m7JWSGAMk/1WJQn96xyD/5GozEzWPIP+bnlFBSFsg/rhuFyGrJxz/+Rp+5FX3HPzkoGrlRMcc/6oTuYx3mxj8o2qZed5vGP6zRMFVeUcY/MWqw+tAHxj+2wlQJzr7FP/V4LkJUdsU/SYwHbWIuxT/6tjxY9+bEP5YwmNgRoMQ/xswtybBZxD+aajgL0xPEPwWp+IV3zsM/ydWUJp2Jwz+vDPrfQkXDP259vqpnAcM/NM8EhQq+wj9AmWByKnvCP3jou3vGOMI/Zco9r932wT9m1jEgb7XBP3iu8OZ5dME/L3HJIP0zwT8gF+zv9/PAPy+2VHtptMA/vqW37lB1wD8Ef256rTbAP43qy6b88L8/FAQZZoV1vz88w4Ou8/q+P8y5jgRGgb4/+7ph9XoIvj+Yk60WkZC9P9dNkQaHGb0/V/2Aa1ujvD+vEC70DC68P48mcVeaubs/SGU1VAJGuz9lVGWxQ9O6P7c42T1dYbo/KPRG0E3wuT9wazNHFIC5P7l05YivELk/O1Nagx6iuD+6xDssYDS4P/Om14Bzx7c/HjwZhldbtz+2FoRIC/C2PyC2MNyNhbY/997KXN4btj8+u5Ht+7K1PzbQWbnlSrU/KdmQ8prjtD9cmEPTGn20Pw6xJZ1kF7Q/np+bmXeysz8Y58YZU06zP9GNlHb26rI/cAXOEGGIsj+MnSxRkiayP0Cjb6iJxbE/klN1j0ZlsT9QylaHyAWxPzsbhxkPp7A/F8j11xlJsD92lmm60NevPzToRJn0Hq8/5bIupZ5nrj8QWDFJzrGtP0p5HgOD/aw/6SEHZLxKrD+F2b4QepmrP4SAasK76ao/OPEbR4E7qj9MfHuCyo6pP213gG6X46g/azk6HOg5qD+eCKu0vJGnP1KvtnkV66Y/QaAmx/JFpj/K0sUTVaKlP+vFlvI8AKU/GWsmFKtfpD//GP9HoMCjP64UP34dI6M/DMBWySOHoj/UEvNftOyhP6GzGZ/QU6E/UdZ8DHq8oD/u+g1ZsiagP5CYr8f2JJ8/aHRReq7/nT8MGzNUkN2cP3BY+lChvps/m06S5uaimj9IKhMPZ4qZP2eZ7FModZg/lvyH2jFjlz93QKJyi1SWP1ECq6Y9SZU/vvCHzlFBlD+EXTEl0jyTPzI6ueHJO5I/X19yVEU+kT/wAh4JUkSQP87Hid79m44/VyduFLm2jD8tyUJV+tiKP72nj2jqAok/9XSq5rY0hz/LFuQLk26FP2JvUcG4sIM/cXaz7Wn7gT/5118p8k6AP8VddPpRV30/NkiX1Okjej8gNuw3nwR3P/0i486X+nM/Q0BXaT0HcT8RS82Bs1hsP//+ofOI2GY/JKPhqGuUYT8lPgxUtStZP7n8jfcKsk8/SwufMhzDPT8=")
