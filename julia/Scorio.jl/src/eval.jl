"""
    bayes(R::AbstractMatrix{<:Integer}, w::AbstractVector{<:Real}, R0::Union{AbstractMatrix{<:Integer}, Nothing}=nothing) -> Tuple{Float64, Float64}

Performance evaluation using the Bayes@N framework.

# Arguments
- `R::AbstractMatrix{Int}`: M×N int matrix with entries in {0,…,C}. Row α are the N outcomes for system α.
- `w::AbstractVector{<:Real}`: length-(C+1) weight vector (w0,…,wC) that maps category k to score wk.
- `R0::Union{AbstractMatrix{Int}, Nothing}`: optional M×D int matrix supplying D prior outcomes per row. If omitted, D=0.

# Returns
- `(mu, sigma)`: performance metric estimate and its uncertainty.

# Notation
δ_{a,b} is the Kronecker delta. For each row α and class k∈{0,…,C}:
- n_{αk}  = Σ_{i=1..N} δ_{k, R_{αi}}                    (counts in R)
- n^0_{αk} = 1 + Σ_{i=1..D} δ_{k, R^0_{αi}}             (Dirichlet(+1) prior)
- ν_{αk}   = n_{αk} + n^0_{αk}

T = 1 + C + D + N  (effective sample size; scalar)

# Estimates
μ = w0 + (1/(M·T)) · Σ_{α=1..M} Σ_{j=0..C} ν_{αj} (w_j − w0)

σ = sqrt{ (1/(M^2·(T+1))) · Σ_{α=1..M} [
          Σ_{j} (ν_{αj}/T) (w_j − w0)^2
          − ( Σ_{j} (ν_{αj}/T) (w_j − w0) )^2 ] }

# Examples
```julia
R = [0 1 2 2 1;
     1 1 0 2 2]
w = [0.0, 0.5, 1.0]
R0 = [0 2;
      1 2]

# With prior (D=2 → T=10)
mu, sigma = bayes(R, w, R0)
# Expected: mu ≈ 0.575, sigma ≈ 0.084275

# Without prior (D=0 → T=8)
mu2, sigma2 = bayes(R, w)
# Expected: mu2 ≈ 0.5625, sigma2 ≈ 0.091998
```
"""
function bayes(
    R::AbstractMatrix{<:Integer},
    w::AbstractVector{<:Real},
    R0::Union{AbstractMatrix{<:Integer}, Nothing}=nothing
)::Tuple{Float64, Float64}
    
    M, N = size(R)
    C = length(w) - 1
    
    # Handle R0 (prior outcomes)
    if isnothing(R0)
        D = 0
        R0m = zeros(Int, M, 0)
    else
        R0m = R0
        if size(R0m, 1) != M
            error("R0 must have the same number of rows (M) as R.")
        end
        D = size(R0m, 2)
    end
    
    # Validate value ranges
    if !isempty(R) && (minimum(R) < 0 || maximum(R) > C)
        error("Entries of R must be integers in [0, C].")
    end
    if !isempty(R0m) && (minimum(R0m) < 0 || maximum(R0m) > C)
        error("Entries of R0 must be integers in [0, C].")
    end
    
    T = 1 + C + D + N
    
    # Helper function to count occurrences of 0..C in each row
    function row_bincount(A::AbstractMatrix{<:Integer}, num_classes::Int)
        M_local, N_local = size(A)
        if N_local == 0
            return zeros(Int, M_local, num_classes)
        end
        out = zeros(Int, M_local, num_classes)
        for i in 1:M_local
            for j in 1:N_local
                out[i, A[i,j]+1] += 1  # Julia is 1-indexed
            end
        end
        return out
    end
    
    # n_{αk} and n^0_{αk}
    n_counts = row_bincount(R, C + 1)
    n0_counts = row_bincount(R0m, C + 1) .+ 1  # add 1 to every class (Dirichlet prior)
    
    # ν_{αk} = n_{αk} + n^0_{αk}
    nu = n_counts .+ n0_counts  # shape: (M, C+1)
    
    # μ = w0 + (1/(M T)) * Σ_α Σ_j ν_{αj} (w_j - w0)
    delta_w = w .- w[1]
    mu = w[1] + sum(nu * delta_w) / (M * T)
    
    # σ = [ (1/(M^2 (T+1))) * Σ_α { Σ_j (ν_{αj}/T)(w_j-w0)^2
    #       - ( Σ_j (ν_{αj}/T)(w_j-w0) )^2 } ]^{1/2}
    nu_over_T = nu ./ T
    termA = sum(nu_over_T .* (delta_w' .^ 2), dims=2)
    termB = (nu_over_T * delta_w) .^ 2
    sigma = sqrt(sum(termA .- termB) / (M^2 * (T + 1)))
    
    return Float64(mu), Float64(sigma)
end


"""
    avg(R::AbstractArray{<:Real}) -> Float64

Returns the naive mean of elements in R. For binary accuracy, encode incorrect=0, correct=1.

# Examples
```julia
R = [0 1 1 0; 1 1 1 1]
avg(R)  # Returns 0.75
```
"""
function avg(R::AbstractArray{<:Real})::Float64
    return Float64(sum(R) / length(R))
end


"""
    pass_at(args...; kwargs...)

Not yet implemented. Placeholder for future functionality.
"""
function pass_at(args...; kwargs...)
    error("Not yet implemented.")
end

export bayes, avg, pass_at
