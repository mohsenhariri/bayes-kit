"""
    competition_ranks_from_scores(scores_in_id_order::AbstractVector{<:Real}; tol::Real=1e-12) -> Vector{Int}

Compute competition ranks from scores.

Given L models with ids 1..L and their scores, returns competition ranks (1,2,3,3,5,...).
Models with tied scores (within tolerance) receive the same rank.

# Arguments
- `scores_in_id_order::AbstractVector{<:Real}`: list/array of scores aligned to ids 1..L
- `tol::Real=1e-12`: tolerance for considering scores as tied

# Returns
- `Vector{Int}`: competition ranks for each model

# Examples
```julia
scores = [0.95, 0.87, 0.87, 0.72, 0.65]
ranks = competition_ranks_from_scores(scores)
# Returns: [1, 2, 2, 4, 5]
```
"""
function competition_ranks_from_scores(
    scores_in_id_order::AbstractVector{<:Real};
    tol::Real=1e-12
)::Vector{Int}
    
    scores = Float64.(scores_in_id_order)
    n = length(scores)
    
    # Get indices that would sort scores in descending order
    order = sortperm(scores, rev=true)
    
    ranks = zeros(Int, n)
    rank = 1
    i = 1
    
    while i <= n
        # Find tie block
        j = i
        while j + 1 <= n && abs(scores[order[j + 1]] - scores[order[i]]) <= tol
            j += 1
        end
        
        # Assign same rank to the whole block
        for t in i:j
            ranks[order[t]] = rank
        end
        
        # Next rank skips by block size
        rank += j - i + 1
        i = j + 1
    end
    
    return ranks
end

export competition_ranks_from_scores
