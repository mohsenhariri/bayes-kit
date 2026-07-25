"""Shared validation and pairwise-count helpers for rank methods."""

_is_rank_sequence(value) = value isa AbstractVector || value isa Tuple

function _typed_nested_array(values, dimensions::Tuple)
    element_type = isempty(values) ? Float64 : foldl(promote_type, typeof.(values))
    return Array{element_type}(undef, dimensions)
end

function _coerce_rank_array_like(R)
    if R isa AbstractArray &&
       (ndims(R) != 1 || eltype(R) <: Number || eltype(R) <: Bool)
        return Array(R)
    end
    _is_rank_sequence(R) || return nothing

    outer = collect(R)
    isempty(outer) && return collect(outer)
    all(_is_rank_sequence, outer) || return collect(outer)
    rows = collect.(outer)
    row_length = length(rows[1])
    all(length(row) == row_length for row in rows) ||
        error("Input R must be a rectangular 2D or 3D array")

    if row_length == 0 || all(value -> !_is_rank_sequence(value), Iterators.flatten(rows))
        flat = collect(Iterators.flatten(rows))
        array = _typed_nested_array(flat, (length(rows), row_length))
        for row in eachindex(rows), column in 1:row_length
            array[row, column] = rows[row][column]
        end
        return array
    end

    all(value -> _is_rank_sequence(value), Iterators.flatten(rows)) ||
        error("Input R must be a rectangular 2D or 3D array")
    planes = [[collect(value) for value in row] for row in rows]
    depth = length(planes[1][1])
    all(length(value) == depth for plane in planes for value in plane) ||
        error("Input R must be a rectangular 2D or 3D array")
    flat = [value for plane in planes for row in plane for value in row]
    array = _typed_nested_array(flat, (length(planes), row_length, depth))
    for model in eachindex(planes), item in 1:row_length, trial in 1:depth
        array[model, item, trial] = planes[model][item][trial]
    end
    return array
end

function validate_input(R; binary_only::Bool=true)
    A = _coerce_rank_array_like(R)
    if isnothing(A)
        error(
            "Input R must be a 2D array of shape (L, M) or 3D array of shape (L, M, N), got shape ()",
        )
    end

    # Promote (L, M) to (L, M, 1)
    if ndims(A) == 2
        A = reshape(A, size(A, 1), size(A, 2), 1)
    elseif ndims(A) != 3
        error(
            "Input R must be a 2D array of shape (L, M) or 3D array of shape (L, M, N), got shape $(size(A))",
        )
    end

    if eltype(A) <: Bool
        A_int = Int.(A)
    else
        if !(eltype(A) <: Number)
            error("Input R must be numeric, got dtype $(eltype(A))")
        end

        if eltype(A) <: Complex
            error("Input R must contain real-valued outcomes")
        end

        if any(x -> !isfinite(x), A)
            error("Input R must not contain NaN or Inf values")
        end

        if eltype(A) <: AbstractFloat
            if any(x -> !(x == 0 || x == 1), A)
                error(
                    "Float inputs must be binary values (0.0 or 1.0). Use integer dtype for multiclass outcomes.",
                )
            end
        elseif binary_only
            if any(x -> !(x == 0 || x == 1), A)
                error("Input R must contain only binary values (0 or 1)")
            end
        end

        A_int = Int.(A)
    end

    L, M, N = size(A_int)
    if L < 2
        error("Need at least 2 models to rank, got L=$L")
    end
    if M < 1
        error("Need at least 1 question, got M=$M")
    end
    if N < 1
        error("Need at least 1 trial, got N=$N")
    end

    return A_int
end

function build_pairwise_wins(R)
    if ndims(R) != 3
        error("Input R must be 3D array of shape (L, M, N), got shape $(size(R))")
    end

    L, M, N = size(R)
    wins = zeros(Float64, L, L)

    @inbounds for i in 1:L
        for j in (i + 1):L
            i_wins = 0
            j_wins = 0
            for m in 1:M, n in 1:N
                ri = R[i, m, n]
                rj = R[j, m, n]
                if ri == 1 && rj == 0
                    i_wins += 1
                elseif rj == 1 && ri == 0
                    j_wins += 1
                end
            end

            wins[i, j] = i_wins
            wins[j, i] = j_wins
        end
    end

    return wins
end

function build_pairwise_counts(R)
    if ndims(R) != 3
        error("Input R must be 3D array of shape (L, M, N), got shape $(size(R))")
    end

    L, M, N = size(R)
    wins = zeros(Float64, L, L)
    ties = zeros(Float64, L, L)

    @inbounds for i in 1:L
        for j in (i + 1):L
            i_wins = 0
            j_wins = 0
            both_same = 0
            for m in 1:M, n in 1:N
                ri = R[i, m, n]
                rj = R[j, m, n]

                if ri == 1 && rj == 0
                    i_wins += 1
                elseif rj == 1 && ri == 0
                    j_wins += 1
                end

                if ri == rj
                    both_same += 1
                end
            end

            wins[i, j] = i_wins
            wins[j, i] = j_wins
            ties[i, j] = both_same
            ties[j, i] = both_same
        end
    end

    return wins, ties
end

function sigmoid(x)
    clipped = clamp.(x, -30.0, 30.0)
    return 1.0 ./ (1.0 .+ exp.(-clipped))
end

"""Return whether every vertex is reachable in both graph directions."""
function is_strongly_connected(adjacency)::Bool
    graph = Bool.(Array(adjacency))
    if ndims(graph) != 2 || size(graph, 1) != size(graph, 2)
        error("adjacency must be a square matrix")
    end

    n_vertices = size(graph, 1)
    n_vertices <= 1 && return true

    function reachable(edges)::BitVector
        seen = falses(n_vertices)
        stack = Int[1]
        seen[1] = true
        while !isempty(stack)
            vertex = pop!(stack)
            for neighbour in 1:n_vertices
                if edges[vertex, neighbour] && !seen[neighbour]
                    seen[neighbour] = true
                    push!(stack, neighbour)
                end
            end
        end
        return seen
    end

    return all(reachable(graph)) && all(reachable(transpose(graph)))
end

"""Average scores for models with identical sufficient-statistic rows."""
function average_equivalent_scores(scores, sufficient_statistics)::Vector{Float64}
    values = Float64.(collect(scores))
    if ndims(values) != 1
        error("scores must be a one-dimensional array")
    end

    statistics = Array(sufficient_statistics)
    if ndims(statistics) < 1 || size(statistics, 1) != length(values)
        error("sufficient_statistics must have one row for every score")
    end

    rows = reshape(statistics, length(values), :)
    result = copy(values)
    groups = Dict{Any,Vector{Int}}()
    for model in eachindex(values)
        key = Tuple(@view rows[model, :])
        push!(get!(groups, key, Int[]), model)
    end
    for members in Base.values(groups)
        if length(members) > 1
            result[members] .= sum(values[members]) / length(members)
        end
    end
    return result
end

function _projection_matches(
    data::AbstractMatrix,
    source_rows::Vector{Int},
    target_rows::Vector{Int},
)::Bool
    n_observations = size(data, 2)
    source_columns = collect(1:n_observations)
    target_columns = collect(1:n_observations)

    # NumPy's lexsort(source[::-1, :]) makes the first selected model row the
    # primary key, followed by later rows.  Julia's stable sort on tuple keys
    # expresses the same column-multiset comparison directly.
    sort!(source_columns; by=column -> Tuple(data[row, column] for row in source_rows))
    sort!(target_columns; by=column -> Tuple(data[row, column] for row in target_rows))

    for row_position in eachindex(source_rows), column_position in 1:n_observations
        if data[source_rows[row_position], source_columns[column_position]] !=
           data[target_rows[row_position], target_columns[column_position]]
            return false
        end
    end
    return true
end

"""Average exact score orbits under simultaneous observation permutations."""
function average_event_exchangeable_scores(scores, observations)::Vector{Float64}
    values = Float64.(collect(scores))
    data_array = Array(observations)
    if ndims(values) != 1
        error("scores must be a one-dimensional array")
    end
    if ndims(data_array) < 2 || size(data_array, 1) != length(values)
        error("observations must have one row for every score")
    end
    data = reshape(data_array, length(values), :)
    n_models = length(values)

    row_signatures = [Tuple(sort(collect(@view data[model, :]))) for model in 1:n_models]
    parent = collect(1:n_models)

    function find_root(index::Int)::Int
        while parent[index] != index
            parent[index] = parent[parent[index]]
            index = parent[index]
        end
        return index
    end

    function union_roots(first::Int, second::Int)
        first_root = find_root(first)
        second_root = find_root(second)
        if first_root != second_root
            parent[second_root] = first_root
        end
        return nothing
    end

    signature_sizes = Dict{Any,Int}()
    for signature in row_signatures
        signature_sizes[signature] = get(signature_sizes, signature, 0) + 1
    end

    function find_automorphism(source::Int, target::Int)
        row_signatures[source] == row_signatures[target] || return nothing

        source_rows = Int[source]
        target_rows = Int[target]
        used_source = falses(n_models)
        used_target = falses(n_models)
        used_source[source] = true
        used_target[target] = true
        mapping = fill(0, n_models)
        mapping[source] = target
        source_order = sort(
            [index for index in 1:n_models if index != source];
            by=index -> signature_sizes[row_signatures[index]],
            alg=Base.Sort.MergeSort,
        )

        _projection_matches(data, source_rows, target_rows) || return nothing

        function search()::Bool
            length(source_rows) == n_models && return true
            next_source = first(index for index in source_order if !used_source[index])
            candidate_source_rows = [source_rows; next_source]
            compatible = Int[]
            for candidate_target in 1:n_models
                if used_target[candidate_target] ||
                   row_signatures[next_source] != row_signatures[candidate_target]
                    continue
                end
                if _projection_matches(
                    data,
                    candidate_source_rows,
                    [target_rows; candidate_target],
                )
                    push!(compatible, candidate_target)
                end
            end
            isempty(compatible) && return false

            used_source[next_source] = true
            push!(source_rows, next_source)
            for candidate_target in compatible
                used_target[candidate_target] = true
                push!(target_rows, candidate_target)
                mapping[next_source] = candidate_target
                search() && return true
                mapping[next_source] = 0
                pop!(target_rows)
                used_target[candidate_target] = false
            end
            pop!(source_rows)
            used_source[next_source] = false
            return false
        end

        return search() ? mapping : nothing
    end

    for first_model in 1:n_models
        for second_model in (first_model + 1):n_models
            find_root(first_model) == find_root(second_model) && continue
            automorphism = find_automorphism(first_model, second_model)
            isnothing(automorphism) && continue
            for (source, target) in enumerate(automorphism)
                union_roots(source, target)
            end
        end
    end

    groups = [find_root(index) for index in 1:n_models]
    return average_equivalent_scores(values, reshape(groups, :, 1))
end
