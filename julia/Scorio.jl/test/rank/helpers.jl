using Test

# ── Data fixture functions (derived from NPZ simulation data) ──

function ordered_binary_R()::Array{Int, 3}
    data = top_p_task_aime25()
    # Python: raw0 = data[0, :24, :10], raw1 = data[1, :24, :10], raw2 = data[2, :24, :10]
    raw0 = data[1, 1:24, 1:10]
    raw1 = data[2, 1:24, 1:10]
    raw2 = data[3, 1:24, 1:10]

    best = max.(raw0, raw1)
    mid_high = copy(raw0)
    mid_low = min.(mid_high, raw1)
    worst = min.(mid_low, raw2)

    R = zeros(Int, 4, 24, 10)
    R[1, :, :] .= best
    R[2, :, :] .= mid_high
    R[3, :, :] .= mid_low
    R[4, :, :] .= worst

    means = [mean(R[l, :, :]) for l in 1:4]
    for i in 1:3
        @assert means[i] >= means[i + 1]
    end
    return R
end

function mean(x)
    return sum(x) / length(x)
end

function ordered_binary_small_R()::Array{Int, 3}
    return ordered_binary_R()[:, 1:10, 1:5]
end

function ordered_binary_matrix()::Array{Int, 2}
    return copy(ordered_binary_small_R()[:, :, 1])
end

function tie_heavy_R()::Array{Int, 3}
    data = top_p_task_aime25()
    # Python: base = data[4, :6, :4]
    base = data[5, 1:6, 1:4]
    R = zeros(Int, 4, size(base, 1), size(base, 2))
    R[1, :, :] .= base
    R[2, :, :] .= base
    R[3, :, :] .= circshift(base, (0, 1))
    R[4, :, :] .= 1 .- base
    return R
end

function equal_information_R()::Array{Int, 3}
    data = top_p_task_aime25()
    # Python: base = data[5, :7, :5]
    base = data[6, 1:7, 1:5]
    R = zeros(Int, 4, size(base, 1), size(base, 2))
    for l in 1:4
        R[l, :, :] .= base
    end
    return R
end

function optimizer_parity_R()::Array{Int, 3}
    R = zeros(Int, 4, 4, 3)
    R[1, :, :] .= [1 0 1; 1 1 0; 0 1 0; 1 0 0]
    R[2, :, :] .= [0 1 0; 1 0 0; 0 0 1; 1 1 0]
    R[3, :, :] .= [1 1 0; 0 0 1; 1 0 1; 0 0 1]
    R[4, :, :] .= [0 0 1; 0 1 1; 1 1 0; 0 1 0]
    return R
end

function multiclass_rank_data()
    data = top_p_task_aime25()
    # Python: R = (data[0:3, :10, :7] + data[3:6, :10, :7]).astype(int)
    R = Int.(data[1:3, 1:10, 1:7] .+ data[4:6, 1:10, 1:7])
    w = [0.0, 0.5, 1.0]

    # Python: R0_shared = (data[6, :10, :3] + data[7, :10, :3]).astype(int)
    R0_shared = Int.(data[7, 1:10, 1:3] .+ data[8, 1:10, 1:3])

    # Python: R0_per_model = (data[8:11, :10, :3] + data[11:14, :10, :3]).astype(int)
    R0_per_model = Int.(data[9:11, 1:10, 1:3] .+ data[12:14, 1:10, 1:3])

    @assert all(0 .<= R .<= 2)
    @assert all(0 .<= R0_shared .<= 2)
    @assert all(0 .<= R0_per_model .<= 2)

    return R, w, R0_shared, R0_per_model
end

# ── Assertion helpers ──

function assert_ranking(ranking; expected_len=nothing)
    @test ranking isa AbstractVector
    values = Float64.(collect(ranking))

    if !isnothing(expected_len)
        @test length(values) == expected_len
    end

    L = length(values)
    @test L >= 2
    @test all(isfinite, values)
    @test all(values .>= 1.0)
    @test all(values .<= Float64(L))
    return nothing
end

function assert_scores(scores; expected_len::Integer)
    @test scores isa AbstractVector
    values = Float64.(collect(scores))
    @test length(values) == Int(expected_len)
    @test all(isfinite, values)
    return nothing
end

function assert_ranking_and_scores(out; expected_len=nothing)
    @test out isa Tuple
    @test length(out) >= 2

    ranking = out[1]
    scores = out[2]

    L = length(ranking)
    target_len = isnothing(expected_len) ? L : Int(expected_len)

    assert_ranking(ranking; expected_len=target_len)
    assert_scores(scores; expected_len=L)

    return ranking, scores
end

function assert_ordering_sanity(ranking; best_idx::Integer=1, worst_idx::Integer=length(ranking))
    values = Float64.(collect(ranking))
    @test values[Int(best_idx)] == minimum(values)
    @test values[Int(worst_idx)] == maximum(values)
    return nothing
end
