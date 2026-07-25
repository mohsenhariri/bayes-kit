using Test
using Scorio

const TEST_FAMILIES = Set(lowercase.(ARGS))
const VALID_TEST_FAMILIES = Set(["eval", "rank", "sinf", "aggregate", "utils"])
const RUN_ALL_TESTS = isempty(TEST_FAMILIES)

unknown_families = setdiff(TEST_FAMILIES, VALID_TEST_FAMILIES)
isempty(unknown_families) || error(
    "Unknown test family: $(join(sort!(collect(unknown_families)), ", ")). " *
    "Expected one of: $(join(sort!(collect(VALID_TEST_FAMILIES)), ", ")).",
)

run_family(name::AbstractString) = RUN_ALL_TESTS || name in TEST_FAMILIES

# Eval and rank share NPZ simulation fixtures. `Pkg.test` makes the declared
# test-only dependency available for both full and focused runs.
if run_family("eval") || run_family("rank")
    include("testdata.jl")
end

@testset "Scorio.jl" begin
    @test Scorio.VERSION == v"0.2.2"

    @test isdefined(Scorio, :Eval)
    @test isdefined(Scorio, :Rank)
    @test isdefined(Scorio, :SInf)
    @test isdefined(Scorio, :Aggregate)
    @test isdefined(Scorio, :Utils)

    @test isdefined(Scorio.Eval, :bayes)
    @test isdefined(Scorio.Eval, :pass_at_k)
    @test isdefined(Scorio.Rank, :avg)
    @test isdefined(Scorio.Rank, :bradley_terry)
    @test isdefined(Scorio.SInf, :should_stop)
    @test isdefined(Scorio.Aggregate, :majority_vote)
    @test isdefined(Scorio.Utils, :rank_scores)
end

run_family("eval") && include("eval/test_eval_apis.jl")
run_family("rank") && include("rank/runtests_rank.jl")
if run_family("sinf")
    include("sinf/test_sinf.jl")
    include("sinf/test_sinf_av.jl")
end
run_family("aggregate") && include("aggregate/test_aggregate.jl")
run_family("utils") && include("test_utils.jl")
