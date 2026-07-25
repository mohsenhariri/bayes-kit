"""Answer aggregation and selection for test-time scaling.

`Scorio.Aggregate` is the Julia port of Python's `scorio.aggregate` package.
It contains per-trace confidence signals, process-reward reductions, fixed-pool
selection and voting rules, and online stopping rules.

Candidate pools use `N`-element vectors for one question or `M × N` matrices
for batches. Returned candidate and token indices deliberately match Python's
0-based convention; `-1` denotes that no valid candidate exists.
"""
module Aggregate

using ..Scorio: _NumpyRNG,
    _numpy_log,
    _numpy_multinomial!,
    _numpy_standard_gamma!,
    _numpy_uniform!

include("aggregate/base.jl")
include("aggregate/confidence.jl")
include("aggregate/prm.jl")
include("aggregate/best_of.jl")
include("aggregate/vote.jl")
include("aggregate/calibration.jl")
include("aggregate/cges.jl")
include("aggregate/online.jl")

export mean_logprob,
    sequence_logprob,
    perplexity,
    self_certainty,
    token_confidence,
    deepconf_confidence,
    token_entropy,
    varentropy,
    max_softmax_probability,
    logprob_margin,
    picsar,
    prm_aggregate,
    best_of_n,
    majority_of_the_bests,
    mob,
    best_of_majority,
    majority_vote,
    weighted_majority_vote,
    softmax_weighted_vote,
    rank_weighted_vote,
    logit_weighted_vote,
    filtered_vote,
    KDEVoteCalibration,
    fit_kde_vote_calibration,
    kde_weighted_vote,
    CGES_OTHER,
    cges_vote,
    cges_stop,
    adaptive_consistency_stop,
    adaptive_consistency_dirichlet_stop,
    adaptive_consistency_crp_stop,
    esc_stop,
    deepconf_stop_threshold,
    deepconf_online_stop

end # module Aggregate
