"""Scorio Julia package."""
module Scorio

const VERSION = v"0.2.2"

include("numpy_rng.jl")
include("eval.jl")
include("rank.jl")
include("sinf.jl")
include("aggregate.jl")
include("utils.jl")

# Short aliases matching Python/JavaScript (`agg`) and Julia module casing.
const Agg = Aggregate
const agg = Aggregate

module Eval
import ..Scorio
using ..Scorio: bayes_ci,
    avg_ci,
    pass_at_k_ci,
    pass_hat_k_ci,
    g_pass_at_k,
    g_pass_at_k_ci,
    g_pass_at_k_tau_ci,
    mg_pass_at_k_ci,
    unanimous_at_k_ci,
    auc_at_k,
    auc_at_k_ci,
    maj_at_k,
    maj_at_k_ci,
    max_at_k,
    max_at_k_ci,
    threshold_spectrum_at_k,
    threshold_spectrum_at_k_ci,
    geom_at_k,
    geom_at_k_ci,
    geom_ds_at_k,
    geom_ds_at_k_ci,
    geo_spectrum_at_k,
    geo_spectrum_at_k_ci,
    geo_spectrum_star_at_k,
    geo_spectrum_star_at_k_ci

# These names are also overloaded by the ranking family at the package root.
# Eval-local wrappers force the scalar 1D/2D implementations so a rank tensor
# receives the same dimensionality error as Python's `scorio.eval` API.
avg(R, w) = invoke(Scorio.avg, Tuple{Any, Any}, R, w)
avg(R; w=nothing) = avg(R, w)

bayes(R, w, R0) =
    invoke(Scorio.bayes, Tuple{Any, Any, Any}, R, w, R0)
bayes(R; w=nothing, R0=nothing) = bayes(R, w, R0)
bayes(R, w; R0=nothing) = bayes(R, w, R0)
pass_at_k(R, k::Integer) = invoke(Scorio.pass_at_k, Tuple{Any, Integer}, R, k)
pass_hat_k(R, k::Integer) = invoke(Scorio.pass_hat_k, Tuple{Any, Integer}, R, k)
g_pass_at_k_tau(R, k::Integer, tau::Real) =
    invoke(Scorio.g_pass_at_k_tau, Tuple{Any, Integer, Real}, R, k, tau)
mg_pass_at_k(R, k::Integer) = invoke(Scorio.mg_pass_at_k, Tuple{Any, Integer}, R, k)
const unanimous_at_k = pass_hat_k

export bayes,
    bayes_ci,
    avg,
    avg_ci,
    pass_at_k,
    pass_at_k_ci,
    pass_hat_k,
    pass_hat_k_ci,
    g_pass_at_k,
    g_pass_at_k_ci,
    g_pass_at_k_tau,
    g_pass_at_k_tau_ci,
    mg_pass_at_k,
    mg_pass_at_k_ci,
    unanimous_at_k,
    unanimous_at_k_ci,
    auc_at_k,
    auc_at_k_ci,
    maj_at_k,
    maj_at_k_ci,
    max_at_k,
    max_at_k_ci,
    threshold_spectrum_at_k,
    threshold_spectrum_at_k_ci,
    geom_at_k,
    geom_at_k_ci,
    geom_ds_at_k,
    geom_ds_at_k_ci,
    geo_spectrum_at_k,
    geo_spectrum_at_k_ci,
    geo_spectrum_star_at_k,
    geo_spectrum_star_at_k_ci
end

module Rank
import ..Scorio
using ..Scorio: Prior,
    GaussianPrior,
    LaplacePrior,
    CauchyPrior,
    UniformPrior,
    CustomPrior,
    EmpiricalPrior

include("rank/public_api.jl")

export Prior,
    GaussianPrior,
    LaplacePrior,
    CauchyPrior,
    UniformPrior,
    CustomPrior,
    EmpiricalPrior,
    avg,
    bayes,
    pass_at_k,
    pass_hat_k,
    g_pass_at_k_tau,
    mg_pass_at_k,
    inverse_difficulty,
    elo,
    glicko,
    trueskill,
    bradley_terry,
    bradley_terry_map,
    bradley_terry_davidson,
    bradley_terry_davidson_map,
    rao_kupper,
    rao_kupper_map,
    thompson,
    bayesian_mcmc,
    borda,
    copeland,
    win_rate,
    minimax,
    schulze,
    ranked_pairs,
    kemeny_young,
    nanson,
    baldwin,
    majority_judgment,
    rasch,
    rasch_map,
    rasch_2pl,
    rasch_2pl_map,
    rasch_3pl,
    rasch_3pl_map,
    rasch_mml,
    rasch_mml_credible,
    dynamic_irt,
    mirt,
    pagerank,
    spectral,
    alpharank,
    nash,
    rank_centrality,
    serial_rank,
    hodge_rank,
    plackett_luce,
    plackett_luce_map,
    davidson_luce,
    davidson_luce_map,
    bradley_terry_luce,
    bradley_terry_luce_map
end

module SInf
using ..Scorio: ranking_confidence,
    ci_from_mu_sigma,
    should_stop,
    should_stop_top1,
    suggest_next_allocation,
    confseq_mean,
    confseq_mean_path,
    fixed_ci_path,
    score_confseq,
    score_confseq_path,
    precision_stop,
    trial_scores,
    question_scores,
    paired_trial_diffs,
    stream_from_tensor,
    compare_paired,
    compare_paired_path,
    decide_better,
    pairwise_confidence,
    empirical_scores,
    should_stop_top1_av,
    should_stop_full_ranking,
    suggest_next_allocation_stratified,
    select_best_fixed_budget,
    should_stop_sampling,
    adaptive_consistency_stop,
    counts_from_answers

export ranking_confidence,
    ci_from_mu_sigma,
    should_stop,
    should_stop_top1,
    suggest_next_allocation,
    confseq_mean,
    confseq_mean_path,
    fixed_ci_path,
    score_confseq,
    score_confseq_path,
    precision_stop,
    trial_scores,
    question_scores,
    paired_trial_diffs,
    stream_from_tensor,
    compare_paired,
    compare_paired_path,
    decide_better,
    pairwise_confidence,
    empirical_scores,
    should_stop_top1_av,
    should_stop_full_ranking,
    suggest_next_allocation_stratified,
    select_best_fixed_budget,
    should_stop_sampling,
    adaptive_consistency_stop,
    counts_from_answers
end

module Utils
using ..Scorio: competition_ranks_from_scores,
    rank_scores,
    compare_rankings,
    lehmer_hash,
    lehmer_unhash,
    ranking_hash,
    unhash_ranking

export competition_ranks_from_scores,
    rank_scores,
    compare_rankings,
    lehmer_hash,
    lehmer_unhash,
    ranking_hash,
    unhash_ranking
end

export bayes,
    bayes_ci,
    avg,
    avg_ci,
    pass_at_k,
    pass_at_k_ci,
    pass_hat_k,
    pass_hat_k_ci,
    g_pass_at_k,
    g_pass_at_k_ci,
    g_pass_at_k_tau,
    g_pass_at_k_tau_ci,
    mg_pass_at_k,
    mg_pass_at_k_ci,
    unanimous_at_k,
    unanimous_at_k_ci,
    auc_at_k,
    auc_at_k_ci,
    maj_at_k,
    maj_at_k_ci,
    max_at_k,
    max_at_k_ci,
    threshold_spectrum_at_k,
    threshold_spectrum_at_k_ci,
    geom_at_k,
    geom_at_k_ci,
    geom_ds_at_k,
    geom_ds_at_k_ci,
    geo_spectrum_at_k,
    geo_spectrum_at_k_ci,
    geo_spectrum_star_at_k,
    geo_spectrum_star_at_k_ci

export ranking_confidence,
    ci_from_mu_sigma,
    should_stop,
    should_stop_top1,
    suggest_next_allocation,
    confseq_mean,
    confseq_mean_path,
    fixed_ci_path,
    score_confseq,
    score_confseq_path,
    precision_stop,
    trial_scores,
    question_scores,
    paired_trial_diffs,
    stream_from_tensor,
    compare_paired,
    compare_paired_path,
    decide_better,
    pairwise_confidence,
    empirical_scores,
    should_stop_top1_av,
    should_stop_full_ranking,
    suggest_next_allocation_stratified,
    select_best_fixed_budget,
    should_stop_sampling,
    adaptive_consistency_stop,
    counts_from_answers

export competition_ranks_from_scores,
    rank_scores,
    compare_rankings,
    lehmer_hash,
    lehmer_unhash,
    ranking_hash,
    unhash_ranking

export Eval, Rank, SInf, Aggregate, Agg, agg, Utils

end # module Scorio
