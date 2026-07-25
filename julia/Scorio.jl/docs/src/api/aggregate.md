# Aggregation (`Scorio.Aggregate`)

Test-time scaling utilities live under `Scorio.Aggregate` (also available as
`Scorio.Agg` and `Scorio.agg`). Candidate pools are vectors for one question or
`M × N` matrices for a batch. To match Python exactly, returned indices are
0-based and `-1` is the no-valid-candidate sentinel.

```@docs
Scorio.Aggregate
```

## Confidence signals

```@docs
Scorio.Aggregate.mean_logprob
Scorio.Aggregate.sequence_logprob
Scorio.Aggregate.perplexity
Scorio.Aggregate.picsar
Scorio.Aggregate.self_certainty
Scorio.Aggregate.token_entropy
Scorio.Aggregate.varentropy
Scorio.Aggregate.max_softmax_probability
Scorio.Aggregate.logprob_margin
Scorio.Aggregate.token_confidence
Scorio.Aggregate.deepconf_confidence
```

## Process rewards

```@docs
Scorio.Aggregate.prm_aggregate
```

## Selection and voting

`majority_of_the_bests` is also available under the short method alias `mob`.

```@docs
Scorio.Aggregate.best_of_n
Scorio.Aggregate.majority_of_the_bests
Scorio.Aggregate.best_of_majority
Scorio.Aggregate.majority_vote
Scorio.Aggregate.weighted_majority_vote
Scorio.Aggregate.softmax_weighted_vote
Scorio.Aggregate.rank_weighted_vote
Scorio.Aggregate.logit_weighted_vote
Scorio.Aggregate.filtered_vote
Scorio.Aggregate.KDEVoteCalibration
Scorio.Aggregate.fit_kde_vote_calibration
Scorio.Aggregate.kde_weighted_vote
Scorio.Aggregate.cges_vote
Scorio.Aggregate.cges_stop
```

## Online stopping

```@docs
Scorio.Aggregate.adaptive_consistency_stop
Scorio.Aggregate.adaptive_consistency_dirichlet_stop
Scorio.Aggregate.adaptive_consistency_crp_stop
Scorio.Aggregate.esc_stop
Scorio.Aggregate.deepconf_stop_threshold
Scorio.Aggregate.deepconf_online_stop
```
