/**
 * Scorio ranking methods — TypeScript port of `scorio.rank`.
 *
 * Ranking estimators for binary (and, for `bayes`, categorical) response
 * tensors of shape `(L, M, N)` (`L` models, `M` questions, `N` trials). Every
 * method returns `{ ranking, scores }`: `ranking[l]` is model `l`'s rank
 * (1 = best) and `scores[l]` the raw method score (larger is better). The
 * `method` option selects the tie convention (`"competition"` by default).
 *
 * Each method is exported under an idiomatic camelCase name and a snake_case
 * alias matching the Python/Julia API.
 */

// ---------------------------------------------------------------------------
// Eval-metric based
// ---------------------------------------------------------------------------
import {
  avg,
  bayes,
  passAtK,
  passHatK,
  gPassAtKTau,
  mgPassAtK,
} from "./evalRanking.js";
import { inverseDifficulty } from "./pointwise.js";

// Pairwise rating systems
import { elo, trueskill, glicko } from "./pairwise.js";

// Bradley-Terry family
import {
  bradleyTerry,
  bradleyTerryMap,
  bradleyTerryDavidson,
  bradleyTerryDavidsonMap,
  raoKupper,
  raoKupperMap,
} from "./bradleyTerry.js";

// Bayesian
import { thompson, bayesianMcmc } from "./bayesian.js";

// Voting
import {
  borda,
  copeland,
  winRate,
  minimax,
  schulze,
  rankedPairs,
  kemenyYoung,
  nanson,
  baldwin,
  majorityJudgment,
} from "./voting.js";

// IRT
import {
  rasch,
  raschMap,
  rasch2pl,
  rasch2plMap,
  rasch3pl,
  rasch3plMap,
  raschMml,
  raschMmlCredible,
  dynamicIrt,
  mirt,
} from "./irt.js";

// Graph / seriation / hodge
import { pagerank, spectral, alpharank, nash } from "./graph.js";
import { rankCentrality } from "./rankCentrality.js";
import { serialRank } from "./serialRank.js";
import { hodgeRank } from "./hodgeRank.js";

// Listwise / Luce
import {
  plackettLuce,
  plackettLuceMap,
  davidsonLuce,
  davidsonLuceMap,
  bradleyTerryLuce,
  bradleyTerryLuceMap,
} from "./listwise.js";

// Priors
import {
  GaussianPrior,
  LaplacePrior,
  CauchyPrior,
  UniformPrior,
  CustomPrior,
  EmpiricalPrior,
} from "./priors.js";

export type { RankResult, RankMethod, BaseRankOptions } from "./internal/result.js";
export type { TensorInput, Tensor3 } from "./internal/tensor.js";
export type { Prior } from "./priors.js";
export {
  GaussianPrior,
  LaplacePrior,
  CauchyPrior,
  UniformPrior,
  CustomPrior,
  EmpiricalPrior,
};

// ---------------------------------------------------------------------------
// Primary camelCase API
// ---------------------------------------------------------------------------
export {
  avg,
  bayes,
  passAtK,
  passHatK,
  gPassAtKTau,
  mgPassAtK,
  inverseDifficulty,
  elo,
  trueskill,
  glicko,
  bradleyTerry,
  bradleyTerryMap,
  bradleyTerryDavidson,
  bradleyTerryDavidsonMap,
  raoKupper,
  raoKupperMap,
  thompson,
  bayesianMcmc,
  borda,
  copeland,
  winRate,
  minimax,
  schulze,
  rankedPairs,
  kemenyYoung,
  nanson,
  baldwin,
  majorityJudgment,
  rasch,
  raschMap,
  rasch2pl,
  rasch2plMap,
  rasch3pl,
  rasch3plMap,
  raschMml,
  raschMmlCredible,
  dynamicIrt,
  mirt,
  pagerank,
  spectral,
  alpharank,
  nash,
  rankCentrality,
  serialRank,
  hodgeRank,
  plackettLuce,
  plackettLuceMap,
  davidsonLuce,
  davidsonLuceMap,
  bradleyTerryLuce,
  bradleyTerryLuceMap,
};

// ---------------------------------------------------------------------------
// snake_case aliases (Python / Julia parity)
// ---------------------------------------------------------------------------
export {
  passAtK as pass_at_k,
  passHatK as pass_hat_k,
  gPassAtKTau as g_pass_at_k_tau,
  mgPassAtK as mg_pass_at_k,
  inverseDifficulty as inverse_difficulty,
  bradleyTerry as bradley_terry,
  bradleyTerryMap as bradley_terry_map,
  bradleyTerryDavidson as bradley_terry_davidson,
  bradleyTerryDavidsonMap as bradley_terry_davidson_map,
  raoKupper as rao_kupper,
  raoKupperMap as rao_kupper_map,
  bayesianMcmc as bayesian_mcmc,
  winRate as win_rate,
  rankedPairs as ranked_pairs,
  kemenyYoung as kemeny_young,
  majorityJudgment as majority_judgment,
  raschMap as rasch_map,
  rasch2pl as rasch_2pl,
  rasch2plMap as rasch_2pl_map,
  rasch3pl as rasch_3pl,
  rasch3plMap as rasch_3pl_map,
  raschMml as rasch_mml,
  raschMmlCredible as rasch_mml_credible,
  dynamicIrt as dynamic_irt,
  rankCentrality as rank_centrality,
  serialRank as serial_rank,
  hodgeRank as hodge_rank,
  plackettLuce as plackett_luce,
  plackettLuceMap as plackett_luce_map,
  davidsonLuce as davidson_luce,
  davidsonLuceMap as davidson_luce_map,
  bradleyTerryLuce as bradley_terry_luce,
  bradleyTerryLuceMap as bradley_terry_luce_map,
};
