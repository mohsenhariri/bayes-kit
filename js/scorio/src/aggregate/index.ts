/**
 * Test-time-scaling answer aggregation — TypeScript port of `scorio.aggregate`.
 *
 * Functions are available in idiomatic camelCase and as snake_case aliases.
 * Candidate-pool selection follows the Python return contract: a scalar for a
 * single question, an array for a batch, and optional representative index /
 * score fields in a fixed tuple order.
 */

import * as confidence from "./confidence.js";
import * as prm from "./prm.js";
import * as bestOf from "./bestOf.js";
import * as vote from "./vote.js";
import * as calibration from "./calibration.js";
import * as cges from "./cges.js";
import * as online from "./online.js";

export type {
  NumericInput,
  TopKLogprobs,
  TokenReducer,
  DeepconfMode,
  PicsarOptions,
  ReducerOptions,
  LogprobMarginOptions,
  DeepconfConfidenceOptions,
} from "./confidence.js";
export type { PrmAggregateMethod, PrmAggregateOptions } from "./prm.js";
export type {
  BestOfNOptions,
  MajorityOfTheBestsOptions,
  BestOfMajorityAggregate,
  BestOfMajorityOptions,
} from "./bestOf.js";
export type {
  MajorityVoteOptions,
  WeightedVoteAggregate,
  WeightedMajorityVoteOptions,
  SoftmaxWeightedVoteOptions,
  RankWeightedVoteOptions,
  LogitTransform,
  LogitWeightedVoteOptions,
  FilteredVoteOptions,
} from "./vote.js";
export type {
  KDEVoteCalibrationInit,
  KDEBandwidthSpecification,
  FitKDEVoteCalibrationOptions,
  KDEWeightedVoteOptions,
} from "./calibration.js";
export { KDEVoteCalibration } from "./calibration.js";
export type { CGESOther, CGESVoteOptions, CGESStopOptions } from "./cges.js";
export { CGES_OTHER } from "./cges.js";
export type {
  AdaptiveConsistencyStopOptions,
  AdaptiveConsistencyDirichletStopOptions,
  AdaptiveConsistencyCrpStopOptions,
} from "./online.js";
export type {
  AnswerInput,
  ScoreInput,
  Selection,
  SelectionIndex,
  SelectionScore,
  PackedSelection,
  SelectionReturnOptions,
  Keep,
} from "./internal/base.js";

// Submodule namespaces mirror `scorio.aggregate.confidence`, `.prm`, etc.
export { confidence, prm, bestOf, vote, calibration, cges, online };

// Primary camelCase API.
export const meanLogprob = confidence.meanLogprob;
export const sequenceLogprob = confidence.sequenceLogprob;
export const perplexity = confidence.perplexity;
export const selfCertainty = confidence.selfCertainty;
export const tokenConfidence = confidence.tokenConfidence;
export const deepconfConfidence = confidence.deepconfConfidence;
export const tokenEntropy = confidence.tokenEntropy;
export const varentropy = confidence.varentropy;
export const maxSoftmaxProbability = confidence.maxSoftmaxProbability;
export const logprobMargin = confidence.logprobMargin;
export const picsar = confidence.picsar;

export const prmAggregate = prm.prmAggregate;

export const bestOfN = bestOf.bestOfN;
export const majorityOfTheBests = bestOf.majorityOfTheBests;
export const mob = bestOf.mob;
export const bestOfMajority = bestOf.bestOfMajority;

export const majorityVote = vote.majorityVote;
export const weightedMajorityVote = vote.weightedMajorityVote;
export const softmaxWeightedVote = vote.softmaxWeightedVote;
export const rankWeightedVote = vote.rankWeightedVote;
export const logitWeightedVote = vote.logitWeightedVote;
export const filteredVote = vote.filteredVote;

export const fitKdeVoteCalibration = calibration.fitKdeVoteCalibration;
export const kdeWeightedVote = calibration.kdeWeightedVote;

export const cgesVote = cges.cgesVote;
export const cgesStop = cges.cgesStop;

export const adaptiveConsistencyStop = online.adaptiveConsistencyStop;
export const adaptiveConsistencyDirichletStop =
  online.adaptiveConsistencyDirichletStop;
export const adaptiveConsistencyCrpStop = online.adaptiveConsistencyCrpStop;
export const escStop = online.escStop;
export const deepconfStopThreshold = online.deepconfStopThreshold;
export const deepconfOnlineStop = online.deepconfOnlineStop;

// snake_case aliases (Python / Julia parity).
export {
  meanLogprob as mean_logprob,
  sequenceLogprob as sequence_logprob,
  selfCertainty as self_certainty,
  tokenConfidence as token_confidence,
  deepconfConfidence as deepconf_confidence,
  tokenEntropy as token_entropy,
  maxSoftmaxProbability as max_softmax_probability,
  logprobMargin as logprob_margin,
  prmAggregate as prm_aggregate,
  bestOfN as best_of_n,
  majorityOfTheBests as majority_of_the_bests,
  bestOfMajority as best_of_majority,
  majorityVote as majority_vote,
  weightedMajorityVote as weighted_majority_vote,
  softmaxWeightedVote as softmax_weighted_vote,
  rankWeightedVote as rank_weighted_vote,
  logitWeightedVote as logit_weighted_vote,
  filteredVote as filtered_vote,
  fitKdeVoteCalibration as fit_kde_vote_calibration,
  kdeWeightedVote as kde_weighted_vote,
  cgesVote as cges_vote,
  cgesStop as cges_stop,
  adaptiveConsistencyStop as adaptive_consistency_stop,
  adaptiveConsistencyDirichletStop as adaptive_consistency_dirichlet_stop,
  adaptiveConsistencyCrpStop as adaptive_consistency_crp_stop,
  escStop as esc_stop,
  deepconfStopThreshold as deepconf_stop_threshold,
  deepconfOnlineStop as deepconf_online_stop,
};
