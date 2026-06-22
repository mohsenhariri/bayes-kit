/**
 * Paired-comparison probabilistic ranking models (Bradley-Terry, Davidson,
 * Rao-Kupper). Port of `scorio/rank/bradley_terry.py`.
 *
 * Latent strengths `pi_i = exp(theta_i)` are fit by minimizing a negative
 * (log-)likelihood — with an optional prior penalty for the MAP variants — over
 * mean-centered log-strengths.
 */

import { minimize } from "./internal/optimize.js";
import { clip } from "./internal/special.js";
import { rankScores } from "./internal/rankScores.js";
import { validatePositiveInt } from "./internal/validate.js";
import {
  buildPairwiseCounts,
  buildPairwiseWins,
  validateInput,
  type TensorInput,
} from "./internal/tensor.js";
import { GaussianPrior, coercePrior, type Prior } from "./priors.js";
import type { BaseRankOptions, RankResult } from "./internal/result.js";

const sum = (a: readonly number[]): number => a.reduce((s, v) => s + v, 0);
const mean = (a: readonly number[]): number => sum(a) / a.length;
const center = (a: readonly number[]): number[] => {
  const m = mean(a);
  return a.map((v) => v - m);
};
const rowSums = (m: number[][]): number[] => m.map((row) => sum(row));

function finalizeStrengths(x: readonly number[]): number[] {
  const c = center(x);
  return c.map((v) => Math.exp(clip(v, -30, 30)));
}

function logPiInit(wins: number[][]): number[] {
  const tw = rowSums(wins).map((v) => Math.max(v, 1));
  const total = sum(tw);
  return tw.map((v) => Math.log(v / total));
}

function isZeroMeanGaussian(prior: Prior): boolean {
  return prior instanceof GaussianPrior && prior.mean === 0;
}

/** Options for the MAP variants. */
export interface BTMapOptions extends BaseRankOptions {
  /** Prior on log-strengths: a variance (zero-mean Gaussian) or a `Prior`. Default `1`. */
  prior?: Prior | number;
  maxIter?: number;
}

/** Options for the ML variants. */
export interface BTMlOptions extends BaseRankOptions {
  maxIter?: number;
}

// --- Bradley-Terry (ML / MAP) -------------------------------------------------

function btNll(wins: number[][], logPi: readonly number[]): number {
  const n = wins.length;
  const lp = center(logPi);
  const capped = lp.map((v) => clip(v, -30, 30));
  const pi = capped.map((v) => Math.exp(v));
  let nll = 0;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      if (i === j) continue;
      const nij = wins[i]![j]!;
      if (nij > 0) nll -= nij * (capped[i]! - Math.log(pi[i]! + pi[j]!));
    }
  }
  return nll;
}

function estimateBtMl(wins: number[][], maxIter: number): number[] {
  const n = wins.length;
  if (sum(rowSums(wins)) <= 0) return new Array<number>(n).fill(1);
  const res = minimize((x) => btNll(wins, x), logPiInit(wins), { maxIter });
  return finalizeStrengths(res.x);
}

function estimateBtMap(wins: number[][], prior: Prior, maxIter: number): number[] {
  const n = wins.length;
  const noDecisive = sum(rowSums(wins)) <= 0;
  if (noDecisive && isZeroMeanGaussian(prior)) return new Array<number>(n).fill(1);
  const res = minimize(
    (x) => btNll(wins, x) + prior.penalty(center(x)),
    logPiInit(wins),
    { maxIter },
  );
  const scores = finalizeStrengths(res.x);
  if (noDecisive && Math.max(...scores) - Math.min(...scores) <= 1e-5) {
    return new Array<number>(n).fill(1);
  }
  return scores;
}

/** Rank models with Bradley-Terry maximum-likelihood strengths. */
export function bradleyTerry(R: TensorInput, options: BTMlOptions = {}): RankResult {
  const method = options.method ?? "competition";
  const wins = buildPairwiseWins(validateInput(R));
  const scores = estimateBtMl(wins, validatePositiveInt("max_iter", options.maxIter ?? 500));
  return { ranking: rankScores(scores, method), scores };
}

/** Rank models with Bradley-Terry MAP estimation. */
export function bradleyTerryMap(R: TensorInput, options: BTMapOptions = {}): RankResult {
  const method = options.method ?? "competition";
  const prior = coercePrior(options.prior ?? 1);
  const wins = buildPairwiseWins(validateInput(R));
  const scores = estimateBtMap(wins, prior, validatePositiveInt("max_iter", options.maxIter ?? 500));
  return { ranking: rankScores(scores, method), scores };
}

// --- Bradley-Terry-Davidson (ML / MAP) ---------------------------------------

function davidsonNll(
  wins: number[][],
  ties: number[][],
  params: readonly number[],
): number {
  const n = wins.length;
  const eps = 1e-10;
  const lp = center(params.slice(0, n));
  const capped = lp.map((v) => clip(v, -30, 30));
  const pi = capped.map((v) => Math.exp(v));
  const theta = Math.exp(clip(params[n]!, -10, 10));
  let nll = 0;
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const nij = wins[i]![j]!;
      const nji = wins[j]![i]!;
      const nTie = ties[i]![j]!;
      let denom = pi[i]! + pi[j]! + theta * Math.sqrt(pi[i]! * pi[j]!);
      denom = Math.max(denom, eps);
      if (nij > 0) nll -= nij * Math.log(Math.max(pi[i]! / denom, eps));
      if (nji > 0) nll -= nji * Math.log(Math.max(pi[j]! / denom, eps));
      if (nTie > 0) {
        const tieProb = (theta * Math.sqrt(pi[i]! * pi[j]!)) / denom;
        nll -= nTie * Math.log(Math.max(tieProb, eps));
      }
    }
  }
  return nll;
}

function estimateDavidson(
  wins: number[][],
  ties: number[][],
  prior: Prior | null,
  maxIter: number,
): number[] {
  const n = wins.length;
  const noDecisive = sum(rowSums(wins)) <= 0;
  if (prior === null) {
    if (noDecisive) return new Array<number>(n).fill(1);
  } else if (noDecisive && isZeroMeanGaussian(prior)) {
    return new Array<number>(n).fill(1);
  }
  const init = [...logPiInit(wins), 0];
  const res = minimize(
    (x) =>
      davidsonNll(wins, ties, x) +
      (prior ? prior.penalty(center(x.slice(0, n))) : 0),
    init,
    { maxIter },
  );
  const scores = finalizeStrengths(res.x.slice(0, n));
  if (prior !== null && noDecisive && Math.max(...scores) - Math.min(...scores) <= 1e-5) {
    return new Array<number>(n).fill(1);
  }
  return scores;
}

/** Rank models with the Bradley-Terry-Davidson tie model (ML). */
export function bradleyTerryDavidson(
  R: TensorInput,
  options: BTMlOptions = {},
): RankResult {
  const method = options.method ?? "competition";
  const { wins, ties } = buildPairwiseCounts(validateInput(R));
  const scores = estimateDavidson(wins, ties, null, validatePositiveInt("max_iter", options.maxIter ?? 500));
  return { ranking: rankScores(scores, method), scores };
}

/** Rank models with the Bradley-Terry-Davidson tie model (MAP). */
export function bradleyTerryDavidsonMap(
  R: TensorInput,
  options: BTMapOptions = {},
): RankResult {
  const method = options.method ?? "competition";
  const prior = coercePrior(options.prior ?? 1);
  const { wins, ties } = buildPairwiseCounts(validateInput(R));
  const scores = estimateDavidson(wins, ties, prior, validatePositiveInt("max_iter", options.maxIter ?? 500));
  return { ranking: rankScores(scores, method), scores };
}

// --- Rao-Kupper (ML / MAP) ---------------------------------------------------

function validateTieStrength(tieStrength: number): number {
  const kappa = Number(tieStrength);
  if (!Number.isFinite(kappa)) throw new Error("tie_strength must be finite.");
  if (kappa < 1) throw new Error("tie_strength must be >= 1.0 for Rao-Kupper");
  return kappa;
}

function raoKupperNll(
  wins: number[][],
  ties: number[][],
  kappa: number,
  logPi: readonly number[],
): number {
  const n = wins.length;
  const eps = 1e-12;
  const lp = center(logPi);
  const pi = lp.map((v) => Math.exp(clip(v, -30, 30)));
  let nll = 0;
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const nij = wins[i]![j]!;
      const nji = wins[j]![i]!;
      const nTie = ties[i]![j]!;
      const denomIj = pi[i]! + kappa * pi[j]!;
      const denomJi = kappa * pi[i]! + pi[j]!;
      const pIj = Math.max(pi[i]! / denomIj, eps);
      const pJi = Math.max(pi[j]! / denomJi, eps);
      let pTie = 0;
      if (kappa > 1) {
        pTie = Math.max(
          ((kappa * kappa - 1) * pi[i]! * pi[j]!) / (denomIj * denomJi),
          eps,
        );
      }
      if (nij > 0) nll -= nij * Math.log(pIj);
      if (nji > 0) nll -= nji * Math.log(pJi);
      if (nTie > 0) {
        if (kappa === 1) return Infinity;
        nll -= nTie * Math.log(pTie);
      }
    }
  }
  return nll;
}

function totalUpperTies(ties: number[][]): number {
  let s = 0;
  for (let i = 0; i < ties.length; i++)
    for (let j = i + 1; j < ties.length; j++) s += ties[i]![j]!;
  return s;
}

function estimateRaoKupper(
  wins: number[][],
  ties: number[][],
  kappa: number,
  prior: Prior | null,
  maxIter: number,
): number[] {
  const n = wins.length;
  if (kappa === 1 && totalUpperTies(ties) > 0) {
    throw new Error("tie_strength=1.0 implies no ties, but tie counts exist");
  }
  const noDecisive = sum(rowSums(wins)) <= 0;
  if (prior === null) {
    if (noDecisive) return new Array<number>(n).fill(1);
  } else if (noDecisive && isZeroMeanGaussian(prior)) {
    return new Array<number>(n).fill(1);
  }
  const init = new Array<number>(n).fill(0);
  const res = minimize(
    (x) =>
      raoKupperNll(wins, ties, kappa, x) + (prior ? prior.penalty(center(x)) : 0),
    init,
    { maxIter },
  );
  const scores = finalizeStrengths(res.x);
  if (prior !== null && noDecisive && Math.max(...scores) - Math.min(...scores) <= 1e-5) {
    return new Array<number>(n).fill(1);
  }
  return scores;
}

/** Options for the Rao-Kupper ML variant. */
export interface RaoKupperOptions extends BTMlOptions {
  /** Rao-Kupper threshold `kappa >= 1`. Default `1.1`. */
  tieStrength?: number;
}

/** Options for the Rao-Kupper MAP variant. */
export interface RaoKupperMapOptions extends BTMapOptions {
  tieStrength?: number;
}

/** Rank models with the Rao-Kupper tie model (ML). */
export function raoKupper(R: TensorInput, options: RaoKupperOptions = {}): RankResult {
  const method = options.method ?? "competition";
  const kappa = validateTieStrength(options.tieStrength ?? 1.1);
  const { wins, ties } = buildPairwiseCounts(validateInput(R));
  const scores = estimateRaoKupper(wins, ties, kappa, null, validatePositiveInt("max_iter", options.maxIter ?? 500));
  return { ranking: rankScores(scores, method), scores };
}

/** Rank models with the Rao-Kupper tie model (MAP). */
export function raoKupperMap(
  R: TensorInput,
  options: RaoKupperMapOptions = {},
): RankResult {
  const method = options.method ?? "competition";
  const kappa = validateTieStrength(options.tieStrength ?? 1.1);
  const prior = coercePrior(options.prior ?? 1);
  const { wins, ties } = buildPairwiseCounts(validateInput(R));
  const scores = estimateRaoKupper(wins, ties, kappa, prior, validatePositiveInt("max_iter", options.maxIter ?? 500));
  return { ranking: rankScores(scores, method), scores };
}
