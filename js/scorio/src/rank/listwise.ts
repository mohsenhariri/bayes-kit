/**
 * Listwise / setwise Luce-family choice models. Port of `scorio/rank/listwise.py`.
 *
 * Plackett-Luce (pairwise reduction) is fit with Hunter's MM updates; the
 * setwise tie models (Davidson-Luce) and the rank-breaking composite likelihood
 * (Bradley-Terry-Luce) are fit with L-BFGS over mean-centered log-strengths.
 */

import { minimize } from "./internal/optimize.js";
import { clip, logaddexp, logsumexp } from "./internal/special.js";
import { rankScores } from "./internal/rankScores.js";
import { validatePositiveInt } from "./internal/validate.js";
import {
  averageEquivalentScores,
  averageEventExchangeableScores,
  buildPairwiseWins,
  isStronglyConnected,
  shape3,
  validateInput,
  type Tensor3,
  type TensorInput,
} from "./internal/tensor.js";
import {
  CauchyPrior,
  GaussianPrior,
  LaplacePrior,
  UniformPrior,
  coercePrior,
  type Prior,
} from "./priors.js";
import type { BaseRankOptions, RankResult } from "./internal/result.js";

const sum = (a: readonly number[]): number => a.reduce((s, v) => s + v, 0);
const defaultIfUndefined = <T>(value: T | undefined, fallback: T): T =>
  value === undefined ? fallback : value;

function coercePythonFloat(value: unknown, name: string): number {
  if (typeof value === "number") return value;
  if (typeof value === "boolean") return value ? 1 : 0;
  if (typeof value === "string" && value.trim().length > 0) {
    const normalized = value.trim().replace(/(?<=\d)_(?=\d)/g, "");
    if (/^[+-]?(?:inf(?:inity)?)$/i.test(normalized)) {
      return normalized.startsWith("-") ? -Infinity : Infinity;
    }
    if (/^[+-]?nan$/i.test(normalized)) return NaN;
    if (/^[+-]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?)$/i.test(normalized)) {
      return Number(normalized);
    }
  }
  throw new TypeError(`${name} must be convertible to a float`);
}

function validatePositiveFloat(name: string, value: unknown): number {
  const converted = coercePythonFloat(value, name);
  if (!Number.isFinite(converted) || converted <= 0) {
    throw new Error(`${name} must be a finite scalar > 0, got ${value}`);
  }
  return converted;
}

const mean = (a: readonly number[]): number => (a.length ? sum(a) / a.length : 0);
const center = (a: readonly number[]): number[] => {
  const m = mean(a);
  return a.map((v) => v - m);
};
const finalize = (x: readonly number[]): number[] =>
  center(x).map((v) => Math.exp(clip(v, -30, 30)));

function isExchangeablePrior(prior: Prior): boolean {
  return (
    prior.constructor === GaussianPrior ||
    prior.constructor === LaplacePrior ||
    prior.constructor === CauchyPrior ||
    prior.constructor === UniformPrior
  );
}

function requireOptimizerSuccess(
  result: { success: boolean; iterations: number },
  modelName: string,
): void {
  if (!result.success) {
    throw new Error(`${modelName} optimization failed after ${result.iterations} iterations`);
  }
}

/** Options for ML listwise variants. */
export interface ListwiseMlOptions extends BaseRankOptions {
  maxIter?: number;
}

/** Options for MAP listwise variants. */
export interface ListwiseMapOptions extends BaseRankOptions {
  prior?: Prior | number;
  maxIter?: number;
}

/** Options for {@link plackettLuce}. */
export interface PlackettLuceOptions extends ListwiseMlOptions {
  tol?: number;
}

/** Options for the Davidson-Luce variants. */
export interface DavidsonLuceMlOptions extends ListwiseMlOptions {
  /** Maximum tie order `D` in the normalization. Default `L-1`. */
  maxTieOrder?: number | null;
}
export interface DavidsonLuceMapOptions extends ListwiseMapOptions {
  maxTieOrder?: number | null;
}

function logPiInit(wins: number[][]): number[] {
  const tw = wins.map((row) => Math.max(sum(row), 1));
  const total = sum(tw);
  return tw.map((v) => Math.log(v / total));
}

/** Per-event `(winners, losers)` partition; events with all/none correct are dropped. */
function extractEvents(R: Tensor3): { winners: number[]; losers: number[] }[] {
  const [L, M, N] = shape3(R);
  const events: { winners: number[]; losers: number[] }[] = [];
  for (let m = 0; m < M; m++) {
    for (let n = 0; n < N; n++) {
      const winners: number[] = [];
      const losers: number[] = [];
      for (let l = 0; l < L; l++) (R[l]![m]![n]! === 1 ? winners : losers).push(l);
      if (winners.length === 0 || winners.length === L) continue;
      events.push({ winners, losers });
    }
  }
  return events;
}

// --- Plackett-Luce (MM ML) ---------------------------------------------------

function mmPlackettLuce(wins: number[][], maxIter: number, tol: number): number[] {
  const L = wins.length;
  const W = wins.map((row) => sum(row));
  const totalWins = sum(W);
  if (totalWins === 0) return new Array<number>(L).fill(1 / L);
  if (!isStronglyConnected(wins.map((row) => row.map((value) => value > 0)))) {
    throw new Error(
      "Plackett-Luce has no finite maximum-likelihood estimate because the directed win graph is not strongly connected; use plackett_luce_map for a regularized estimate.",
    );
  }

  let pi = W.map((v) => Math.max(v / totalWins, 1e-10));
  const nComp = Array.from({ length: L }, (_, i) =>
    Array.from({ length: L }, (_, j) => wins[i]![j]! + wins[j]![i]!),
  );

  let converged = false;
  for (let it = 0; it < maxIter; it++) {
    const piOld = pi.slice();
    for (let i = 0; i < L; i++) {
      let denom = 0;
      for (let j = 0; j < L; j++) {
        if (i === j) continue;
        if (nComp[i]![j]! > 0) denom += nComp[i]![j]! / (piOld[i]! + piOld[j]!);
      }
      pi[i] = denom > 0 ? W[i]! / denom : piOld[i]!;
    }
    const piSum = sum(pi);
    if (piSum > 0) pi = pi.map((v) => v / piSum);
    pi = pi.map((v) => Math.max(v, 1e-10));
    let maxChange = 0;
    for (let i = 0; i < L; i++) maxChange = Math.max(maxChange, Math.abs(pi[i]! - piOld[i]!));
    if (maxChange < tol) {
      converged = true;
      break;
    }
  }
  if (!converged) {
    throw new Error("plackett_luce MM optimization did not converge within max_iter");
  }
  return pi;
}

/** Rank models with Plackett-Luce maximum likelihood (pairwise MM). */
export function plackettLuce(
  R: TensorInput,
  options: PlackettLuceOptions = {},
): RankResult {
  const method = defaultIfUndefined(options.method, "competition");
  const tensor = validateInput(R);
  const maxIter = validatePositiveInt(
    "max_iter",
    defaultIfUndefined(options.maxIter, 10000),
  );
  const tol = validatePositiveFloat(
    "tol",
    defaultIfUndefined(options.tol, 1e-8),
  );
  const wins = buildPairwiseWins(tensor);
  const scores = averageEquivalentScores(mmPlackettLuce(wins, maxIter, tol), tensor);
  return { ranking: rankScores(scores, method), scores };
}

/** Rank models with Plackett-Luce MAP estimation (pairwise reduction). */
export function plackettLuceMap(
  R: TensorInput,
  options: ListwiseMapOptions = {},
): RankResult {
  const method = defaultIfUndefined(options.method, "competition");
  const tensor = validateInput(R);
  const maxIter = validatePositiveInt(
    "max_iter",
    defaultIfUndefined(options.maxIter, 500),
  );
  const prior = coercePrior(defaultIfUndefined(options.prior, 1));
  const wins = buildPairwiseWins(tensor);
  if (prior.constructor === UniformPrior) {
    const scores = averageEquivalentScores(
      mmPlackettLuce(
        wins,
        maxIter,
        1e-8,
      ),
      tensor,
    );
    return { ranking: rankScores(scores, method), scores };
  }
  const L = wins.length;
  const nll = (logPiRaw: readonly number[]): number => {
    const lp = center(logPiRaw);
    let s = 0;
    for (let i = 0; i < L; i++) {
      for (let j = 0; j < L; j++) {
        if (i === j) continue;
        const nij = wins[i]![j]!;
        if (nij > 0) s -= nij * (lp[i]! - logaddexp(lp[i]!, lp[j]!));
      }
    }
    return s + prior.penalty(lp);
  };
  const res = minimize(nll, logPiInit(wins), { maxIter });
  requireOptimizerSuccess(res, "plackett_luce_map");
  let scores = finalize(res.x);
  if (isExchangeablePrior(prior)) scores = averageEquivalentScores(scores, tensor);
  return { ranking: rankScores(scores, method), scores };
}

// --- Davidson-Luce (setwise ties) --------------------------------------------

function logElementarySymmetricSum(logX: readonly number[], k: number): number {
  if (k < 0) throw new Error("k must be >= 0");
  if (k === 0) return 0;
  const n = logX.length;
  if (k > n) return -Infinity;
  const logE = new Array<number>(k + 1).fill(-Infinity);
  logE[0] = 0;
  for (let i = 0; i < n; i++) {
    const upper = Math.min(k, i + 1);
    for (let j = upper; j >= 1; j--) {
      logE[j] = logaddexp(logE[j]!, logE[j - 1]! + logX[i]!);
    }
  }
  return logE[k]!;
}

function logDenominatorDavidsonLuce(
  logAlpha: readonly number[],
  logDeltaParams: readonly number[],
  comparisonSet: readonly number[],
  maxTieOrder: number,
): number {
  const D = Math.min(maxTieOrder, comparisonSet.length);
  const terms: number[] = [];
  for (let t = 1; t <= D; t++) {
    let logDeltaT = 0;
    if (t !== 1) {
      const idx = t - 2;
      logDeltaT = idx < logDeltaParams.length ? logDeltaParams[idx]! : 0;
    }
    const logX = comparisonSet.map((i) => logAlpha[i]! / t);
    const logEt = logElementarySymmetricSum(logX, t);
    if (logEt === -Infinity) continue;
    terms.push(logDeltaT + logEt);
  }
  return logsumexp(terms);
}

function estimateDavidsonLuce(
  events: { winners: number[]; losers: number[] }[],
  L: number,
  maxTieOrder: number,
  maxIter: number,
  prior: Prior | null,
): number[] {
  if (events.length === 0) return new Array<number>(L).fill(1 / L);
  const uniformPrior = prior !== null && prior.constructor === UniformPrior;
  if (prior === null || uniformPrior) {
    const directedChoices = Array.from({ length: L }, () =>
      new Array<boolean>(L).fill(false),
    );
    for (const { winners, losers } of events) {
      for (const winner of winners) {
        for (const loser of losers) directedChoices[winner]![loser] = true;
        for (const coWinner of winners) {
          if (winner !== coWinner) directedChoices[winner]![coWinner] = true;
        }
      }
    }
    if (!isStronglyConnected(directedChoices)) {
      throw new Error(
        "Davidson-Luce has no finite maximum-likelihood strength estimate because its win/co-winner graph is not strongly connected; use davidson_luce_map with a proper prior.",
      );
    }
    if (uniformPrior) {
      return estimateDavidsonLuce(events, L, maxTieOrder, maxIter, null);
    }
  }
  const comparisonSet = Array.from({ length: L }, (_, i) => i);
  const nll = (params: readonly number[]): number => {
    const logAlpha = center(params.slice(0, L));
    const logDeltaParams = params.slice(L);
    let s = 0;
    for (const { winners } of events) {
      const t = winners.length;
      if (t < 1 || t > maxTieOrder) {
        throw new Error("Observed winner-set size exceeds max_tie_order");
      }
      const logDeltaT = t === 1 ? 0 : logDeltaParams[t - 2]!;
      const logNumerator = logDeltaT + mean(winners.map((w) => logAlpha[w]!));
      const logDenom = logDenominatorDavidsonLuce(
        logAlpha,
        logDeltaParams,
        comparisonSet,
        maxTieOrder,
      );
      s -= logNumerator - logDenom;
    }
    if (prior) s += prior.penalty(logAlpha);
    return s;
  };
  const init = [
    ...new Array<number>(L).fill(0),
    ...new Array<number>(Math.max(maxTieOrder - 1, 0)).fill(0),
  ];
  const res = minimize(nll, init, { maxIter });
  requireOptimizerSuccess(
    res,
    prior === null ? "davidson_luce" : "davidson_luce_map",
  );
  return finalize(res.x.slice(0, L));
}

function davidsonEquivalenceStatistics(
  events: { winners: number[]; losers: number[] }[],
  L: number,
  maxTieOrder: number,
): number[][] {
  const counts = Array.from({ length: L }, () =>
    new Array<number>(maxTieOrder).fill(0),
  );
  for (const { winners } of events) {
    for (const winner of winners) counts[winner]![winners.length - 1]! += 1;
  }
  return counts;
}

function resolveMaxTieOrder(
  L: number,
  given: number | null | undefined,
): number {
  const mto = validatePositiveInt(
    "max_tie_order",
    given === null || given === undefined ? Math.max(L - 1, 1) : given,
  );
  if (mto > L) throw new Error(`max_tie_order must be <= number of models (${L})`);
  return mto;
}

/** Rank models with Davidson-Luce maximum likelihood (setwise ties). */
export function davidsonLuce(
  R: TensorInput,
  options: DavidsonLuceMlOptions = {},
): RankResult {
  const method = defaultIfUndefined(options.method, "competition");
  const tensor = validateInput(R);
  const maxIter = validatePositiveInt(
    "max_iter",
    defaultIfUndefined(options.maxIter, 500),
  );
  const events = extractEvents(tensor);
  const [L] = shape3(tensor);
  const mto = resolveMaxTieOrder(L, options.maxTieOrder);
  if (events.some(({ winners }) => winners.length > mto)) {
    throw new Error(
      "Observed winner-set size exceeds max_tie_order; increase max_tie_order so every observed event has positive model probability.",
    );
  }
  const scores = averageEquivalentScores(estimateDavidsonLuce(
    events,
    L,
    mto,
    maxIter,
    null,
  ), davidsonEquivalenceStatistics(events, L, mto));
  return { ranking: rankScores(scores, method), scores };
}

/** Rank models with Davidson-Luce MAP estimation. */
export function davidsonLuceMap(
  R: TensorInput,
  options: DavidsonLuceMapOptions = {},
): RankResult {
  const method = defaultIfUndefined(options.method, "competition");
  const tensor = validateInput(R);
  const maxIter = validatePositiveInt(
    "max_iter",
    defaultIfUndefined(options.maxIter, 500),
  );
  const [L] = shape3(tensor);
  const prior = coercePrior(defaultIfUndefined(options.prior, 1));
  const events = extractEvents(tensor);
  const mto = resolveMaxTieOrder(L, options.maxTieOrder);
  if (events.some(({ winners }) => winners.length > mto)) {
    throw new Error(
      "Observed winner-set size exceeds max_tie_order; increase max_tie_order so every observed event has positive model probability.",
    );
  }
  let scores = estimateDavidsonLuce(
    events,
    L,
    mto,
    maxIter,
    prior,
  );
  if (isExchangeablePrior(prior)) {
    scores = averageEquivalentScores(
      scores,
      davidsonEquivalenceStatistics(events, L, mto),
    );
  }
  return { ranking: rankScores(scores, method), scores };
}

// --- Bradley-Terry-Luce (composite likelihood) -------------------------------

function estimateBtl(
  events: { winners: number[]; losers: number[] }[],
  L: number,
  maxIter: number,
  prior: Prior | null,
): number[] {
  if (events.length === 0) return new Array<number>(L).fill(1 / L);
  const uniformPrior = prior !== null && prior.constructor === UniformPrior;
  if (prior === null || uniformPrior) {
    const directedChoices = Array.from({ length: L }, () =>
      new Array<boolean>(L).fill(false),
    );
    for (const { winners, losers } of events) {
      for (const winner of winners) {
        for (const loser of losers) directedChoices[winner]![loser] = true;
      }
    }
    if (!isStronglyConnected(directedChoices)) {
      throw new Error(
        "Bradley-Terry-Luce has no finite maximum-likelihood estimate because the directed choice graph is not strongly connected; use bradley_terry_luce_map for a regularized estimate.",
      );
    }
    if (uniformPrior) return estimateBtl(events, L, maxIter, null);
  }
  const nll = (logPiRaw: readonly number[]): number => {
    const logPi = center(logPiRaw);
    let s = 0;
    for (const { winners, losers } of events) {
      const logSumLosers = logsumexp(losers.map((l) => logPi[l]!));
      for (const w of winners) {
        s -= logPi[w]!;
        s += logaddexp(logPi[w]!, logSumLosers);
      }
    }
    if (prior) s += prior.penalty(logPi);
    return s;
  };
  const res = minimize(nll, new Array<number>(L).fill(0), { maxIter });
  requireOptimizerSuccess(
    res,
    prior === null ? "bradley_terry_luce" : "bradley_terry_luce_map",
  );
  return finalize(res.x);
}

/** Rank models with Bradley-Terry-Luce composite-likelihood ML. */
export function bradleyTerryLuce(
  R: TensorInput,
  options: ListwiseMlOptions = {},
): RankResult {
  const method = defaultIfUndefined(options.method, "competition");
  const tensor = validateInput(R);
  const maxIter = validatePositiveInt(
    "max_iter",
    defaultIfUndefined(options.maxIter, 500),
  );
  const events = extractEvents(tensor);
  const [L] = shape3(tensor);
  const scores = averageEventExchangeableScores(
    estimateBtl(
      events,
      L,
      maxIter,
      null,
    ),
    tensor,
  );
  return { ranking: rankScores(scores, method), scores };
}

/** Rank models with Bradley-Terry-Luce composite-likelihood MAP estimation. */
export function bradleyTerryLuceMap(
  R: TensorInput,
  options: ListwiseMapOptions = {},
): RankResult {
  const method = defaultIfUndefined(options.method, "competition");
  const tensor = validateInput(R);
  const maxIter = validatePositiveInt(
    "max_iter",
    defaultIfUndefined(options.maxIter, 500),
  );
  const prior = coercePrior(defaultIfUndefined(options.prior, 1));
  const events = extractEvents(tensor);
  const [L] = shape3(tensor);
  let scores = estimateBtl(
    events,
    L,
    maxIter,
    prior,
  );
  if (isExchangeablePrior(prior)) {
    scores = averageEventExchangeableScores(scores, tensor);
  }
  return { ranking: rankScores(scores, method), scores };
}
