/**
 * Bayesian ranking methods. Port of `scorio/rank/bayesian.py`.
 *
 * These methods are Monte Carlo: `thompson` ranks by posterior expected rank
 * from Beta-Binomial draws, and `bayesianMcmc` ranks by Bradley-Terry posterior
 * means from random-walk Metropolis-Hastings. Results are seeded and
 * reproducible but, unlike the deterministic methods, are not expected to match
 * the NumPy reference bit-for-bit (the RNG differs).
 */

import { rankScores } from "./internal/rankScores.js";
import { SeededRng } from "./internal/rng.js";
import {
  averageEquivalentScores,
  buildPairwiseWins,
  shape3,
  validateInput,
  type TensorInput,
} from "./internal/tensor.js";
import type { BaseRankOptions, RankResult } from "./internal/result.js";

const sum = (a: readonly number[]): number => a.reduce((s, v) => s + v, 0);

const PYTHON_FLOAT_PATTERN = /^[+-]?(?:(?:(?:\d(?:_?\d)*(?:\.(?:\d(?:_?\d)*)?)?|\.\d(?:_?\d)*)(?:[eE][+-]?\d(?:_?\d)*)?)|inf(?:inity)?|nan)$/i;

function pythonFloat(value: unknown, errorMessage: string): number {
  if (typeof value === "number") return value;
  if (typeof value === "boolean") return value ? 1 : 0;
  if (typeof value !== "string") throw new TypeError(errorMessage);

  const text = value.trim();
  if (!PYTHON_FLOAT_PATTERN.test(text)) throw new Error(errorMessage);
  if (/^[+-]?nan$/i.test(text)) return Number.NaN;
  if (/^[+-]?inf(?:inity)?$/i.test(text)) return text.startsWith("-") ? -Infinity : Infinity;
  return Number(text.replace(/_/g, ""));
}

function defaultIfUndefined<T>(value: T | undefined, fallback: T): T {
  return value === undefined ? fallback : value;
}

/** Accept the scalar seed forms accepted by `numpy.random.default_rng`. */
function pythonSeed(value: unknown): number {
  // Python's `None` requests fresh entropy rather than the default fixed seed.
  if (value === null) return Math.floor(Math.random() * 0x1_0000_0000);
  if (typeof value === "boolean") return value ? 1 : 0;
  if (typeof value !== "number" || !Number.isFinite(value) || !Number.isInteger(value)) {
    throw new TypeError("seed must be a nonnegative integer or null");
  }
  if (value < 0) throw new Error("seed must be a nonnegative integer or null");
  return value;
}

/** Options for {@link thompson}. */
export interface ThompsonOptions extends BaseRankOptions {
  nSamples?: number;
  priorAlpha?: number;
  priorBeta?: number;
  seed?: number;
}

/** Rank models by Thompson-sampling posterior expected rank. */
export function thompson(R: TensorInput, options: ThompsonOptions = {}): RankResult {
  const method = defaultIfUndefined(options.method, "competition");
  const tensor = validateInput(R);

  const nSamples = defaultIfUndefined(options.nSamples, 10000);
  if (typeof nSamples !== "number" || !Number.isInteger(nSamples)) {
    throw new TypeError(`n_samples must be an integer, got ${String(nSamples)}`);
  }
  if (nSamples < 1) {
    throw new Error(`n_samples must be >= 1, got ${nSamples}`);
  }
  const priorAlpha = pythonFloat(
    defaultIfUndefined(options.priorAlpha, 1),
    "prior_alpha must be > 0 and finite.",
  );
  if (!Number.isFinite(priorAlpha) || priorAlpha <= 0) throw new Error("prior_alpha must be > 0 and finite.");
  const priorBeta = pythonFloat(
    defaultIfUndefined(options.priorBeta, 1),
    "prior_beta must be > 0 and finite.",
  );
  if (!Number.isFinite(priorBeta) || priorBeta <= 0) throw new Error("prior_beta must be > 0 and finite.");

  const [L, M, N] = shape3(tensor);
  const seed = pythonSeed(defaultIfUndefined(options.seed, 42));
  const rng = new SeededRng(seed);

  const successes = tensor.map((mat) => {
    let s = 0;
    for (const row of mat) for (const v of row) s += v;
    return s;
  });
  const total = M * N;
  const postAlphas = successes.map((s) => priorAlpha + s);
  const postBetas = successes.map((s) => priorBeta + (total - s));

  // np.allclose(a, a[0]) semantics: |a_i - a_0| <= atol + rtol·|a_0|.
  const allClose = (a: number[]) =>
    a.every((v) => Math.abs(v - a[0]!) <= 1e-8 + 1e-5 * Math.abs(a[0]!));
  if (allClose(postAlphas) && allClose(postBetas)) {
    const scores = new Array<number>(L).fill(-(L + 1) / 2);
    return { ranking: rankScores(scores, method), scores };
  }

  const rankSums = new Array<number>(L).fill(0);
  for (let t = 0; t < nSamples; t++) {
    const samples = postAlphas.map((a, i) => rng.beta(a, postBetas[i]!));
    const order = Array.from({ length: L }, (_, i) => i).sort(
      (a, b) => samples[b]! - samples[a]!,
    );
    for (let p = 0; p < L; p++) rankSums[order[p]!]! += p + 1;
  }
  const scores = averageEquivalentScores(
    rankSums.map((s) => -s / nSamples),
    postAlphas.map((alpha, index) => [alpha, postBetas[index]!]),
  );
  return { ranking: rankScores(scores, method), scores };
}

/** Options for {@link bayesianMcmc}. */
export interface BayesianMcmcOptions extends BaseRankOptions {
  nSamples?: number;
  burnin?: number;
  priorVar?: number;
  seed?: number;
}

/** Rank models via Bayesian Bradley-Terry posterior means from MCMC. */
export function bayesianMcmc(R: TensorInput, options: BayesianMcmcOptions = {}): RankResult {
  const method = defaultIfUndefined(options.method, "competition");
  const tensor = validateInput(R);

  const nSamples = defaultIfUndefined(options.nSamples, 5000);
  if (typeof nSamples !== "number" || !Number.isInteger(nSamples)) {
    throw new TypeError(`n_samples must be an integer, got ${String(nSamples)}`);
  }
  if (nSamples < 1) throw new Error(`n_samples must be >= 1, got ${nSamples}`);
  const burnin = defaultIfUndefined(options.burnin, 1000);
  if (typeof burnin !== "number" || !Number.isInteger(burnin)) {
    throw new TypeError(`burnin must be an integer, got ${String(burnin)}`);
  }
  if (burnin < 0) throw new Error(`burnin must be >= 0, got ${burnin}`);
  const priorVar = pythonFloat(
    defaultIfUndefined(options.priorVar, 1),
    "prior_var must be > 0 and finite.",
  );
  if (!Number.isFinite(priorVar) || priorVar <= 0) throw new Error("prior_var must be > 0 and finite.");

  const [L] = shape3(tensor);
  const seed = pythonSeed(defaultIfUndefined(options.seed, 42));
  const rng = new SeededRng(seed);
  const wins = buildPairwiseWins(tensor);

  if (sum(wins.map((row) => sum(row))) <= 0) {
    const scores = new Array<number>(L).fill(0);
    return { ranking: rankScores(scores, method), scores };
  }

  const logLikelihood = (theta: number[]): number => {
    let ll = 0;
    for (let i = 0; i < L; i++)
      for (let j = 0; j < L; j++) {
        if (i === j || wins[i]![j]! === 0) continue;
        const diff = theta[j]! - theta[i]!;
        let logP: number;
        if (diff > 20) logP = -diff;
        else if (diff < -20) logP = 0;
        else logP = -Math.log(1 + Math.exp(diff));
        ll += wins[i]![j]! * logP;
      }
    return ll;
  };
  const logPrior = (theta: number[]): number =>
    (-0.5 * sum(theta.map((t) => t * t))) / priorVar;
  const logPosterior = (theta: number[]): number => logLikelihood(theta) + logPrior(theta);

  let thetaCurrent = new Array<number>(L).fill(0);
  let logPostCurrent = logPosterior(thetaCurrent);
  const samples: number[][] = [];
  let proposalStd = 0.1;
  let accepted = 0;

  const totalIter = nSamples + burnin;
  for (let iteration = 0; iteration < totalIter; iteration++) {
    const thetaProposed = thetaCurrent.map((t) => t + rng.normal(0, proposalStd));
    const logPostProposed = logPosterior(thetaProposed);
    const logAccept = logPostProposed - logPostCurrent;
    if (Math.log(rng.random()) < Math.min(logAccept, 0)) {
      thetaCurrent = thetaProposed;
      logPostCurrent = logPostProposed;
      accepted += 1;
    }
    if (iteration >= burnin) samples.push(thetaCurrent.slice());
    if (iteration > 0 && iteration % 500 === 0 && iteration < burnin) {
      const acceptRate = accepted / iteration;
      if (acceptRate < 0.2) proposalStd *= 0.8;
      else if (acceptRate > 0.5) proposalStd *= 1.2;
    }
  }

  let scores = new Array<number>(L).fill(0);
  for (const s of samples) for (let i = 0; i < L; i++) scores[i]! += s[i]!;
  for (let i = 0; i < L; i++) scores[i]! /= samples.length;
  scores = averageEquivalentScores(scores, tensor);
  return { ranking: rankScores(scores, method), scores };
}
