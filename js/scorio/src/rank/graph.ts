/**
 * Graph-based ranking methods. Port of `scorio/rank/graph.py`.
 *
 * Pairwise win probabilities are turned into a graph / Markov chain and ranked
 * by a stationary distribution (`pagerank`, `alpharank`), a Perron-style
 * spectral score (`spectral`), or a zero-sum equilibrium (`nash`).
 */

import { matVec, l1Diff } from "./internal/linalg.js";
import { solveMaximinStrategy } from "./internal/lp.js";
import { validatePositiveFloat, validatePositiveInt } from "./internal/validate.js";
import { rankScores } from "./internal/rankScores.js";
import {
  buildPairwiseCounts,
  shape3,
  validateInput,
  zeros2,
  type Tensor3,
  type TensorInput,
} from "./internal/tensor.js";
import type { BaseRankOptions, RankResult } from "./internal/result.js";

const sum = (a: readonly number[]): number => a.reduce((s, v) => s + v, 0);

/** Empirical tied-split pairwise win-probability matrix `P̂`, with `P̂[i][i]=0.5`. */
function pairwiseWinProbabilities(R: Tensor3): number[][] {
  const { wins, ties } = buildPairwiseCounts(R);
  const L = wins.length;
  const P = Array.from({ length: L }, () => new Array<number>(L).fill(0.5));
  for (let i = 0; i < L; i++) {
    for (let j = 0; j < L; j++) {
      if (i === j) continue;
      const total = wins[i]![j]! + wins[j]![i]! + ties[i]![j]!;
      if (total > 0) P[i]![j] = (wins[i]![j]! + 0.5 * ties[i]![j]!) / total;
    }
  }
  return P;
}

/** Stationary distribution of a row-stochastic `C` via `π ← πC`. */
function powerStationaryRowStochastic(
  C: number[][],
  maxIter: number,
  tol: number,
): number[] {
  const n = C.length;
  if (n === 0) return [];
  let pi = new Array<number>(n).fill(1 / n);
  for (let it = 0; it < maxIter; it++) {
    const piNew = new Array<number>(n).fill(0);
    for (let i = 0; i < n; i++) {
      const p = pi[i]!;
      if (p === 0) continue;
      const row = C[i]!;
      for (let j = 0; j < n; j++) piNew[j]! += p * row[j]!;
    }
    const s = sum(piNew);
    if (s <= 0 || piNew.some((v) => !Number.isFinite(v))) {
      return new Array<number>(n).fill(1 / n);
    }
    for (let j = 0; j < n; j++) piNew[j]! /= s;
    if (l1Diff(piNew, pi) < tol) return piNew;
    pi = piNew;
  }
  return pi;
}

/** Options for {@link pagerank}. */
export interface PageRankOptions extends BaseRankOptions {
  damping?: number;
  maxIter?: number;
  tol?: number;
  /** Teleportation vector of length `L` (nonnegative). Default uniform. */
  teleport?: readonly number[];
}

/** Rank models with PageRank on the pairwise win-probability graph. */
export function pagerank(R: TensorInput, options: PageRankOptions = {}): RankResult {
  const method = options.method ?? "competition";
  const damping = options.damping ?? 0.85;
  if (!Number.isFinite(damping) || !(damping > 0 && damping < 1)) {
    throw new Error("damping must be in (0, 1)");
  }
  const maxIter = validatePositiveInt("max_iter", options.maxIter ?? 100);
  const tol = validatePositiveFloat("tol", options.tol ?? 1e-6);
  const tensor = validateInput(R);
  const [L] = shape3(tensor);

  let e: number[];
  if (options.teleport === undefined) {
    e = new Array<number>(L).fill(1 / L);
  } else {
    e = options.teleport.slice();
    if (e.length !== L) throw new Error(`teleport must have shape (L=${L},)`);
    if (e.some((v) => !Number.isFinite(v))) throw new Error("teleport must contain finite values");
    if (e.some((v) => v < 0)) throw new Error("teleport must be nonnegative");
    const s = sum(e);
    if (s <= 0) throw new Error("teleport must sum to a positive value");
    e = e.map((v) => v / s);
  }

  const Phat = pairwiseWinProbabilities(tensor);
  const W = Phat.map((row) => row.slice());
  for (let i = 0; i < L; i++) W[i]![i] = 0;

  // Column-stochastic transition matrix P[i][j] = P(to i from j).
  const P = zeros2(L, L);
  for (let j = 0; j < L; j++) {
    let colSum = 0;
    for (let i = 0; i < L; i++) colSum += W[i]![j]!;
    for (let i = 0; i < L; i++) P[i]![j] = colSum > 0 ? W[i]![j]! / colSum : 1 / L;
  }

  let r = new Array<number>(L).fill(1 / L);
  for (let it = 0; it < maxIter; it++) {
    const Pr = matVec(P, r);
    const rNew = Pr.map((v, i) => damping * v + (1 - damping) * e[i]!);
    if (l1Diff(rNew, r) < tol) {
      r = rNew;
      break;
    }
    r = rNew;
  }
  return { ranking: rankScores(r, method), scores: r };
}

/** Options for {@link spectral}. */
export interface SpectralOptions extends BaseRankOptions {
  maxIter?: number;
  tol?: number;
}

/** Rank models by spectral centrality of pairwise win probabilities. */
export function spectral(R: TensorInput, options: SpectralOptions = {}): RankResult {
  const method = options.method ?? "competition";
  const maxIter = validatePositiveInt("max_iter", options.maxIter ?? 10000);
  const tol = validatePositiveFloat("tol", options.tol ?? 1e-12);
  const tensor = validateInput(R);
  const [L] = shape3(tensor);

  const Phat = pairwiseWinProbabilities(tensor);
  const W = Phat.map((row) => row.slice());
  for (let i = 0; i < L; i++) W[i]![i] = 0;
  for (let i = 0; i < L; i++) W[i]![i] = sum(W[i]!);

  let v = new Array<number>(L).fill(1 / L);
  for (let it = 0; it < maxIter; it++) {
    const vNew = matVec(W, v);
    const s = sum(vNew);
    if (s <= 0 || vNew.some((x) => !Number.isFinite(x))) {
      const uniform = new Array<number>(L).fill(1 / L);
      return { ranking: rankScores(uniform, method), scores: uniform };
    }
    for (let i = 0; i < L; i++) vNew[i]! /= s;
    if (l1Diff(vNew, v) < tol) {
      return { ranking: rankScores(vNew, method), scores: vNew };
    }
    v = vNew;
  }
  return { ranking: rankScores(v, method), scores: v };
}

/** Options for {@link alpharank}. */
export interface AlphaRankOptions extends BaseRankOptions {
  alpha?: number;
  populationSize?: number;
  maxIter?: number;
  tol?: number;
}

/** Rank models with single-population alpha-Rank. */
export function alpharank(R: TensorInput, options: AlphaRankOptions = {}): RankResult {
  const method = options.method ?? "competition";
  const alpha = options.alpha ?? 1;
  const maxIter = validatePositiveInt("max_iter", options.maxIter ?? 100000);
  const tol = validatePositiveFloat("tol", options.tol ?? 1e-12);
  const m = validatePositiveInt("population_size", options.populationSize ?? 50, 2);
  if (!Number.isFinite(alpha) || alpha < 0) throw new Error("alpha must be >= 0");

  const tensor = validateInput(R);
  const [L] = shape3(tensor);
  const Phat = pairwiseWinProbabilities(tensor);
  const payoffSum = 1;
  const eta = 1 / (L - 1);

  const rho = (payoffRs: number): number => {
    const u = alpha * (m / (m - 1)) * (payoffRs - 0.5 * payoffSum);
    if (Math.abs(u) < 1e-14) return 1 / m;
    if (u > 50) return 1;
    if (u < -50) return 0;
    const num = -Math.expm1(-u);
    const den = -Math.expm1(-m * u);
    if (den === 0) return 1 / m;
    const out = num / den;
    return out < 0 ? 0 : out > 1 ? 1 : out;
  };

  const C = zeros2(L, L);
  for (let resident = 0; resident < L; resident++) {
    for (let r = 0; r < L; r++) {
      if (r === resident) continue;
      C[resident]![r] = eta * rho(Phat[r]![resident]!);
    }
    C[resident]![resident] = 1 - sum(C[resident]!);
  }

  const pi = powerStationaryRowStochastic(C, maxIter, tol).map((v) => Math.max(v, 0));
  const total = sum(pi);
  const scores =
    total > 0 ? pi.map((v) => v / total) : new Array<number>(L).fill(1 / L);
  return { ranking: rankScores(scores, method), scores };
}

/** Options for {@link nash}. */
export interface NashOptions extends BaseRankOptions {
  /** Which per-model summary to rank by. Default `"vs_equilibrium"`. */
  scoreType?: "vs_equilibrium" | "equilibrium" | "advantage_vs_equilibrium";
}

/** Rank models via a Nash equilibrium on the zero-sum meta-game. */
export function nash(R: TensorInput, options: NashOptions = {}): RankResult {
  const method = options.method ?? "competition";
  const scoreType = options.scoreType ?? "vs_equilibrium";
  if (
    scoreType !== "vs_equilibrium" &&
    scoreType !== "equilibrium" &&
    scoreType !== "advantage_vs_equilibrium"
  ) {
    throw new Error(
      'score_type must be one of "vs_equilibrium", "equilibrium", "advantage_vs_equilibrium"',
    );
  }
  const tensor = validateInput(R);
  const [L] = shape3(tensor);
  const Phat = pairwiseWinProbabilities(tensor);

  const A = zeros2(L, L);
  for (let i = 0; i < L; i++)
    for (let j = 0; j < L; j++) A[i]![j] = i === j ? 0 : 2 * Phat[i]![j]! - 1;

  const isZero = A.every((row) => row.every((v) => Math.abs(v) <= 1e-14));
  let equilibrium: number[];
  if (isZero) {
    equilibrium = new Array<number>(L).fill(1 / L);
  } else {
    const x = solveMaximinStrategy(A);
    equilibrium = x ?? new Array<number>(L).fill(1 / L);
  }

  let scores: number[];
  if (scoreType === "equilibrium") scores = equilibrium;
  else if (scoreType === "advantage_vs_equilibrium") scores = matVec(A, equilibrium);
  else scores = matVec(Phat, equilibrium);

  return { ranking: rankScores(scores, method), scores };
}
