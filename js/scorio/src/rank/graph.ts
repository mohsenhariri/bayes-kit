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

function pythonTruthy(value: unknown): boolean {
  if (value == null) return false;
  if (typeof value === "boolean") return value;
  if (typeof value === "number") return value !== 0;
  if (typeof value === "bigint") return value !== 0n;
  if (typeof value === "string" || Array.isArray(value)) return value.length > 0;
  return true;
}

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
  teleport?: readonly number[] | null;
}

/** Rank models with PageRank on the pairwise win-probability graph. */
export function pagerank(R: TensorInput, options: PageRankOptions = {}): RankResult {
  const method = defaultIfUndefined(options.method, "competition");
  const damping = pythonFloat(
    defaultIfUndefined(options.damping, 0.85),
    "damping must be in (0, 1)",
  );
  if (!Number.isFinite(damping) || !(damping > 0 && damping < 1)) {
    throw new Error("damping must be in (0, 1)");
  }
  const maxIter = validatePositiveInt(
    "max_iter",
    defaultIfUndefined(options.maxIter, 100),
  );
  const tol = validatePositiveFloat(
    "tol",
    pythonFloat(
      defaultIfUndefined(options.tol, 1e-6),
      "tol must be a positive finite scalar",
    ),
  );
  const tensor = validateInput(R);
  const [L] = shape3(tensor);

  let e: number[];
  if (options.teleport == null) {
    e = new Array<number>(L).fill(1 / L);
  } else {
    if (!Array.isArray(options.teleport)) {
      throw new Error(`teleport must have shape (L=${L},)`);
    }
    // NumPy applies dtype=float coercion before checking the vector shape.
    e = (options.teleport as readonly unknown[]).map((value) =>
      pythonFloat(value, "teleport must contain finite values"),
    );
    if (e.length !== L) throw new Error(`teleport must have shape (L=${L},)`);
    if (e.some((v) => !Number.isFinite(v))) throw new Error("teleport must contain finite values");
    if (e.some((v) => v < 0)) throw new Error("teleport must be nonnegative");
    const s = sum(e);
    if (s <= 0) throw new Error("teleport must sum to a positive value");
    e = e.map((v) => v / s);
  }

  const Phat = pairwiseWinProbabilities(tensor);
  const Q = Phat.map((row) => row.slice());
  for (let i = 0; i < L; i++) Q[i]![i] = 0;
  for (let i = 0; i < L; i++) Q[i]![i] = sum(Q[i]!);
  for (let i = 0; i < L; i++)
    for (let j = 0; j < L; j++) Q[i]![j]! /= L - 1;

  let r = new Array<number>(L).fill(1 / L);
  for (let it = 0; it < maxIter; it++) {
    const Pr = matVec(Q, r);
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
  const method = defaultIfUndefined(options.method, "competition");
  const maxIter = validatePositiveInt(
    "max_iter",
    defaultIfUndefined(options.maxIter, 10000),
  );
  const tol = validatePositiveFloat(
    "tol",
    pythonFloat(
      defaultIfUndefined(options.tol, 1e-12),
      "tol must be a positive finite scalar",
    ),
  );
  const tensor = validateInput(R);
  const [L] = shape3(tensor);

  const { wins, ties } = buildPairwiseCounts(tensor);
  const shiftedA = zeros2(L, L);
  for (let i = 0; i < L; i++) {
    for (let j = 0; j < L; j++) {
      if (i === j) {
        shiftedA[i]![j] = 1;
        continue;
      }
      const total = wins[i]![j]! + wins[j]![i]! + ties[i]![j]!;
      shiftedA[i]![j] = (wins[i]![j]! + 0.5 * ties[i]![j]! + 1) / (total + 2);
    }
  }

  let v = new Array<number>(L).fill(1 / L);
  for (let it = 0; it < maxIter; it++) {
    const vNew = matVec(shiftedA, v);
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
  const method = defaultIfUndefined(options.method, "competition");
  const maxIter = validatePositiveInt(
    "max_iter",
    defaultIfUndefined(options.maxIter, 100000),
  );
  const tol = validatePositiveFloat(
    "tol",
    pythonFloat(
      defaultIfUndefined(options.tol, 1e-12),
      "tol must be a positive finite scalar",
    ),
  );
  const m = validatePositiveInt(
    "population_size",
    defaultIfUndefined(options.populationSize, 50),
    2,
  );
  const alpha = pythonFloat(
    defaultIfUndefined(options.alpha, 1),
    "alpha must be >= 0",
  );
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
  /** Retained for Python API parity; unused by the LP solver. Default `100`. */
  nIter?: number;
  /** Retained for Python API parity; unused by the LP solver. Default `0.1`. */
  temperature?: number;
  /** Currently only `"lp"` is supported. */
  solver?: string;
  /** Which per-model summary to rank by. Default `"vs_equilibrium"`. */
  scoreType?: "vs_equilibrium" | "equilibrium" | "advantage_vs_equilibrium";
  /** Include the equilibrium mixture in the result. Default `false`. */
  returnEquilibrium?: boolean;
}

/** Nash result; `equilibrium` is present when `returnEquilibrium` is true. */
export interface NashResult extends RankResult {
  equilibrium?: number[];
}

/** Rank models via a Nash equilibrium on the zero-sum meta-game. */
export function nash(R: TensorInput, options: NashOptions = {}): NashResult {
  const method = defaultIfUndefined(options.method, "competition");
  validatePositiveInt("n_iter", defaultIfUndefined(options.nIter, 100));
  const temperature = pythonFloat(
    defaultIfUndefined(options.temperature, 0.1),
    "temperature must be a positive finite scalar",
  );
  if (!Number.isFinite(temperature) || temperature <= 0) {
    throw new Error("temperature must be a positive finite scalar");
  }
  const tensor = validateInput(R);
  const [L] = shape3(tensor);

  const solver = String(defaultIfUndefined(options.solver, "lp"));
  if (solver !== "lp") throw new Error('solver must be "lp"');
  const scoreType = String(
    defaultIfUndefined(options.scoreType, "vs_equilibrium"),
  );
  if (
    scoreType !== "vs_equilibrium" &&
    scoreType !== "equilibrium" &&
    scoreType !== "advantage_vs_equilibrium"
  ) {
    throw new Error(
      'score_type must be one of "vs_equilibrium", "equilibrium", "advantage_vs_equilibrium"',
    );
  }
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
    if (x === null || x.some((value) => !Number.isFinite(value))) {
      throw new Error("Nash equilibrium linear program failed: solver did not find a finite solution");
    }

    // For complete binary response events A[i,j] is the accuracy difference.
    // The maximin face can therefore contain many arbitrary LP vertices.  The
    // Python contract selects the label-invariant mixture over all maximum-
    // accuracy models instead.
    const totals = tensor.map((model) =>
      model.reduce(
        (modelTotal, question) =>
          modelTotal + question.reduce((questionTotal, value) => questionTotal + value, 0),
        0,
      ),
    );
    const maximum = Math.max(...totals);
    const maximizers = totals.map((value) => value === maximum);
    const count = maximizers.reduce((total, value) => total + (value ? 1 : 0), 0);
    equilibrium = maximizers.map((value) => (value ? 1 / count : 0));
  }

  let scores: number[];
  if (scoreType === "equilibrium") scores = equilibrium;
  else if (scoreType === "advantage_vs_equilibrium") scores = matVec(A, equilibrium);
  else scores = matVec(Phat, equilibrium);

  const result: NashResult = { ranking: rankScores(scores, method), scores };
  if (pythonTruthy(options.returnEquilibrium)) result.equilibrium = equilibrium;
  return result;
}
