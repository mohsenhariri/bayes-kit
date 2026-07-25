/**
 * HodgeRank — statistical ranking via combinatorial Hodge theory
 * (Jiang, Lim, Yao & Ye, 2009). Port of `scorio/rank/hodge_rank.py`.
 *
 * Treats aggregated pairwise outcomes as an edge flow and finds global
 * potentials whose gradient best matches that flow in weighted least squares,
 * via the minimum-norm solution of the weighted graph Laplacian normal
 * equations (a Moore-Penrose pseudoinverse).
 */

import { matVec, pinvSymmetric } from "./internal/linalg.js";
import { rankScores } from "./internal/rankScores.js";
import {
  buildPairwiseCounts,
  shape3,
  validateInput,
  zeros2,
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

/** Options for {@link hodgeRank}. */
export interface HodgeRankOptions extends BaseRankOptions {
  /** `"binary"` (default) `P(j>i)-P(i>j)`, or `"log_odds"`. */
  pairwiseStat?: "binary" | "log_odds";
  /** Edge weights: `"total"` (default), `"decisive"`, or `"uniform"`. */
  weightMethod?: "total" | "decisive" | "uniform";
  /** Additive smoothing for `pairwiseStat="log_odds"`. Default `0.5`. */
  epsilon?: number;
  /** Include weighted least-squares residual diagnostics. Default `false`. */
  returnDiagnostics?: boolean;
}

/** Weighted residual diagnostics for the fitted Hodge potential. */
export interface HodgeDiagnostics {
  residualL2: number;
  relativeResidualL2: number;
}

/** HodgeRank result; `diagnostics` is present when requested. */
export interface HodgeRankResult extends RankResult {
  diagnostics?: HodgeDiagnostics;
}

/** Rank models with ℓ₂ HodgeRank on the pairwise-comparison graph. */
export function hodgeRank(R: TensorInput, options: HodgeRankOptions = {}): HodgeRankResult {
  const method = defaultIfUndefined(options.method, "competition");
  const tensor = validateInput(R);
  const [L] = shape3(tensor);
  const { wins, ties } = buildPairwiseCounts(tensor);

  const pairwiseStat = String(
    defaultIfUndefined(options.pairwiseStat, "binary"),
  );

  const total = zeros2(L, L);
  for (let i = 0; i < L; i++)
    for (let j = 0; j < L; j++) total[i]![j] = wins[i]![j]! + wins[j]![i]! + ties[i]![j]!;

  // Skew-symmetric edge flow Y.
  const Y = zeros2(L, L);
  if (pairwiseStat === "binary") {
    for (let i = 0; i < L; i++)
      for (let j = 0; j < L; j++) {
        if (i === j) continue;
        if (total[i]![j]! > 0) Y[i]![j] = (wins[j]![i]! - wins[i]![j]!) / total[i]![j]!;
      }
  } else if (pairwiseStat === "log_odds") {
    const epsilon = pythonFloat(
      defaultIfUndefined(options.epsilon, 0.5),
      "epsilon must be > 0 for log-odds smoothing",
    );
    if (!Number.isFinite(epsilon) || epsilon <= 0) {
      throw new Error("epsilon must be > 0 for log-odds smoothing");
    }
    for (let i = 0; i < L; i++)
      for (let j = 0; j < L; j++) {
        if (i === j) continue;
        if (total[i]![j]! > 0) {
          const num = total[i]![j]! - wins[i]![j]! + epsilon;
          const den = total[i]![j]! - wins[j]![i]! + epsilon;
          Y[i]![j] = Math.log(num / den);
        }
      }
  } else {
    throw new Error('pairwise_stat must be one of: "binary", "log_odds"');
  }

  // Symmetric nonnegative edge weights.
  const weightMethod = String(
    defaultIfUndefined(options.weightMethod, "total"),
  );
  const w = zeros2(L, L);
  for (let i = 0; i < L; i++)
    for (let j = 0; j < L; j++) {
      if (i === j) continue;
      if (weightMethod === "total") w[i]![j] = total[i]![j]!;
      else if (weightMethod === "decisive") w[i]![j] = wins[i]![j]! + wins[j]![i]!;
      else if (weightMethod === "uniform") w[i]![j] = total[i]![j]! > 0 ? 1 : 0;
      else throw new Error('weight_method must be one of: "total", "decisive", "uniform"');
    }

  let anyPositive = false;
  for (let i = 0; i < L; i++) for (let j = 0; j < L; j++) if (w[i]![j]! > 0) anyPositive = true;
  if (!anyPositive) {
    const scores = new Array<number>(L).fill(1 / L);
    const result: HodgeRankResult = { ranking: rankScores(scores, method), scores };
    if (pythonTruthy(options.returnDiagnostics)) {
      result.diagnostics = { residualL2: 0, relativeResidualL2: 0 };
    }
    return result;
  }

  // Weighted Laplacian and divergence.
  const Lap = zeros2(L, L);
  for (let i = 0; i < L; i++) {
    let deg = 0;
    for (let j = 0; j < L; j++) {
      if (i === j) continue;
      Lap[i]![j] = -w[i]![j]!;
      deg += w[i]![j]!;
    }
    Lap[i]![i] = deg;
  }
  const div = new Array<number>(L).fill(0);
  for (let i = 0; i < L; i++) {
    let s = 0;
    for (let j = 0; j < L; j++) s += w[i]![j]! * Y[i]![j]!;
    div[i] = s;
  }

  // Match NumPy's default `pinv` singular-value cutoff.
  const pinv = pinvSymmetric(Lap, 1e-15);
  const scores = matVec(pinv, div).map((v) => -v);
  const result: HodgeRankResult = { ranking: rankScores(scores, method), scores };
  if (pythonTruthy(options.returnDiagnostics)) {
    let residualSquared = 0;
    let flowSquared = 0;
    for (let i = 0; i < L; i++) {
      for (let j = i + 1; j < L; j++) {
        if (w[i]![j]! <= 0) continue;
        const gradient = scores[j]! - scores[i]!;
        const residual = Y[i]![j]! - gradient;
        residualSquared += w[i]![j]! * residual * residual;
        flowSquared += w[i]![j]! * Y[i]![j]! * Y[i]![j]!;
      }
    }
    const residualL2 = Math.sqrt(residualSquared);
    const denominator = Math.sqrt(flowSquared);
    result.diagnostics = {
      residualL2,
      relativeResidualL2: denominator > 0 ? residualL2 / denominator : 0,
    };
  }
  return result;
}
