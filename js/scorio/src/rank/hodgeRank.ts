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

/** Options for {@link hodgeRank}. */
export interface HodgeRankOptions extends BaseRankOptions {
  /** `"binary"` (default) `P(j>i)-P(i>j)`, or `"log_odds"`. */
  pairwiseStat?: "binary" | "log_odds";
  /** Edge weights: `"total"` (default), `"decisive"`, or `"uniform"`. */
  weightMethod?: "total" | "decisive" | "uniform";
  /** Additive smoothing for `pairwiseStat="log_odds"`. Default `0.5`. */
  epsilon?: number;
}

/** Rank models with ℓ₂ HodgeRank on the pairwise-comparison graph. */
export function hodgeRank(R: TensorInput, options: HodgeRankOptions = {}): RankResult {
  const method = options.method ?? "competition";
  const pairwiseStat = options.pairwiseStat ?? "binary";
  const weightMethod = options.weightMethod ?? "total";
  const epsilon = options.epsilon ?? 0.5;

  const tensor = validateInput(R);
  const [L] = shape3(tensor);
  const { wins, ties } = buildPairwiseCounts(tensor);

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
    return { ranking: rankScores(scores, method), scores };
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

  // The weighted Laplacian is singular (constant null space) but `div` is
  // orthogonal to it, so the minimum-norm solution is unique. NumPy's
  // `pinv` default cutoff is 1e-15·σ_max; we use a slightly looser 1e-12 to
  // robustly zero the (exactly-zero) null eigenvalue given the Jacobi
  // eigensolver's precision, which gives numpy-identical scores on real graphs.
  const pinv = pinvSymmetric(Lap, 1e-12);
  const scores = matVec(pinv, div).map((v) => -v);
  return { ranking: rankScores(scores, method), scores };
}
