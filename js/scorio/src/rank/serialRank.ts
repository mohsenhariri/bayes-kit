/**
 * SerialRank — spectral ranking by seriation (Fogel, d'Aspremont & Vojnovic).
 * Port of `scorio/rank/serial_rank.py`.
 *
 * Builds a similarity matrix from pairwise comparisons, forms its graph
 * Laplacian, and orders models by a Fiedler vector (the eigenvector of the
 * second-smallest Laplacian eigenvalue).
 */

import { eigSymmetric, matMul, transpose } from "./internal/linalg.js";
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

function comparisonMatrix(
  wins: number[][],
  ties: number[][],
  comparison: string,
): number[][] {
  const L = wins.length;
  const C = zeros2(L, L);
  if (comparison === "prob_diff" || comparison === "fractional") {
    for (let i = 0; i < L; i++)
      for (let j = 0; j < L; j++) {
        if (i === j) continue;
        const total = wins[i]![j]! + wins[j]![i]! + ties[i]![j]!;
        if (total > 0) C[i]![j] = (wins[i]![j]! - wins[j]![i]!) / total;
      }
    return C;
  }
  if (comparison === "sign" || comparison === "majority") {
    for (let i = 0; i < L; i++)
      for (let j = 0; j < L; j++) {
        if (i === j) continue;
        C[i]![j] = Math.sign(wins[i]![j]! - wins[j]![i]!);
      }
    return C;
  }
  throw new Error('comparison must be "prob_diff" or "sign"');
}

function modelAccuracy(R: Tensor3): number[] {
  const [, , N] = shape3(R);
  return R.map((mat) => {
    let s = 0;
    let n = 0;
    for (const row of mat) for (const v of row) s += v;
    n = mat.length * N;
    return s / n;
  });
}

/** Lexicographic comparison key for orientation `(upsets, weightedUpsets, -corr)`. */
function orientationKey(scores: number[], C: number[][]): [number, number, number] {
  const n = scores.length;
  const cVals: number[] = [];
  const preds: number[] = [];
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const c = C[i]![j]!;
      if (c === 0) continue;
      cVals.push(c);
      preds.push(Math.sign(scores[i]! - scores[j]!));
    }
  }
  if (cVals.length === 0) return [0, 0, 0];
  let upsets = 0;
  let wUpsets = 0;
  let corr = 0;
  for (let k = 0; k < cVals.length; k++) {
    const c = cVals[k]!;
    const pred = preds[k]!;
    const disagree = pred === 0 || pred * c < 0;
    if (disagree) {
      upsets += 1;
      wUpsets += Math.abs(c);
    }
    corr += pred * c;
  }
  return [upsets, wUpsets, -corr];
}

function keyLeq(a: [number, number, number], b: [number, number, number]): boolean {
  if (a[0] !== b[0]) return a[0] < b[0];
  if (a[1] !== b[1]) return a[1] < b[1];
  return a[2] <= b[2];
}

function std(a: readonly number[]): number {
  const m = sum(a) / a.length;
  return Math.sqrt(sum(a.map((v) => (v - m) ** 2)) / a.length);
}

/** Options for {@link serialRank}. */
export interface SerialRankOptions extends BaseRankOptions {
  /** `"prob_diff"` (default) or `"sign"` aggregation of comparisons. */
  comparison?: "prob_diff" | "sign";
}

/** Rank models with SerialRank spectral seriation. */
export function serialRank(R: TensorInput, options: SerialRankOptions = {}): RankResult {
  const method = options.method ?? "competition";
  const comparison = options.comparison ?? "prob_diff";
  const tensor = validateInput(R);
  const [L] = shape3(tensor);
  const { wins, ties } = buildPairwiseCounts(tensor);
  const C = comparisonMatrix(wins, ties, comparison);

  // S = ½ (L·11ᵀ + C Cᵀ); Laplacian Ls = diag(S1) - S.
  const CCt = matMul(C, transpose(C));
  const S = zeros2(L, L);
  for (let i = 0; i < L; i++)
    for (let j = 0; j < L; j++) S[i]![j] = 0.5 * (L + CCt[i]![j]!);
  const Ls = zeros2(L, L);
  for (let i = 0; i < L; i++) {
    const d = sum(S[i]!);
    for (let j = 0; j < L; j++) Ls[i]![j] = (i === j ? d : 0) - S[i]![j]!;
  }

  const { values, vectors } = eigSymmetric(Ls);
  // Fiedler vector = eigenvector of the second-smallest eigenvalue.
  let v = vectors.map((row) => row[1]!);
  let unique: boolean;
  if (L === 2) {
    unique = true;
  } else {
    const scale = Math.max(1, Math.max(...values.map((w) => Math.abs(w))));
    const eigengap = values[2]! - values[1]!;
    unique = Number.isFinite(eigengap) && eigengap > 1e-10 * scale;
  }

  const allFinite = v.every((x) => Number.isFinite(x));
  // np.allclose(v, v[0]) semantics (rtol=1e-5, atol=1e-8): treat a
  // near-constant Fiedler vector as degenerate, not just an exactly-equal one.
  const allEqual = v.every(
    (x) => Math.abs(x - v[0]!) <= 1e-8 + 1e-5 * Math.abs(v[0]!),
  );
  if (!unique || !allFinite || allEqual) {
    const scores = modelAccuracy(tensor);
    return { ranking: rankScores(scores, method), scores };
  }

  const keyPos = orientationKey(v, C);
  const keyNeg = orientationKey(v.map((x) => -x), C);
  let scores = keyLeq(keyPos, keyNeg) ? v : v.map((x) => -x);

  if (std(scores) < 1e-12) scores = modelAccuracy(tensor);
  return { ranking: rankScores(scores, method), scores };
}
