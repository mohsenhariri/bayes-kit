/**
 * Bayes@N family — Bayesian posterior mean and uncertainty for repeated
 * categorical outcomes with an optional prior. Port of `scorio/eval/bayes.py`.
 *
 * Reference: Hariri et al. (2026), "Don't Pass@k: A Bayesian Framework for
 * Large Language Model Evaluation," ICLR 2026.
 */

import { normalCredibleInterval, type Bounds } from "./internal/ci.js";
import {
  asMatrix,
  asPriorMatrix,
  rowBincount,
  validateMatrixRange,
  type Matrix,
} from "./internal/validate.js";

function detectBinaryWeights(R: readonly (readonly number[])[]): number[] {
  const seen = new Set<number>();
  for (const row of R) for (const v of row) seen.add(v);
  const isBinary = seen.size <= 2 && [...seen].every((v) => v === 0 || v === 1);
  if (!isBinary) {
    const vals = [...seen].sort((a, b) => a - b).join(", ");
    throw new Error(
      `R contains more than 2 unique values (${vals}), so weight vector 'w' must be provided. ` +
        `Please specify a weight vector of length ${seen.size} to map each category to a score.`,
    );
  }
  return [0.0, 1.0];
}

/**
 * Bayes@N posterior mean (`mu`) and standard deviation (`sigma`).
 *
 * @param R  `M x N` integer matrix with entries in `{0,...,C}`.
 * @param w  Optional length-`(C+1)` weight vector. If omitted, `R` must be
 *           binary and `[0, 1]` is used.
 * @param R0 Optional `M x D` matrix of prior outcomes per row.
 * @returns `[mu, sigma]`.
 */
export function bayes(
  R: Matrix,
  w?: readonly number[] | null,
  R0?: Matrix | null,
): [number, number] {
  const Rm = asMatrix(R);
  const wv = w == null ? detectBinaryWeights(Rm) : w.map(Number);
  const M = Rm.length;
  const N = Rm[0]!.length;
  const C = wv.length - 1;

  let R0m: number[][];
  let D: number;
  if (R0 == null) {
    D = 0;
    R0m = Rm.map(() => []);
  } else {
    R0m = asPriorMatrix(R0, M);
    if (R0m.length !== M) {
      throw new Error("R0 must have the same number of rows (M) as R.");
    }
    D = R0m[0]!.length;
  }

  validateMatrixRange(Rm, 0, C, "R");
  validateMatrixRange(R0m, 0, C, "R0");

  const T = 1 + C + D + N;

  const nCounts = rowBincount(Rm, C + 1);
  const n0Counts = rowBincount(R0m, C + 1).map((row) => row.map((c) => c + 1));

  const deltaW = wv.map((wj) => wj - wv[0]!);

  let muAccum = 0;
  let sigmaAccum = 0;
  for (let a = 0; a < M; a++) {
    let dot = 0; // sum_j (nu/T) * deltaW
    let sq = 0; // sum_j (nu/T) * deltaW^2
    for (let j = 0; j <= C; j++) {
      const nu = nCounts[a]![j]! + n0Counts[a]![j]!;
      muAccum += nu * deltaW[j]!;
      const nuOverT = nu / T;
      dot += nuOverT * deltaW[j]!;
      sq += nuOverT * deltaW[j]! * deltaW[j]!;
    }
    sigmaAccum += sq - dot * dot;
  }

  const mu = wv[0]! + muAccum / (M * T);
  const sigma = Math.sqrt(sigmaAccum / (M * M * (T + 1)));
  return [mu, sigma];
}

/**
 * Bayes@N posterior mean, standard deviation, and credible interval.
 *
 * @returns `[mu, sigma, lo, hi]`.
 */
export function bayesCi(
  R: Matrix,
  w?: readonly number[] | null,
  R0?: Matrix | null,
  confidence = 0.95,
  bounds?: Bounds | null,
): [number, number, number, number] {
  const [mu, sigma] = bayes(R, w, R0);
  const [lo, hi] = normalCredibleInterval(mu, sigma, confidence, true, bounds);
  return [mu, sigma, lo, hi];
}
