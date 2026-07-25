/**
 * Avg@N family — weighted average with Bayes@N uncertainty rescaled to the
 * avg scale. Port of `scorio/eval/avg.py`.
 */

import { bayes } from "./bayes.js";
import { normalCredibleInterval, type Bounds } from "./internal/ci.js";
import {
  asMatrix,
  validateBinary,
  validateMatrixRange,
  type Matrix,
} from "./internal/validate.js";

function weightedMean(R: readonly (readonly number[])[], wv?: number[]): number {
  let total = 0;
  let count = 0;
  for (const row of R) {
    for (const v of row) {
      total += wv === undefined ? v : wv[v]!;
      count += 1;
    }
  }
  return total / count;
}

/**
 * Avg@N with a Bayesian uncertainty estimate (uniform prior, no `R0`).
 *
 * @param R `M x N` integer matrix with entries in `{0,...,C}`.
 * @param w Optional length-`(C+1)` weight vector. If omitted, `R` must be
 *          binary and `[0, 1]` is used.
 * @returns `[a, sigma]` where `a` is the (weighted) average.
 */
export function avg(R: Matrix, w?: readonly number[] | null): [number, number] {
  const Rm = asMatrix(R);
  let wv: number[];
  if (w == null) {
    validateBinary(Rm);
    wv = [0.0, 1.0];
  } else {
    wv = w.map(Number);
  }
  const N = Rm[0]!.length;
  const C = wv.length - 1;
  if (N <= 0) {
    throw new Error("R must have at least one column (N>=1)");
  }
  validateMatrixRange(Rm, 0, C, "R");

  const [, sigmaBayes] = bayes(Rm, wv);
  const T = 1 + C + N; // D = 0
  const sigmaAvg = (T / N) * sigmaBayes;
  return [weightedMean(Rm, wv), sigmaAvg];
}

/**
 * Avg@N with Bayesian `sigma` and a normal-approximation credible interval.
 *
 * @returns `[a, sigma, lo, hi]`.
 */
export function avgCi(
  R: Matrix,
  w?: readonly number[] | null,
  confidence = 0.95,
  bounds?: Bounds | null,
): [number, number, number, number] {
  const [a, sigma] = avg(R, w);
  const [lo, hi] = normalCredibleInterval(a, sigma, confidence, true, bounds);
  return [a, sigma, lo, hi];
}
