/**
 * Credible-interval helpers, mirroring the CI utilities in
 * `scorio/eval/utils.py`.
 */

import { ndtri } from "./math.js";

/** Optional `[lo, hi]` clipping bounds for a credible interval. */
export type Bounds = readonly [number, number];

/**
 * Standard-normal z value for a desired confidence level.
 *
 * @param confidence Confidence in (0, 1), e.g. 0.95.
 * @param twoSided   Two-sided central interval (default) vs. one-sided tail.
 */
export function zValue(confidence: number, twoSided = true): number {
  if (!(confidence > 0 && confidence < 1)) {
    throw new Error(`confidence must be in (0,1); got ${confidence}`);
  }
  return twoSided ? ndtri(0.5 + 0.5 * confidence) : ndtri(confidence);
}

/**
 * Gaussian-approximate Bayesian credible interval from posterior mean/std.
 *
 * @returns `[lo, hi]`, optionally clipped to `bounds`.
 */
export function normalCredibleInterval(
  mu: number,
  sigma: number,
  credibility = 0.95,
  twoSided = true,
  bounds?: Bounds | null,
): [number, number] {
  if (sigma < 0) {
    throw new Error(`sigma must be >= 0; got ${sigma}`);
  }
  const z = zValue(credibility, twoSided);
  let lo: number;
  let hi: number;
  if (twoSided) {
    lo = mu - z * sigma;
    hi = mu + z * sigma;
  } else {
    lo = -Infinity;
    hi = mu + z * sigma;
  }
  if (bounds != null) {
    const [bLo, bHi] = bounds;
    if (bLo > bHi) {
      throw new Error("bounds must satisfy bounds[0] <= bounds[1]");
    }
    lo = Math.max(lo, bLo);
    hi = Math.min(hi, bHi);
  }
  return [lo, hi];
}
