/**
 * Majority-vote metrics for binary outcomes — Maj@k (probability that more
 * than half of `k` selected traces are correct) with a Bayesian credible
 * interval. A binary-outcome wrapper around the generalized threshold-pass
 * family (G-Pass@k_τ). Port of `scorio/eval/maj.py`.
 *
 * Reference: Liu, J., Liu, H., Xiao, L., et al. (2024). Are Your LLMs Capable
 * of Stable Reasoning? arXiv:2412.13147.
 */

import { gPassAtKTau, gPassAtKTauCi } from "./gpass.js";
import { type Bounds } from "./internal/ci.js";
import { type Matrix } from "./internal/validate.js";

/** Threshold τ such that `ceil(τ k)` is a strict majority of `k`. */
function majorityTau(k: number): number {
  return (Math.floor(k / 2) + 1) / k;
}

/**
 * Maj@k: strict-majority correctness over `k` samples — probability that a
 * uniformly sampled subset of `k` observed traces contains strictly more than
 * half correct solutions. Equivalent to G-Pass@k_τ with τ = (⌊k/2⌋+1)/k.
 */
export function majAtK(R: Matrix, k: number): number {
  return gPassAtKTau(R, k, majorityTau(k));
}

/**
 * Bayesian posterior summary `[mu, sigma, lo, hi]` for Maj@k, reusing the
 * generalized threshold-pass posterior at the strict-majority threshold
 * τ = (⌊k/2⌋+1)/k.
 */
export function majAtKCi(
  R: Matrix,
  k: number,
  confidence = 0.95,
  bounds: Bounds = [0.0, 1.0],
  alpha0 = 1.0,
  beta0 = 1.0,
): [number, number, number, number] {
  return gPassAtKTauCi(R, k, majorityTau(k), confidence, bounds, alpha0, beta0);
}
