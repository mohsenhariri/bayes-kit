/**
 * Pass family — Pass@k (probability at least one of k samples passes) and
 * Pass^k / Pass-hat@k (probability all k pass), with Beta-posterior credible
 * intervals. Port of `scorio/eval/pass_at_k.py`.
 *
 * References: Chen et al. (2021), arXiv:2107.03374; Yao et al. (2024),
 * arXiv:2406.12045.
 */

import {
  betaRatio,
  hypergeomAtLeastOne,
  hypergeomPmf,
} from "./internal/math.js";
import { normalCredibleInterval, type Bounds } from "./internal/ci.js";
import {
  asMatrix,
  rowSums,
  validateBinary,
  type Matrix,
} from "./internal/validate.js";

function checkK(k: number, N: number): void {
  if (!(k >= 1 && k <= N)) {
    throw new Error(`k must satisfy 1 <= k <= N (N=${N}); got k=${k}`);
  }
}

/** Unbiased Pass@k: probability at least one of `k` selected samples passes. */
export function passAtK(R: Matrix, k: number): number {
  const Rm = asMatrix(R);
  validateBinary(Rm);
  const N = Rm[0]!.length;
  checkK(k, N);
  if (!Number.isInteger(k)) return NaN;
  const nu = rowSums(Rm);
  const vals = nu.map((v) => hypergeomAtLeastOne(N, v, k));
  return vals.reduce((s, v) => s + v, 0) / vals.length;
}

/** Pass^k (Pass-hat@k): probability all `k` selected samples pass. */
export function passHatK(R: Matrix, k: number): number {
  const Rm = asMatrix(R);
  validateBinary(Rm);
  const N = Rm[0]!.length;
  checkK(k, N);
  if (!Number.isInteger(k)) return NaN;
  const nu = rowSums(Rm);
  const vals = nu.map((v) => hypergeomPmf(N, v, k, k));
  return vals.reduce((s, v) => s + v, 0) / vals.length;
}

/** Per-row Beta posterior parameters `[alpha, beta]` for binary outcomes. */
function binaryBetaPosterior(
  Rm: readonly (readonly number[])[],
  alpha0: number,
  beta0: number,
): { alpha: number[]; beta: number[]; N: number } {
  validateBinary(Rm);
  const N = Rm[0]!.length;
  const c = rowSums(Rm);
  return {
    alpha: c.map((ci) => alpha0 + ci),
    beta: c.map((ci) => beta0 + (N - ci)),
    N,
  };
}

/** Posterior mean/std for the i.i.d. Pass@k quantity `1 - (1-p)^k`. */
function passAtKBayes(
  R: Matrix,
  k: number,
  alpha0: number,
  beta0: number,
): [number, number] {
  const Rm = asMatrix(R);
  const { alpha, beta, N } = binaryBetaPosterior(Rm, alpha0, beta0);
  checkK(k, N);
  const M = Rm.length;
  let meanSum = 0;
  let varSum = 0;
  for (let i = 0; i < M; i++) {
    const a = alpha[i]!;
    const b = beta[i]!;
    const eQk = betaRatio(a, b, 0, k); // E[(1-p)^k]
    const eQ2k = betaRatio(a, b, 0, 2 * k); // E[(1-p)^(2k)]
    const m = 1 - eQk;
    const e2 = 1 - 2 * eQk + eQ2k;
    meanSum += m;
    varSum += Math.max(0, e2 - m * m);
  }
  return [meanSum / M, Math.sqrt(varSum) / M];
}

/** Posterior mean/std for the i.i.d. Pass^k quantity `p^k`. */
function passHatKBayes(
  R: Matrix,
  k: number,
  alpha0: number,
  beta0: number,
): [number, number] {
  const Rm = asMatrix(R);
  const { alpha, beta, N } = binaryBetaPosterior(Rm, alpha0, beta0);
  checkK(k, N);
  const M = Rm.length;
  let meanSum = 0;
  let varSum = 0;
  for (let i = 0; i < M; i++) {
    const a = alpha[i]!;
    const b = beta[i]!;
    const ePk = betaRatio(a, b, k, 0); // E[p^k]
    const eP2k = betaRatio(a, b, 2 * k, 0); // E[p^(2k)]
    meanSum += ePk;
    varSum += Math.max(0, eP2k - ePk * ePk);
  }
  return [meanSum / M, Math.sqrt(varSum) / M];
}

/** Bayesian `[mu, sigma, lo, hi]` for i.i.d. Pass@k. */
export function passAtKCi(
  R: Matrix,
  k: number,
  confidence = 0.95,
  bounds: Bounds | null = [0.0, 1.0],
  alpha0 = 1.0,
  beta0 = 1.0,
): [number, number, number, number] {
  const [mu, sigma] = passAtKBayes(R, k, alpha0, beta0);
  const [lo, hi] = normalCredibleInterval(mu, sigma, confidence, true, bounds);
  return [mu, sigma, lo, hi];
}

/** Bayesian `[mu, sigma, lo, hi]` for i.i.d. Pass^k. */
export function passHatKCi(
  R: Matrix,
  k: number,
  confidence = 0.95,
  bounds: Bounds | null = [0.0, 1.0],
  alpha0 = 1.0,
  beta0 = 1.0,
): [number, number, number, number] {
  const [mu, sigma] = passHatKBayes(R, k, alpha0, beta0);
  const [lo, hi] = normalCredibleInterval(mu, sigma, confidence, true, bounds);
  return [mu, sigma, lo, hi];
}
