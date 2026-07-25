/**
 * Generalized pass-family metrics for binary outcomes — G-Pass@k, the
 * thresholded G-Pass@k_τ (at least `ceil(τk)` successes), and mG-Pass@k
 * (the mean over thresholds `τ ∈ [0.5, 1.0]`), with Beta-posterior credible
 * intervals. Port of `scorio/eval/gpass.py`.
 *
 * References: Liu et al. (2024), arXiv:2412.13147; Yao et al. (2024),
 * arXiv:2406.12045.
 */

import {
  betaRatio,
  comb,
  hypergeomPmf,
  hypergeomSf,
  logBetaRatio,
} from "./internal/math.js";
import { normalCredibleInterval, type Bounds } from "./internal/ci.js";
import {
  asMatrix,
  rowSums,
  validateBinary,
  type Matrix,
} from "./internal/validate.js";
import { passAtK, passHatK, passHatKCi } from "./passAtK.js";

function checkK(k: number, N: number): void {
  if (!(k >= 1 && k <= N)) {
    throw new Error(`k must satisfy 1 <= k <= N (N=${N}); got k=${k}`);
  }
}

function checkTau(tau: number): void {
  if (!(tau >= 0.0 && tau <= 1.0)) {
    throw new Error(`tau must be in [0, 1]; got ${tau}`);
  }
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

/**
 * G-Pass@k: the all-success (`τ = 1`) threshold, an alias for Pass^k.
 *
 * Included for literature using the G-Pass@k naming convention.
 */
export function gPassAtK(R: Matrix, k: number): number {
  return passHatK(R, k);
}

/**
 * G-Pass@k_τ: average probability of at least `ceil(τk)` successes among `k`
 * selected samples (unbiased hypergeometric estimator). `τ = 0` reduces to
 * Pass@k and `τ = 1` to Pass^k.
 */
export function gPassAtKTau(R: Matrix, k: number, tau: number): number {
  const Rm = asMatrix(R);
  validateBinary(Rm);
  const N = Rm[0]!.length;
  checkTau(tau);
  checkK(k, N);
  if (!Number.isInteger(k)) return NaN;

  if (tau <= 0.0) {
    return passAtK(Rm, k);
  }

  const nu = rowSums(Rm);
  const j0 = Math.max(1, Math.ceil(tau * k));
  const M = Rm.length;
  const vals = nu.map((v) => hypergeomSf(N, v, k, j0));
  return vals.reduce((s, v) => s + v, 0) / M;
}

/**
 * mG-Pass@k: the mean generalized pass metric, `2 ∫_{0.5}^{1} G-Pass@k_τ dτ`.
 *
 * The integral over `τ ∈ [0.5, 1.0]` collapses to the closed form
 * `(2/k) · Σ_{j=m+1}^{k} (j - m) · P(X = j)`, where `m = ceil(k/2)` is the
 * majority threshold and `X ~ Hypergeometric(N, ν, k)`.
 */
export function mgPassAtK(R: Matrix, k: number): number {
  const Rm = asMatrix(R);
  validateBinary(Rm);
  const N = Rm[0]!.length;
  checkK(k, N);
  if (!Number.isInteger(k)) {
    throw new TypeError("'number' object cannot be interpreted as an integer");
  }

  const nu = rowSums(Rm);

  const majority = Math.ceil(0.5 * k);
  if (majority >= k) {
    return 0.0;
  }

  const M = Rm.length;
  const vals = new Array<number>(M).fill(0);
  for (let j = majority + 1; j <= k; j++) {
    for (let i = 0; i < M; i++) {
      const v = nu[i]!;
      const pmf = hypergeomPmf(N, v, k, j);
      vals[i]! += (j - majority) * pmf;
    }
  }
  return vals.reduce((s, v) => s + (v * 2.0) / k, 0) / M;
}

/**
 * Posterior mean/std of `Σ_j coeff_j · 1{X >= ...}` for a hypergeometric-style
 * weighted sum whose i.i.d. Beta-Binomial moments are
 * `E[(success-budget) terms] = Beta(a+j, b+k-j)/Beta(a, b)`.
 *
 * The first and second moments need `Beta(a+x, b+k-x)/Beta(a,b)` (`tK`) and
 * `Beta(a+x, b+2k-x)/Beta(a,b)` (`t2K`). Evaluate each required ratio directly
 * in log space. A forward recurrence from x=0 is tempting, but its seed can
 * underflow for large k and then incorrectly leaves every later moment at zero.
 */
function iidWeightedMoments(
  alpha: readonly number[],
  beta: readonly number[],
  M: number,
  js: readonly number[],
  coeff: readonly number[],
  k: number,
): [number, number] {
  let meanSum = 0;
  let varSum = 0;
  for (let i = 0; i < M; i++) {
    const a = alpha[i]!;
    const b = beta[i]!;

    const logTK = new Array<number>(k + 1);
    for (let x = 0; x <= k; x++) logTK[x] = logBetaRatio(a, b, x, k - x);

    const logT2K = new Array<number>(2 * k + 1);
    for (let x = 0; x <= 2 * k; x++) {
      logT2K[x] = logBetaRatio(a, b, x, 2 * k - x);
    }

    let m = 0;
    for (let idx = 0; idx < js.length; idx++) {
      const c = coeff[idx]!;
      if (c !== 0.0) m += Math.exp(Math.log(c) + logTK[js[idx]!]!);
    }

    let e2 = 0;
    for (let idxJ = 0; idxJ < js.length; idxJ++) {
      const cJ = coeff[idxJ]!;
      if (cJ === 0.0) continue;
      const j = js[idxJ]!;
      for (let idxL = 0; idxL < js.length; idxL++) {
        const cL = coeff[idxL]!;
        if (cL === 0.0) continue;
        // Preserve Python's floating-point evaluation order. For sufficiently
        // large k the coefficient product overflows while the beta moment
        // underflows, producing NaN; Python's max(0.0, NaN) then clips that
        // row variance to zero.
        e2 += cJ * cL * Math.exp(logT2K[j + js[idxL]!]!);
      }
    }

    meanSum += m;
    const rawVariance = e2 - m * m;
    varSum += Number.isNaN(rawVariance) ? 0.0 : Math.max(0.0, rawVariance);
  }
  return [meanSum / M, Math.sqrt(varSum) / M];
}

/** Posterior mean/std for the i.i.d. G-Pass@k_τ quantity. */
function gPassAtKTauBayes(
  R: Matrix,
  k: number,
  tau: number,
  alpha0: number,
  beta0: number,
): [number, number] {
  const Rm = asMatrix(R);
  const { alpha, beta, N } = binaryBetaPosterior(Rm, alpha0, beta0);
  checkTau(tau);
  checkK(k, N);

  if (tau <= 0.0) {
    return passAtKBayes(Rm, k, alpha0, beta0);
  }
  if (tau >= 1.0) {
    return passHatKBayes(Rm, k, alpha0, beta0);
  }
  if (!Number.isInteger(k)) {
    throw new TypeError("'number' object cannot be interpreted as an integer");
  }

  const M = Rm.length;
  const j0 = Math.ceil(tau * k);
  const js: number[] = [];
  for (let j = j0; j <= k; j++) js.push(j);
  const coeff = js.map((j) => comb(k, j));

  return iidWeightedMoments(alpha, beta, M, js, coeff, k);
}

/** Posterior mean/std for the i.i.d. mG-Pass@k quantity. */
function mgPassAtKBayes(
  R: Matrix,
  k: number,
  alpha0: number,
  beta0: number,
): [number, number] {
  const Rm = asMatrix(R);
  const { alpha, beta, N } = binaryBetaPosterior(Rm, alpha0, beta0);
  checkK(k, N);
  if (!Number.isInteger(k)) {
    throw new TypeError("'number' object cannot be interpreted as an integer");
  }

  const majority = Math.ceil(0.5 * k);
  if (majority >= k) {
    return [0.0, 0.0];
  }

  const M = Rm.length;
  const js: number[] = [];
  for (let j = majority + 1; j <= k; j++) js.push(j);
  const coeff = js.map((j) => (2.0 / k) * (j - majority) * comb(k, j));

  return iidWeightedMoments(alpha, beta, M, js, coeff, k);
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
    const eQk = betaRatio(a, b, 0, k);
    const eQ2k = betaRatio(a, b, 0, 2 * k);
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
    const ePk = betaRatio(a, b, k, 0);
    const eP2k = betaRatio(a, b, 2 * k, 0);
    meanSum += ePk;
    varSum += Math.max(0, eP2k - ePk * ePk);
  }
  return [meanSum / M, Math.sqrt(varSum) / M];
}

/**
 * Bayesian `[mu, sigma, lo, hi]` for G-Pass@k (alias for Pass^k posterior).
 */
export function gPassAtKCi(
  R: Matrix,
  k: number,
  confidence = 0.95,
  bounds: Bounds | null = [0.0, 1.0],
  alpha0 = 1.0,
  beta0 = 1.0,
): [number, number, number, number] {
  return passHatKCi(R, k, confidence, bounds, alpha0, beta0);
}

/** Bayesian `[mu, sigma, lo, hi]` for thresholded G-Pass@k_τ. */
export function gPassAtKTauCi(
  R: Matrix,
  k: number,
  tau: number,
  confidence = 0.95,
  bounds: Bounds | null = [0.0, 1.0],
  alpha0 = 1.0,
  beta0 = 1.0,
): [number, number, number, number] {
  const [mu, sigma] = gPassAtKTauBayes(R, k, tau, alpha0, beta0);
  const [lo, hi] = normalCredibleInterval(mu, sigma, confidence, true, bounds);
  return [mu, sigma, lo, hi];
}

/** Bayesian `[mu, sigma, lo, hi]` for mG-Pass@k. */
export function mgPassAtKCi(
  R: Matrix,
  k: number,
  confidence = 0.95,
  bounds: Bounds | null = [0.0, 1.0],
  alpha0 = 1.0,
  beta0 = 1.0,
): [number, number, number, number] {
  const [mu, sigma] = mgPassAtKBayes(R, k, alpha0, beta0);
  const [lo, hi] = normalCredibleInterval(mu, sigma, confidence, true, bounds);
  return [mu, sigma, lo, hi];
}
