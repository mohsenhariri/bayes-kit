/**
 * AUC@K — normalized area under the Pass@j curve for budgets j = 1, ..., k.
 *
 * For a binary outcome matrix `R` in {0,1}^(M x N), AUC@K averages per-question
 * Pass@j values using trapezoidal weights matching Eq. (7) of Hu et al. (2026).
 * The Bayesian summary computes posterior `mu` and `sigma` under a Beta model
 * for each question's latent success rate. Port of `scorio/eval/auc.py`.
 *
 * References: Hu et al. (2026), arXiv:2601.08763.
 */

import { comb } from "./internal/math.js";
import { normalCredibleInterval, type Bounds } from "./internal/ci.js";
import {
  asMatrix,
  rowSums,
  validateBinary,
  type Matrix,
} from "./internal/validate.js";
import { passAtK, passAtKCi } from "./passAtK.js";

function checkK(N: number, k: number): void {
  if (!(k >= 1 && k <= N) || !Number.isInteger(k)) {
    throw new Error(`k must satisfy 1 <= k <= N (N=${N}); got k=${k}`);
  }
}

/** Vectorized Pass@k value from a per-row success count `nu`. */
function passAtKFromCount(nu: number, N: number, k: number): number {
  const denom = comb(N, k);
  return 1.0 - comb(N - nu, k) / denom;
}

/** Eq. (7) trapezoidal-rule coefficients for AUC@K over Pass@1..Pass@K. */
function aucAtKCoefficients(k: number): number[] {
  if (k < 1) {
    throw new Error(`k must be >= 1; got ${k}`);
  }
  if (k === 1) {
    return [1.0];
  }
  const coeff = new Array<number>(k).fill(1.0 / (k - 1));
  coeff[0] = 0.5 / (k - 1);
  coeff[k - 1] = 0.5 / (k - 1);
  return coeff;
}

/**
 * AUC@K point estimate: average over questions of the trapezoidal area under
 * the Pass@1..Pass@k curve. For `k = 1`, AUC@1 = Pass@1.
 *
 * References: Hu et al. (2026), arXiv:2601.08763.
 *
 * @param R `M x N` binary matrix with entries in {0, 1}.
 * @param k Maximum sampling budget with `1 <= k <= N`.
 * @returns The average AUC@K score across all `M` questions.
 */
export function aucAtK(R: Matrix, k: number): number {
  const Rm = asMatrix(R);
  validateBinary(Rm);
  const N = Rm[0]!.length;
  checkK(N, k);

  if (k === 1) {
    return passAtK(Rm, 1);
  }

  const nu = rowSums(Rm);
  const coeff = aucAtKCoefficients(k);

  const vals = nu.map((nuI) => {
    let acc = 0;
    for (let j = 1; j <= k; j++) {
      acc += coeff[j - 1]! * passAtKFromCount(nuI, N, j);
    }
    return acc;
  });
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

/** Posterior mean/std for AUC@K (the trapezoidal-weighted sum of Pass@j). */
function aucAtKBayes(
  R: Matrix,
  k: number,
  alpha0: number,
  beta0: number,
): [number, number] {
  const Rm = asMatrix(R);
  const { alpha, beta, N } = binaryBetaPosterior(Rm, alpha0, beta0);
  checkK(N, k);
  const M = Rm.length;
  const coeff = aucAtKCoefficients(k);

  let meanSum = 0;
  let varSum = 0;
  // Eq. (7) is a weighted sum of Pass@j terms, and for Bernoulli success rate
  // p we use Pass@j(p) = 1 - (1 - p)^j.
  for (let i = 0; i < M; i++) {
    const a = alpha[i]!;
    const b = beta[i]!;

    // r[s] = E[(1-p)^s] = Beta(a, b+s)/Beta(a, b). Using the recurrence
    //   r[s] = r[s-1] * (b + s - 1) / (a + b + s - 1),   r[0] = 1,
    // precomputes every moment in O(k) arithmetic with no gammaln calls.
    const r = new Array<number>(2 * k + 1);
    r[0] = 1.0;
    for (let s = 1; s <= 2 * k; s++) {
      r[s] = (r[s - 1]! * (b + s - 1)) / (a + b + s - 1);
    }

    let dotCoeffEq = 0;
    for (let j = 1; j <= k; j++) {
      dotCoeffEq += coeff[j - 1]! * r[j]!;
    }
    const m = 1.0 - dotCoeffEq;

    let e2 = 1.0 - 2.0 * dotCoeffEq;
    for (let j = 1; j <= k; j++) {
      const cJ = coeff[j - 1]!;
      for (let l = 1; l <= k; l++) {
        e2 += cJ * coeff[l - 1]! * r[j + l]!;
      }
    }

    meanSum += m;
    varSum += Math.max(0.0, e2 - m * m);
  }

  const mu = meanSum / M;
  const sigma = Math.sqrt(varSum) / M;
  return [mu, sigma];
}

/**
 * Bayesian posterior summary `[mu, sigma, lo, hi]` for the latent AUC@K target.
 *
 * Each question's success probability is a latent Bernoulli parameter with a
 * Beta prior; that uncertainty is propagated through the AUC@K weighted sum of
 * i.i.d. Pass@j targets. For `k = 1`, AUC@1 is Pass@1, so this returns
 * `passAtKCi` with `k = 1`.
 *
 * References: Hu et al. (2026), arXiv:2601.08763.
 *
 * @param R `M x N` binary matrix with entries in {0, 1}.
 * @param k Maximum sampling budget with `1 <= k <= N`.
 * @param confidence Credibility level of the interval.
 * @param bounds `[lo, hi]` clipping bounds for the interval.
 * @param alpha0 Beta prior parameter.
 * @param beta0 Beta prior parameter.
 * @returns `[mu, sigma, lo, hi]`.
 */
export function aucAtKCi(
  R: Matrix,
  k: number,
  confidence = 0.95,
  bounds: Bounds | null = [0.0, 1.0],
  alpha0 = 1.0,
  beta0 = 1.0,
): [number, number, number, number] {
  if (k === 1) {
    return passAtKCi(R, 1, confidence, bounds, alpha0, beta0);
  }

  const [mu, sigma] = aucAtKBayes(R, k, alpha0, beta0);
  const [lo, hi] = normalCredibleInterval(mu, sigma, confidence, true, bounds);
  return [mu, sigma, lo, hi];
}
