/**
 * Max@k family — the continuous-reward generalization of Pass@k. Instead of
 * asking whether at least one sampled response is correct, Max@k scores the
 * best response among `k` sampled traces under a user-specified reward scale.
 * Port of `scorio/eval/max_reward.py`.
 *
 * The point estimator matches the appendix evaluation formula in Bagirov et
 * al. (2025), "The Best of N Worlds: Aligning Reinforcement Learning with
 * Best-of-N Sampling via max@k Optimization" (Appendix C.1 / Listing 1,
 * arXiv:2510.23393). The companion `*_ci` function is a `scorio` Bayesian
 * extension using the same grouped-Dirichlet posterior as `bayes`.
 */

import { gammaln, comb, betaRatio } from "./internal/math.js";
import { normalCredibleInterval, type Bounds } from "./internal/ci.js";
import { asMatrix, validateMatrixRange, type Matrix } from "./internal/validate.js";
import { bayesCi } from "./bayes.js";

/** Numerically stable `log(sum(exp(values)))`, mirroring `scipy.special.logsumexp`. */
function logSumExp(values: readonly number[]): number {
  let max = -Infinity;
  for (const v of values) if (v > max) max = v;
  if (max === -Infinity) return -Infinity;
  let sum = 0;
  for (const v of values) sum += Math.exp(v - max);
  return max + Math.log(sum);
}

/** Per-row counts of the values `0..length-1` over a possibly-empty matrix. */
function rowBincountWide(
  A: readonly (readonly number[])[],
  length: number,
): number[][] {
  return A.map((row) => {
    const counts = new Array<number>(length).fill(0);
    for (const v of row) counts[v]! += 1;
    return counts;
  });
}

/** Normalized `R`, weight vector `w`, and prior matrix `R0`. */
interface CategoricalInput {
  Rm: number[][];
  wv: number[];
  R0m: number[][];
}

/** Normalize R, w, and R0 for weighted categorical metrics. */
function prepareCategoricalInput(
  R: Matrix,
  w?: readonly number[],
  R0?: Matrix,
): CategoricalInput {
  const Rm = asMatrix(R);

  let wv: number[];
  if (w === undefined) {
    const seen = new Set<number>();
    for (const row of Rm) for (const v of row) seen.add(v);
    const isBinary = seen.size <= 2 && [...seen].every((v) => v === 0 || v === 1);
    if (!isBinary) {
      const vals = [...seen].sort((a, b) => a - b).join(", ");
      throw new Error(
        `R contains more than 2 unique values (${vals}), so weight vector 'w' must be provided. ` +
          `Please specify a weight vector of length ${seen.size} to map each category to a score.`,
      );
    }
    wv = [0.0, 1.0];
  } else {
    wv = w.map(Number);
  }

  const M = Rm.length;
  const C = wv.length - 1;
  validateMatrixRange(Rm, 0, C, "R");

  let R0m: number[][];
  if (R0 === undefined) {
    R0m = Rm.map(() => []);
  } else {
    R0m = asMatrix(R0);
    if (R0m.length !== M) {
      throw new Error("R0 must have the same number of rows (M) as R.");
    }
    validateMatrixRange(R0m, 0, C, "R0");
  }

  return { Rm, wv, R0m };
}

function validateK(N: number, k: number): void {
  if (!(k >= 1 && k <= N) || !Number.isInteger(k)) {
    throw new Error(`k must satisfy 1 <= k <= N (N=${N}); got k=${k}`);
  }
}

/** Sorted unique values of `w`, with each entry's index into that sorted list. */
function uniqueLevels(wv: readonly number[]): { levels: number[]; inverse: number[] } {
  const sorted = [...new Set(wv)].sort((a, b) => a - b);
  const index = new Map<number, number>();
  sorted.forEach((v, i) => index.set(v, i));
  return { levels: sorted, inverse: wv.map((v) => index.get(v)!) };
}

/** Grouped Dirichlet posterior parameters and the unique reward levels. */
function groupedPosteriorParams(
  R: Matrix,
  w?: readonly number[],
  R0?: Matrix,
): { gamma: number[][]; levels: number[] } {
  const { Rm, wv, R0m } = prepareCategoricalInput(R, w, R0);
  const C = wv.length - 1;

  const { levels, inverse } = uniqueLevels(wv);
  const L = levels.length;

  const nCounts = rowBincountWide(Rm, C + 1);
  const n0Counts = rowBincountWide(R0m, C + 1).map((row) => row.map((c) => c + 1));

  const gamma = Rm.map((_, row) => {
    const g = new Array<number>(L).fill(0);
    for (let cat = 0; cat <= C; cat++) {
      g[inverse[cat]!]! += nCounts[row]![cat]! + n0Counts[row]![cat]!;
    }
    return g;
  });

  return { gamma, levels };
}

/**
 * `E[X^k (X+Y)^k]` for `X, Y` from a 3-part Dirichlet partition.
 *
 * `X` has parameter `a`, `Y` has parameter `b`, and the omitted remainder has
 * parameter `total - a - b`. Follows from the multinomial expansion of
 * `(X+Y)^k` and Dirichlet raw moments.
 */
function dirichletNestedCumulativeMoment(
  total: number,
  a: number,
  b: number,
  k: number,
): number {
  if (b <= 0.0) {
    throw new Error("b must be > 0 for nested cumulative moments");
  }
  const logTerms: number[] = [];
  for (let r = 0; r <= k; r++) {
    logTerms.push(
      gammaln(k + 1.0) -
        gammaln(r + 1.0) -
        gammaln(k - r + 1.0) +
        gammaln(a + k + r) -
        gammaln(a) +
        gammaln(b + k - r) -
        gammaln(b) -
        (gammaln(total + 2.0 * k) - gammaln(total)),
    );
  }
  return Math.exp(logSumExp(logTerms));
}

/**
 * Max@k: expected best reward among `k` sampled traces.
 *
 * When `w = [0, 1]`, Max@k reduces exactly to Pass@k. More generally, the
 * reward vector `w` maps categorical outcomes to arbitrary real-valued scores,
 * and Max@k averages the best score obtainable from a subset of size `k`.
 *
 * The finite-sample estimator matches Bagirov et al. (2025), Appendix C.1 /
 * Listing 1 (arXiv:2510.23393).
 *
 * @param R `M x N` categorical outcome matrix with integer entries in
 *          `{0,...,C}`.
 * @param k Number of selected samples, with `1 <= k <= N`.
 * @param w Optional reward vector of shape `(C+1,)`. If omitted, `R` must be
 *          binary and `[0, 1]` is used.
 * @returns Average Max@k score across prompts.
 */
export function maxAtK(R: Matrix, k: number, w?: readonly number[]): number {
  const { Rm, wv } = prepareCategoricalInput(R, w);
  const N = Rm[0]!.length;
  validateK(N, k);

  const denom = comb(N, k);
  // coeff[i - (k-1)] = C(i, k-1) / C(N, k) for i in {k-1, ..., N-1}.
  const coeff: number[] = [];
  for (let i = k - 1; i < N; i++) coeff.push(comb(i, k - 1) / denom);

  let acc = 0;
  for (const row of Rm) {
    const sorted = row.map((c) => wv[c]!).sort((a, b) => a - b);
    let val = 0;
    for (let j = 0; j < coeff.length; j++) {
      val += coeff[j]! * sorted[k - 1 + j]!;
    }
    acc += val;
  }
  return acc / Rm.length;
}

/** Posterior mean/std for Max@k under a grouped Dirichlet posterior. */
function maxAtKBayes(
  R: Matrix,
  k: number,
  w?: readonly number[],
  R0?: Matrix,
): { mu: number; sigma: number; levels: number[] } {
  const { gamma, levels } = groupedPosteriorParams(R, w, R0);
  const M = gamma.length;
  const L = gamma[0]!.length;
  const total = gamma[0]!.reduce((s, v) => s + v, 0);

  if (k < 1 || !Number.isInteger(k)) {
    throw new Error(`k must be an integer >= 1; got ${k}`);
  }
  // The posterior moments describe the latent distribution, so k is not
  // restricted by the observed sample size once the posterior is defined.

  if (L === 1) {
    return { mu: levels[0]!, sigma: 0.0, levels };
  }

  const gaps: number[] = [];
  for (let i = 1; i < L; i++) gaps.push(levels[i]! - levels[i - 1]!);
  const top = levels[L - 1]!;

  const means = new Array<number>(M);
  const vars_ = new Array<number>(M);

  for (let row = 0; row < M; row++) {
    const gammaRow = gamma[row]!;
    // cum[idx] = A_l parameters for l = 1..L-1 (cumulative sums, excluding last).
    const cum: number[] = [];
    let running = 0;
    for (let i = 0; i < L - 1; i++) {
      running += gammaRow[i]!;
      cum.push(running);
    }

    const eAk = new Array<number>(L - 1);
    const eA2k = new Array<number>(L - 1);
    for (let idx = 0; idx < L - 1; idx++) {
      const a = cum[idx]!;
      const b = total - a;
      eAk[idx] = betaRatio(a, b, k, 0);
      eA2k[idx] = betaRatio(a, b, 2 * k, 0);
    }

    let dotGapsEAk = 0;
    for (let i = 0; i < L - 1; i++) dotGapsEAk += gaps[i]! * eAk[i]!;
    const m = top - dotGapsEAk;

    // cross is the (L-1)x(L-1) matrix of cumulative cross moments.
    const cross: number[][] = Array.from({ length: L - 1 }, () =>
      new Array<number>(L - 1).fill(0),
    );
    for (let i = 0; i < L - 1; i++) {
      cross[i]![i] = eA2k[i]!;
      for (let j = i + 1; j < L - 1; j++) {
        const a = cum[i]!;
        const b = cum[j]! - cum[i]!;
        const moment = dirichletNestedCumulativeMoment(total, a, b, k);
        cross[i]![j] = moment;
        cross[j]![i] = moment;
      }
    }

    let e2 = top * top - 2.0 * top * dotGapsEAk;
    // gaps @ cross @ gaps
    let quad = 0;
    for (let i = 0; i < L - 1; i++) {
      let rowDot = 0;
      for (let j = 0; j < L - 1; j++) rowDot += cross[i]![j]! * gaps[j]!;
      quad += gaps[i]! * rowDot;
    }
    e2 += quad;

    means[row] = m;
    vars_[row] = Math.max(0.0, e2 - m * m);
  }

  const mu = means.reduce((s, v) => s + v, 0) / M;
  const sigma = Math.sqrt(vars_.reduce((s, v) => s + v, 0)) / M;
  return { mu, sigma, levels };
}

/**
 * Bayesian posterior summary for {@link maxAtK}, returning `[mu, sigma, lo, hi]`.
 *
 * The posterior uses the same Dirichlet-plus-one construction as `bayes`. When
 * `k = 1`, Max@1 reduces to the single-draw expected score, so this function
 * agrees with `bayesCi`. This uncertainty model is a `scorio` extension and is
 * not part of Bagirov et al. (2025).
 *
 * @param R `M x N` categorical outcome matrix with integer entries in
 *          `{0,...,C}`.
 * @param k Selection count; defined for any integer `k >= 1`. `k = 1` matches
 *          `bayesCi`.
 * @param w Optional reward vector of shape `(C+1,)`. If omitted, `R` must be
 *          binary and `[0, 1]` is used.
 * @param R0 Optional `M x D` matrix of prior outcomes.
 * @param confidence Credibility level for the normal-approximation interval.
 * @param bounds Optional `[lo, hi]` clipping bounds. If omitted, the interval
 *          is clipped to the minimum and maximum reward levels in `w`.
 * @returns `[mu, sigma, lo, hi]`.
 */
export function maxAtKCi(
  R: Matrix,
  k: number,
  w?: readonly number[],
  R0?: Matrix,
  confidence = 0.95,
  bounds?: Bounds,
): [number, number, number, number] {
  if (k === 1) {
    return bayesCi(R, w, R0, confidence, bounds);
  }

  const { mu, sigma, levels } = maxAtKBayes(R, k, w, R0);
  const effectiveBounds: Bounds =
    bounds === undefined
      ? [Math.min(...levels), Math.max(...levels)]
      : bounds;
  const [lo, hi] = normalCredibleInterval(mu, sigma, confidence, true, effectiveBounds);
  return [mu, sigma, lo, hi];
}
