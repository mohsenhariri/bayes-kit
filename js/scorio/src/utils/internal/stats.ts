/** Dependency-free statistical primitives used by `scorio/utils`. */

import { gammaln } from "../../eval/internal/math.js";
import { normCdf } from "../../rank/internal/special.js";
import { rankdata } from "../../rank/internal/rankScores.js";

export type Correlation = [statistic: number, pvalue: number];

function tieStats(values: readonly number[]): {
  pairs: number;
  cubic: number;
  variance: number;
} {
  const counts = new Map<number, number>();
  for (const value of values) counts.set(value, (counts.get(value) ?? 0) + 1);
  let pairs = 0;
  let cubic = 0;
  let variance = 0;
  for (const count of counts.values()) {
    if (count <= 1) continue;
    pairs += (count * (count - 1)) / 2;
    cubic += count * (count - 1) * (count - 2);
    variance += count * (count - 1) * (2 * count + 5);
  }
  return { pairs, cubic, variance };
}

function factorialNumber(n: number): number {
  let value = 1;
  for (let i = 2; i <= n; i++) value *= i;
  return value;
}

/** Exact two-sided p-value for an untied Kendall statistic. */
function kendallExactPvalue(n: number, discordant: number): number {
  const total = (n * (n - 1)) / 2;
  const tail = Math.min(discordant, total - discordant);
  const factorial = factorialNumber(n);
  if (tail === 0) return Math.min(1, 2 / factorial);
  if (tail === 1 && n > 33) return Math.min(1, (2 * n) / factorial);

  // Mahonian inversion counts. The `auto` policy reaches this branch only for
  // n <= 33, so all values remain comfortably within finite IEEE-754 range.
  let counts = [1];
  for (let size = 2; size <= n; size++) {
    const next = new Array<number>(counts.length + size - 1).fill(0);
    let window = 0;
    for (let k = 0; k < next.length; k++) {
      if (k < counts.length) window += counts[k]!;
      if (k >= size && k - size < counts.length) window -= counts[k - size]!;
      next[k] = window;
    }
    counts = next;
  }
  let cumulative = 0;
  for (let i = 0; i <= tail; i++) cumulative += counts[i]!;
  return Math.min(1, (2 * cumulative) / factorial);
}

/** SciPy-compatible Kendall tau-b and its default two-sided p-value. */
export function kendallTau(x: readonly number[], y: readonly number[]): Correlation {
  const n = x.length;
  if (n <= 1) return [NaN, NaN];

  let concordant = 0;
  let discordant = 0;
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const product = (x[i]! - x[j]!) * (y[i]! - y[j]!);
      if (product > 0) concordant += 1;
      else if (product < 0) discordant += 1;
    }
  }

  const total = (n * (n - 1)) / 2;
  const xStats = tieStats(x);
  const yStats = tieStats(y);
  if (xStats.pairs === total || yStats.pairs === total) return [NaN, NaN];

  const conMinusDis = concordant - discordant;
  const denominator = Math.sqrt(total - xStats.pairs) * Math.sqrt(total - yStats.pairs);
  const tau = Math.max(-1, Math.min(1, conMinusDis / denominator));

  const hasTies = xStats.pairs !== 0 || yStats.pairs !== 0;
  if (!hasTies && (n <= 33 || Math.min(discordant, total - discordant) <= 1)) {
    return [tau, kendallExactPvalue(n, discordant)];
  }

  // This form follows scipy.stats.kendalltau. `conMinusDis` above is
  // algebraically identical to its tie-adjusted count.
  const m = n * (n - 1);
  const variance =
    (m * (2 * n + 5) - xStats.variance - yStats.variance) / 18 +
    (2 * xStats.pairs * yStats.pairs) / m +
    (xStats.cubic * yStats.cubic) / (9 * m * (n - 2));
  const z = conMinusDis / Math.sqrt(variance);
  return [tau, Math.min(1, 2 * normCdf(-Math.abs(z)))];
}

const BETA_FPMIN = 1e-300;
const BETA_EPSILON = 3e-16;
const BETA_MAX_ITERATIONS = 400;

function betaContinuedFraction(a: number, b: number, x: number): number {
  const qab = a + b;
  const qap = a + 1;
  const qam = a - 1;
  let c = 1;
  let d = 1 - (qab * x) / qap;
  if (Math.abs(d) < BETA_FPMIN) d = BETA_FPMIN;
  d = 1 / d;
  let h = d;

  for (let m = 1; m <= BETA_MAX_ITERATIONS; m++) {
    const m2 = 2 * m;
    let aa = (m * (b - m) * x) / ((qam + m2) * (a + m2));
    d = 1 + aa * d;
    if (Math.abs(d) < BETA_FPMIN) d = BETA_FPMIN;
    c = 1 + aa / c;
    if (Math.abs(c) < BETA_FPMIN) c = BETA_FPMIN;
    d = 1 / d;
    h *= d * c;

    aa = (-(a + m) * (qab + m) * x) / ((a + m2) * (qap + m2));
    d = 1 + aa * d;
    if (Math.abs(d) < BETA_FPMIN) d = BETA_FPMIN;
    c = 1 + aa / c;
    if (Math.abs(c) < BETA_FPMIN) c = BETA_FPMIN;
    d = 1 / d;
    const delta = d * c;
    h *= delta;
    if (Math.abs(delta - 1) <= BETA_EPSILON) break;
  }
  return h;
}

function regularizedBeta(x: number, a: number, b: number): number {
  if (Number.isNaN(x) || Number.isNaN(a) || Number.isNaN(b)) return NaN;
  if (x <= 0) return x === 0 ? 0 : NaN;
  if (x >= 1) return x === 1 ? 1 : NaN;
  const front = Math.exp(
    gammaln(a + b) -
      gammaln(a) -
      gammaln(b) +
      a * Math.log(x) +
      b * Math.log1p(-x),
  );
  if (x < (a + 1) / (a + b + 2)) {
    return (front * betaContinuedFraction(a, b, x)) / a;
  }
  return 1 - (front * betaContinuedFraction(b, a, 1 - x)) / b;
}

function pearson(x: readonly number[], y: readonly number[]): number {
  const n = x.length;
  let meanX = 0;
  let meanY = 0;
  for (let i = 0; i < n; i++) {
    meanX += x[i]!;
    meanY += y[i]!;
  }
  meanX /= n;
  meanY /= n;

  let covariance = 0;
  let varianceX = 0;
  let varianceY = 0;
  for (let i = 0; i < n; i++) {
    const dx = x[i]! - meanX;
    const dy = y[i]! - meanY;
    covariance += dx * dy;
    varianceX += dx * dx;
    varianceY += dy * dy;
  }
  if (varianceX === 0 || varianceY === 0) return NaN;
  return Math.max(-1, Math.min(1, covariance / Math.sqrt(varianceX * varianceY)));
}

/** SciPy-compatible Spearman rho and asymptotic two-sided p-value. */
export function spearmanR(x: readonly number[], y: readonly number[]): Correlation {
  const n = x.length;
  if (n <= 1) return [NaN, NaN];
  const rho = pearson(rankdata(x, "average"), rankdata(y, "average"));
  if (Number.isNaN(rho)) return [NaN, NaN];
  const degrees = n - 2;
  if (degrees <= 0) return [rho, NaN];
  const t = rho * Math.sqrt(degrees / ((rho + 1) * (1 - rho)));
  if (!Number.isFinite(t)) return [rho, 0];
  const tSquared = t * t;
  // Equivalent incomplete-beta forms, selected to avoid losing precision
  // when either ratio rounds too close to one. This mirrors the accuracy of
  // scipy.special.stdtr used by scipy.stats.spearmanr.
  const pvalue =
    tSquared < degrees
      ? 1 - regularizedBeta(tSquared / (degrees + tSquared), 0.5, degrees / 2)
      : regularizedBeta(degrees / (degrees + tSquared), degrees / 2, 0.5);
  return [rho, pvalue];
}

function weightedTauForLexicographicRank(
  x: readonly number[],
  y: readonly number[],
  primary: "x" | "y",
): number {
  const n = x.length;
  const order = Array.from({ length: n }, (_, i) => i);
  order.sort((i, j) => {
    const firstI = primary === "x" ? x[i]! : y[i]!;
    const firstJ = primary === "x" ? x[j]! : y[j]!;
    if (firstI !== firstJ) return firstJ - firstI;
    const secondI = primary === "x" ? y[i]! : x[i]!;
    const secondJ = primary === "x" ? y[j]! : x[j]!;
    return secondI !== secondJ ? secondJ - secondI : i - j;
  });
  const ranks = new Array<number>(n);
  for (let rank = 0; rank < n; rank++) ranks[order[rank]!] = rank;

  let totalWeight = 0;
  let xTieWeight = 0;
  let yTieWeight = 0;
  let numerator = 0;
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const weight = 1 / (ranks[i]! + 1) + 1 / (ranks[j]! + 1);
      totalWeight += weight;
      const dx = x[i]! - x[j]!;
      const dy = y[i]! - y[j]!;
      if (dx === 0) xTieWeight += weight;
      if (dy === 0) yTieWeight += weight;
      if (dx * dy > 0) numerator += weight;
      else if (dx * dy < 0) numerator -= weight;
    }
  }
  const denominator = Math.sqrt(
    (totalWeight - xTieWeight) * (totalWeight - yTieWeight),
  );
  if (denominator === 0) return NaN;
  return Math.max(-1, Math.min(1, numerator / denominator));
}

/** SciPy's default additive-hyperbolic weighted Kendall tau. */
export function weightedKendallTau(
  x: readonly number[],
  y: readonly number[],
): Correlation {
  if (x.length <= 1) return [NaN, NaN];
  const xy = weightedTauForLexicographicRank(x, y, "x");
  const yx = weightedTauForLexicographicRank(x, y, "y");
  return [(xy + yx) / 2, NaN];
}
