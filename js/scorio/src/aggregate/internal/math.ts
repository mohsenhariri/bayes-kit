/** Numeric helpers needed by the dependency-free aggregation port. */

import { gammaln } from "../../eval/internal/math.js";

/** Arithmetic mean of a non-empty array. */
export function mean(values: readonly number[]): number {
  let total = 0;
  for (const value of values) total += value;
  return total / values.length;
}

/** NumPy's default (`method="linear"`) quantile for a non-empty array. */
export function quantile(values: readonly number[], q: number): number {
  const ordered = [...values].sort((a, b) => a - b);
  const index = (ordered.length - 1) * q;
  const lo = Math.floor(index);
  const hi = Math.ceil(index);
  if (lo === hi) return ordered[lo]!;
  const fraction = index - lo;
  return ordered[lo]! + fraction * (ordered[hi]! - ordered[lo]!);
}

// Regularized incomplete beta I_x(a, b), via the Numerical Recipes continued
// fraction. This is the only scipy.special primitive used by aggregate.
const FPMIN = 1e-300;
const EPSILON = 3e-16;
const MAX_ITERATIONS = 300;

function betaContinuedFraction(a: number, b: number, x: number): number {
  const qab = a + b;
  const qap = a + 1;
  const qam = a - 1;
  let c = 1;
  let d = 1 - (qab * x) / qap;
  if (Math.abs(d) < FPMIN) d = FPMIN;
  d = 1 / d;
  let h = d;

  for (let m = 1; m <= MAX_ITERATIONS; m++) {
    const m2 = 2 * m;
    let aa = (m * (b - m) * x) / ((qam + m2) * (a + m2));
    d = 1 + aa * d;
    if (Math.abs(d) < FPMIN) d = FPMIN;
    c = 1 + aa / c;
    if (Math.abs(c) < FPMIN) c = FPMIN;
    d = 1 / d;
    h *= d * c;

    aa = (-(a + m) * (qab + m) * x) / ((a + m2) * (qap + m2));
    d = 1 + aa * d;
    if (Math.abs(d) < FPMIN) d = FPMIN;
    c = 1 + aa / c;
    if (Math.abs(c) < FPMIN) c = FPMIN;
    d = 1 / d;
    const delta = d * c;
    h *= delta;
    if (Math.abs(delta - 1) < EPSILON) break;
  }
  return h;
}

/**
 * Upper tail `P(Binomial(n, 1/2) >= start)` when `start > n/2`.
 *
 * The terms decrease from `start`, so a forward recurrence plus compensated
 * summation stays accurate for the large, nearly tied integer counts where a
 * generic incomplete-beta continued fraction loses several decimal places.
 */
function binomialHalfUpperTail(n: number, start: number): number {
  const logTerm =
    gammaln(n + 1) -
    gammaln(start + 1) -
    gammaln(n - start + 1) -
    n * Math.LN2;
  let term = Math.exp(logTerm);
  if (term === 0) return 0;

  let sum = term;
  let compensation = 0;
  for (let j = start; j < n; j++) {
    term *= (n - j) / (j + 1);
    if (term === 0) break;
    const adjusted = term - compensation;
    const next = sum + adjusted;
    compensation = next - sum - adjusted;
    sum = next;
    if (term <= Number.EPSILON * sum && j > n / 2) break;
  }
  return sum;
}

/** Regularized incomplete beta `I_x(a, b)` (`scipy.special.betainc`). */
export function betainc(a: number, b: number, x: number): number {
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  // Adaptive-Consistency always reaches this integer/x=1/2 branch. The
  // beta-binomial identity is both more accurate for large counts and gives
  // the exact symmetry value 0.5 for tied counts.
  if (
    x === 0.5 &&
    Number.isSafeInteger(a) &&
    Number.isSafeInteger(b) &&
    a > 0 &&
    b > 0
  ) {
    if (a === b) return 0.5;
    const n = a + b - 1;
    if (a > b) return binomialHalfUpperTail(n, a);
    return 1 - binomialHalfUpperTail(n, b);
  }
  const logFront =
    gammaln(a + b) -
    gammaln(a) -
    gammaln(b) +
    a * Math.log(x) +
    b * Math.log1p(-x);
  const front = Math.exp(logFront);
  if (x < (a + 1) / (a + b + 2)) {
    return (front * betaContinuedFraction(a, b, x)) / a;
  }
  return 1 - (front * betaContinuedFraction(b, a, 1 - x)) / b;
}
