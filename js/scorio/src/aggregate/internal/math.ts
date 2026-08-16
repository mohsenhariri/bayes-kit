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
  if (lo >= ordered.length - 1) return ordered[ordered.length - 1]!;
  const fraction = index - lo;
  const low = ordered[lo]!;
  const high = ordered[lo + 1]!;
  // NumPy's `_lerp` switches endpoints at gamma >= 0.5 so the interpolation is
  // anchored to the nearer sample; keeping the same branch keeps the result
  // bit-identical rather than merely close.
  return fraction < 0.5
    ? low + (high - low) * fraction
    : high - (high - low) * (1 - fraction);
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

// Loader's saddle-point form of the binomial pmf. Differencing three `gammaln`
// values instead loses about `n * eps` of relative accuracy (the log-gamma of
// n = 10000 is ~82099, so a 1-ulp slip there is a 2e-11 relative error in the
// exponentiated term), which would dominate the whole tail sum.
const LN_SQRT_2PI = 0.918938533204672741780329736406;
const STIRLING_HALVES = [
  0.0, 0.1534264097200273452913848, 0.0810614667953272582196702,
  0.0548141210519176538961390, 0.0413406959554092940938221,
  0.03316287351993628748511048, 0.02767792568499833914878929,
  0.02374616365629749597132920, 0.02079067210376509311152277,
  0.01848845053267318523077934, 0.01664469118982119216319487,
  0.01513497322191737887351255, 0.01387612882307074799874573,
  0.01281046524292022692424986, 0.01189670994589177009505572,
  0.01110455975820691732662991, 0.010411265261972096497478567,
  0.009799416126158803298389475, 0.009255462182712732917728637,
  0.008768700134139385462952823, 0.008330563433362871256469318,
  0.007934114564314020547248100, 0.007573675487951840794972024,
  0.007244554301320383179543912, 0.006942840107209529865664152,
  0.006665247032707682442354394, 0.006408994188004207068439631,
  0.006171712263039457647532867, 0.005951370112758847735624416,
  0.005746216513010115682023589, 0.005554733551962801371038690,
];

/** `log(n!) - log(sqrt(2*pi*n) * (n/e)^n)`, the Stirling series remainder. */
export function stirlingError(n: number): number {
  const s0 = 0.083333333333333333333;
  const s1 = 0.00277777777777777777778;
  const s2 = 0.00079365079365079365079365;
  const s3 = 0.000595238095238095238095238;
  const s4 = 0.0008417508417508417508417508;
  if (n <= 15) {
    const halves = n + n;
    if (halves === Math.floor(halves)) return STIRLING_HALVES[halves]!;
    return gammaln(n + 1) - (n + 0.5) * Math.log(n) + n - LN_SQRT_2PI;
  }
  const squared = n * n;
  if (n > 500) return (s0 - s1 / squared) / n;
  if (n > 80) return (s0 - (s1 - s2 / squared) / squared) / n;
  if (n > 35) return (s0 - (s1 - (s2 - s3 / squared) / squared) / squared) / n;
  return (s0 - (s1 - (s2 - (s3 - s4 / squared) / squared) / squared) / squared) / n;
}

/** Loader's deviance `x*log(x/np) + np - x`, series-summed when `x ~= np`. */
export function binomialDeviance(x: number, np: number): number {
  if (Math.abs(x - np) < 0.1 * (x + np)) {
    const ratio = (x - np) / (x + np);
    const squared = ratio * ratio;
    let sum = (x - np) * ratio;
    let term = 2 * x * ratio;
    for (let j = 1; j < 1000; j++) {
      term *= squared;
      const next = sum + term / (2 * j + 1);
      if (next === sum) return next;
      sum = next;
    }
  }
  return x * Math.log(x / np) + np - x;
}

/** `C(n, k) / 2^n`, evaluated without differencing large log-gamma values. */
function binomialHalfPmf(n: number, k: number): number {
  if (k === 0 || k === n) return 0.5 ** n;
  const half = n / 2;
  const exponent =
    stirlingError(n) -
    stirlingError(k) -
    stirlingError(n - k) -
    binomialDeviance(k, half) -
    binomialDeviance(n - k, half);
  return Math.exp(exponent) / Math.sqrt((2 * Math.PI * k * (n - k)) / n);
}

/**
 * Upper tail `P(Binomial(n, 1/2) >= start)` when `start > n/2`.
 *
 * The terms decrease from `start`, so a forward recurrence plus compensated
 * summation stays accurate for the large, nearly tied integer counts where a
 * generic incomplete-beta continued fraction loses several decimal places.
 */
function binomialHalfUpperTail(n: number, start: number): number {
  let term = binomialHalfPmf(n, start);
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
