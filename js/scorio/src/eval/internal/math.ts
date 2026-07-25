/**
 * Special-function primitives used by the eval metrics.
 *
 * These reimplement the small slice of `scipy.special` that the Python
 * reference relies on (`gammaln`, `betaln`, `comb`, `ndtri`) so that the
 * package has zero runtime dependencies.
 */

// Lanczos approximation coefficients (g = 7, n = 9). Accurate to ~1e-15.
const LANCZOS_G = 7;
const LANCZOS_COEFFS = [
  0.99999999999980993, 676.5203681218851, -1259.1392167224028,
  771.32342877765313, -176.61502916214059, 12.507343278686905,
  -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
];

/** Natural log of the absolute value of the Gamma function, `lgamma(x)`. */
export function gammaln(x: number): number {
  if (Number.isNaN(x)) return NaN;
  // Reflection formula for x < 0.5. Use |sin| because this returns the log of
  // the absolute value of Gamma; without it, negative non-integers (where
  // sin(pi x) < 0) would yield log of a negative number, i.e. NaN.
  if (x < 0.5) {
    return (
      Math.log(Math.PI / Math.abs(Math.sin(Math.PI * x))) - gammaln(1 - x)
    );
  }
  x -= 1;
  let a = LANCZOS_COEFFS[0]!;
  const t = x + LANCZOS_G + 0.5;
  for (let i = 1; i < LANCZOS_COEFFS.length; i++) {
    a += LANCZOS_COEFFS[i]! / (x + i);
  }
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
}

/** Natural log of the Beta function: `betaln(a, b) = lgamma(a)+lgamma(b)-lgamma(a+b)`. */
export function betaln(a: number, b: number): number {
  return gammaln(a) + gammaln(b) - gammaln(a + b);
}

/**
 * Binomial coefficient `C(n, k)` as a float, matching
 * `scipy.special.comb(n, k, exact=False)`: returns 0 when `k < 0`, `n < 0`,
 * or `k > n`.
 */
export function comb(n: number, k: number): number {
  if (k < 0 || n < 0 || k > n) return 0;
  if (k === 0 || k === n) return 1;
  return Math.exp(gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1));
}

/** Natural logarithm of the binomial coefficient, with invalid choices at -Infinity. */
export function logComb(n: number, k: number): number {
  if (k < 0 || n < 0 || k > n) return -Infinity;
  if (k === 0 || k === n) return 0.0;
  const kk = Math.min(k, n - k);
  return gammaln(n + 1) - gammaln(kk + 1) - gammaln(n - kk + 1);
}

function hypergeomSupport(
  population: number,
  successes: number,
  draws: number,
): [number, number] {
  return [Math.max(0, draws - (population - successes)), Math.min(draws, successes)];
}

/** Log PMF for X ~ Hypergeometric(population, successes, draws). */
export function logHypergeomPmf(
  population: number,
  successes: number,
  draws: number,
  observed: number,
): number {
  const [lo, hi] = hypergeomSupport(population, successes, draws);
  if (observed < lo || observed > hi) return -Infinity;
  return (
    logComb(successes, observed) +
    logComb(population - successes, draws - observed) -
    logComb(population, draws)
  );
}

/** PMF for X ~ Hypergeometric(population, successes, draws). */
export function hypergeomPmf(
  population: number,
  successes: number,
  draws: number,
  observed: number,
): number {
  const logP = logHypergeomPmf(population, successes, draws, observed);
  if (logP === -Infinity) return 0.0;
  return Math.min(1.0, Math.max(0.0, Math.exp(logP)));
}

/** P(X >= minSuccesses) for a hypergeometric variate, summed in log space. */
export function hypergeomSf(
  population: number,
  successes: number,
  draws: number,
  minSuccesses: number,
): number {
  const [lo, hi] = hypergeomSupport(population, successes, draws);
  const start = Math.max(lo, minSuccesses);
  if (start <= lo) return 1.0;
  if (start > hi) return 0.0;

  let maxLog = -Infinity;
  const logs: number[] = [];
  for (let observed = start; observed <= hi; observed++) {
    const logP = logHypergeomPmf(population, successes, draws, observed);
    logs.push(logP);
    if (logP > maxLog) maxLog = logP;
  }

  // Kahan summation keeps broad tails close to one without allowing roundoff
  // to escape the probability range.
  let sum = 0.0;
  let correction = 0.0;
  for (const logP of logs) {
    const term = Math.exp(logP - maxLog) - correction;
    const next = sum + term;
    correction = next - sum - term;
    sum = next;
  }
  return Math.min(1.0, Math.max(0.0, Math.exp(maxLog) * sum));
}

/** P(X >= 1), evaluated without subtracting two nearly equal binomial coefficients. */
export function hypergeomAtLeastOne(
  population: number,
  successes: number,
  draws: number,
): number {
  if (successes <= 0) return 0.0;
  if (draws > population - successes) return 1.0;
  const logPZero = logComb(population - successes, draws) - logComb(population, draws);
  return Math.min(1.0, Math.max(0.0, -Math.expm1(logPZero)));
}

/**
 * `Beta(alpha + a, beta + b) / Beta(alpha, beta)`, computed stably in log
 * space. Used for closed-form posterior moments of `p^a (1-p)^b`.
 */
export function betaRatio(
  alpha: number,
  beta: number,
  a: number,
  b: number,
): number {
  return Math.exp(logBetaRatio(alpha, beta, a, b));
}

/** Logarithm of `Beta(alpha+a, beta+b) / Beta(alpha,beta)`. */
export function logBetaRatio(
  alpha: number,
  beta: number,
  a: number,
  b: number,
): number {
  return betaln(alpha + a, beta + b) - betaln(alpha, beta);
}

/**
 * Inverse of the standard-normal CDF (`scipy.special.ndtri` / probit).
 *
 * Peter Acklam's rational approximation; relative error < 1.15e-9 across the
 * full range, which is far tighter than the credible-interval use needs.
 */
export function ndtri(p: number): number {
  if (p <= 0) return p === 0 ? -Infinity : NaN;
  if (p >= 1) return p === 1 ? Infinity : NaN;

  // Rational approximation coefficients.
  const a = [
    -3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2,
    1.38357751867269e2, -3.066479806614716e1, 2.506628277459239,
  ];
  const b = [
    -5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2,
    6.680131188771972e1, -1.328068155288572e1,
  ];
  const c = [
    -7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838,
    -2.549732539343734, 4.374664141464968, 2.938163982698783,
  ];
  const d = [
    7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996,
    3.754408661907416,
  ];

  const pLow = 0.02425;
  const pHigh = 1 - pLow;
  let q: number;
  let r: number;

  if (p < pLow) {
    q = Math.sqrt(-2 * Math.log(p));
    return (
      (((((c[0]! * q + c[1]!) * q + c[2]!) * q + c[3]!) * q + c[4]!) * q +
        c[5]!) /
      ((((d[0]! * q + d[1]!) * q + d[2]!) * q + d[3]!) * q + 1)
    );
  }
  if (p <= pHigh) {
    q = p - 0.5;
    r = q * q;
    return (
      ((((((a[0]! * r + a[1]!) * r + a[2]!) * r + a[3]!) * r + a[4]!) * r +
        a[5]!) *
        q) /
      (((((b[0]! * r + b[1]!) * r + b[2]!) * r + b[3]!) * r + b[4]!) * r + 1)
    );
  }
  q = Math.sqrt(-2 * Math.log(1 - p));
  return (
    -(((((c[0]! * q + c[1]!) * q + c[2]!) * q + c[3]!) * q + c[4]!) * q +
      c[5]!) /
    ((((d[0]! * q + d[1]!) * q + d[2]!) * q + d[3]!) * q + 1)
  );
}
