/**
 * Special functions used by the ranking estimators (`erf`/normal CDF & PDF,
 * `logaddexp`, `logsumexp`, clipping). The standard-normal quantile (`ndtri`)
 * is reused from the eval port.
 */

export { ndtri as normPpf } from "../../eval/internal/math.js";

const SQRT_2 = Math.SQRT2;
const SQRT_2PI = Math.sqrt(2 * Math.PI);
const TWO_OVER_SQRT_PI = 2 / Math.sqrt(Math.PI);

/** `erf` via Maclaurin series for `|x| < 1`, accurate to full double precision. */
function erfSeries(z: number): number {
  // erf(z) = (2/√π) Σ_{n≥0} (-1)^n z^{2n+1} / (n! (2n+1))
  let term = z; // n = 0 term before the 1/(2n+1) factor is folded in below
  let sum = z;
  const z2 = z * z;
  for (let n = 1; n < 200; n++) {
    term *= -z2 / n;
    const add = term / (2 * n + 1);
    sum += add;
    if (Math.abs(add) < 1e-18 * Math.abs(sum)) break;
  }
  return TWO_OVER_SQRT_PI * sum;
}

/** `erfc` for `z >= 1` via the Lentz continued fraction (accurate into the tail). */
function erfcLarge(z: number): number {
  // erfc(z) = (e^{-z²}/√π) · 1/(z + (1/2)/(z + 1/(z + (3/2)/(z + 2/(z + ...)))))
  const tiny = 1e-300;
  let f = z;
  if (f === 0) f = tiny;
  let c = f;
  let d = 0;
  for (let i = 1; i < 300; i++) {
    const a = i / 2;
    // Continued fraction terms alternate between b = z (even) and the running
    // numerator a; here we use the standard erfc CF with partial numerators a_i
    // and partial denominators z.
    d = z + a * d;
    if (d === 0) d = tiny;
    c = z + a / c;
    if (c === 0) c = tiny;
    d = 1 / d;
    const delta = c * d;
    f *= delta;
    if (Math.abs(delta - 1) < 1e-16) break;
  }
  return (Math.exp(-z * z) / Math.sqrt(Math.PI)) * (1 / f);
}

/** Complementary error function, accurate across the whole real line. */
export function erfc(x: number): number {
  const z = Math.abs(x);
  const base = z < 1 ? 1 - erfSeries(z) : erfcLarge(z);
  return x >= 0 ? base : 2 - base;
}

/** Error function, accurate across the whole real line. */
export function erf(x: number): number {
  if (x === 0) return 0;
  const z = Math.abs(x);
  const val = z < 1 ? erfSeries(z) : 1 - erfcLarge(z);
  return x >= 0 ? val : -val;
}

/** Standard-normal CDF `Φ(x)`, computed via `erfc` to stay accurate in both tails. */
export function normCdf(x: number): number {
  return 0.5 * erfc(-x / SQRT_2);
}

/** Standard-normal PDF `φ(x)`. */
export function normPdf(x: number): number {
  return Math.exp(-0.5 * x * x) / SQRT_2PI;
}

/** `log(exp(a) + exp(b))` computed stably. */
export function logaddexp(a: number, b: number): number {
  if (a === -Infinity) return b;
  if (b === -Infinity) return a;
  const hi = a > b ? a : b;
  const lo = a > b ? b : a;
  return hi + Math.log1p(Math.exp(lo - hi));
}

/** `log(Σ exp(values))` computed stably. */
export function logsumexp(values: readonly number[]): number {
  if (values.length === 0) return -Infinity;
  let max = -Infinity;
  for (const v of values) if (v > max) max = v;
  if (max === -Infinity) return -Infinity;
  let sum = 0;
  for (const v of values) sum += Math.exp(v - max);
  return max + Math.log(sum);
}

/** Clamp `x` into `[lo, hi]` (NumPy `np.clip`). */
export function clip(x: number, lo: number, hi: number): number {
  return x < lo ? lo : x > hi ? hi : x;
}
