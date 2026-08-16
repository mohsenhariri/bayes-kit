/** Online stopping rules for candidate sampling and trace generation. */

import {
  groupConfidences,
  type TopKLogprobs,
  tokenConfidence,
} from "./confidence.js";
import { isValidAnswer } from "./internal/base.js";
import {
  betainc,
  binomialDeviance,
  quantile,
  stirlingError,
} from "./internal/math.js";
import {
  asFiniteVector,
  type NumericInput,
} from "./internal/numeric.js";
import { gammaln, ndtri } from "../eval/internal/math.js";
import {
  pythonComparableNumber,
  pythonInt,
  pythonTruthy,
} from "./internal/runtime.js";

function orderedAnswerCounts<T>(answers: Iterable<T>): number[] {
  const tally = new Map<T, number>();
  for (const answer of answers) {
    if (!isValidAnswer(answer)) continue;
    tally.set(answer, (tally.get(answer) ?? 0) + 1);
  }
  return [...tally.values()];
}

function topTwoCounts<T>(answers: Iterable<T>): [number, number] {
  const counts = orderedAnswerCounts(answers).sort((a, b) => b - a);
  return [counts[0] ?? 0, counts[1] ?? 0];
}

export interface AdaptiveConsistencyStopOptions<ReturnProb extends boolean = false> {
  threshold?: number;
  returnProb?: ReturnProb;
}

/** Bayesian Adaptive-Consistency stop based on the top-two answer counts. */
export function adaptiveConsistencyStop<T, ReturnProb extends boolean = false>(
  answers: Iterable<T>,
  options: AdaptiveConsistencyStopOptions<ReturnProb> = {},
): ReturnProb extends true ? [boolean, number] : boolean {
  const { threshold = 0.95 } = options;
  const boundary = pythonComparableNumber(threshold, "threshold");
  if (!(boundary > 0 && boundary < 1)) {
    throw new Error(`threshold must be in (0, 1); got ${threshold}.`);
  }
  const [first, second] = topTwoCounts(answers);
  const probability =
    first === 0 ? 0 : 1 - betainc(first + 1, second + 1, 0.5);
  const stop = probability >= boundary;
  return (pythonTruthy(options.returnProb) ? [stop, probability] : stop) as ReturnProb extends true
    ? [boolean, number]
    : boolean;
}

const GAMMA_EPSILON = 1e-14;
const GAMMA_FPMIN = 1e-300;
const GAMMA_MIN_ITERATIONS = 500;
const GAMMA_INVERSION_STEPS = 60;

/**
 * `log(x^a e^{-x} / Gamma(a))`, the prefactor shared by both gamma expansions.
 *
 * Differencing `a * log(x)` against `gammaln(a)` loses about `gammaln(a) * eps`
 * of accuracy, and `gammaln(a)` reaches ~1e6 for the shapes this integral sees.
 * Since `x^{a-1} e^{-x} / Gamma(a)` is the Poisson pmf at `a - 1` with mean `x`,
 * Loader's saddle-point form evaluates it without that subtraction.
 */
function logGammaPrefactor(a: number, x: number): number {
  if (a > 1) {
    return (
      Math.log(x) -
      stirlingError(a - 1) -
      binomialDeviance(a - 1, x) -
      0.5 * Math.log(2 * Math.PI * (a - 1))
    );
  }
  return -x + a * Math.log(x) - gammaln(a);
}

/** Regularized lower incomplete gamma P(a, x), for positive a and x >= 0. */
function regularizedGammaP(a: number, x: number): number {
  if (x <= 0) return 0;
  if (x === Infinity) return 1;
  const logFactor = logGammaPrefactor(a, x);
  // Both expansions need O(sqrt(a)) terms when x sits near a, so a fixed cap
  // silently truncates exactly where the Dirichlet integrand is most sensitive.
  const limit = Math.max(
    GAMMA_MIN_ITERATIONS,
    Math.ceil(40 * Math.sqrt(a)) + 100,
  );

  if (x < a + 1) {
    let ap = a;
    let term = 1 / a;
    let sum = term;
    for (let iteration = 1; iteration <= limit; iteration++) {
      ap += 1;
      term *= x / ap;
      sum += term;
      if (Math.abs(term) <= Math.abs(sum) * GAMMA_EPSILON) break;
    }
    return Math.max(0, Math.min(1, sum * Math.exp(logFactor)));
  }

  let b = x + 1 - a;
  let c = 1 / GAMMA_FPMIN;
  let d = 1 / (Math.abs(b) < GAMMA_FPMIN ? GAMMA_FPMIN : b);
  let h = d;
  for (let iteration = 1; iteration <= limit; iteration++) {
    const an = -iteration * (iteration - a);
    b += 2;
    d = an * d + b;
    if (Math.abs(d) < GAMMA_FPMIN) d = GAMMA_FPMIN;
    c = b + an / c;
    if (Math.abs(c) < GAMMA_FPMIN) c = GAMMA_FPMIN;
    d = 1 / d;
    const delta = d * c;
    h *= delta;
    if (Math.abs(delta - 1) <= GAMMA_EPSILON) break;
  }
  const q = Math.exp(logFactor) * h;
  return Math.max(0, Math.min(1, 1 - q));
}

/** Gamma(shape, 1) density. */
function gammaDensity(shape: number, x: number): number {
  return x <= 0 ? 0 : Math.exp(logGammaPrefactor(shape, x)) / x;
}

function gammaQuantile(shape: number, probability: number): number {
  if (probability <= 0) return 0;
  if (probability >= 1) return Infinity;
  // Wilson-Hilferty seeds a safeguarded Newton iteration. Newton converges in a
  // handful of steps where plain bisection needs ~90, and each step costs a
  // full O(sqrt(shape))-term CDF evaluation.
  const scale = 1 / (9 * shape);
  let x = shape * (1 - scale + ndtri(probability) * Math.sqrt(scale)) ** 3;
  if (!Number.isFinite(x) || x <= 0) x = shape;

  let low = 0;
  let high = Infinity;
  for (let iteration = 0; iteration < GAMMA_INVERSION_STEPS; iteration++) {
    const cdf = regularizedGammaP(shape, x);
    if (cdf < probability) low = x;
    else high = x;
    const density = gammaDensity(shape, x);
    let next = density > 0 ? x + (probability - cdf) / density : NaN;
    if (!Number.isFinite(next) || !(next > low && next < high)) {
      next = Number.isFinite(high) ? 0.5 * (low + high) : Math.max(2 * x, low + 1);
    }
    const converged = Math.abs(next - x) <= Math.abs(next) * 1e-15;
    x = next;
    if (converged) break;
  }
  return x;
}

// Adaptive 15-point Gauss-Kronrod quadrature, matching the `epsabs`/`epsrel`/
// `limit` settings Python passes to `scipy.integrate.quad`. The integrand below
// concentrates almost all of its variation in a narrow layer near one endpoint,
// which a fixed-depth Simpson rule resolves only to a few parts in 1e8; an
// error-driven subdivision spends its panels where that variation actually is.
const KRONROD_NODES = [
  0.991455371120812639206854697526329,
  0.949107912342758524526189684047851,
  0.864864423359769072789712788640926,
  0.741531185599394439863864773280788,
  0.586087235467691130294144838258730,
  0.405845151377397166906606412076961,
  0.207784955007898467600689403773245,
  0.0,
];
const KRONROD_WEIGHTS = [
  0.022935322010529224963732008058970,
  0.063092092629978553290700663189204,
  0.104790010322250183839876322541518,
  0.140653259715525918745189590510238,
  0.169004726639267902826583426598550,
  0.190350578064785409913256402421014,
  0.204432940075298892414161999234649,
  0.209482141084727828012999174891714,
];
const GAUSS_WEIGHTS = [
  0.129484966168869693270611432679082,
  0.279705391489276667901467771423780,
  0.381830050505118944950369775488975,
  0.417959183673469387755102040816327,
];
const QUADRATURE_EPSABS = 1e-10;
const QUADRATURE_EPSREL = 1e-10;
const QUADRATURE_LIMIT = 250;
const MACHINE_EPSILON = 2.220446049250313e-16;
const UNDERFLOW = 2.2250738585072014e-308;

interface QuadraturePanel {
  left: number;
  right: number;
  value: number;
  error: number;
}

/** One QUADPACK `dqk15` panel: the Kronrod value and its Gauss-difference error. */
function kronrod15(
  fn: (value: number) => number,
  left: number,
  right: number,
): { value: number; error: number } {
  const center = 0.5 * (left + right);
  const halfLength = 0.5 * (right - left);
  const centerValue = fn(center);

  const lower = new Array<number>(7);
  const upper = new Array<number>(7);
  let gauss = centerValue * GAUSS_WEIGHTS[3]!;
  let kronrod = centerValue * KRONROD_WEIGHTS[7]!;
  let absoluteIntegral = Math.abs(kronrod);

  for (let j = 0; j < 7; j++) {
    const offset = halfLength * KRONROD_NODES[j]!;
    const below = fn(center - offset);
    const above = fn(center + offset);
    lower[j] = below;
    upper[j] = above;
    const total = below + above;
    // Odd Kronrod nodes (1-based even) coincide with the 7-point Gauss nodes.
    if (j % 2 === 1) gauss += GAUSS_WEIGHTS[(j - 1) / 2]! * total;
    kronrod += KRONROD_WEIGHTS[j]! * total;
    absoluteIntegral += KRONROD_WEIGHTS[j]! * (Math.abs(below) + Math.abs(above));
  }

  const halfKronrod = kronrod * 0.5;
  let oscillation = KRONROD_WEIGHTS[7]! * Math.abs(centerValue - halfKronrod);
  for (let j = 0; j < 7; j++) {
    oscillation +=
      KRONROD_WEIGHTS[j]! *
      (Math.abs(lower[j]! - halfKronrod) + Math.abs(upper[j]! - halfKronrod));
  }

  const magnitude = Math.abs(halfLength);
  const value = kronrod * halfLength;
  const scaledAbsolute = absoluteIntegral * magnitude;
  const scaledOscillation = oscillation * magnitude;
  let error = Math.abs((kronrod - gauss) * halfLength);
  if (scaledOscillation !== 0 && error !== 0) {
    error =
      scaledOscillation *
      Math.min(1, ((200 * error) / scaledOscillation) ** 1.5);
  }
  if (scaledAbsolute > UNDERFLOW / (50 * MACHINE_EPSILON)) {
    error = Math.max(MACHINE_EPSILON * 50 * scaledAbsolute, error);
  }
  return { value, error };
}

/** Error-driven bisection over Gauss-Kronrod panels (QUADPACK `dqag` core). */
function adaptiveQuadrature(
  fn: (value: number) => number,
  left: number,
  right: number,
): { value: number; error: number } {
  const first = kronrod15(fn, left, right);
  const panels: QuadraturePanel[] = [{ left, right, ...first }];
  let value = first.value;
  let error = first.error;

  for (let iteration = 1; iteration < QUADRATURE_LIMIT; iteration++) {
    if (error <= Math.max(QUADRATURE_EPSABS, QUADRATURE_EPSREL * Math.abs(value))) {
      break;
    }
    let worst = 0;
    for (let index = 1; index < panels.length; index++) {
      if (panels[index]!.error > panels[worst]!.error) worst = index;
    }
    const panel = panels[worst]!;
    const middle = 0.5 * (panel.left + panel.right);
    if (!(middle > panel.left && middle < panel.right)) break;
    const below = kronrod15(fn, panel.left, middle);
    const above = kronrod15(fn, middle, panel.right);
    value += below.value + above.value - panel.value;
    error += below.error + above.error - panel.error;
    panels[worst] = { left: panel.left, right: middle, ...below };
    panels.push({ left: middle, right: panel.right, ...above });
  }
  return { value, error };
}

function dirichletLeaderProbability(counts: readonly number[]): number {
  if (counts.length === 0) return 0;
  let leader = 0;
  for (let index = 1; index < counts.length; index++) {
    if (counts[index]! > counts[leader]!) leader = index;
  }
  const leaderShape = counts[leader]! + 1;
  const otherShapes = counts
    .filter((_, index) => index !== leader)
    .map((count) => count + 1);
  if (otherShapes.every((shape) => shape === leaderShape)) {
    return 1 / counts.length;
  }
  const integrand = (probability: number): number => {
    if (probability <= 0) return 0;
    if (probability >= 1) return 1;
    const value = gammaQuantile(leaderShape, probability);
    let logProduct = 0;
    for (const shape of otherShapes) {
      const cdf = regularizedGammaP(shape, value);
      if (cdf <= 0) return 0;
      logProduct += Math.log(cdf);
    }
    return Math.exp(logProduct);
  };

  const { value, error } = adaptiveQuadrature(integrand, 0, 1);
  const tolerance = Math.max(1e-8, 1e-7 * Math.abs(value));
  if (!Number.isFinite(value) || !Number.isFinite(error) || error > tolerance) {
    throw new Error(
      "Dirichlet leader-probability integration did not converge " +
        `(estimated error ${error}).`,
    );
  }
  return Math.max(0, Math.min(1, value));
}

export interface AdaptiveConsistencyDirichletStopOptions<
  ReturnProb extends boolean = false,
> extends AdaptiveConsistencyStopOptions<ReturnProb> {}

/** Full observed-support Dirichlet Adaptive-Consistency stopping criterion. */
export function adaptiveConsistencyDirichletStop<
  T,
  ReturnProb extends boolean = false,
>(
  answers: Iterable<T>,
  options: AdaptiveConsistencyDirichletStopOptions<ReturnProb> = {},
): ReturnProb extends true ? [boolean, number] : boolean {
  const requested = options.threshold === undefined ? 0.95 : options.threshold;
  const threshold = pythonComparableNumber(requested, "threshold");
  if (!(threshold > 0 && threshold < 1)) {
    throw new Error(`threshold must be in (0, 1); got ${requested}.`);
  }
  const counts = orderedAnswerCounts(answers);
  let probability: number;
  if (counts.length === 0) {
    probability = 0;
  } else if (counts.length < 3) {
    const ordered = [...counts].sort((left, right) => right - left);
    const first = ordered[0]!;
    const second = ordered[1] ?? 0;
    probability = 1 - betainc(first + 1, second + 1, 0.5);
  } else {
    probability = dirichletLeaderProbability(counts);
  }
  const stop = counts.length > 0 && probability >= threshold;
  return (pythonTruthy(options.returnProb) ? [stop, probability] : stop) as ReturnProb extends true
    ? [boolean, number]
    : boolean;
}

class CrpRandom {
  private state: number;
  private readonly unseeded: boolean;

  constructor(seed: number | null) {
    this.unseeded = seed === null;
    this.state = seed === null ? 0 : seed >>> 0;
  }

  random(): number {
    if (this.unseeded) return Math.random();
    this.state = (this.state + 0x6d2b79f5) | 0;
    let value = this.state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  }

  normal(): number {
    let first = this.random();
    const second = this.random();
    if (first < 1e-300) first = 1e-300;
    return Math.sqrt(-2 * Math.log(first)) * Math.cos(2 * Math.PI * second);
  }

  gamma(shape: number): number {
    if (shape < 1) {
      const uniform = Math.max(this.random(), 1e-300);
      return this.gamma(shape + 1) * uniform ** (1 / shape);
    }
    const d = shape - 1 / 3;
    const c = 1 / Math.sqrt(9 * d);
    for (;;) {
      let normal: number;
      let base: number;
      do {
        normal = this.normal();
        base = 1 + c * normal;
      } while (base <= 0);
      const cube = base ** 3;
      const uniform = this.random();
      if (uniform < 1 - 0.0331 * normal ** 4) return d * cube;
      if (
        Math.log(Math.max(uniform, 1e-300)) <
        0.5 * normal * normal + d * (1 - cube + Math.log(cube))
      ) {
        return d * cube;
      }
    }
  }
}

function crpLeaderProbability(
  counts: readonly number[],
  horizon: number,
  nAlpha: number,
  nSimulations: number,
  seed: number | null,
): number {
  const observed = counts.reduce((total, count) => total + count, 0);
  const remaining = horizon - observed;
  let leader = 0;
  for (let index = 1; index < counts.length; index++) {
    if (counts[index]! > counts[leader]!) leader = index;
  }
  const rate = 1 + 0.5772156649015329 + Math.log(observed);
  const random = new CrpRandom(seed);
  let successes = 0;
  for (let alphaIndex = 0; alphaIndex < nAlpha; alphaIndex++) {
    const alpha = random.gamma(counts.length) / rate;
    for (let simulation = 0; simulation < nSimulations; simulation++) {
      const clusters = [...counts];
      let customers = observed;
      for (let drawIndex = 0; drawIndex < remaining; drawIndex++) {
        const draw = random.random() * (customers + alpha);
        if (draw >= customers) {
          clusters.push(1);
        } else {
          let cumulative = 0;
          let chosen = clusters.length - 1;
          for (let cluster = 0; cluster < clusters.length; cluster++) {
            cumulative += clusters[cluster]!;
            if (draw < cumulative) {
              chosen = cluster;
              break;
            }
          }
          clusters[chosen]! += 1;
        }
        customers += 1;
      }
      let finalLeader = 0;
      for (let cluster = 1; cluster < clusters.length; cluster++) {
        if (clusters[cluster]! > clusters[finalLeader]!) finalLeader = cluster;
      }
      if (finalLeader === leader) successes += 1;
    }
  }
  return successes / (nAlpha * nSimulations);
}

export interface AdaptiveConsistencyCrpStopOptions<
  ReturnProb extends boolean = false,
> extends AdaptiveConsistencyStopOptions<ReturnProb> {
  horizon?: number;
  nAlpha?: number;
  nSimulations?: number;
  seed?: number | null;
}

/** Finite-horizon CRP Adaptive-Consistency Monte Carlo comparator. */
export function adaptiveConsistencyCrpStop<
  T,
  ReturnProb extends boolean = false,
>(
  answers: Iterable<T>,
  options: AdaptiveConsistencyCrpStopOptions<ReturnProb> = {},
): ReturnProb extends true ? [boolean, number] : boolean {
  const {
    threshold = 0.95,
    horizon = 40,
    nAlpha = 100,
    nSimulations = 1000,
    seed = 0,
  } = options;
  const boundary = pythonComparableNumber(threshold, "threshold");
  if (!(boundary > 0 && boundary < 1)) {
    throw new Error(`threshold must be in (0, 1); got ${threshold}.`);
  }
  for (const [value, name] of [
    [horizon, "horizon"],
    [nAlpha, "n_alpha"],
    [nSimulations, "n_simulations"],
  ] as const) {
    if (!Number.isInteger(value) || value < 1) {
      throw new Error(`${name} must be an integer >= 1; got ${value}.`);
    }
  }
  if (seed !== null && (!Number.isInteger(seed) || seed < 0)) {
    throw new Error(`seed must be a non-negative integer or null; got ${seed}.`);
  }

  const counts = orderedAnswerCounts(answers);
  if (counts.length === 0) {
    return (pythonTruthy(options.returnProb) ? [false, 0] : false) as ReturnProb extends true
      ? [boolean, number]
      : boolean;
  }
  const observed = counts.reduce((total, count) => total + count, 0);
  if (observed >= horizon) {
    return (pythonTruthy(options.returnProb) ? [true, 1] : true) as ReturnProb extends true
      ? [boolean, number]
      : boolean;
  }
  const probability = crpLeaderProbability(
    counts,
    horizon,
    nAlpha,
    nSimulations,
    seed,
  );
  const stop = probability >= boundary;
  return (pythonTruthy(options.returnProb) ? [stop, probability] : stop) as ReturnProb extends true
    ? [boolean, number]
    : boolean;
}

/** Stop when a non-empty answer window is entirely valid and unanimous. */
export function escStop<T>(windowAnswers: Iterable<T>): boolean {
  const answers = [...windowAnswers];
  if (answers.length === 0) return false;
  const first = answers[0]!;
  if (!isValidAnswer(first)) return false;
  for (let i = 1; i < answers.length; i++) {
    if (!isValidAnswer(answers[i]!) || answers[i] !== first) return false;
  }
  return true;
}

/** Quantile threshold retaining the most-confident `keep` fraction of warmups. */
export function deepconfStopThreshold(
  warmupConfidences: NumericInput,
  options: { keep?: number } = {},
): number {
  const requested = options.keep === undefined ? 0.1 : options.keep;
  const keep = pythonComparableNumber(requested, "keep");
  if (!(keep > 0 && keep <= 1)) {
    throw new Error(`keep must be in (0, 1]; got ${keep}.`);
  }
  return quantile(
    asFiniteVector(
      warmupConfidences,
      "warmup_confidences",
      "need at least one warmup confidence.",
    ),
    1 - keep,
  );
}

/** First completed window whose DeepConf group confidence is below threshold. */
export function deepconfOnlineStop(
  topkLogprobs: TopKLogprobs,
  threshold: number,
  options: { window?: number } = {},
): number | null {
  const confidence = tokenConfidence(topkLogprobs);
  const requested = options.window === undefined ? 2048 : options.window;
  const width = Math.min(pythonInt(requested, "window"), confidence.length);
  const groups = groupConfidences(confidence, width);
  const boundary = pythonComparableNumber(threshold, "threshold");
  for (let start = 0; start < groups.length; start++) {
    if (groups[start]! < boundary) return start + width - 1;
  }
  return null;
}
