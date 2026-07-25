/** Online stopping rules for candidate sampling and trace generation. */

import {
  groupConfidences,
  type TopKLogprobs,
  tokenConfidence,
} from "./confidence.js";
import { isValidAnswer } from "./internal/base.js";
import { betainc, quantile } from "./internal/math.js";
import {
  asFiniteVector,
  type NumericInput,
} from "./internal/numeric.js";
import { gammaln } from "../eval/internal/math.js";
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
const GAMMA_MAX_ITERATIONS = 500;
const GAMMA_NORMAL_APPROXIMATION_SHAPE = 10_000;

/** Standard-normal CDF expressed through the accurately convergent a=1/2 case. */
function normalCdf(value: number): number {
  if (value === 0) return 0.5;
  const magnitude = regularizedGammaP(0.5, (value * value) / 2);
  return value > 0 ? (1 + magnitude) / 2 : (1 - magnitude) / 2;
}

/** Regularized lower incomplete gamma P(a, x), for positive a and x >= 0. */
function regularizedGammaP(a: number, x: number): number {
  if (x <= 0) return 0;
  if (x === Infinity) return 1;
  // The Lanczos subtraction in the exact series loses useful digits once both
  // a and x are very large and close.  Wilson-Hilferty is asymptotically
  // accurate in precisely that regime and preserves SciPy parity for the
  // large-count Dirichlet integral.
  if (a >= GAMMA_NORMAL_APPROXIMATION_SHAPE) {
    const z =
      (Math.cbrt(x / a) - (1 - 1 / (9 * a))) /
      Math.sqrt(1 / (9 * a));
    return normalCdf(z);
  }
  const logFactor = -x + a * Math.log(x) - gammaln(a);
  if (x < a + 1) {
    let ap = a;
    let term = 1 / a;
    let sum = term;
    for (let iteration = 1; iteration <= GAMMA_MAX_ITERATIONS; iteration++) {
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
  for (let iteration = 1; iteration <= GAMMA_MAX_ITERATIONS; iteration++) {
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

function gammaQuantile(shape: number, probability: number): number {
  if (probability <= 0) return 0;
  if (probability >= 1) return Infinity;
  let low = 0;
  let high = Math.max(1, shape);
  while (regularizedGammaP(shape, high) < probability) {
    high *= 2;
    if (!Number.isFinite(high)) return Infinity;
  }
  for (let iteration = 0; iteration < 90; iteration++) {
    const middle = (low + high) / 2;
    if (regularizedGammaP(shape, middle) < probability) low = middle;
    else high = middle;
  }
  return (low + high) / 2;
}

function adaptiveSimpson(
  fn: (value: number) => number,
  left: number,
  right: number,
  tolerance: number,
  whole: number,
  fLeft: number,
  fMiddle: number,
  fRight: number,
  depth: number,
): number {
  const middle = (left + right) / 2;
  const leftMiddle = (left + middle) / 2;
  const rightMiddle = (middle + right) / 2;
  const fLeftMiddle = fn(leftMiddle);
  const fRightMiddle = fn(rightMiddle);
  const leftArea =
    ((middle - left) * (fLeft + 4 * fLeftMiddle + fMiddle)) / 6;
  const rightArea =
    ((right - middle) * (fMiddle + 4 * fRightMiddle + fRight)) / 6;
  const delta = leftArea + rightArea - whole;
  if (depth <= 0 || Math.abs(delta) <= 15 * tolerance) {
    return leftArea + rightArea + delta / 15;
  }
  return (
    adaptiveSimpson(
      fn,
      left,
      middle,
      tolerance / 2,
      leftArea,
      fLeft,
      fLeftMiddle,
      fMiddle,
      depth - 1,
    ) +
    adaptiveSimpson(
      fn,
      middle,
      right,
      tolerance / 2,
      rightArea,
      fMiddle,
      fRightMiddle,
      fRight,
      depth - 1,
    )
  );
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
  const cache = new Map<number, number>();
  const integrand = (probability: number): number => {
    if (probability <= 0) return 0;
    if (probability >= 1) return 1;
    const cached = cache.get(probability);
    if (cached !== undefined) return cached;
    const value = gammaQuantile(leaderShape, probability);
    let logProduct = 0;
    for (const shape of otherShapes) {
      const cdf = regularizedGammaP(shape, value);
      if (cdf <= 0) {
        cache.set(probability, 0);
        return 0;
      }
      logProduct += Math.log(cdf);
    }
    const result = Math.exp(logProduct);
    cache.set(probability, result);
    return result;
  };
  const fLeft = 0;
  const fMiddle = integrand(0.5);
  const fRight = 1;
  const whole = (fLeft + 4 * fMiddle + fRight) / 6;
  return Math.max(
    0,
    Math.min(
      1,
      adaptiveSimpson(
        integrand,
        0,
        1,
        1e-10,
        whole,
        fLeft,
        fMiddle,
        fRight,
        20,
      ),
    ),
  );
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
