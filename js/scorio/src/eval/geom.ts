/**
 * Geometric pass/spectrum metrics for binary outcomes. Port of
 * `scorio/eval/geom.py`.
 *
 * Implements finite-bank geometric and threshold-spectrum metrics together with
 * approximate Beta-Bernoulli posterior summaries for latent resampling
 * quantities. The GeoSpectrum family blends Pass@k with a threshold-spectrum
 * summary as `Pass@k^lam * S_{w,k}^{1-lam}`.
 */

import { betaRatio, comb } from "./internal/math.js";
import { normalCredibleInterval, type Bounds } from "./internal/ci.js";
import {
  asMatrix,
  rowSums,
  validateBinary,
  type Matrix,
} from "./internal/validate.js";
import { passAtK, passHatK } from "./passAtK.js";

type Weights = readonly number[];

function weightedGeometricMean(
  x: number,
  y: number,
  xWeight: number,
  yWeight: number,
): number {
  if (xWeight === 0.0 && yWeight === 0.0) {
    throw new Error("at least one power must be non-zero");
  }

  if (x === 0.0 && xWeight < 0.0) {
    if (y === 0.0 && yWeight > 0.0) {
      return 0.0;
    }
    throw new Error(
      `x_power must be non-negative when x is zero; got x_power=${xWeight}`,
    );
  }

  if (y === 0.0 && yWeight < 0.0) {
    if (x === 0.0 && xWeight > 0.0) {
      return 0.0;
    }
    throw new Error(
      `y_power must be non-negative when y is zero; got y_power=${yWeight}`,
    );
  }

  return Math.pow(x, xWeight) * Math.pow(y, yWeight);
}

function validateBetaPrior(alpha0: number, beta0: number): void {
  if (alpha0 <= 0.0 || beta0 <= 0.0) {
    throw new Error(
      `alpha0 and beta0 must both be > 0 for a Beta prior; got ${alpha0}, ${beta0}`,
    );
  }
}

function validateFiniteBankK(N: number, k: number): void {
  if (!(k >= 1 && k <= N) || !Number.isInteger(k)) {
    throw new Error(`k must satisfy 1 <= k <= N (N=${N}); got k=${k}`);
  }
}

function validateLatentK(k: number): void {
  if (!Number.isInteger(k) || k < 1) {
    throw new Error(`k must be >= 1; got k=${k}`);
  }
}

function resolveLambda(lam: number, lambda?: number): number {
  if (lambda !== undefined) {
    if (lam !== 0.5) {
      throw new Error("Specify at most one of 'lam' and 'lambda_'.");
    }
    lam = lambda;
  }
  if (!(lam >= 0.0 && lam <= 1.0)) {
    throw new Error(`lam must be in [0, 1]; got ${lam}`);
  }
  return lam;
}

/** Endpoint weights `w_r = 1{r = k}`. */
function unanimousSpectrumWeights(k: number): number[] {
  validateLatentK(k);
  const weights = new Array<number>(k).fill(0.0);
  weights[k - 1] = 1.0;
  return weights;
}

/** Upper-half weights `w_r = (2/k) 1{r >= ceil(k/2) + 1}` used by GeoSpectrum*@k. */
function mgSpectrumWeights(k: number): number[] {
  validateLatentK(k);
  const weights = new Array<number>(k).fill(0.0);
  const start = Math.ceil(k / 2.0);
  for (let i = start; i < k; i++) {
    weights[i] = 2.0 / k;
  }
  return weights;
}

function validateSpectrumWeights(weights: Weights, k: number): number[] {
  const w = Array.from(weights, Number);
  if (w.length !== k) {
    throw new Error(
      `weights must be a length-${k} 1D array; got shape (${w.length},)`,
    );
  }
  if (!w.every((v) => Number.isFinite(v))) {
    throw new Error("weights must be finite");
  }
  if (w.some((v) => v < 0.0)) {
    throw new Error("weights must be non-negative");
  }
  const weightSum = w.reduce((s, v) => s + v, 0);
  if (weightSum > 1.0 + 1e-12) {
    throw new Error(`weights must satisfy sum(weights) <= 1; got sum=${weightSum}`);
  }
  return w;
}

/** Cumulative event-score levels `A_j = sum_{r<=j} w_r` with `A_0 = 0`. */
function eventScoreLevels(weights: readonly number[]): number[] {
  const levels = new Array<number>(weights.length + 1);
  levels[0] = 0.0;
  let acc = 0.0;
  for (let i = 0; i < weights.length; i++) {
    acc += weights[i]!;
    levels[i + 1] = acc;
  }
  return levels;
}

/**
 * Finite-bank threshold-spectrum summary `S_{w,k}(R)`, averaged across questions.
 */
export function thresholdSpectrumAtK(
  R: Matrix,
  k: number,
  weights: Weights,
): number {
  const Rm = asMatrix(R);
  validateBinary(Rm);
  const N = Rm[0]!.length;
  validateFiniteBankK(N, k);
  const w = validateSpectrumWeights(weights, k);

  const nu = rowSums(Rm);
  const levels = eventScoreLevels(w);
  const denom = comb(N, k);
  const vals = nu.map(() => 0.0);
  for (let j = 1; j <= k; j++) {
    const credit = levels[j]!;
    if (credit === 0.0) continue;
    for (let i = 0; i < nu.length; i++) {
      const v = nu[i]!;
      vals[i]! += (credit * comb(v, j) * comb(N - v, k - j)) / denom;
    }
  }
  return vals.reduce((s, v) => s + v, 0) / vals.length;
}

/**
 * Dataset-level Pass/Unanimous geometric blend: `Pass@k^a * Unanimous@k^b`.
 * Defaults to the geometric mean (`a = b = 1/2`).
 */
export function geomDsAtK(
  R: Matrix,
  k: number,
  passPower = 0.5,
  unanimousPower = 0.5,
): number {
  const passScore = passAtK(R, k);
  const unanimousScore = passHatK(R, k);
  return weightedGeometricMean(passScore, unanimousScore, passPower, unanimousPower);
}

/**
 * Questionwise Geom@k averaged across questions: forms the per-question
 * geometric blend of Pass@k and Unanimous@k before averaging.
 */
export function geomAtK(
  R: Matrix,
  k: number,
  passPower = 0.5,
  unanimousPower = 0.5,
): number {
  const Rm = asMatrix(R);
  validateBinary(Rm);
  const N = Rm[0]!.length;
  validateFiniteBankK(N, k);

  const nu = rowSums(Rm);
  const denom = comb(N, k);
  let sum = 0.0;
  for (let i = 0; i < Rm.length; i++) {
    const v = nu[i]!;
    const passVal = 1.0 - comb(N - v, k) / denom;
    const unanimousVal = comb(v, k) / denom;
    sum += weightedGeometricMean(passVal, unanimousVal, passPower, unanimousPower);
  }
  return sum / Rm.length;
}

/**
 * `GeoSpectrum_{lam,w}@k(R) = Pass@k^lam * S_{w,k}^{1-lam}` on the observed
 * finite bank. With `weights` omitted, uses the upper-half mG weights. The
 * `lambda` alias overrides `lam` when provided.
 */
export function geoSpectrumAtK(
  R: Matrix,
  k: number,
  lam = 0.5,
  weights?: Weights,
  lambda?: number,
): number {
  lam = resolveLambda(lam, lambda);
  const passScore = passAtK(R, k);
  if (lam === 1.0) {
    return passScore;
  }
  const w =
    weights === undefined
      ? mgSpectrumWeights(k)
      : validateSpectrumWeights(weights, k);
  const spectrumScore = thresholdSpectrumAtK(R, k, w);
  return weightedGeometricMean(passScore, spectrumScore, lam, 1.0 - lam);
}

interface RowMoments {
  meanPass: number[];
  varPass: number[];
  meanSpec: number[];
  varSpec: number[];
  covPs: number[];
}

/** Per-row Beta posterior parameters for binary outcomes. */
function binaryBetaPosteriorParams(
  Rm: readonly (readonly number[])[],
  alpha0: number,
  beta0: number,
): { alpha: number[]; beta: number[] } {
  validateBinary(Rm);
  const N = Rm[0]!.length;
  const c = rowSums(Rm);
  return {
    alpha: c.map((ci) => alpha0 + ci),
    beta: c.map((ci) => beta0 + (N - ci)),
  };
}

/** Per-question posterior moments for latent Pass@k and spectrum scores. */
function passAndSpectrumRowPosteriorMoments(
  R: Matrix,
  k: number,
  weights: readonly number[],
  alpha0: number,
  beta0: number,
): RowMoments {
  validateLatentK(k);
  validateBetaPrior(alpha0, beta0);

  const Rm = asMatrix(R);
  validateBinary(Rm);
  const M = Rm.length;
  const w = validateSpectrumWeights(weights, k);

  const { alpha, beta } = binaryBetaPosteriorParams(Rm, alpha0, beta0);
  const levels = eventScoreLevels(w);
  const coeff = new Array<number>(k + 1).fill(0.0);
  for (let j = 1; j <= k; j++) {
    coeff[j] = levels[j]! * comb(k, j);
  }
  const activeJs: number[] = [];
  for (let j = 1; j <= k; j++) {
    if (coeff[j] !== 0.0) activeJs.push(j);
  }

  const meanPass = new Array<number>(M);
  const varPass = new Array<number>(M);
  const meanSpec = new Array<number>(M);
  const varSpec = new Array<number>(M);
  const covPs = new Array<number>(M);

  for (let i = 0; i < M; i++) {
    const aI = alpha[i]!;
    const bI = beta[i]!;

    // tK[x] = Beta(aI+x, bI+k-x)/Beta(aI,bI); t2K[x] = Beta(aI+x, bI+2k-x)/Beta(aI,bI).
    // Each is built from one betaRatio seed via the recurrence
    //   t[x] = t[x-1] * (aI + x - 1) / (sumTo - x),
    // removing the O(k^2) gammaln-bearing betaRatio calls in the moment sums.
    const tK = new Array<number>(k + 1);
    tK[0] = betaRatio(aI, bI, 0, k);
    for (let x = 1; x <= k; x++) tK[x] = (tK[x - 1]! * (aI + x - 1)) / (bI + k - x);

    const t2K = new Array<number>(2 * k + 1);
    t2K[0] = betaRatio(aI, bI, 0, 2 * k);
    for (let x = 1; x <= 2 * k; x++) {
      t2K[x] = (t2K[x - 1]! * (aI + x - 1)) / (bI + 2 * k - x);
    }

    const eqk = tK[0]!;
    const mPass = 1.0 - eqk;
    const vPass = Math.max(0.0, t2K[0]! - eqk * eqk);

    let mSpec = 0.0;
    let e2Spec = 0.0;
    let ePs = 0.0;

    for (const j of activeJs) {
      const cJ = coeff[j]!;
      const momentJ = tK[j]!;
      mSpec += cJ * momentJ;
      ePs += cJ * (momentJ - t2K[j]!);
      for (const l of activeJs) {
        e2Spec += cJ * coeff[l]! * t2K[j + l]!;
      }
    }

    const vSpec = Math.max(0.0, e2Spec - mSpec * mSpec);
    const cov = ePs - mPass * mSpec;

    meanPass[i] = mPass;
    varPass[i] = vPass;
    meanSpec[i] = mSpec;
    varSpec[i] = vSpec;
    covPs[i] = cov;
  }

  return { meanPass, varPass, meanSpec, varSpec, covPs };
}

interface DatasetMoments {
  muPass: number;
  varPass: number;
  muSpec: number;
  varSpec: number;
  cov: number;
}

/** Dataset-level posterior moments for latent Pass@k and spectrum scores. */
function passAndSpectrumPosteriorMoments(
  R: Matrix,
  k: number,
  weights: readonly number[],
  alpha0: number,
  beta0: number,
): DatasetMoments {
  const { meanPass, varPass, meanSpec, varSpec, covPs } =
    passAndSpectrumRowPosteriorMoments(R, k, weights, alpha0, beta0);
  const M = meanPass.length;
  const mean = (a: number[]) => a.reduce((s, v) => s + v, 0) / M;
  const sumDiv = (a: number[]) => a.reduce((s, v) => s + v, 0) / (M * M);
  return {
    muPass: mean(meanPass),
    varPass: sumDiv(varPass),
    muSpec: mean(meanSpec),
    varSpec: sumDiv(varSpec),
    cov: sumDiv(covPs),
  };
}

/** Approximate posterior mean/std for latent `GeoSpectrum_{lam,w}@k`. */
function geoSpectrumAtKBayes(
  R: Matrix,
  k: number,
  lam: number,
  weights: readonly number[],
  alpha0: number,
  beta0: number,
): [number, number] {
  lam = resolveLambda(lam);
  const { muPass, varPass, muSpec, varSpec, cov } =
    passAndSpectrumPosteriorMoments(R, k, weights, alpha0, beta0);

  if (lam === 0.0) {
    return [muSpec, Math.sqrt(Math.max(0.0, varSpec))];
  }
  if (lam === 1.0) {
    return [muPass, Math.sqrt(Math.max(0.0, varPass))];
  }

  const mu = weightedGeometricMean(muPass, muSpec, lam, 1.0 - lam);
  if (mu === 0.0) {
    return [0.0, 0.0];
  }

  const gradPass = lam * Math.pow(muPass, lam - 1.0) * Math.pow(muSpec, 1.0 - lam);
  const gradSpec = (1.0 - lam) * Math.pow(muPass, lam) * Math.pow(muSpec, -lam);
  const sigma2 =
    gradPass * gradPass * varPass +
    gradSpec * gradSpec * varSpec +
    2.0 * gradPass * gradSpec * cov;
  return [mu, Math.sqrt(Math.max(0.0, sigma2))];
}

/** Approximate posterior mean/std for latent questionwise Geom@k. */
function geomAtKBayes(
  R: Matrix,
  k: number,
  passPower: number,
  unanimousPower: number,
  alpha0: number,
  beta0: number,
): [number, number] {
  const { meanPass, varPass, meanSpec, varSpec, covPs } =
    passAndSpectrumRowPosteriorMoments(
      R,
      k,
      unanimousSpectrumWeights(k),
      alpha0,
      beta0,
    );
  const M = meanPass.length;
  const means = new Array<number>(M);
  const variances = new Array<number>(M);

  for (let i = 0; i < M; i++) {
    const muPass = meanPass[i]!;
    const muUnanimous = meanSpec[i]!;
    const mu = weightedGeometricMean(muPass, muUnanimous, passPower, unanimousPower);
    means[i] = mu;
    if (mu === 0.0) {
      variances[i] = 0.0;
      continue;
    }

    let gradPass = 0.0;
    if (passPower !== 0.0) {
      gradPass =
        passPower *
        Math.pow(muPass, passPower - 1.0) *
        Math.pow(muUnanimous, unanimousPower);
    }

    let gradUnanimous = 0.0;
    if (unanimousPower !== 0.0) {
      gradUnanimous =
        unanimousPower *
        Math.pow(muPass, passPower) *
        Math.pow(muUnanimous, unanimousPower - 1.0);
    }

    variances[i] = Math.max(
      0.0,
      gradPass * gradPass * varPass[i]! +
        gradUnanimous * gradUnanimous * varSpec[i]! +
        2.0 * gradPass * gradUnanimous * covPs[i]!,
    );
  }

  const mu = means.reduce((s, v) => s + v, 0) / M;
  const sigma = Math.sqrt(variances.reduce((s, v) => s + v, 0)) / M;
  return [mu, sigma];
}

/** Approximate posterior mean/std for latent dataset-level Geom@k. */
function geomDsAtKBayes(
  R: Matrix,
  k: number,
  passPower: number,
  unanimousPower: number,
  alpha0: number,
  beta0: number,
): [number, number] {
  const { muPass, varPass, muSpec, varSpec, cov } =
    passAndSpectrumPosteriorMoments(
      R,
      k,
      unanimousSpectrumWeights(k),
      alpha0,
      beta0,
    );
  const muUnanimous = muSpec;
  const varUnanimous = varSpec;
  const covPu = cov;

  const mu = weightedGeometricMean(muPass, muUnanimous, passPower, unanimousPower);
  if (mu === 0.0) {
    return [0.0, 0.0];
  }

  let gradPass = 0.0;
  if (passPower !== 0.0) {
    gradPass =
      passPower *
      Math.pow(muPass, passPower - 1.0) *
      Math.pow(muUnanimous, unanimousPower);
  }

  let gradUnanimous = 0.0;
  if (unanimousPower !== 0.0) {
    gradUnanimous =
      unanimousPower *
      Math.pow(muPass, passPower) *
      Math.pow(muUnanimous, unanimousPower - 1.0);
  }

  const sigma2 =
    gradPass * gradPass * varPass +
    gradUnanimous * gradUnanimous * varUnanimous +
    2.0 * gradPass * gradUnanimous * covPu;
  return [mu, Math.sqrt(Math.max(0.0, sigma2))];
}

/**
 * Approximate posterior `[mu, sigma, lo, hi]` for the latent spectrum
 * `S_{w,k}(p)`. Defined for any integer `k >= 1` (no `k <= N` restriction).
 */
export function thresholdSpectrumAtKCi(
  R: Matrix,
  k: number,
  weights: Weights,
  confidence = 0.95,
  bounds: Bounds = [0.0, 1.0],
  alpha0 = 1.0,
  beta0 = 1.0,
): [number, number, number, number] {
  const w = validateSpectrumWeights(weights, k);
  const { muSpec, varSpec } = passAndSpectrumPosteriorMoments(
    R,
    k,
    w,
    alpha0,
    beta0,
  );
  const sigma = Math.sqrt(Math.max(0.0, varSpec));
  const [lo, hi] = normalCredibleInterval(muSpec, sigma, confidence, true, bounds);
  return [muSpec, sigma, lo, hi];
}

/** Approximate posterior `[mu, sigma, lo, hi]` for the questionwise Geom@k target. */
export function geomAtKCi(
  R: Matrix,
  k: number,
  passPower = 0.5,
  unanimousPower = 0.5,
  confidence = 0.95,
  bounds: Bounds = [0.0, 1.0],
  alpha0 = 1.0,
  beta0 = 1.0,
): [number, number, number, number] {
  const [mu, sigma] = geomAtKBayes(R, k, passPower, unanimousPower, alpha0, beta0);
  const [lo, hi] = normalCredibleInterval(mu, sigma, confidence, true, bounds);
  return [mu, sigma, lo, hi];
}

/** Approximate posterior `[mu, sigma, lo, hi]` for the dataset-level Geom@k target. */
export function geomDsAtKCi(
  R: Matrix,
  k: number,
  passPower = 0.5,
  unanimousPower = 0.5,
  confidence = 0.95,
  bounds: Bounds = [0.0, 1.0],
  alpha0 = 1.0,
  beta0 = 1.0,
): [number, number, number, number] {
  const [mu, sigma] = geomDsAtKBayes(R, k, passPower, unanimousPower, alpha0, beta0);
  const [lo, hi] = normalCredibleInterval(mu, sigma, confidence, true, bounds);
  return [mu, sigma, lo, hi];
}

/**
 * Approximate posterior `[mu, sigma, lo, hi]` for latent `GeoSpectrum_{lam,w}@k`.
 * With `weights` omitted, uses the upper-half mG weights. The `lambda` alias
 * overrides `lam` when provided.
 */
export function geoSpectrumAtKCi(
  R: Matrix,
  k: number,
  lam = 0.5,
  weights?: Weights,
  lambda?: number,
  confidence = 0.95,
  bounds: Bounds = [0.0, 1.0],
  alpha0 = 1.0,
  beta0 = 1.0,
): [number, number, number, number] {
  lam = resolveLambda(lam, lambda);
  let w: number[];
  if (lam !== 1.0) {
    w =
      weights === undefined
        ? mgSpectrumWeights(k)
        : validateSpectrumWeights(weights, k);
  } else {
    // GeoSpectrum_{1,w}@k is exactly Pass@k, so `w` is irrelevant.
    w = unanimousSpectrumWeights(k);
  }

  const [mu, sigma] = geoSpectrumAtKBayes(R, k, lam, w, alpha0, beta0);
  const [lo, hi] = normalCredibleInterval(mu, sigma, confidence, true, bounds);
  return [mu, sigma, lo, hi];
}

/**
 * Explicit alias for the default `GeoSpectrum*@k` operating point (lam = 0.5
 * with the upper-half mG spectrum weights).
 */
export function geoSpectrumStarAtK(R: Matrix, k: number): number {
  return geoSpectrumAtK(R, k, 0.5, mgSpectrumWeights(k));
}

/** Approximate posterior `[mu, sigma, lo, hi]` for latent `GeoSpectrum*@k`. */
export function geoSpectrumStarAtKCi(
  R: Matrix,
  k: number,
  confidence = 0.95,
  bounds: Bounds = [0.0, 1.0],
  alpha0 = 1.0,
  beta0 = 1.0,
): [number, number, number, number] {
  return geoSpectrumAtKCi(
    R,
    k,
    0.5,
    undefined,
    undefined,
    confidence,
    bounds,
    alpha0,
    beta0,
  );
}
