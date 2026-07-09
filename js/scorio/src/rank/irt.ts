/**
 * Item Response Theory (IRT) ranking methods. Port of `scorio/rank/irt.py`.
 *
 * Latent model abilities `theta` and item parameters (difficulty `b`,
 * discrimination `a`, guessing `c`) are estimated under binary IRT families via
 * joint MLE / MAP (L-BFGS) or marginal MLE (EM + Gauss-Hermite quadrature with
 * EAP scoring). Rankings are induced by ability scores.
 */

import { eigSymmetric, hermgauss } from "./internal/linalg.js";
import { minimize } from "./internal/optimize.js";
import { validatePositiveInt } from "./internal/validate.js";
import { clip } from "./internal/special.js";
import { rankScores } from "./internal/rankScores.js";
import {
  shape3,
  sigmoid,
  validateInput,
  type Tensor3,
  type TensorInput,
} from "./internal/tensor.js";
import { coercePrior, type Prior } from "./priors.js";
import type { BaseRankOptions, RankResult } from "./internal/result.js";

const sum = (a: readonly number[]): number => a.reduce((s, v) => s + v, 0);
const mean = (a: readonly number[]): number => sum(a) / a.length;
const center = (a: readonly number[]): number[] => {
  const m = mean(a);
  return a.map((v) => v - m);
};
const logit = (p: number): number => Math.log(p / (1 - p));

/** Per-(model, item) correct counts `(L, M)` and the trial count `N`. */
function toBinomialCounts(R: Tensor3): { k: number[][]; nTrials: number } {
  const [L, M, N] = shape3(R);
  const k = Array.from({ length: L }, (_, l) =>
    Array.from({ length: M }, (_, m) => {
      let s = 0;
      for (let n = 0; n < N; n++) s += R[l]![m]![n]!;
      return s;
    }),
  );
  return { k, nTrials: N };
}

/** Initial `(theta, beta)` from observed proportions, shared by all JMLE inits. */
function abilityDifficultyInit(k: number[][], nTrials: number): {
  theta: number[];
  beta: number[];
} {
  const L = k.length;
  const M = k[0]!.length;
  const pLM = k.map((row) => row.map((v) => clip((v + 0.5) / (nTrials + 1), 1e-6, 1 - 1e-6)));
  const modelScores = pLM.map((row) => mean(row));
  const questionDiff = Array.from({ length: M }, (_, m) =>
    mean(pLM.map((row) => row[m]!)),
  );
  return {
    theta: modelScores.map((p) => logit(p)),
    beta: questionDiff.map((p) => -logit(p)),
  };
}

function binomialNll(k: number[][], nTrials: number, prob: number[][]): number {
  const L = k.length;
  const M = k[0]!.length;
  let nll = 0;
  for (let l = 0; l < L; l++)
    for (let m = 0; m < M; m++) {
      const p = clip(prob[l]![m]!, 1e-10, 1 - 1e-10);
      nll -= k[l]![m]! * Math.log(p) + (nTrials - k[l]![m]!) * Math.log(1 - p);
    }
  return nll;
}

/** Options for the JMLE / MAP IRT variants. */
export interface IrtOptions extends BaseRankOptions {
  maxIter?: number;
}
export interface IrtMapOptions extends IrtOptions {
  prior?: Prior | number;
}
export interface Irt2plOptions extends IrtOptions {
  regDiscrimination?: number;
}
export interface Irt2plMapOptions extends IrtMapOptions {
  regDiscrimination?: number;
}
export interface Irt3plOptions extends IrtOptions {
  fixGuessing?: number | null;
  regDiscrimination?: number;
  regGuessing?: number;
  guessingUpper?: number;
}
export interface Irt3plMapOptions extends Irt3plOptions {
  prior?: Prior | number;
}

// --- Rasch (1PL) -------------------------------------------------------------

function estimateRasch(
  k: number[][],
  nTrials: number,
  maxIter: number,
  prior: Prior | null,
): { theta: number[]; beta: number[] } {
  const L = k.length;
  const M = k[0]!.length;
  const init = abilityDifficultyInit(k, nTrials);
  const nll = (params: readonly number[]): number => {
    const theta = params.slice(0, L);
    const beta = center(params.slice(L));
    const prob = theta.map((th) => beta.map((b) => sigmoid(th - b)));
    let v = binomialNll(k, nTrials, prob);
    if (prior) v += prior.penalty(theta);
    return v;
  };
  const res = minimize(nll, [...init.theta, ...init.beta], { maxIter });
  void M;
  return { theta: res.x.slice(0, L), beta: center(res.x.slice(L)) };
}

/** Rank models with Rasch (1PL) IRT via joint MLE. */
export function rasch(R: TensorInput, options: IrtOptions = {}): RankResult {
  const method = options.method ?? "competition";
  const { k, nTrials } = toBinomialCounts(validateInput(R));
  const { theta } = estimateRasch(k, nTrials, validatePositiveInt("max_iter", options.maxIter ?? 500), null);
  return { ranking: rankScores(theta, method), scores: theta };
}

/** Rank models with Rasch (1PL) IRT via MAP estimation. */
export function raschMap(R: TensorInput, options: IrtMapOptions = {}): RankResult {
  const method = options.method ?? "competition";
  const prior = coercePrior(options.prior ?? 1);
  const { k, nTrials } = toBinomialCounts(validateInput(R));
  const { theta } = estimateRasch(k, nTrials, validatePositiveInt("max_iter", options.maxIter ?? 500), prior);
  return { ranking: rankScores(theta, method), scores: theta };
}

// --- 2PL ---------------------------------------------------------------------

function estimate2pl(
  k: number[][],
  nTrials: number,
  maxIter: number,
  regDiscrimination: number,
  prior: Prior | null,
): { theta: number[] } {
  const L = k.length;
  const M = k[0]!.length;
  const init = abilityDifficultyInit(k, nTrials);
  const nll = (params: readonly number[]): number => {
    const theta = params.slice(0, L);
    const beta = center(params.slice(L, L + M));
    const logA = params.slice(L + M);
    const a = logA.map((v) => Math.exp(clip(v, -3, 3)));
    const prob = theta.map((th) => beta.map((b, m) => sigmoid(a[m]! * (th - b))));
    let v = binomialNll(k, nTrials, prob);
    v += regDiscrimination * sum(logA.map((x) => x * x));
    if (prior) v += prior.penalty(theta);
    return v;
  };
  const init2 = [...init.theta, ...init.beta, ...new Array<number>(M).fill(0)];
  const res = minimize(nll, init2, { maxIter });
  return { theta: res.x.slice(0, L) };
}

/** Rank models with 2PL IRT via joint (optionally regularized) JMLE. */
export function rasch2pl(R: TensorInput, options: Irt2plOptions = {}): RankResult {
  const method = options.method ?? "competition";
  const { k, nTrials } = toBinomialCounts(validateInput(R));
  const { theta } = estimate2pl(
    k,
    nTrials,
    validatePositiveInt("max_iter", options.maxIter ?? 500),
    options.regDiscrimination ?? 0.01,
    null,
  );
  return { ranking: rankScores(theta, method), scores: theta };
}

/** Rank models with 2PL IRT via MAP estimation. */
export function rasch2plMap(R: TensorInput, options: Irt2plMapOptions = {}): RankResult {
  const method = options.method ?? "competition";
  const prior = coercePrior(options.prior ?? 1);
  const { k, nTrials } = toBinomialCounts(validateInput(R));
  const { theta } = estimate2pl(
    k,
    nTrials,
    validatePositiveInt("max_iter", options.maxIter ?? 500),
    options.regDiscrimination ?? 0.01,
    prior,
  );
  return { ranking: rankScores(theta, method), scores: theta };
}

// --- 3PL ---------------------------------------------------------------------

function validateGuessingUpper(g: number): number {
  if (!Number.isFinite(g) || !(g > 0 && g < 1)) {
    throw new Error("guessing_upper must be in (0, 1) and finite.");
  }
  return g;
}

function validateFixGuessing(fg: number | null | undefined, gUpper: number): number | null {
  if (fg === null || fg === undefined) return null;
  if (!Number.isFinite(fg) || !(fg >= 0 && fg <= gUpper)) {
    throw new Error(`fix_guessing must be in [0, guessing_upper=${gUpper}] and finite.`);
  }
  return fg;
}

function estimate3pl(
  k: number[][],
  nTrials: number,
  maxIter: number,
  fixGuessing: number | null,
  regDiscrimination: number,
  regGuessing: number,
  guessingUpper: number,
  prior: Prior | null,
): { theta: number[] } {
  const L = k.length;
  const M = k[0]!.length;
  const init = abilityDifficultyInit(k, nTrials);
  const nll = (params: readonly number[]): number => {
    const theta = params.slice(0, L);
    const beta = center(params.slice(L, L + M));
    const logA = params.slice(L + M, L + 2 * M);
    const a = logA.map((v) => Math.exp(clip(v, -3, 3)));
    let logitC: number[] = [];
    let c: number[];
    if (fixGuessing === null) {
      logitC = params.slice(L + 2 * M);
      c = logitC.map((v) => guessingUpper * sigmoid(v));
    } else {
      c = new Array<number>(M).fill(fixGuessing);
    }
    const prob = theta.map((th) =>
      beta.map((b, m) => {
        const base = sigmoid(a[m]! * (th - b));
        return c[m]! + (1 - c[m]!) * base;
      }),
    );
    let v = binomialNll(k, nTrials, prob);
    v += regDiscrimination * sum(logA.map((x) => x * x));
    if (fixGuessing === null) v += regGuessing * sum(logitC.map((x) => x * x));
    if (prior) v += prior.penalty(theta);
    return v;
  };
  const init2 =
    fixGuessing === null
      ? [
          ...init.theta,
          ...init.beta,
          ...new Array<number>(M).fill(0),
          ...new Array<number>(M).fill(0),
        ]
      : [...init.theta, ...init.beta, ...new Array<number>(M).fill(0)];
  const res = minimize(nll, init2, { maxIter });
  return { theta: res.x.slice(0, L) };
}

/** Rank models with 3PL IRT via joint (optionally regularized) JMLE. */
export function rasch3pl(R: TensorInput, options: Irt3plOptions = {}): RankResult {
  const method = options.method ?? "competition";
  const guessingUpper = validateGuessingUpper(options.guessingUpper ?? 0.5);
  const fixGuessing = validateFixGuessing(options.fixGuessing, guessingUpper);
  const { k, nTrials } = toBinomialCounts(validateInput(R));
  const { theta } = estimate3pl(
    k,
    nTrials,
    validatePositiveInt("max_iter", options.maxIter ?? 500),
    fixGuessing,
    options.regDiscrimination ?? 0.01,
    options.regGuessing ?? 0.1,
    guessingUpper,
    null,
  );
  return { ranking: rankScores(theta, method), scores: theta };
}

/** Rank models with 3PL IRT via MAP estimation. */
export function rasch3plMap(R: TensorInput, options: Irt3plMapOptions = {}): RankResult {
  const method = options.method ?? "competition";
  const prior = coercePrior(options.prior ?? 1);
  const guessingUpper = validateGuessingUpper(options.guessingUpper ?? 0.5);
  const fixGuessing = validateFixGuessing(options.fixGuessing, guessingUpper);
  const { k, nTrials } = toBinomialCounts(validateInput(R));
  const { theta } = estimate3pl(
    k,
    nTrials,
    validatePositiveInt("max_iter", options.maxIter ?? 500),
    fixGuessing,
    options.regDiscrimination ?? 0.01,
    options.regGuessing ?? 0.1,
    guessingUpper,
    prior,
  );
  return { ranking: rankScores(theta, method), scores: theta };
}

// --- MML (EM + Gauss-Hermite quadrature) -------------------------------------

interface MmlResult {
  abilities: number[];
  beta: number[];
  posterior: number[][];
  thetaQ: number[];
}

function estimateRaschMml(
  k: number[][],
  nTrials: number,
  maxIter: number,
  emIter: number,
  nQuadrature: number,
): MmlResult {
  const L = k.length;
  const M = k[0]!.length;
  const { nodes, weights } = hermgauss(nQuadrature);
  const thetaQ = nodes.map((x) => Math.SQRT2 * x);
  const wQ = weights.map((w) => w / Math.sqrt(Math.PI));

  // Initialize difficulties.
  const pLM = k.map((row) => row.map((v) => clip((v + 0.5) / (nTrials + 1), 1e-6, 1 - 1e-6)));
  const questionDiff = Array.from({ length: M }, (_, m) => mean(pLM.map((row) => row[m]!)));
  let beta = questionDiff.map((qd) => -Math.log((qd + 0.01) / (1 - qd + 0.01)));
  beta = center(beta);

  const computePosterior = (): number[][] => {
    const logLik = Array.from({ length: L }, () => new Array<number>(nQuadrature).fill(0));
    for (let q = 0; q < nQuadrature; q++) {
      for (let l = 0; l < L; l++) {
        let s = 0;
        for (let m = 0; m < M; m++) {
          const prob = clip(sigmoid(thetaQ[q]! - beta[m]!), 1e-10, 1 - 1e-10);
          s += k[l]![m]! * Math.log(prob) + (nTrials - k[l]![m]!) * Math.log(1 - prob);
        }
        logLik[l]![q] = s;
      }
    }
    return logLik.map((row) => {
      const maxV = Math.max(...row);
      const lik = row.map((v, q) => Math.exp(v - maxV) * wQ[q]!);
      const total = sum(lik);
      return lik.map((v) => v / total);
    });
  };

  let posterior = computePosterior();
  for (let em = 0; em < emIter; em++) {
    posterior = computePosterior();
    for (let m = 0; m < M; m++) {
      const itemNll = (b: readonly number[]): number => {
        let nll = 0;
        for (let q = 0; q < nQuadrature; q++) {
          const prob = clip(sigmoid(thetaQ[q]! - b[0]!), 1e-10, 1 - 1e-10);
          for (let l = 0; l < L; l++) {
            const logP =
              k[l]![m]! * Math.log(prob) + (nTrials - k[l]![m]!) * Math.log(1 - prob);
            nll -= posterior[l]![q]! * logP;
          }
        }
        return nll;
      };
      const res = minimize(itemNll, [beta[m]!], { maxIter });
      beta[m] = res.x[0]!;
    }
    beta = center(beta);
  }

  posterior = computePosterior();
  const abilities = posterior.map((row) => sum(row.map((p, q) => p * thetaQ[q]!)));
  return { abilities, beta, posterior, thetaQ };
}

/** Options for the MML variants. */
export interface MmlOptions extends BaseRankOptions {
  maxIter?: number;
  emIter?: number;
  nQuadrature?: number;
}
export interface MmlCredibleOptions extends MmlOptions {
  /** Posterior quantile `q` in `(0, 1)`. Default `0.05`. */
  quantile?: number;
}

/** Rank models with Rasch MML (EM + quadrature) and EAP scoring. */
export function raschMml(R: TensorInput, options: MmlOptions = {}): RankResult {
  const method = options.method ?? "competition";
  const nQuadrature = options.nQuadrature ?? 21;
  if (nQuadrature < 2) throw new Error("n_quadrature must be >= 2");
  const { k, nTrials } = toBinomialCounts(validateInput(R));
  const { abilities } = estimateRaschMml(
    k,
    nTrials,
    validatePositiveInt("max_iter", options.maxIter ?? 100),
    validatePositiveInt("em_iter", options.emIter ?? 20),
    nQuadrature,
  );
  return { ranking: rankScores(abilities, method), scores: abilities };
}

/** Rank models by a posterior quantile under Rasch MML. */
export function raschMmlCredible(
  R: TensorInput,
  options: MmlCredibleOptions = {},
): RankResult {
  const method = options.method ?? "competition";
  const quantile = options.quantile ?? 0.05;
  if (!(quantile > 0 && quantile < 1)) throw new Error("quantile must be in (0, 1)");
  const nQuadrature = options.nQuadrature ?? 21;
  if (nQuadrature < 2) throw new Error("n_quadrature must be >= 2");
  const { k, nTrials } = toBinomialCounts(validateInput(R));
  const { posterior, thetaQ } = estimateRaschMml(
    k,
    nTrials,
    validatePositiveInt("max_iter", options.maxIter ?? 100),
    validatePositiveInt("em_iter", options.emIter ?? 20),
    nQuadrature,
  );

  // Posterior quantile for each model over the (sorted) quadrature grid.
  const order = Array.from({ length: thetaQ.length }, (_, i) => i).sort(
    (a, b) => thetaQ[a]! - thetaQ[b]!,
  );
  const thetaSorted = order.map((i) => thetaQ[i]!);
  const scores = posterior.map((row) => {
    const postSorted = order.map((i) => row[i]!);
    let cum = 0;
    let j = thetaSorted.length - 1;
    for (let idx = 0; idx < postSorted.length; idx++) {
      cum += postSorted[idx]!;
      if (cum >= quantile) {
        j = idx;
        break;
      }
    }
    return thetaSorted[j]!;
  });
  return { ranking: rankScores(scores, method), scores };
}

// --- Dynamic IRT -------------------------------------------------------------

const DYNAMIC_ALIASES: Record<string, string> = {
  baseline: "initial",
  start: "initial",
  end: "final",
  average: "mean",
  delta: "gain",
  trend: "gain",
};

function validateScoreTarget(target: string): string {
  let t = String(target).trim().toLowerCase();
  t = DYNAMIC_ALIASES[t] ?? t;
  if (t !== "initial" && t !== "final" && t !== "mean" && t !== "gain") {
    throw new Error(
      "score_target must be one of {'initial', 'final', 'mean', 'gain'} " +
        "(aliases: baseline, start, end, average, delta, trend).",
    );
  }
  return t;
}

function scoreDynamicPath(thetaPath: number[][], target: string): number[] {
  if (target === "initial") return thetaPath.map((row) => row[0]!);
  if (target === "final") return thetaPath.map((row) => row[row.length - 1]!);
  if (target === "mean") return thetaPath.map((row) => mean(row));
  return thetaPath.map((row) => row[row.length - 1]! - row[0]!);
}

function validateTimePoints(
  timePoints: readonly number[] | undefined,
  nTime: number,
): number[] {
  let raw: number[];
  if (timePoints === undefined) {
    raw =
      nTime === 1
        ? [0]
        : Array.from({ length: nTime }, (_, i) => i / (nTime - 1));
  } else {
    raw = timePoints.slice();
    if (raw.length !== nTime) {
      throw new Error("time_points must be a 1D array with length equal to R.shape[2].");
    }
    if (raw.some((v) => !Number.isFinite(v))) {
      throw new Error("time_points must contain only finite values.");
    }
    for (let i = 1; i < raw.length; i++)
      if (raw[i]! - raw[i - 1]! <= 0) throw new Error("time_points must be strictly increasing.");
  }
  if (nTime < 2) return new Array<number>(nTime).fill(0);
  const span = raw[raw.length - 1]! - raw[0]!;
  if (!Number.isFinite(span) || span <= 0) {
    throw new Error("time_points must span a positive interval.");
  }
  return raw.map((v) => (v - raw[0]!) / span);
}

function estimateGrowth(
  R: Tensor3,
  timeUnit: number[],
  maxIter: number,
  slopeReg: number,
): { theta0: number[]; theta1: number[] } {
  const [L, M, N] = shape3(R);
  if (N < 2) {
    const { k } = toBinomialCounts(R);
    const { theta } = estimateRasch(k, N, maxIter, null);
    return { theta0: theta, theta1: new Array<number>(L).fill(0) };
  }
  const p0 = Array.from({ length: L }, (_, l) =>
    clip(mean(R[l]!.map((row) => row[0]!)), 1e-6, 1 - 1e-6),
  );
  const theta0Init = p0.map((p) => logit(p));
  const pM = Array.from({ length: M }, (_, m) => {
    let s = 0;
    for (let l = 0; l < L; l++) for (let n = 0; n < N; n++) s += R[l]![m]![n]!;
    return clip(s / (L * N), 1e-6, 1 - 1e-6);
  });
  const betaInit = pM.map((p) => -logit(p));
  const init = [...theta0Init, ...new Array<number>(L).fill(0), ...betaInit];

  const nll = (params: readonly number[]): number => {
    const theta0 = params.slice(0, L);
    const theta1 = params.slice(L, 2 * L);
    const beta = center(params.slice(2 * L));
    let v = 0;
    for (let l = 0; l < L; l++)
      for (let m = 0; m < M; m++)
        for (let n = 0; n < N; n++) {
          const diff = theta0[l]! + theta1[l]! * timeUnit[n]! - beta[m]!;
          const p = clip(sigmoid(diff), 1e-10, 1 - 1e-10);
          const r = R[l]![m]![n]!;
          v -= r * Math.log(p) + (1 - r) * Math.log(1 - p);
        }
    v += slopeReg * sum(theta1.map((x) => x * x));
    return v;
  };
  const res = minimize(nll, init, { maxIter });
  return { theta0: res.x.slice(0, L), theta1: res.x.slice(L, 2 * L) };
}

function estimateStateSpace(
  R: Tensor3,
  timeUnit: number[],
  maxIter: number,
  stateReg: number,
): number[][] {
  const [L, M, N] = shape3(R);
  if (N < 2) {
    const { k } = toBinomialCounts(R);
    const { theta } = estimateRasch(k, N, maxIter, null);
    return theta.map((t) => [t]);
  }
  const pLN = Array.from({ length: L }, (_, l) =>
    Array.from({ length: N }, (_, n) =>
      clip(mean(R[l]!.map((row) => row[n]!)), 1e-6, 1 - 1e-6),
    ),
  );
  const thetaInit = pLN.flatMap((row) => row.map((p) => logit(p)));
  const pM = Array.from({ length: M }, (_, m) => {
    let s = 0;
    for (let l = 0; l < L; l++) for (let n = 0; n < N; n++) s += R[l]![m]![n]!;
    return clip(s / (L * N), 1e-6, 1 - 1e-6);
  });
  const betaInit = pM.map((p) => -logit(p));
  const dt = Array.from({ length: N - 1 }, (_, i) => timeUnit[i + 1]! - timeUnit[i]!);
  const init = [...thetaInit, ...betaInit];

  const nll = (params: readonly number[]): number => {
    const theta: number[][] = [];
    for (let l = 0; l < L; l++) theta.push(params.slice(l * N, (l + 1) * N));
    const beta = center(params.slice(L * N));
    let v = 0;
    for (let l = 0; l < L; l++)
      for (let m = 0; m < M; m++)
        for (let n = 0; n < N; n++) {
          const p = clip(sigmoid(theta[l]![n]! - beta[m]!), 1e-10, 1 - 1e-10);
          const r = R[l]![m]![n]!;
          v -= r * Math.log(p) + (1 - r) * Math.log(1 - p);
        }
    for (let l = 0; l < L; l++)
      for (let n = 0; n < N - 1; n++) {
        const step = (theta[l]![n + 1]! - theta[l]![n]!) / Math.sqrt(dt[n]!);
        v += stateReg * step * step;
      }
    for (let l = 0; l < L; l++) v += 1e-3 * theta[l]![0]! * theta[l]![0]!;
    return v;
  };
  const res = minimize(nll, init, { maxIter });
  const theta: number[][] = [];
  for (let l = 0; l < L; l++) theta.push(res.x.slice(l * N, (l + 1) * N));
  return theta;
}

/** Options for {@link dynamicIrt}. */
export interface DynamicIrtOptions extends BaseRankOptions {
  variant?: "linear" | "growth" | "state_space";
  maxIter?: number;
  timePoints?: readonly number[];
  scoreTarget?: string;
  slopeReg?: number;
  stateReg?: number;
  assumeTimeAxis?: boolean;
}

/** Rank models with dynamic (longitudinal) IRT variants. */
export function dynamicIrt(R: TensorInput, options: DynamicIrtOptions = {}): RankResult {
  const method = options.method ?? "competition";
  const maxIter = validatePositiveInt("max_iter", options.maxIter ?? 500);
  const variant = String(options.variant ?? "linear").trim().toLowerCase();
  const tensor = validateInput(R);
  const [, , N] = shape3(tensor);
  const scoreTarget = validateScoreTarget(options.scoreTarget ?? "final");
  const slopeReg = options.slopeReg ?? 0.01;
  const stateReg = options.stateReg ?? 1;

  let scores: number[];
  if (variant === "linear") {
    if (scoreTarget !== "final") {
      throw new Error(
        "score_target is only used for longitudinal variants ('growth' and 'state_space').",
      );
    }
    const { k } = toBinomialCounts(tensor);
    scores = estimateRasch(k, N, maxIter, null).theta;
  } else if (variant === "growth") {
    if (!options.assumeTimeAxis) {
      throw new Error(
        "variant='growth' interprets axis-2 as ordered longitudinal time. " +
          "Set assume_time_axis=True to proceed.",
      );
    }
    const timeUnit = validateTimePoints(options.timePoints, N);
    const { theta0, theta1 } = estimateGrowth(tensor, timeUnit, maxIter, slopeReg);
    const thetaPath = theta0.map((t0, l) => timeUnit.map((t) => t0 + theta1[l]! * t));
    scores = scoreDynamicPath(thetaPath, scoreTarget);
  } else if (variant === "state_space") {
    if (!options.assumeTimeAxis) {
      throw new Error(
        "variant='state_space' interprets axis-2 as ordered longitudinal time. " +
          "Set assume_time_axis=True to proceed.",
      );
    }
    const timeUnit = validateTimePoints(options.timePoints, N);
    const thetaPath = estimateStateSpace(tensor, timeUnit, maxIter, stateReg);
    scores = scoreDynamicPath(thetaPath, scoreTarget);
  } else {
    throw new Error(`Unknown variant: ${variant}. Use 'linear', 'growth', or 'state_space'.`);
  }

  return { ranking: rankScores(scores, method), scores };
}

// --- Multidimensional IRT (MIRT) ---------------------------------------------

/** Options for compensatory multidimensional IRT. */
export interface MirtOptions extends BaseRankOptions {
  /** Number of latent ability dimensions `D` (default 2). */
  nFactors?: number;
  /** `"2pl"` (no guessing) or `"3pl"` (item pseudo-guessing). */
  model?: "2pl" | "3pl";
  /** Max L-BFGS iterations per EM M-step. */
  maxIter?: number;
  /** Max EM iterations. */
  emIter?: number;
  /** Gauss-Hermite nodes per dimension (grid is `nQuadrature ** nFactors`). */
  nQuadrature?: number;
  /** Fixed guessing in `[0, guessingUpper]` (3PL only); otherwise estimated. */
  fixGuessing?: number | null;
  /** L2 (ridge) penalty on slope vectors. */
  regDiscrimination?: number;
  /** L2 penalty on guessing logits (3PL only). */
  regGuessing?: number;
  /** Upper bound for item guessing in `(0, 1)`. */
  guessingUpper?: number;
  /** Convergence tolerance on the max item-parameter change between EM steps. */
  tol?: number;
}

/** `D`-dimensional product Gauss-Hermite grid (nodes `G x D`, log weights `G`). */
function buildProductQuadrature(
  nFactors: number,
  nQuadrature: number,
): { grid: number[][]; logW: number[] } {
  const { nodes, weights } = hermgauss(nQuadrature);
  const nodes1d = nodes.map((x) => Math.sqrt(2) * x);
  const logw1d = weights.map((w) => Math.log(w) - 0.5 * Math.log(Math.PI));

  const G = nQuadrature ** nFactors;
  const grid: number[][] = Array.from({ length: G }, () => new Array<number>(nFactors).fill(0));
  const logW = new Array<number>(G).fill(0);
  const idx = new Array<number>(nFactors).fill(0);
  for (let g = 0; g < G; g++) {
    let acc = 0;
    for (let d = 0; d < nFactors; d++) {
      grid[g]![d] = nodes1d[idx[d]!]!;
      acc += logw1d[idx[d]!]!;
    }
    logW[g] = acc;
    for (let d = 0; d < nFactors; d++) {
      idx[d]!++;
      if (idx[d]! < nQuadrature) break;
      idx[d] = 0;
    }
  }
  return { grid, logW };
}

interface MirtConfig {
  nFactors: number;
  model: "2pl" | "3pl";
  maxIter: number;
  emIter: number;
  nQuadrature: number;
  fixGuessing: number | null;
  regDiscrimination: number;
  regGuessing: number;
  guessingUpper: number;
  tol: number;
}

/**
 * Estimate a compensatory MIRT model via marginal-MLE EM with EAP scoring, and
 * return the rotation-invariant reference-composite ranking scores.
 */
function estimateMirt(k: number[][], nTrials: number, cfg: MirtConfig): { scores: number[] } {
  const L = k.length;
  const M = k[0]!.length;
  const D = cfg.nFactors;
  const { guessingUpper, regDiscrimination, regGuessing } = cfg;
  const estimateC = cfg.model === "3pl" && cfg.fixGuessing === null;
  const cFixed =
    cfg.model === "3pl" && cfg.fixGuessing !== null
      ? new Array<number>(M).fill(cfg.fixGuessing)
      : null;

  const { grid, logW } = buildProductQuadrature(D, cfg.nQuadrature);
  const G = grid.length;

  // Initialization: intercepts from item easiness, slopes from the leading
  // singular directions of the centered logit matrix (via eig of ZᵀZ).
  const pLM = k.map((row) => row.map((v) => clip((v + 0.5) / (nTrials + 1), 1e-6, 1 - 1e-6)));
  const z = pLM.map((row) => row.map((p) => Math.log(p / (1 - p))));
  const d0 = Array.from({ length: M }, (_, m) => mean(z.map((r) => r[m]!)));
  const zc = z.map((row) => row.map((v, m) => v - d0[m]!));
  const gram = Array.from({ length: M }, (_, i) =>
    Array.from({ length: M }, (_, j) => {
      let s = 0;
      for (let l = 0; l < L; l++) s += zc[l]![i]! * zc[l]![j]!;
      return s;
    }),
  );
  const { values, vectors } = eigSymmetric(gram); // ascending eigenvalues
  const a = Array.from({ length: M }, () => new Array<number>(D).fill(0));
  for (let dd = 0; dd < D; dd++) {
    const col = M - 1 - dd; // top-D singular directions
    if (col < 0) break;
    const sv = Math.sqrt(Math.max(values[col]!, 0)); // singular value
    const scale = Math.sqrt(sv);
    for (let m = 0; m < M; m++) a[m]![dd] = clip(vectors[m]![col]! * scale, -3, 3);
  }
  let d = d0.slice();
  let gamma = new Array<number>(M).fill(0);

  const currentC = (g: readonly number[]): number[] | null =>
    estimateC ? g.map((v) => guessingUpper * sigmoid(v)) : cFixed;

  // Probabilities at every (grid node, item): `G x M`, clamped.
  const probsAt = (aM: number[][], dM: readonly number[], cM: number[] | null): number[][] => {
    const p = Array.from({ length: G }, () => new Array<number>(M));
    for (let g = 0; g < G; g++)
      for (let m = 0; m < M; m++) {
        let lin = dM[m]!;
        for (let dd = 0; dd < D; dd++) lin += grid[g]![dd]! * aM[m]![dd]!;
        const s = sigmoid(lin);
        const pp = cM ? cM[m]! + (1 - cM[m]!) * s : s;
        p[g]![m] = clip(pp, 1e-10, 1 - 1e-10);
      }
    return p;
  };

  const posterior = (aM: number[][], dM: readonly number[], cM: number[] | null): number[][] => {
    const p = probsAt(aM, dM, cM);
    const post = Array.from({ length: L }, () => new Array<number>(G).fill(0));
    for (let l = 0; l < L; l++) {
      const ll = new Array<number>(G);
      let mx = -Infinity;
      for (let g = 0; g < G; g++) {
        let s = logW[g]!;
        for (let m = 0; m < M; m++)
          s += k[l]![m]! * Math.log(p[g]![m]!) + (nTrials - k[l]![m]!) * Math.log(1 - p[g]![m]!);
        ll[g] = s;
        if (s > mx) mx = s;
      }
      let denom = 0;
      for (let g = 0; g < G; g++) {
        const e = Math.exp(ll[g]! - mx);
        post[l]![g] = e;
        denom += e;
      }
      for (let g = 0; g < G; g++) post[l]![g]! /= denom;
    }
    return post;
  };

  const unpack = (params: readonly number[]): { aM: number[][]; dM: number[]; gM: number[] } => {
    const aM = Array.from({ length: M }, (_, m) =>
      Array.from({ length: D }, (_, dd) => params[m * D + dd]!),
    );
    const dM = params.slice(M * D, M * D + M);
    const gM = estimateC ? params.slice(M * D + M, M * D + 2 * M) : gamma;
    return { aM, dM, gM };
  };

  for (let it = 0; it < cfg.emIter; it++) {
    // E-step.
    const post = posterior(a, d, currentC(gamma));
    const f = new Array<number>(G).fill(0);
    for (let g = 0; g < G; g++) {
      let s = 0;
      for (let l = 0; l < L; l++) s += post[l]![g]!;
      f[g] = nTrials * s;
    }
    const r = Array.from({ length: G }, (_, g) =>
      Array.from({ length: M }, (_, m) => {
        let s = 0;
        for (let l = 0; l < L; l++) s += post[l]![g]! * k[l]![m]!;
        return s;
      }),
    );

    // M-step: maximize the expected complete-data likelihood for the separable
    // item parameters jointly (L-BFGS, numerical gradient).
    const obj = (params: readonly number[]): number => {
      const { aM, dM, gM } = unpack(params);
      const cM = estimateC ? gM.map((v) => guessingUpper * sigmoid(v)) : cFixed;
      const p = probsAt(aM, dM, cM);
      let nll = 0;
      for (let g = 0; g < G; g++)
        for (let m = 0; m < M; m++)
          nll -= r[g]![m]! * Math.log(p[g]![m]!) + (f[g]! - r[g]![m]!) * Math.log(1 - p[g]![m]!);
      for (let m = 0; m < M; m++)
        for (let dd = 0; dd < D; dd++) nll += regDiscrimination * aM[m]![dd]! * aM[m]![dd]!;
      if (estimateC) for (let m = 0; m < M; m++) nll += regGuessing * gM[m]! * gM[m]!;
      return nll;
    };

    const x0 = [
      ...a.flatMap((row) => row),
      ...d,
      ...(estimateC ? gamma : []),
    ];
    const res = minimize(obj, x0, { maxIter: cfg.maxIter });
    const next = unpack(res.x);

    let delta = 0;
    for (let m = 0; m < M; m++) {
      for (let dd = 0; dd < D; dd++) delta = Math.max(delta, Math.abs(next.aM[m]![dd]! - a[m]![dd]!));
      delta = Math.max(delta, Math.abs(next.dM[m]! - d[m]!));
      if (estimateC) delta = Math.max(delta, Math.abs(next.gM[m]! - gamma[m]!));
    }
    a.splice(0, M, ...next.aM);
    d = next.dM;
    gamma = next.gM;
    if (delta < cfg.tol) break;
  }

  // Final E-step: EAP abilities, then rotation-invariant reference composite.
  const post = posterior(a, d, currentC(gamma));
  const theta = Array.from({ length: L }, (_, l) =>
    Array.from({ length: D }, (_, dd) => {
      let s = 0;
      for (let g = 0; g < G; g++) s += post[l]![g]! * grid[g]![dd]!;
      return s;
    }),
  );
  const aBar = Array.from({ length: D }, (_, dd) => mean(a.map((row) => row[dd]!)));
  const scores = theta.map((row) => row.reduce((s, v, dd) => s + v * aBar[dd]!, 0));
  return { scores };
}

/**
 * Rank models with compensatory multidimensional IRT (MIRT) via marginal-MLE
 * EM. Each model has a `D`-dimensional ability vector and each item a slope
 * vector `a` and intercept `d`, with `P = c + (1-c)·σ(aᵀθ + d)` (`c = 0` for
 * 2PL). Abilities are scored by the rotation-invariant reference composite
 * `āᵀθ` (projection onto the mean slope direction). Port of `scorio.rank.mirt`.
 */
export function mirt(R: TensorInput, options: MirtOptions = {}): RankResult {
  const method = options.method ?? "competition";
  const nFactors = validatePositiveInt("n_factors", options.nFactors ?? 2, 1);
  const maxIter = validatePositiveInt("max_iter", options.maxIter ?? 50);
  const emIter = validatePositiveInt("em_iter", options.emIter ?? 100);
  const nQuadrature = validatePositiveInt("n_quadrature", options.nQuadrature ?? 15, 2);

  const model = options.model ?? "2pl";
  if (model !== "2pl" && model !== "3pl") {
    throw new Error("model must be '2pl' or '3pl'.");
  }
  const fixGuessingRaw = options.fixGuessing ?? null;
  if (model === "2pl" && fixGuessingRaw !== null) {
    throw new Error("fixGuessing is only valid for model='3pl'.");
  }
  const guessingUpper = validateGuessingUpper(options.guessingUpper ?? 0.5);
  const fixGuessing = validateFixGuessing(fixGuessingRaw, guessingUpper);

  const gridSize = nQuadrature ** nFactors;
  if (gridSize > 200000) {
    throw new Error(
      `Product quadrature grid would have ${gridSize} nodes ` +
        `(n_quadrature=${nQuadrature} ** n_factors=${nFactors}). Reduce n_factors ` +
        "or n_quadrature; compensatory MML-EM is intended for a small number of factors.",
    );
  }

  const { k, nTrials } = toBinomialCounts(validateInput(R));
  const M = k[0]!.length;
  if (nFactors > M) {
    throw new Error(`n_factors=${nFactors} cannot exceed number of questions M=${M}.`);
  }

  const { scores } = estimateMirt(k, nTrials, {
    nFactors,
    model,
    maxIter,
    emIter,
    nQuadrature,
    fixGuessing,
    regDiscrimination: options.regDiscrimination ?? 0.01,
    regGuessing: options.regGuessing ?? 0.1,
    guessingUpper,
    tol: options.tol ?? 1e-4,
  });
  return { ranking: rankScores(scores, method), scores };
}
