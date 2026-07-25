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
  averageEquivalentScores,
  averageEventExchangeableScores,
  isStronglyConnected,
  shape3,
  sigmoid,
  validateInput,
  type Tensor3,
  type TensorInput,
} from "./internal/tensor.js";
import {
  CauchyPrior,
  GaussianPrior,
  LaplacePrior,
  UniformPrior,
  coercePrior,
  type Prior,
} from "./priors.js";
import type { BaseRankOptions, RankResult } from "./internal/result.js";

const LOG_DISCRIMINATION_BOUND = 8;
const MAX_STABLE_IRT_LOCATION = 50;

const defaultIfUndefined = <T>(value: T | undefined, fallback: T): T =>
  value === undefined ? fallback : value;

function pythonTruthy(value: unknown): boolean {
  if (value === null || value === undefined) return false;
  if (typeof value === "boolean") return value;
  if (typeof value === "number") return value !== 0 || Number.isNaN(value);
  if (typeof value === "string" || Array.isArray(value)) return value.length > 0;
  return true;
}

function defineNonEnumerableAlias(
  target: object,
  key: PropertyKey,
  value: unknown,
): void {
  Object.defineProperty(target, key, {
    value,
    enumerable: false,
    configurable: true,
  });
}

const sum = (a: readonly number[]): number => a.reduce((s, v) => s + v, 0);
const mean = (a: readonly number[]): number => sum(a) / a.length;
const center = (a: readonly number[]): number[] => {
  const m = mean(a);
  return a.map((v) => v - m);
};
const logit = (p: number): number => Math.log(p / (1 - p));

function coercePythonFloat(value: unknown, name: string): number {
  if (typeof value === "number") return value;
  if (typeof value === "boolean") return value ? 1 : 0;
  if (typeof value === "string" && value.trim().length > 0) {
    const normalized = value.trim().replace(/(?<=\d)_(?=\d)/g, "");
    if (/^[+-]?(?:inf(?:inity)?)$/i.test(normalized)) {
      return normalized.startsWith("-") ? -Infinity : Infinity;
    }
    if (/^[+-]?nan$/i.test(normalized)) return NaN;
    if (
      /^[+-]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?)$/i.test(
        normalized,
      )
    ) {
      return Number(normalized);
    }
  }
  throw new TypeError(`${name} must be convertible to a float`);
}

function validateNonnegativeFloat(name: string, value: unknown): number {
  const converted = coercePythonFloat(value, name);
  if (!Number.isFinite(converted) || converted < 0) {
    throw new Error(`${name} must be a finite scalar >= 0.0, got ${value}`);
  }
  return converted;
}

/** IRT's Python helper intentionally treats booleans as numeric variances. */
function coerceAbilityPrior(prior: unknown): Prior {
  if (typeof prior === "boolean") {
    return coercePrior(prior ? 1 : 0);
  }
  return coercePrior(prior as Prior | number);
}

function requireFinitePersonMle(k: number[][], nTrials: number, name: string): void {
  const maximum = k[0]!.length * nTrials;
  if (k.some((row) => {
    const total = sum(row);
    return total === 0 || total === maximum;
  })) {
    throw new Error(
      `${name} has no finite ability MLE for an all-correct or all-wrong ` +
        "model row; use the corresponding MAP or MML estimator.",
    );
  }
}

function requireFiniteItemEstimates(k: number[][], nTrials: number, name: string): void {
  const L = k.length;
  const M = k[0]!.length;
  for (let m = 0; m < M; m++) {
    let total = 0;
    for (let l = 0; l < L; l++) total += k[l]![m]!;
    if (total === 0 || total === L * nTrials) {
      throw new Error(
        `${name} has no finite item-parameter estimate for an all-correct or ` +
          "all-wrong question; remove that non-informative question or use " +
          "rasch_mml, which handles boundary items explicitly.",
      );
    }
  }
}

/**
 * Detect complete/quasi separation for the two-way person/item fixed effects.
 * Each binary cell induces an order constraint between its person and item
 * nodes; a non-trivial separating direction exists exactly when that directed
 * constraint graph is not strongly connected.
 */
function requireNoFixedEffectSeparation(
  k: number[][],
  nTrials: number,
  name: string,
): void {
  const L = k.length;
  const M = k[0]!.length;
  const adjacency = Array.from({ length: L + M }, () =>
    new Array<boolean>(L + M).fill(false),
  );
  for (let l = 0; l < L; l++) {
    for (let m = 0; m < M; m++) {
      const person = l;
      const item = L + m;
      const count = k[l]![m]!;
      if (count === nTrials) adjacency[item]![person] = true;
      else if (count === 0) adjacency[person]![item] = true;
      else {
        adjacency[item]![person] = true;
        adjacency[person]![item] = true;
      }
    }
  }
  if (!isStronglyConnected(adjacency)) {
    throw new Error(
      `${name} has no finite joint location estimate because the binary ` +
        "response pattern is completely or quasi-separated; use a MAP or " +
        "MML estimator with proper regularization.",
    );
  }
}

function requireOptimizerSuccess(
  result: { x: number[]; fun: number; success: boolean },
  modelName: string,
): void {
  if (
    !result.success ||
    !Number.isFinite(result.fun) ||
    result.x.some((value) => !Number.isFinite(value))
  ) {
    throw new Error(`${modelName} optimization failed to converge.`);
  }
}

/**
 * The in-tree L-BFGS occasionally exhausts its iteration counter a fraction
 * before SciPy's L-BFGS-B reports success, even though both have reached the
 * same stationary M-step.  Accept that backend-only status difference after
 * independently auditing the returned point; genuine non-convergence still
 * fails on the projected first-order condition.
 */
function requireOptimizerSuccessOrStationarity(
  result: { x: number[]; fun: number; success: boolean },
  objective: (params: readonly number[]) => number,
  modelName: string,
): void {
  if (
    !Number.isFinite(result.fun) ||
    result.x.some((value) => !Number.isFinite(value))
  ) {
    throw new Error(`${modelName} optimization failed to converge.`);
  }
  if (result.success) return;

  const gradientNorm = Math.max(
    ...numericalGradient(objective, result.x).map(Math.abs),
  );
  if (!Number.isFinite(gradientNorm) || gradientNorm > 5e-4) {
    throw new Error(
      `${modelName} optimization stopped before reaching a stationary ` +
        `solution (projected gradient ${gradientNorm}).`,
    );
  }
}

function numericalGradient(
  objective: (params: readonly number[]) => number,
  point: readonly number[],
): number[] {
  const base = objective(point);
  return point.map((value, index) => {
    const step = Math.sqrt(Number.EPSILON) * Math.max(1, Math.abs(value));
    const upper = point.slice();
    upper[index] = value + step;
    return (objective(upper) - base) / step;
  });
}

function requireStableNonconvexSolution(
  result: { x: number[]; fun: number; success: boolean; iterations?: number },
  objective: (params: readonly number[]) => number,
  nModels: number,
  nItems: number,
  modelName: string,
  exactGradient?: readonly number[],
  iterationBudget?: number,
  allowBackendBudgetMismatch = false,
): void {
  if (!Number.isFinite(result.fun) || result.x.some((value) => !Number.isFinite(value))) {
    throw new Error(`${modelName} optimization failed to converge.`);
  }
  const gradient = exactGradient?.slice() ?? numericalGradient(objective, result.x);
  const discriminationStart = nModels + nItems;
  for (let index = discriminationStart; index < discriminationStart + nItems; index++) {
    const value = result.x[index]!;
    if (
      (value <= -LOG_DISCRIMINATION_BOUND + 1e-8 && gradient[index]! > 0) ||
      (value >= LOG_DISCRIMINATION_BOUND - 1e-8 && gradient[index]! < 0)
    ) {
      gradient[index] = 0;
    }
  }
  const gradientNorm = Math.max(...gradient.map(Math.abs));
  if (!result.success && !allowBackendBudgetMismatch) {
    throw new Error(`${modelName} optimization failed to converge.`);
  }
  if (
    !allowBackendBudgetMismatch &&
    iterationBudget !== undefined &&
    result.iterations !== undefined &&
    result.iterations >= Math.ceil(0.9 * iterationBudget)
  ) {
    throw new Error(`${modelName} optimization failed to converge within max_iter.`);
  }
  if (!Number.isFinite(gradientNorm) || gradientNorm > 5e-4) {
    throw new Error(
      `${modelName} optimization stopped before reaching a stationary solution ` +
        `(projected gradient ${gradientNorm}).`,
    );
  }

  const theta = result.x.slice(0, nModels);
  const beta = center(result.x.slice(nModels, nModels + nItems));
  const logA = result.x.slice(discriminationStart, discriminationStart + nItems);
  if (
    [...theta, ...beta].some(
      (value) => !Number.isFinite(value) || Math.abs(value) > MAX_STABLE_IRT_LOCATION,
    ) ||
    logA.some(
      (value) =>
        !Number.isFinite(value) ||
        Math.abs(value) >= LOG_DISCRIMINATION_BOUND - 1e-6,
    )
  ) {
    throw new Error(
      `${modelName} did not have a stable interior joint estimate; ` +
        "ability/difficulty parameters saturated or an item discrimination " +
        "reached the numerical search boundary.",
    );
  }
}

function hasNontrivialModelItemAutomorphism(k: number[][]): boolean {
  const probe = Array.from({ length: k.length }, (_, index) => index);
  const averaged = averageEventExchangeableScores(probe, k);
  return averaged.some((value, index) => value !== probe[index]);
}

interface CanonicalCounts {
  k: number[][];
  /** Canonical index -> caller's model index. */
  modelOrder: number[];
  /** Canonical index -> caller's item index. */
  itemOrder: number[];
}

function compareNumberArrays(
  left: readonly number[],
  right: readonly number[],
): number {
  for (let index = 0; index < left.length; index++) {
    if (left[index] !== right[index]) return left[index]! - right[index]!;
  }
  return 0;
}

function visitPermutations(
  length: number,
  visit: (permutation: readonly number[]) => void,
): void {
  const permutation = Array.from({ length }, (_, index) => index);
  const recurse = (start: number): void => {
    if (start === length) {
      visit(permutation);
      return;
    }
    for (let index = start; index < length; index++) {
      [permutation[start], permutation[index]] = [
        permutation[index]!,
        permutation[start]!,
      ];
      recurse(start + 1);
      [permutation[start], permutation[index]] = [
        permutation[index]!,
        permutation[start]!,
      ];
    }
  };
  recurse(0);
}

/**
 * Put the weighted model/item bipartite graph into a deterministic order before
 * numerical optimization.  For the small model sets used by joint IRT we can
 * canonicalize exactly by enumerating the smaller label side and sorting the
 * other side by its induced column/row word.  A color-refinement fallback
 * keeps larger problems inexpensive.
 */
function canonicalizeIrtCounts(
  input: number[][],
  canonicalizeModels: boolean,
): CanonicalCounts {
  const L = input.length;
  const M = input[0]!.length;
  let bestModelOrder = Array.from({ length: L }, (_, index) => index);
  let bestItemOrder = Array.from({ length: M }, (_, index) => index);
  let bestWord: number[] | null = null;
  const consider = (modelOrder: readonly number[], itemOrder: readonly number[]): void => {
    const word = modelOrder.flatMap((model) =>
      itemOrder.map((item) => input[model]![item]!),
    );
    if (bestWord === null || compareNumberArrays(word, bestWord) < 0) {
      bestWord = word;
      bestModelOrder = modelOrder.slice();
      bestItemOrder = itemOrder.slice();
    }
  };

  if (!canonicalizeModels) {
    bestItemOrder.sort((left, right) =>
      compareNumberArrays(
        input.map((row) => row[left]!),
        input.map((row) => row[right]!),
      ),
    );
    consider(bestModelOrder, bestItemOrder);
  } else if (L <= 8) {
    visitPermutations(L, (modelOrder) => {
      const itemOrder = Array.from({ length: M }, (_, index) => index).sort(
        (left, right) =>
          compareNumberArrays(
            modelOrder.map((model) => input[model]![left]!),
            modelOrder.map((model) => input[model]![right]!),
          ),
      );
      consider(modelOrder, itemOrder);
    });
  } else if (M <= 8) {
    visitPermutations(M, (itemOrder) => {
      const modelOrder = Array.from({ length: L }, (_, index) => index).sort(
        (left, right) =>
          compareNumberArrays(
            itemOrder.map((item) => input[left]![item]!),
            itemOrder.map((item) => input[right]![item]!),
          ),
      );
      consider(modelOrder, itemOrder);
    });
  } else {
    const assignColors = (keys: readonly string[]): number[] => {
      const unique = [...new Set(keys)].sort();
      const color = new Map(unique.map((key, index) => [key, index]));
      return keys.map((key) => color.get(key)!);
    };
    let modelColors = assignColors(
      input.map((row) => row.slice().sort((a, b) => a - b).join(",")),
    );
    let itemColors = assignColors(
      Array.from({ length: M }, (_, item) =>
        input
          .map((row) => row[item]!)
          .sort((a, b) => a - b)
          .join(","),
      ),
    );
    for (let iteration = 0; iteration < L + M; iteration++) {
      const nextModelColors = assignColors(
        input.map((row) =>
          row
            .map((value, item) => `${itemColors[item]}:${value}`)
            .sort()
            .join("|"),
        ),
      );
      const nextItemColors = assignColors(
        Array.from({ length: M }, (_, item) =>
          input
            .map((row, model) => `${nextModelColors[model]}:${row[item]}`)
            .sort()
            .join("|"),
        ),
      );
      if (
        nextModelColors.every((value, index) => value === modelColors[index]) &&
        nextItemColors.every((value, index) => value === itemColors[index])
      ) {
        break;
      }
      modelColors = nextModelColors;
      itemColors = nextItemColors;
    }
    bestModelOrder.sort((left, right) => modelColors[left]! - modelColors[right]!);
    bestItemOrder.sort((left, right) => itemColors[left]! - itemColors[right]!);
    consider(bestModelOrder, bestItemOrder);
  }

  return {
    k: bestModelOrder.map((model) =>
      bestItemOrder.map((item) => input[model]![item]!),
    ),
    modelOrder: bestModelOrder,
    itemOrder: bestItemOrder,
  };
}

function restoreCanonicalVector<T>(
  values: readonly T[],
  order: readonly number[],
): T[] {
  const restored = new Array<T>(values.length);
  for (let index = 0; index < values.length; index++) {
    restored[order[index]!] = values[index]!;
  }
  return restored;
}

interface NonconvexResult {
  x: number[];
  fun: number;
  success: boolean;
  iterations: number;
}

/**
 * Port of Python's deterministic multi-start audit for suspicious 2PL/3PL
 * fits.  Starts are derived from the top eigenspace of the centered response
 * Gram matrix, so the audit is equivariant to model and item relabelling.
 */
function auditNonconvexIdentifiability(
  baseResult: NonconvexResult,
  objective: (params: readonly number[]) => number,
  paramsInit: readonly number[],
  maxIter: number,
  nModels: number,
  nItems: number,
  k: number[][],
  modelName: string,
  exactGradient?: (params: readonly number[]) => readonly number[],
  exchangeable = true,
): NonconvexResult {
  const hasAutomorphism =
    exchangeable && hasNontrivialModelItemAutomorphism(k);
  const components = (candidate: NonconvexResult): {
    theta: number[];
    beta: number[];
    logA: number[];
  } => ({
    theta: candidate.x.slice(0, nModels),
    beta: center(candidate.x.slice(nModels, nModels + nItems)),
    logA: candidate.x.slice(
      nModels + nItems,
      nModels + 2 * nItems,
    ),
  });
  const validateCandidate = (candidate: NonconvexResult): void => {
    requireStableNonconvexSolution(
      candidate,
      objective,
      nModels,
      nItems,
      modelName,
      exactGradient?.(candidate.x),
      maxIter,
      hasAutomorphism,
    );
  };

  const base = components(baseResult);
  const adjustedTheta = exchangeable
    ? averageEventExchangeableScores(base.theta, k)
    : base.theta;
  const suspicious =
    Math.max(...base.logA.map(Math.abs)) > 4 ||
    Math.max(...[...base.theta, ...base.beta].map(Math.abs)) > 10 ||
    Math.max(
      ...adjustedTheta.map((value, index) =>
        Math.abs(value - base.theta[index]!),
      ),
    ) > 1e-4 ||
    hasAutomorphism;
  if (!suspicious) return baseResult;

  const itemMeans = Array.from({ length: nItems }, (_, item) =>
    mean(k.map((row) => row[item]!)),
  );
  const centeredCounts = k.map((row) =>
    row.map((value, item) => value - itemMeans[item]!),
  );
  const gram = Array.from({ length: nModels }, (_, left) =>
    Array.from({ length: nModels }, (_, right) =>
      sum(
        centeredCounts[left]!.map(
          (value, item) => value * centeredCounts[right]![item]!,
        ),
      ),
    ),
  );
  const { values, vectors } = eigSymmetric(gram);
  const largest = values[values.length - 1]!;
  const topColumns = values
    .map((value, index) => ({ value, index }))
    .filter(
      ({ value }) =>
        value >= largest - 1e-10 * Math.max(1, largest),
    )
    .map(({ index }) => index);
  const directions: number[][] = [];
  if (largest > Number.EPSILON) {
    for (let probe = 0; probe < nModels; probe++) {
      const direction = Array.from({ length: nModels }, (_, model) =>
        sum(
          topColumns.map(
            (column) =>
              vectors[model]![column]! * vectors[probe]![column]!,
          ),
        ),
      );
      const norm = Math.sqrt(sum(direction.map((value) => value * value)));
      if (norm <= 1e-10) continue;
      for (let index = 0; index < direction.length; index++) {
        direction[index]! /= norm;
      }
      const duplicate = directions.some(
        (prior) =>
          direction.every(
            (value, index) => Math.abs(value - prior[index]!) <= 1e-10,
          ) ||
          direction.every(
            (value, index) => Math.abs(value + prior[index]!) <= 1e-10,
          ),
      );
      if (!duplicate) directions.push(direction);
    }
  }

  const candidates: NonconvexResult[] = [baseResult];
  for (const direction of directions) {
    for (const sign of [-1, 1]) {
      const start = paramsInit.slice();
      for (let model = 0; model < nModels; model++) {
        start[model]! += sign * direction[model]!;
      }
      try {
        const candidate = minimize(objective, start, {
          maxIter,
          ftol: 1e-14,
          gtol: 1e-9,
          m: 30,
        });
        validateCandidate(candidate);
        candidates.push(candidate);
      } catch {
        // Failed alternative starts are not evidence against an otherwise
        // stable base fit, matching the Python audit.
      }
    }
  }

  const bestValue = Math.min(...candidates.map((candidate) => candidate.fun));
  const objectiveTolerance = 1e-7 * Math.max(1, Math.abs(bestValue));
  const nearBest = candidates.filter(
    (candidate) => candidate.fun <= bestValue + objectiveTolerance,
  );
  const rankings = new Set<string>();
  for (const candidate of nearBest) {
    let theta = components(candidate).theta;
    if (exchangeable) theta = averageEventExchangeableScores(theta, k);
    rankings.add(rankScores(theta, "competition").join(","));
  }
  if (rankings.size > 1) {
    throw new Error(
      `${modelName} has multiple equally good nonconvex solutions that ` +
        "imply different rankings; the ranking is not identified. Use a " +
        "Rasch or MML estimator, or report a sensitivity analysis.",
    );
  }

  const invariantTieBreak = (
    candidate: NonconvexResult,
  ): { norm: number; signature: number[] } => {
    const { theta, beta, logA } = components(candidate);
    return {
      norm: Math.sqrt(sum(candidate.x.map((value) => value * value))),
      signature: [
        ...theta.slice().sort((a, b) => a - b),
        ...beta.slice().sort((a, b) => a - b),
        ...logA.slice().sort((a, b) => a - b),
      ].map((value) => Math.round(value * 1e10) / 1e10),
    };
  };
  return nearBest.reduce((best, candidate) => {
    const a = invariantTieBreak(candidate);
    const b = invariantTieBreak(best);
    if (a.norm !== b.norm) return a.norm < b.norm ? candidate : best;
    for (let index = 0; index < a.signature.length; index++) {
      if (a.signature[index] !== b.signature[index]) {
        return a.signature[index]! < b.signature[index]! ? candidate : best;
      }
    }
    return best;
  });
}

function priorGradient(prior: Prior, theta: readonly number[]): number[] {
  if (prior.constructor === GaussianPrior) {
    const gaussian = prior as GaussianPrior;
    return theta.map((value) => (value - gaussian.mean) / gaussian.var);
  }
  if (prior.constructor === LaplacePrior) {
    const laplace = prior as LaplacePrior;
    return theta.map((value) => Math.sign(value - laplace.loc) / laplace.scale);
  }
  if (prior.constructor === CauchyPrior) {
    const cauchy = prior as CauchyPrior;
    return theta.map((value) => {
      const z = (value - cauchy.loc) / cauchy.scale;
      return (2 * z) / (cauchy.scale * (1 + z * z));
    });
  }
  if (prior.constructor === UniformPrior) return new Array<number>(theta.length).fill(0);

  return theta.map((value, index) => {
    const step = Math.sqrt(Number.EPSILON) * Math.max(1, Math.abs(value));
    const upper = theta.slice();
    const lower = theta.slice();
    upper[index] = value + step;
    lower[index] = value - step;
    return (prior.penalty(upper) - prior.penalty(lower)) / (2 * step);
  });
}

function priorIsExchangeable(prior: Prior): boolean {
  return (
    prior.constructor === GaussianPrior ||
    prior.constructor === LaplacePrior ||
    prior.constructor === CauchyPrior ||
    prior.constructor === UniformPrior
  );
}

function onePlEquivalenceStatistics(k: number[][]): number[][] {
  return k.map((row) => [sum(row)]);
}

export interface IrtResult<ItemParams> extends RankResult {
  itemParams?: ItemParams;
}

export interface RaschItemParams {
  difficulty: number[];
}

export interface TwoPlItemParams extends RaschItemParams {
  discrimination: number[];
}

export interface ThreePlItemParams extends TwoPlItemParams {
  guessing: number[];
}

export interface RaschMmlItemParams extends RaschItemParams {
  /** Python-compatible key. */
  ability_sd: number[];
  /** Idiomatic JS alias. */
  abilitySd: number[];
}

export interface DynamicIrtItemParams extends RaschItemParams {
  baseline?: number[];
  slope?: number[];
  /** Python-compatible keys. */
  ability_path?: number[][];
  time_points?: number[];
  /** Idiomatic JS aliases. */
  abilityPath?: number[][];
  timePoints?: number[];
  gain?: number[];
}

export interface MirtItemParams extends TwoPlItemParams {
  slopes: number[][];
  intercept: number[];
  abilities: number[][];
  /** Python-compatible key. */
  ability_sd: number[][];
  /** Idiomatic JS alias. */
  abilitySd: number[][];
  guessing?: number[];
}

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
  /** Include fitted item parameters in the result. Default `false`. */
  returnItemParams?: boolean;
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
  requireOptimizerSuccess(res, prior ? "rasch_map" : "rasch");
  void M;
  return { theta: res.x.slice(0, L), beta: center(res.x.slice(L)) };
}

/** Rank models with Rasch (1PL) IRT via joint MLE. */
export function rasch(R: TensorInput, options: IrtOptions = {}): IrtResult<RaschItemParams> {
  const method = defaultIfUndefined(options.method, "competition");
  const maxIter = validatePositiveInt(
    "max_iter",
    defaultIfUndefined(options.maxIter, 500),
  );
  const { k, nTrials } = toBinomialCounts(validateInput(R));
  requireFinitePersonMle(k, nTrials, "Rasch");
  requireFiniteItemEstimates(k, nTrials, "Rasch");
  requireNoFixedEffectSeparation(k, nTrials, "Rasch");
  const { theta, beta } = estimateRasch(
    k,
    nTrials,
    maxIter,
    null,
  );
  const scores = averageEquivalentScores(theta, onePlEquivalenceStatistics(k));
  const result: IrtResult<RaschItemParams> = {
    ranking: rankScores(scores, method),
    scores,
  };
  if (pythonTruthy(options.returnItemParams)) result.itemParams = { difficulty: beta };
  return result;
}

/** Rank models with Rasch (1PL) IRT via MAP estimation. */
export function raschMap(
  R: TensorInput,
  options: IrtMapOptions = {},
): IrtResult<RaschItemParams> {
  const method = defaultIfUndefined(options.method, "competition");
  const maxIter = validatePositiveInt(
    "max_iter",
    defaultIfUndefined(options.maxIter, 500),
  );
  const { k, nTrials } = toBinomialCounts(validateInput(R));
  const prior = coerceAbilityPrior(defaultIfUndefined(options.prior, 1));
  const uniform = prior.constructor === UniformPrior;
  if (uniform) {
    requireFinitePersonMle(k, nTrials, "Rasch");
    requireFiniteItemEstimates(k, nTrials, "Rasch");
    requireNoFixedEffectSeparation(k, nTrials, "Rasch");
  } else {
    requireFiniteItemEstimates(k, nTrials, "Rasch MAP");
  }
  const { theta, beta } = estimateRasch(
    k,
    nTrials,
    maxIter,
    uniform ? null : prior,
  );
  const scores = priorIsExchangeable(prior)
    ? averageEquivalentScores(theta, onePlEquivalenceStatistics(k))
    : theta;
  const result: IrtResult<RaschItemParams> = {
    ranking: rankScores(scores, method),
    scores,
  };
  if (pythonTruthy(options.returnItemParams)) result.itemParams = { difficulty: beta };
  return result;
}

// --- 2PL ---------------------------------------------------------------------

function estimate2pl(
  k: number[][],
  nTrials: number,
  maxIter: number,
  regDiscrimination: number,
  prior: Prior | null,
  modelName: string,
): { theta: number[]; beta: number[]; discrimination: number[] } {
  const exchangeable = prior === null || priorIsExchangeable(prior);
  const canonical = canonicalizeIrtCounts(k, exchangeable);
  k = canonical.k;
  const L = k.length;
  const M = k[0]!.length;
  const init = abilityDifficultyInit(k, nTrials);
  const nll = (params: readonly number[]): number => {
    const theta = params.slice(0, L);
    const beta = center(params.slice(L, L + M));
    const logA = params.slice(L + M);
    const a = logA.map((v) => Math.exp(clip(v, -LOG_DISCRIMINATION_BOUND, LOG_DISCRIMINATION_BOUND)));
    const prob = theta.map((th) => beta.map((b, m) => sigmoid(a[m]! * (th - b))));
    let v = binomialNll(k, nTrials, prob);
    v += regDiscrimination * sum(logA.map((x) => x * x));
    if (prior) v += prior.penalty(theta);
    return v;
  };
  const init2 = [...init.theta, ...init.beta, ...new Array<number>(M).fill(0)];
  const hasAutomorphism =
    exchangeable && hasNontrivialModelItemAutomorphism(k);
  let res = minimize(nll, init2, { maxIter, ftol: 1e-14, gtol: 1e-9, m: 30 });
  if (hasAutomorphism) {
    const gradientNorm = Math.max(
      ...numericalGradient(nll, res.x).map(Math.abs),
    );
    if (!res.success || gradientNorm > 5e-4) {
      const continued = minimize(nll, res.x, {
        maxIter,
        ftol: 1e-14,
        gtol: 1e-9,
        m: 30,
      });
      if (continued.fun <= res.fun) res = continued;
    }
  }
  requireStableNonconvexSolution(
    res,
    nll,
    L,
    M,
    modelName,
    undefined,
    maxIter,
    hasAutomorphism,
  );
  res = auditNonconvexIdentifiability(
    res,
    nll,
    init2,
    maxIter,
    L,
    M,
    k,
    modelName,
    undefined,
    exchangeable,
  );
  const beta = restoreCanonicalVector(
    center(res.x.slice(L, L + M)),
    canonical.itemOrder,
  );
  const discrimination = restoreCanonicalVector(res.x
    .slice(L + M)
    .map((value) => Math.exp(clip(value, -LOG_DISCRIMINATION_BOUND, LOG_DISCRIMINATION_BOUND))), canonical.itemOrder);
  return {
    theta: restoreCanonicalVector(res.x.slice(0, L), canonical.modelOrder),
    beta,
    discrimination,
  };
}

/** Rank models with 2PL IRT via joint (optionally regularized) JMLE. */
export function rasch2pl(
  R: TensorInput,
  options: Irt2plOptions = {},
): IrtResult<TwoPlItemParams> {
  const method = defaultIfUndefined(options.method, "competition");
  const maxIter = validatePositiveInt(
    "max_iter",
    defaultIfUndefined(options.maxIter, 500),
  );
  const regDiscrimination = validateNonnegativeFloat(
    "reg_discrimination",
    defaultIfUndefined(options.regDiscrimination, 0.01),
  );
  if (regDiscrimination === 0) {
    throw new Error(
      "reg_discrimination must be positive for 2PL joint estimation; " +
      "without it, the ability/discrimination scale is not identified.",
    );
  }
  const { k, nTrials } = toBinomialCounts(validateInput(R));
  requireFinitePersonMle(k, nTrials, "2PL");
  requireFiniteItemEstimates(k, nTrials, "2PL");
  requireNoFixedEffectSeparation(k, nTrials, "2PL");
  const { theta, beta, discrimination } = estimate2pl(
    k,
    nTrials,
    maxIter,
    regDiscrimination,
    null,
    "rasch_2pl",
  );
  const scores = averageEventExchangeableScores(theta, k);
  const result: IrtResult<TwoPlItemParams> = {
    ranking: rankScores(scores, method),
    scores,
  };
  if (pythonTruthy(options.returnItemParams)) {
    result.itemParams = { difficulty: beta, discrimination };
  }
  return result;
}

/** Rank models with 2PL IRT via MAP estimation. */
export function rasch2plMap(
  R: TensorInput,
  options: Irt2plMapOptions = {},
): IrtResult<TwoPlItemParams> {
  const method = defaultIfUndefined(options.method, "competition");
  const maxIter = validatePositiveInt(
    "max_iter",
    defaultIfUndefined(options.maxIter, 500),
  );
  const regDiscrimination = validateNonnegativeFloat(
    "reg_discrimination",
    defaultIfUndefined(options.regDiscrimination, 0.01),
  );
  const { k, nTrials } = toBinomialCounts(validateInput(R));
  const prior = coerceAbilityPrior(defaultIfUndefined(options.prior, 1));
  const uniform = prior.constructor === UniformPrior;
  if (uniform) {
    if (regDiscrimination === 0) {
      throw new Error(
        "reg_discrimination must be positive for 2PL joint estimation; " +
          "without it, the ability/discrimination scale is not identified.",
      );
    }
    requireFinitePersonMle(k, nTrials, "2PL");
    requireFiniteItemEstimates(k, nTrials, "2PL");
    requireNoFixedEffectSeparation(k, nTrials, "2PL");
  } else {
    requireFiniteItemEstimates(k, nTrials, "2PL MAP");
  }
  const { theta, beta, discrimination } = estimate2pl(
    k,
    nTrials,
    maxIter,
    regDiscrimination,
    uniform ? null : prior,
    uniform ? "rasch_2pl" : "rasch_2pl_map",
  );
  const scores = priorIsExchangeable(prior)
    ? averageEventExchangeableScores(theta, k)
    : theta;
  const result: IrtResult<TwoPlItemParams> = {
    ranking: rankScores(scores, method),
    scores,
  };
  if (pythonTruthy(options.returnItemParams)) {
    result.itemParams = { difficulty: beta, discrimination };
  }
  return result;
}

// --- 3PL ---------------------------------------------------------------------

function validateGuessingUpper(g: number): number {
  const converted = coercePythonFloat(g, "guessing_upper");
  if (!Number.isFinite(converted) || !(converted > 0 && converted < 1)) {
    throw new Error("guessing_upper must be in (0, 1) and finite.");
  }
  return converted;
}

function validateFixGuessing(fg: number | null | undefined, gUpper: number): number | null {
  if (fg === null || fg === undefined) return null;
  const converted = coercePythonFloat(fg, "fix_guessing");
  if (!Number.isFinite(converted) || !(converted >= 0 && converted <= gUpper)) {
    throw new Error(`fix_guessing must be in [0, guessing_upper=${gUpper}] and finite.`);
  }
  return converted;
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
  modelName: string,
): {
  theta: number[];
  beta: number[];
  discrimination: number[];
  guessing: number[];
} {
  const exchangeable = prior === null || priorIsExchangeable(prior);
  const canonical = canonicalizeIrtCounts(k, exchangeable);
  k = canonical.k;
  const L = k.length;
  const M = k[0]!.length;
  const init = abilityDifficultyInit(k, nTrials);
  const nll = (params: readonly number[]): number => {
    const theta = params.slice(0, L);
    const beta = center(params.slice(L, L + M));
    const logA = params.slice(L + M, L + 2 * M);
    const a = logA.map((v) =>
      Math.exp(clip(v, -LOG_DISCRIMINATION_BOUND, LOG_DISCRIMINATION_BOUND)),
    );
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
  const gradientAt = (params: readonly number[]): number[] => {
    const theta = params.slice(0, L);
    const beta = center(params.slice(L, L + M));
    const logA = params.slice(L + M, L + 2 * M);
    const a = logA.map((value) =>
      Math.exp(clip(value, -LOG_DISCRIMINATION_BOUND, LOG_DISCRIMINATION_BOUND)),
    );
    const guessingLogits =
      fixGuessing === null ? params.slice(L + 2 * M) : new Array<number>(M).fill(0);
    const unitGuessing = guessingLogits.map(sigmoid);
    const guessing =
      fixGuessing === null
        ? unitGuessing.map((value) => guessingUpper * value)
        : new Array<number>(M).fill(fixGuessing);
    const gradTheta = new Array<number>(L).fill(0);
    const gradBeta = new Array<number>(M).fill(0);
    const gradLogA = logA.map((value) => 2 * regDiscrimination * value);
    const gradGuessing = new Array<number>(M).fill(0);

    for (let l = 0; l < L; l++) {
      for (let m = 0; m < M; m++) {
        const itemLogit = a[m]! * (theta[l]! - beta[m]!);
        const base = sigmoid(itemLogit);
        const probability = Math.max(
          guessing[m]! + (1 - guessing[m]!) * base,
          Number.MIN_VALUE,
        );
        const residual = nTrials * probability - k[l]![m]!;
        const gradLogit = (residual * base) / probability;
        gradTheta[l]! += gradLogit * a[m]!;
        gradBeta[m]! -= gradLogit * a[m]!;
        gradLogA[m]! += gradLogit * itemLogit;
        if (fixGuessing === null) {
          gradGuessing[m]! += residual / (probability * (1 - guessing[m]!));
        }
      }
    }
    const betaGradientMean = mean(gradBeta);
    for (let m = 0; m < M; m++) gradBeta[m]! -= betaGradientMean;
    if (prior) {
      const priorGrad = priorGradient(prior, theta);
      for (let l = 0; l < L; l++) gradTheta[l]! += priorGrad[l]!;
    }
    if (fixGuessing === null) {
      for (let m = 0; m < M; m++) {
        const derivative = guessingUpper * unitGuessing[m]! * (1 - unitGuessing[m]!);
        gradGuessing[m] =
          gradGuessing[m]! * derivative + 2 * regGuessing * guessingLogits[m]!;
      }
    }
    return [
      ...gradTheta,
      ...gradBeta,
      ...gradLogA,
      ...(fixGuessing === null ? gradGuessing : []),
    ];
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
  const hasAutomorphism =
    exchangeable && hasNontrivialModelItemAutomorphism(k);
  let res = minimize(nll, init2, { maxIter, ftol: 1e-14, gtol: 1e-9, m: 30 });
  if (hasAutomorphism) {
    const gradientNorm = Math.max(...gradientAt(res.x).map(Math.abs));
    if (!res.success || gradientNorm > 5e-4) {
      const continued = minimize(nll, res.x, {
        maxIter,
        ftol: 1e-14,
        gtol: 1e-9,
        m: 30,
      });
      if (continued.fun <= res.fun) res = continued;
    }
  }
  requireStableNonconvexSolution(
    res,
    nll,
    L,
    M,
    modelName,
    gradientAt(res.x),
    maxIter,
    hasAutomorphism,
  );
  res = auditNonconvexIdentifiability(
    res,
    nll,
    init2,
    maxIter,
    L,
    M,
    k,
    modelName,
    gradientAt,
    exchangeable,
  );
  const beta = restoreCanonicalVector(
    center(res.x.slice(L, L + M)),
    canonical.itemOrder,
  );
  const discrimination = restoreCanonicalVector(
    res.x
      .slice(L + M, L + 2 * M)
      .map((value) =>
        Math.exp(clip(value, -LOG_DISCRIMINATION_BOUND, LOG_DISCRIMINATION_BOUND)),
      ),
    canonical.itemOrder,
  );
  let guessing: number[];
  if (fixGuessing === null) {
    const logits = res.x.slice(L + 2 * M);
    if (logits.some((value) => Math.abs(value) > 30)) {
      throw new Error(
        `${modelName} guessing parameters saturated at a boundary; use ` +
          "stronger guessing regularization or fix_guessing.",
      );
    }
    guessing = restoreCanonicalVector(
      logits.map((value) => guessingUpper * sigmoid(value)),
      canonical.itemOrder,
    );
  } else {
    guessing = new Array<number>(M).fill(fixGuessing);
  }
  return {
    theta: restoreCanonicalVector(res.x.slice(0, L), canonical.modelOrder),
    beta,
    discrimination,
    guessing,
  };
}

/** Rank models with 3PL IRT via joint (optionally regularized) JMLE. */
export function rasch3pl(
  R: TensorInput,
  options: Irt3plOptions = {},
): IrtResult<ThreePlItemParams> {
  const method = defaultIfUndefined(options.method, "competition");
  const maxIter = validatePositiveInt(
    "max_iter",
    defaultIfUndefined(options.maxIter, 500),
  );
  const regDiscrimination = validateNonnegativeFloat(
    "reg_discrimination",
    defaultIfUndefined(options.regDiscrimination, 0.01),
  );
  const regGuessing = validateNonnegativeFloat(
    "reg_guessing",
    defaultIfUndefined(options.regGuessing, 0.1),
  );
  const guessingUpper = validateGuessingUpper(
    defaultIfUndefined(options.guessingUpper, 0.5),
  );
  const fixGuessing = validateFixGuessing(options.fixGuessing, guessingUpper);
  if (regDiscrimination === 0) {
    throw new Error(
      "reg_discrimination must be positive for 3PL joint estimation; " +
        "without it, the ability/discrimination scale is not identified.",
    );
  }
  if (fixGuessing === null && regGuessing === 0) {
    throw new Error(
      "reg_guessing must be positive when 3PL guessing parameters are " +
      "estimated, so boundary guessing logits cannot diverge.",
    );
  }
  const { k, nTrials } = toBinomialCounts(validateInput(R));
  requireFinitePersonMle(k, nTrials, "3PL");
  requireFiniteItemEstimates(k, nTrials, "3PL");
  requireNoFixedEffectSeparation(k, nTrials, "3PL");
  const { theta, beta, discrimination, guessing } = estimate3pl(
    k,
    nTrials,
    maxIter,
    fixGuessing,
    regDiscrimination,
    regGuessing,
    guessingUpper,
    null,
    "rasch_3pl",
  );
  const scores = averageEventExchangeableScores(theta, k);
  const result: IrtResult<ThreePlItemParams> = {
    ranking: rankScores(scores, method),
    scores,
  };
  if (pythonTruthy(options.returnItemParams)) {
    result.itemParams = { difficulty: beta, discrimination, guessing };
  }
  return result;
}

/** Rank models with 3PL IRT via MAP estimation. */
export function rasch3plMap(
  R: TensorInput,
  options: Irt3plMapOptions = {},
): IrtResult<ThreePlItemParams> {
  const method = defaultIfUndefined(options.method, "competition");
  const maxIter = validatePositiveInt(
    "max_iter",
    defaultIfUndefined(options.maxIter, 500),
  );
  const regDiscrimination = validateNonnegativeFloat(
    "reg_discrimination",
    defaultIfUndefined(options.regDiscrimination, 0.01),
  );
  const regGuessing = validateNonnegativeFloat(
    "reg_guessing",
    defaultIfUndefined(options.regGuessing, 0.1),
  );
  const guessingUpper = validateGuessingUpper(
    defaultIfUndefined(options.guessingUpper, 0.5),
  );
  const fixGuessing = validateFixGuessing(options.fixGuessing, guessingUpper);
  if (fixGuessing === null && regGuessing === 0) {
    throw new Error(
      "reg_guessing must be positive when 3PL guessing parameters are " +
      "estimated, so boundary guessing logits cannot diverge.",
    );
  }
  const { k, nTrials } = toBinomialCounts(validateInput(R));
  const prior = coerceAbilityPrior(defaultIfUndefined(options.prior, 1));
  const uniform = prior.constructor === UniformPrior;
  if (uniform) {
    if (regDiscrimination === 0) {
      throw new Error(
        "reg_discrimination must be positive for 3PL joint estimation; " +
          "without it, the ability/discrimination scale is not identified.",
      );
    }
    requireFinitePersonMle(k, nTrials, "3PL");
    requireFiniteItemEstimates(k, nTrials, "3PL");
    requireNoFixedEffectSeparation(k, nTrials, "3PL");
  } else {
    requireFiniteItemEstimates(k, nTrials, "3PL MAP");
  }
  const { theta, beta, discrimination, guessing } = estimate3pl(
    k,
    nTrials,
    maxIter,
    fixGuessing,
    regDiscrimination,
    regGuessing,
    guessingUpper,
    uniform ? null : prior,
    uniform ? "rasch_3pl" : "rasch_3pl_map",
  );
  const scores = priorIsExchangeable(prior)
    ? averageEventExchangeableScores(theta, k)
    : theta;
  const result: IrtResult<ThreePlItemParams> = {
    ranking: rankScores(scores, method),
    scores,
  };
  if (pythonTruthy(options.returnItemParams)) {
    result.itemParams = { difficulty: beta, discrimination, guessing };
  }
  return result;
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

  const itemTotals = Array.from({ length: M }, (_, m) => {
    let total = 0;
    for (let l = 0; l < L; l++) total += k[l]![m]!;
    return total;
  });
  const informative = itemTotals.map(
    (total) => total !== 0 && total !== L * nTrials,
  );
  if (informative.some((value) => !value)) {
    const beta = itemTotals.map((total) =>
      total === 0 ? Infinity : total === L * nTrials ? -Infinity : 0,
    );
    if (informative.some(Boolean)) {
      const subK = k.map((row) => row.filter((_, m) => informative[m]));
      const sub = estimateRaschMml(subK, nTrials, maxIter, emIter, nQuadrature);
      let subIndex = 0;
      for (let m = 0; m < M; m++) {
        if (informative[m]) beta[m] = sub.beta[subIndex++]!;
      }
      return { ...sub, beta };
    }

    const { nodes, weights } = hermgauss(nQuadrature);
    const thetaQ = nodes.map((x) => Math.SQRT2 * x);
    const normalized = weights.map((weight) => weight / Math.sqrt(Math.PI));
    const posterior = Array.from({ length: L }, () => normalized.slice());
    const abilities = posterior.map((row) =>
      sum(row.map((probability, q) => probability * thetaQ[q]!)),
    );
    return { abilities, beta, posterior, thetaQ };
  }

  const { nodes, weights } = hermgauss(nQuadrature);
  const thetaQ = nodes.map((x) => Math.SQRT2 * x);
  const wQ = weights.map((w) => w / Math.sqrt(Math.PI));

  // Initialize difficulties.
  const pLM = k.map((row) => row.map((v) => clip((v + 0.5) / (nTrials + 1), 1e-6, 1 - 1e-6)));
  const questionDiff = Array.from({ length: M }, (_, m) => mean(pLM.map((row) => row[m]!)));
  let beta = questionDiff.map((qd) => -Math.log((qd + 0.01) / (1 - qd + 0.01)));

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
      requireOptimizerSuccess(res, "rasch_mml item M-step");
      beta[m] = res.x[0]!;
    }
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
  /** Include difficulty and posterior ability SD in the result. */
  returnItemParams?: boolean;
}
export interface MmlCredibleOptions
  extends Omit<MmlOptions, "returnItemParams"> {
  /** Posterior quantile `q` in `(0, 1)`. Default `0.05`. */
  quantile?: number;
}

/** Rank models with Rasch MML (EM + quadrature) and EAP scoring. */
export function raschMml(
  R: TensorInput,
  options: MmlOptions = {},
): IrtResult<RaschMmlItemParams> {
  const method = defaultIfUndefined(options.method, "competition");
  const maxIter = validatePositiveInt(
    "max_iter",
    defaultIfUndefined(options.maxIter, 100),
  );
  const emIter = validatePositiveInt(
    "em_iter",
    defaultIfUndefined(options.emIter, 20),
  );
  const nQuadrature = validatePositiveInt(
    "n_quadrature",
    defaultIfUndefined(options.nQuadrature, 21),
    2,
  );
  const { k, nTrials } = toBinomialCounts(validateInput(R));
  const { abilities, beta, posterior, thetaQ } = estimateRaschMml(
    k,
    nTrials,
    maxIter,
    emIter,
    nQuadrature,
  );
  const scores = averageEquivalentScores(abilities, onePlEquivalenceStatistics(k));
  const result: IrtResult<RaschMmlItemParams> = {
    ranking: rankScores(scores, method),
    scores,
  };
  if (pythonTruthy(options.returnItemParams)) {
    const abilitySd = posterior.map((row, l) => {
      const secondMoment = sum(row.map((probability, q) => probability * thetaQ[q]! ** 2));
      return Math.sqrt(Math.max(secondMoment - abilities[l]! ** 2, 0));
    });
    const itemParams = {
      difficulty: beta,
      ability_sd: abilitySd,
    } as RaschMmlItemParams;
    defineNonEnumerableAlias(itemParams, "abilitySd", abilitySd);
    result.itemParams = itemParams;
  }
  return result;
}

/** Rank models by a posterior quantile under Rasch MML. */
export function raschMmlCredible(
  R: TensorInput,
  options: MmlCredibleOptions = {},
): RankResult {
  if (Object.prototype.hasOwnProperty.call(options, "returnItemParams")) {
    throw new Error("rasch_mml_credible does not accept returnItemParams.");
  }
  const method = defaultIfUndefined(options.method, "competition");
  const quantile = defaultIfUndefined(options.quantile, 0.05);
  if (typeof quantile !== "number" || !Number.isFinite(quantile)) {
    throw new TypeError("quantile must be a finite scalar in (0, 1)");
  }
  if (!(quantile > 0 && quantile < 1)) throw new Error("quantile must be in (0, 1)");
  const maxIter = validatePositiveInt(
    "max_iter",
    defaultIfUndefined(options.maxIter, 100),
  );
  const emIter = validatePositiveInt(
    "em_iter",
    defaultIfUndefined(options.emIter, 20),
  );
  const nQuadrature = validatePositiveInt(
    "n_quadrature",
    defaultIfUndefined(options.nQuadrature, 21),
    2,
  );
  const { k, nTrials } = toBinomialCounts(validateInput(R));
  const { posterior, thetaQ } = estimateRaschMml(
    k,
    nTrials,
    maxIter,
    emIter,
    nQuadrature,
  );

  // Posterior quantile for each model over the (sorted) quadrature grid.
  const order = Array.from({ length: thetaQ.length }, (_, i) => i).sort(
    (a, b) => thetaQ[a]! - thetaQ[b]!,
  );
  const thetaSorted = order.map((i) => thetaQ[i]!);
  let scores = posterior.map((row) => {
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
  scores = averageEquivalentScores(scores, onePlEquivalenceStatistics(k));
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
  timePoints: readonly number[] | null | undefined,
  nTime: number,
): { raw: number[]; timeUnit: number[] } {
  let raw: number[];
  if (timePoints === undefined || timePoints === null) {
    raw =
      nTime === 1
        ? [0]
        : Array.from({ length: nTime }, (_, i) => i / (nTime - 1));
  } else {
    if (!Array.isArray(timePoints)) {
      throw new Error(
        "time_points must be a 1D array with length equal to R.shape[2].",
      );
    }
    raw = timePoints.map((value) =>
      coercePythonFloat(value, "time_points"),
    );
    if (raw.length !== nTime) {
      throw new Error("time_points must be a 1D array with length equal to R.shape[2].");
    }
    if (raw.some((v) => !Number.isFinite(v))) {
      throw new Error("time_points must contain only finite values.");
    }
    for (let i = 1; i < raw.length; i++)
      if (raw[i]! - raw[i - 1]! <= 0) throw new Error("time_points must be strictly increasing.");
  }
  if (nTime < 2) return { raw, timeUnit: new Array<number>(nTime).fill(0) };
  const span = raw[raw.length - 1]! - raw[0]!;
  if (!Number.isFinite(span) || span <= 0) {
    throw new Error("time_points must span a positive interval.");
  }
  return { raw, timeUnit: raw.map((v) => (v - raw[0]!) / span) };
}

function estimateGrowth(
  R: Tensor3,
  timeUnit: number[],
  maxIter: number,
  slopeReg: number,
): { theta0: number[]; theta1: number[]; beta: number[] } {
  const [L, M, N] = shape3(R);
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
  requireOptimizerSuccess(res, "dynamic_irt growth");
  return {
    theta0: res.x.slice(0, L),
    theta1: res.x.slice(L, 2 * L),
    beta: center(res.x.slice(2 * L)),
  };
}

function estimateStateSpace(
  R: Tensor3,
  timeUnit: number[],
  maxIter: number,
  stateReg: number,
): { thetaPath: number[][]; beta: number[] } {
  const [L, M, N] = shape3(R);
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
  requireOptimizerSuccess(res, "dynamic_irt state_space");
  const theta: number[][] = [];
  for (let l = 0; l < L; l++) theta.push(res.x.slice(l * N, (l + 1) * N));
  return { thetaPath: theta, beta: center(res.x.slice(L * N)) };
}

/** Options for {@link dynamicIrt}. */
export interface DynamicIrtOptions extends BaseRankOptions {
  variant?: "linear" | "growth" | "state_space";
  maxIter?: number;
  timePoints?: readonly number[] | null;
  scoreTarget?: string;
  slopeReg?: number;
  stateReg?: number;
  assumeTimeAxis?: boolean;
  /** Include static or longitudinal fitted parameters in the result. */
  returnItemParams?: boolean;
}

/** Rank models with dynamic (longitudinal) IRT variants. */
export function dynamicIrt(
  R: TensorInput,
  options: DynamicIrtOptions = {},
): IrtResult<DynamicIrtItemParams> {
  const method = defaultIfUndefined(options.method, "competition");
  const maxIter = validatePositiveInt(
    "max_iter",
    defaultIfUndefined(options.maxIter, 500),
  );
  const variant = String(defaultIfUndefined(options.variant, "linear"))
    .trim()
    .toLowerCase();
  const tensor = validateInput(R);
  const [, , N] = shape3(tensor);
  const scoreTarget = validateScoreTarget(
    defaultIfUndefined(options.scoreTarget, "final"),
  );
  const slopeReg = validateNonnegativeFloat(
    "slope_reg",
    defaultIfUndefined(options.slopeReg, 0.01),
  );
  const stateReg = validateNonnegativeFloat(
    "state_reg",
    defaultIfUndefined(options.stateReg, 1),
  );
  const { k } = toBinomialCounts(tensor);
  if (variant !== "linear") requireFiniteItemEstimates(k, N, "Dynamic IRT");

  let scores: number[];
  let itemParams: DynamicIrtItemParams;
  if (variant === "linear") {
    if (scoreTarget !== "final") {
      throw new Error(
        "score_target is only used for longitudinal variants ('growth' and 'state_space').",
      );
    }
    requireFinitePersonMle(k, N, "Rasch");
    requireFiniteItemEstimates(k, N, "Rasch");
    requireNoFixedEffectSeparation(k, N, "Rasch");
    const fitted = estimateRasch(k, N, maxIter, null);
    scores = averageEquivalentScores(fitted.theta, onePlEquivalenceStatistics(k));
    itemParams = { difficulty: fitted.beta };
  } else if (variant === "growth") {
    if (!pythonTruthy(options.assumeTimeAxis)) {
      throw new Error(
        "variant='growth' interprets axis-2 as ordered longitudinal time. " +
        "Set assume_time_axis=True to proceed.",
      );
    }
    if (N < 2) {
      throw new Error("Longitudinal dynamic IRT requires at least two time points.");
    }
    if (slopeReg === 0) {
      throw new Error(
        "slope_reg must be positive for variant='growth' so temporal " +
          "separation cannot produce an infinite slope estimate.",
      );
    }
    requireFinitePersonMle(k, N, "Dynamic growth IRT");
    requireNoFixedEffectSeparation(k, N, "Dynamic growth IRT");
    const { raw, timeUnit } = validateTimePoints(options.timePoints, N);
    let { theta0, theta1, beta } = estimateGrowth(
      tensor,
      timeUnit,
      maxIter,
      slopeReg,
    );
    const correctByTime = tensor.map((model) =>
      Array.from({ length: N }, (_, n) => sum(model.map((item) => item[n]!))),
    );
    const equivalence = correctByTime.map((row) => [
      sum(row),
      sum(row.map((value, n) => value * timeUnit[n]!)),
    ]);
    theta0 = averageEquivalentScores(theta0, equivalence);
    theta1 = averageEquivalentScores(theta1, equivalence);
    const thetaPath = theta0.map((t0, l) => timeUnit.map((t) => t0 + theta1[l]! * t));
    scores = scoreDynamicPath(thetaPath, scoreTarget);
    itemParams = {
      difficulty: beta,
      baseline: theta0,
      slope: theta1,
      ability_path: thetaPath,
      time_points: raw,
    } as DynamicIrtItemParams;
    defineNonEnumerableAlias(itemParams, "abilityPath", thetaPath);
    defineNonEnumerableAlias(itemParams, "timePoints", raw);
  } else if (variant === "state_space") {
    if (!pythonTruthy(options.assumeTimeAxis)) {
      throw new Error(
        "variant='state_space' interprets axis-2 as ordered longitudinal time. " +
        "Set assume_time_axis=True to proceed.",
      );
    }
    if (N < 2) {
      throw new Error("Longitudinal dynamic IRT requires at least two time points.");
    }
    if (stateReg === 0) {
      throw new Error(
        "state_reg must be positive for variant='state_space' so each latent " +
          "trajectory has a proper random-walk penalty.",
      );
    }
    const { raw, timeUnit } = validateTimePoints(options.timePoints, N);
    const fitted = estimateStateSpace(tensor, timeUnit, maxIter, stateReg);
    const thetaPath = fitted.thetaPath;
    const equivalence = tensor.map((model) =>
      Array.from({ length: N }, (_, n) => sum(model.map((item) => item[n]!))),
    );
    for (let n = 0; n < N; n++) {
      const averaged = averageEquivalentScores(
        thetaPath.map((row) => row[n]!),
        equivalence,
      );
      for (let l = 0; l < thetaPath.length; l++) thetaPath[l]![n] = averaged[l]!;
    }
    scores = scoreDynamicPath(thetaPath, scoreTarget);
    itemParams = {
      difficulty: fitted.beta,
      ability_path: thetaPath,
      time_points: raw,
      gain: thetaPath.map((row) => row[row.length - 1]! - row[0]!),
    } as DynamicIrtItemParams;
    defineNonEnumerableAlias(itemParams, "abilityPath", thetaPath);
    defineNonEnumerableAlias(itemParams, "timePoints", raw);
  } else {
    throw new Error(`Unknown variant: ${variant}. Use 'linear', 'growth', or 'state_space'.`);
  }

  const result: IrtResult<DynamicIrtItemParams> = {
    ranking: rankScores(scores, method),
    scores,
  };
  if (pythonTruthy(options.returnItemParams)) result.itemParams = itemParams;
  return result;
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
  /** Include item parameters and multidimensional posterior summaries. */
  returnItemParams?: boolean;
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
function estimateMirt(
  k: number[][],
  nTrials: number,
  cfg: MirtConfig,
): {
  scores: number[];
  theta: number[][];
  thetaSd: number[][];
  slopes: number[][];
  intercept: number[];
  guessing: number[];
  discrimination: number[];
  difficulty: number[];
} {
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
    let res = minimize(obj, x0, { maxIter: cfg.maxIter });
    // A continuation preserves the fitted objective while compensating for
    // the smaller per-call line-search/history budget of the local optimizer.
    // SciPy's L-BFGS-B reaches these same 3PL M-steps within one call.
    if (!res.success) res = minimize(obj, res.x, { maxIter: cfg.maxIter });
    requireOptimizerSuccessOrStationarity(res, obj, "mirt item M-step");
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
  const thetaSd = Array.from({ length: L }, (_, l) =>
    Array.from({ length: D }, (_, dd) => {
      let secondMoment = 0;
      for (let g = 0; g < G; g++) {
        secondMoment += post[l]![g]! * grid[g]![dd]! ** 2;
      }
      return Math.sqrt(Math.max(secondMoment - theta[l]![dd]! ** 2, 0));
    }),
  );

  // Resolve the per-axis sign symmetry with non-negative mean slopes.
  for (let dd = 0; dd < D; dd++) {
    if (sum(a.map((row) => row[dd]!)) < 0) {
      for (let m = 0; m < M; m++) a[m]![dd] = -a[m]![dd]!;
      for (let l = 0; l < L; l++) theta[l]![dd] = -theta[l]![dd]!;
    }
  }

  const guessing = currentC(gamma) ?? new Array<number>(M).fill(0);
  const discrimination = a.map((row) =>
    Math.sqrt(sum(row.map((value) => value * value))),
  );
  const difficulty = d.map(
    (intercept, m) => -intercept / Math.max(discrimination[m]!, 1e-12),
  );
  const aBar = Array.from({ length: D }, (_, dd) => mean(a.map((row) => row[dd]!)));
  const scores = theta.map((row) => row.reduce((s, v, dd) => s + v * aBar[dd]!, 0));
  return {
    scores,
    theta,
    thetaSd,
    slopes: a,
    intercept: d,
    guessing,
    discrimination,
    difficulty,
  };
}

/**
 * Rank models with compensatory multidimensional IRT (MIRT) via marginal-MLE
 * EM. Each model has a `D`-dimensional ability vector and each item a slope
 * vector `a` and intercept `d`, with `P = c + (1-c)·σ(aᵀθ + d)` (`c = 0` for
 * 2PL). Abilities are scored by the rotation-invariant reference composite
 * `āᵀθ` (projection onto the mean slope direction). Port of `scorio.rank.mirt`.
 */
export function mirt(
  R: TensorInput,
  options: MirtOptions = {},
): IrtResult<MirtItemParams> {
  const method = defaultIfUndefined(options.method, "competition");
  const nFactors = validatePositiveInt(
    "n_factors",
    defaultIfUndefined(options.nFactors, 2),
    1,
  );
  const maxIter = validatePositiveInt(
    "max_iter",
    defaultIfUndefined(options.maxIter, 50),
  );
  const emIter = validatePositiveInt(
    "em_iter",
    defaultIfUndefined(options.emIter, 100),
  );
  const nQuadrature = validatePositiveInt(
    "n_quadrature",
    defaultIfUndefined(options.nQuadrature, 15),
    2,
  );

  const regDiscrimination = validateNonnegativeFloat(
    "reg_discrimination",
    defaultIfUndefined(options.regDiscrimination, 0.01),
  );
  const regGuessing = validateNonnegativeFloat(
    "reg_guessing",
    defaultIfUndefined(options.regGuessing, 0.1),
  );
  const tol = validateNonnegativeFloat(
    "tol",
    defaultIfUndefined(options.tol, 1e-4),
  );
  const guessingUpper = validateGuessingUpper(
    defaultIfUndefined(options.guessingUpper, 0.5),
  );

  const model = String(defaultIfUndefined(options.model, "2pl"))
    .trim()
    .toLowerCase();
  if (model !== "2pl" && model !== "3pl") {
    throw new Error("model must be '2pl' or '3pl'.");
  }
  const fixGuessingRaw = defaultIfUndefined(options.fixGuessing, null);
  if (model === "2pl" && fixGuessingRaw !== null) {
    throw new Error("fixGuessing is only valid for model='3pl'.");
  }
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
  requireFiniteItemEstimates(k, nTrials, "MIRT");
  const M = k[0]!.length;
  if (nFactors > M) {
    throw new Error(`n_factors=${nFactors} cannot exceed number of questions M=${M}.`);
  }

  const canonical = canonicalizeIrtCounts(k, true);
  const fitted = estimateMirt(canonical.k, nTrials, {
    nFactors,
    model,
    maxIter,
    emIter,
    nQuadrature,
    fixGuessing,
    regDiscrimination,
    regGuessing,
    guessingUpper,
    tol,
  });
  fitted.scores = restoreCanonicalVector(fitted.scores, canonical.modelOrder);
  fitted.theta = restoreCanonicalVector(fitted.theta, canonical.modelOrder);
  fitted.thetaSd = restoreCanonicalVector(fitted.thetaSd, canonical.modelOrder);
  fitted.slopes = restoreCanonicalVector(fitted.slopes, canonical.itemOrder);
  fitted.intercept = restoreCanonicalVector(fitted.intercept, canonical.itemOrder);
  fitted.guessing = restoreCanonicalVector(fitted.guessing, canonical.itemOrder);
  fitted.discrimination = restoreCanonicalVector(
    fitted.discrimination,
    canonical.itemOrder,
  );
  fitted.difficulty = restoreCanonicalVector(
    fitted.difficulty,
    canonical.itemOrder,
  );
  const scores = averageEventExchangeableScores(fitted.scores, k);
  for (let dd = 0; dd < nFactors; dd++) {
    const abilities = averageEventExchangeableScores(
      fitted.theta.map((row) => row[dd]!),
      k,
    );
    const abilitySd = averageEventExchangeableScores(
      fitted.thetaSd.map((row) => row[dd]!),
      k,
    );
    for (let l = 0; l < fitted.theta.length; l++) {
      fitted.theta[l]![dd] = abilities[l]!;
      fitted.thetaSd[l]![dd] = abilitySd[l]!;
    }
  }
  const result: IrtResult<MirtItemParams> = {
    ranking: rankScores(scores, method),
    scores,
  };
  if (pythonTruthy(options.returnItemParams)) {
    const itemParams = {
      difficulty: fitted.difficulty,
      discrimination: fitted.discrimination,
      slopes: fitted.slopes,
      intercept: fitted.intercept,
      abilities: fitted.theta,
      ability_sd: fitted.thetaSd,
      ...(model === "3pl" ? { guessing: fitted.guessing } : {}),
    } as MirtItemParams;
    defineNonEnumerableAlias(itemParams, "abilitySd", fitted.thetaSd);
    result.itemParams = itemParams;
  }
  return result;
}
