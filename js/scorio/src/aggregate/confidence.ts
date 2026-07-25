/** Confidence and uncertainty signals derived from one trace's token scores. */

import { mean, quantile } from "./internal/math.js";
import {
  asFiniteVector,
  type NumericInput,
} from "./internal/numeric.js";
import {
  pythonComparableNumber,
  pythonInt,
  pythonTruthy,
} from "./internal/runtime.js";

export type { NumericInput } from "./internal/numeric.js";

/** One top-k row or a rectangular/ragged sequence of top-k rows. */
export type TopKLogprobs =
  | readonly number[]
  | readonly (readonly number[])[];

export type TokenReducer = "mean" | "min" | "max";
export type DeepconfMode = "mean" | "tail" | "bottom_group" | "lowest_group";

function asTopK(input: TopKLogprobs): number[][] {
  if (!Array.isArray(input) || input.length === 0) {
    throw new Error("need at least one token (T >= 1).");
  }
  const values = input as readonly unknown[];
  const nested = Array.isArray(values[0]);
  let rows: number[][];
  if (nested) {
    if (!values.every(Array.isArray)) {
      throw new Error("topk_logprobs must be a top-k row or a sequence of rows.");
    }
    rows = (values as readonly (readonly number[])[]).map((row) => [...row]);
  } else {
    if (values.some(Array.isArray)) {
      throw new Error("topk_logprobs must be a top-k row or a sequence of rows.");
    }
    rows = [[...(input as readonly number[])]];
  }
  for (const row of rows) {
    if (row.length === 0) {
      throw new Error("every position needs at least one top-k candidate.");
    }
    if (row.some((value) => typeof value !== "number" || !Number.isFinite(value))) {
      throw new Error("topk_logprobs must all be finite.");
    }
  }
  return rows;
}

function maximum(values: readonly number[]): number {
  let out = -Infinity;
  for (const value of values) if (value > out) out = value;
  return out;
}

function minimum(values: readonly number[]): number {
  let out = Infinity;
  for (const value of values) if (value < out) out = value;
  return out;
}

function reduce(values: readonly number[], how: TokenReducer): number {
  if (how === "mean") return mean(values);
  if (how === "min") return minimum(values);
  if (how === "max") return maximum(values);
  throw new Error(`aggregate must be one of mean, min, max; got ${String(how)}.`);
}

function normalizedLogprobs(row: readonly number[]): number[] {
  const rowMax = maximum(row);
  const shifted = row.map((value) => value - rowMax);
  let total = 0;
  for (const value of shifted) total += Math.exp(value);
  const logNormalizer = Math.log(total);
  return shifted.map((value) => value - logNormalizer);
}

function perTokenEntropy(rows: readonly (readonly number[])[]): number[] {
  return rows.map((row) => {
    let entropy = 0;
    for (const logProbability of normalizedLogprobs(row)) {
      entropy -= Math.exp(logProbability) * logProbability;
    }
    return entropy;
  });
}

function perTokenVarentropy(rows: readonly (readonly number[])[]): number[] {
  return rows.map((row) => {
    const logProbabilities = normalizedLogprobs(row);
    let entropy = 0;
    for (const logProbability of logProbabilities) {
      const probability = Math.exp(logProbability);
      entropy -= probability * logProbability;
    }
    let variance = 0;
    for (const logProbability of logProbabilities) {
      const probability = Math.exp(logProbability);
      const centeredSurprisal = -logProbability - entropy;
      variance += probability * centeredSurprisal * centeredSurprisal;
    }
    return variance;
  });
}

function perTokenSelfCertainty(rows: readonly (readonly number[])[]): number[] {
  return rows.map((row) => {
    const logProbabilities = normalizedLogprobs(row);
    let meanLogProbability = 0;
    for (const logProbability of logProbabilities) {
      meanLogProbability += logProbability;
    }
    meanLogProbability /= logProbabilities.length;
    return -Math.log(logProbabilities.length) - meanLogProbability;
  });
}

function perTokenConfidence(rows: readonly (readonly number[])[]): number[] {
  return rows.map((row) => -mean(row));
}

function perTokenMaxProbability(rows: readonly (readonly number[])[]): number[] {
  return rows.map((row) => Math.exp(maximum(row)));
}

function perTokenMargin(
  rows: readonly (readonly number[])[],
  useProbability: boolean,
): number[] {
  return rows.map((row) => {
    if (row.length < 2) return 0;
    let top1 = -Infinity;
    let top2 = -Infinity;
    for (const value of row) {
      if (value > top1) {
        top2 = top1;
        top1 = value;
      } else if (value > top2) {
        top2 = value;
      }
    }
    return useProbability
      ? Math.exp(top1) - Math.exp(top2)
      : top1 - top2;
  });
}

/** Mean chosen-token log-probability (higher is more confident). */
export function meanLogprob(logprobs: NumericInput): number {
  return mean(asFiniteVector(logprobs, "logprobs"));
}

/** Total chosen-token sequence log-likelihood (higher is more confident). */
export function sequenceLogprob(logprobs: NumericInput): number {
  let total = 0;
  for (const value of asFiniteVector(logprobs, "logprobs")) total += value;
  return total;
}

/** Sequence perplexity; unlike most signals here, lower is more confident. */
export function perplexity(logprobs: NumericInput): number {
  return Math.exp(-meanLogprob(logprobs));
}

export interface PicsarOptions {
  answerStart?: number | null;
  normalizeReasoning?: boolean;
}

/** PiCSAR reasoning-plus-answer log-likelihood selector. */
export function picsar(
  logprobs: NumericInput,
  options: PicsarOptions = {},
): number {
  const { answerStart = null, normalizeReasoning = false } = options;
  const values = asFiniteVector(logprobs, "logprobs");
  if (answerStart === null || answerStart === undefined) {
    let total = 0;
    for (const value of values) total += value;
    return total;
  }
  const split = pythonComparableNumber(answerStart, "answer_start");
  if (!Number.isInteger(split) || split < 0 || split > values.length) {
    throw new Error(
      `answer_start must be an integer in [0, ${values.length}]; got ${answerStart}.`,
    );
  }
  let reasoning = 0;
  for (let i = 0; i < split; i++) reasoning += values[i]!;
  if (pythonTruthy(normalizeReasoning) && split > 0) reasoning /= split;
  let answer = 0;
  for (let i = split; i < values.length; i++) answer += values[i]!;
  return reasoning + answer;
}

export interface ReducerOptions {
  aggregate?: TokenReducer;
}

/** KL-from-uniform over each renormalized top-k law, reduced over the trace. */
export function selfCertainty(
  topkLogprobs: TopKLogprobs,
  options: ReducerOptions = {},
): number {
  return reduce(
    perTokenSelfCertainty(asTopK(topkLogprobs)),
    options.aggregate === undefined ? "mean" : options.aggregate,
  );
}

/** Shannon entropy of the renormalized top-k law (lower is more confident). */
export function tokenEntropy(
  topkLogprobs: TopKLogprobs,
  options: ReducerOptions = {},
): number {
  return reduce(
    perTokenEntropy(asTopK(topkLogprobs)),
    options.aggregate === undefined ? "mean" : options.aggregate,
  );
}

/** Variance of surprisal under the renormalized top-k law. */
export function varentropy(
  topkLogprobs: TopKLogprobs,
  options: ReducerOptions = {},
): number {
  return reduce(
    perTokenVarentropy(asTopK(topkLogprobs)),
    options.aggregate === undefined ? "mean" : options.aggregate,
  );
}

/** Maximum raw softmax probability per position, reduced over the trace. */
export function maxSoftmaxProbability(
  topkLogprobs: TopKLogprobs,
  options: ReducerOptions = {},
): number {
  return reduce(
    perTokenMaxProbability(asTopK(topkLogprobs)),
    options.aggregate === undefined ? "mean" : options.aggregate,
  );
}

export interface LogprobMarginOptions extends ReducerOptions {
  useProb?: boolean;
}

/** Top-1 minus top-2 margin in log-probability or probability space. */
export function logprobMargin(
  topkLogprobs: TopKLogprobs,
  options: LogprobMarginOptions = {},
): number {
  return reduce(
    perTokenMargin(asTopK(topkLogprobs), pythonTruthy(options.useProb)),
    options.aggregate === undefined ? "mean" : options.aggregate,
  );
}

/** DeepConf per-token confidence `-mean(top-k log probabilities)`. */
export function tokenConfidence(topkLogprobs: TopKLogprobs): number[] {
  return perTokenConfidence(asTopK(topkLogprobs));
}

/** Sliding-window means, with windows longer than the trace clamped to `T`. */
export function groupConfidences(
  confidence: readonly number[],
  window: number,
): number[] {
  const width = Math.min(pythonInt(window, "window"), confidence.length);
  if (width <= 0) throw new Error(`window must be positive; got ${window}.`);
  if (width === confidence.length) return [mean(confidence)];

  // Mirror NumPy's `cumsum(insert(conf, 0, 0.0))` expression used by Python,
  // including its floating-point operation order.
  const cumulative = new Array<number>(confidence.length + 1);
  cumulative[0] = 0;
  for (let i = 0; i < confidence.length; i++) {
    cumulative[i + 1] = cumulative[i]! + confidence[i]!;
  }
  const groups = new Array<number>(confidence.length - width + 1);
  for (let start = 0; start < groups.length; start++) {
    groups[start] =
      (cumulative[start + width]! - cumulative[start]!) / width;
  }
  return groups;
}

export interface DeepconfConfidenceOptions {
  mode?: DeepconfMode;
  window?: number;
  tailTokens?: number;
  bottomQuantile?: number;
}

/** DeepConf mean, tail, bottom-group, or lowest-group trace confidence. */
export function deepconfConfidence(
  topkLogprobs: TopKLogprobs,
  options: DeepconfConfidenceOptions = {},
): number {
  const {
    mode = "mean",
    window = 2048,
    tailTokens = 2048,
    bottomQuantile = 0.1,
  } = options;
  const confidence = tokenConfidence(topkLogprobs);
  if (mode === "mean") return mean(confidence);
  if (mode === "tail") {
    const tail = Math.min(
      Math.max(pythonInt(tailTokens, "tail_tokens"), 1),
      confidence.length,
    );
    return mean(confidence.slice(-tail));
  }
  if (mode === "lowest_group" || mode === "bottom_group") {
    const groups = groupConfidences(confidence, window);
    if (mode === "lowest_group") return minimum(groups);
    const fraction = pythonComparableNumber(bottomQuantile, "bottom_quantile");
    if (!(fraction > 0 && fraction <= 1)) {
      throw new Error(`bottom_quantile must be in (0, 1]; got ${bottomQuantile}.`);
    }
    const threshold = quantile(groups, fraction);
    return mean(groups.filter((value) => value <= threshold));
  }
  throw new Error(
    `mode must be 'mean', 'tail', 'bottom_group', or 'lowest_group'; got ${String(mode)}.`,
  );
}
