/** KDE-calibrated weighted voting for scalar verifier probabilities. */

import {
  type AnswerInput,
  normalizePool,
  packSelection,
  type PackedSelection,
  type ScoreInput,
  type SelectionReturnOptions,
  validIndices,
} from "./internal/base.js";
import type { NumericInput } from "./internal/numeric.js";
import { pythonFloat } from "./internal/runtime.js";

interface Flattened {
  values: number[];
  shape: number[];
}

function sameShape(left: readonly number[], right: readonly number[]): boolean {
  return (
    left.length === right.length &&
    left.every((dimension, index) => dimension === right[index])
  );
}

function flattenNumeric(input: NumericInput, name: string): Flattened {
  if (typeof input === "number") return { values: [input], shape: [] };
  if (!Array.isArray(input)) throw new Error(`${name} must contain numbers.`);
  if (input.length === 0) return { values: [], shape: [0] };
  const children = input.map((entry) => flattenNumeric(entry, name));
  const childShape = children[0]!.shape;
  if (children.some((child) => !sameShape(child.shape, childShape))) {
    throw new Error(`${name} must be a rectangular numeric array.`);
  }
  return {
    values: children.flatMap((child) => child.values),
    shape: [input.length, ...childShape],
  };
}

function flattenCorrect(input: unknown, name = "correct"): {
  values: (number | boolean)[];
  shape: number[];
} {
  if (typeof input === "number" || typeof input === "boolean") {
    return { values: [input], shape: [] };
  }
  if (!Array.isArray(input)) {
    throw new Error(`${name} must contain only boolean or 0/1 values.`);
  }
  if (input.length === 0) return { values: [], shape: [0] };
  const children = input.map((entry) => flattenCorrect(entry, name));
  const childShape = children[0]!.shape;
  if (children.some((child) => !sameShape(child.shape, childShape))) {
    throw new Error(`${name} must be a rectangular array.`);
  }
  return {
    values: children.flatMap((child) => child.values),
    shape: [input.length, ...childShape],
  };
}

function rebuildShape(values: readonly number[], shape: readonly number[]): NumericInput {
  let offset = 0;
  const build = (dimensions: readonly number[]): NumericInput => {
    if (dimensions.length === 0) return values[offset++]!;
    const [length, ...rest] = dimensions;
    return Array.from({ length: length! }, () => build(rest));
  };
  return build(shape);
}

function probabilityValues(input: NumericInput, name: string): Flattened {
  const flattened = flattenNumeric(input, name);
  if (
    flattened.values.some(
      (value) => !Number.isFinite(value) || !(value > 0 && value < 1),
    )
  ) {
    throw new Error(`${name} must all be finite and strictly in (0, 1).`);
  }
  return flattened;
}

function logit(value: number): number {
  return Math.log(value) - Math.log1p(-value);
}

function logGaussianKde(
  queryLogits: readonly number[],
  samples: readonly number[],
  bandwidth: number,
): number[] {
  const normalizer = Math.log(
    samples.length * bandwidth * Math.sqrt(2 * Math.PI),
  );
  return queryLogits.map((query) => {
    const terms = samples.map((sample) => {
      const standardized = (query - sample) / bandwidth;
      return -0.5 * standardized * standardized;
    });
    let maximum = -Infinity;
    for (const term of terms) if (term > maximum) maximum = term;
    if (maximum === -Infinity) return -Infinity;
    let sum = 0;
    for (const term of terms) sum += Math.exp(term - maximum);
    return maximum + Math.log(sum) - normalizer;
  });
}

function readonlyVector(values: readonly number[], name: string): readonly number[] {
  if (!Array.isArray(values) || values.some(Array.isArray)) {
    throw new Error(`${name} must be one-dimensional.`);
  }
  return Object.freeze([...values]);
}

export interface KDEVoteCalibrationInit {
  correctLogits: readonly number[];
  incorrectLogits: readonly number[];
  correctBandwidth: number;
  incorrectBandwidth: number;
  binEdges: readonly number[];
  binProbability: readonly number[];
  kernel?: "gaussian";
  binning?: "quantile";
}

/** Fitted immutable state for non-parametric KDE weighted voting. */
export class KDEVoteCalibration {
  readonly correctLogits: readonly number[];
  readonly incorrectLogits: readonly number[];
  readonly correctBandwidth: number;
  readonly incorrectBandwidth: number;
  readonly binEdges: readonly number[];
  readonly binProbability: readonly number[];
  readonly kernel: "gaussian";
  readonly binning: "quantile";

  constructor(init: KDEVoteCalibrationInit) {
    const correct = readonlyVector(init.correctLogits, "correct_logits");
    const incorrect = readonlyVector(init.incorrectLogits, "incorrect_logits");
    const edges = readonlyVector(init.binEdges, "bin_edges");
    const probabilities = readonlyVector(init.binProbability, "bin_probability");
    const kernel = init.kernel === undefined ? "gaussian" : init.kernel;
    const binning = init.binning === undefined ? "quantile" : init.binning;

    if (correct.length === 0 || incorrect.length === 0) {
      throw new Error("KDE calibration needs correct and incorrect samples.");
    }
    if (
      correct.some((value) => !Number.isFinite(value)) ||
      incorrect.some((value) => !Number.isFinite(value))
    ) {
      throw new Error("KDE logit samples must all be finite.");
    }
    const correctBandwidth = pythonFloat(
      init.correctBandwidth,
      "correct_bandwidth",
    );
    const incorrectBandwidth = pythonFloat(
      init.incorrectBandwidth,
      "incorrect_bandwidth",
    );
    if (
      !Number.isFinite(correctBandwidth) ||
      correctBandwidth <= 0 ||
      !Number.isFinite(incorrectBandwidth) ||
      incorrectBandwidth <= 0
    ) {
      throw new Error("KDE bandwidths must be finite and > 0.");
    }
    if (edges.length !== probabilities.length + 1 || edges.length < 2) {
      throw new Error("bin_edges must contain exactly one more value than bins.");
    }
    if (edges[0] !== -Infinity) throw new Error("bin_edges must start at -inf.");
    if (edges[edges.length - 1] !== Infinity) {
      throw new Error("bin_edges must end at +inf.");
    }
    for (let index = 1; index < edges.length; index++) {
      if (!(edges[index]! > edges[index - 1]!)) {
        throw new Error("bin_edges must be strictly increasing.");
      }
    }
    if (
      probabilities.some(
        (value) => !Number.isFinite(value) || value < 0 || value > 1,
      )
    ) {
      throw new Error("bin_probability values must be finite and in [0, 1].");
    }
    if (kernel !== "gaussian") {
      throw new Error("only the implemented 'gaussian' kernel is valid.");
    }
    if (binning !== "quantile") {
      throw new Error("only the implemented 'quantile' binning is valid.");
    }

    this.correctLogits = correct;
    this.incorrectLogits = incorrect;
    this.correctBandwidth = correctBandwidth;
    this.incorrectBandwidth = incorrectBandwidth;
    this.binEdges = edges;
    this.binProbability = probabilities;
    this.kernel = kernel;
    this.binning = binning;
    Object.freeze(this);
  }

  get nBins(): number {
    return this.binProbability.length;
  }

  /** Python-compatible attribute alias. */
  get n_bins(): number {
    return this.nBins;
  }

  get correct_logits(): readonly number[] {
    return this.correctLogits;
  }

  get incorrect_logits(): readonly number[] {
    return this.incorrectLogits;
  }

  get correct_bandwidth(): number {
    return this.correctBandwidth;
  }

  get incorrect_bandwidth(): number {
    return this.incorrectBandwidth;
  }

  get bin_edges(): readonly number[] {
    return this.binEdges;
  }

  get bin_probability(): readonly number[] {
    return this.binProbability;
  }

  calibratedProbability(scores: NumericInput): NumericInput {
    const { values, shape } = probabilityValues(scores, "scores");
    const internal = this.binEdges.slice(1, -1);
    const output = values.map((value) => {
      let low = 0;
      let high = internal.length;
      while (low < high) {
        const middle = (low + high) >>> 1;
        if (value < internal[middle]!) high = middle;
        else low = middle + 1;
      }
      return this.binProbability[low]!;
    });
    return rebuildShape(output, shape);
  }

  calibrated_probability(scores: NumericInput): NumericInput {
    return this.calibratedProbability(scores);
  }

  logDensityRatio(scores: NumericInput): NumericInput {
    const { values, shape } = probabilityValues(scores, "scores");
    const logits = values.map(logit);
    const correct = logGaussianKde(
      logits,
      this.correctLogits,
      this.correctBandwidth,
    );
    const incorrect = logGaussianKde(
      logits,
      this.incorrectLogits,
      this.incorrectBandwidth,
    );
    return rebuildShape(
      correct.map((value, index) => value - incorrect[index]!),
      shape,
    );
  }

  log_density_ratio(scores: NumericInput): NumericInput {
    return this.logDensityRatio(scores);
  }

  weights(scores: readonly number[], options: { nAnswers: number }): number[] {
    const nAnswers = options.nAnswers;
    if (!Number.isInteger(nAnswers) || nAnswers < 2) {
      throw new Error(
        nAnswers < 2
          ? "n_answers must be >= 2 for the KDE weight formula."
          : `n_answers must be an integer >= 2; got ${nAnswers}.`,
      );
    }
    const { values, shape } = probabilityValues(scores, "scores");
    if (shape.length !== 1 || values.length === 0) {
      throw new Error("scores must be a nonempty 1D response pool.");
    }
    const calibrated = this.calibratedProbability(values) as number[];
    let qHat = 0;
    for (const value of calibrated) qHat += value;
    qHat /= calibrated.length;
    const offset =
      qHat === 0
        ? -Infinity
        : qHat === 1
          ? Infinity
          : Math.log(qHat) + Math.log(nAnswers - 1) - Math.log1p(-qHat);
    const ratio = this.logDensityRatio(values) as number[];
    return ratio.map((value) => value + offset);
  }
}

export type KDEBandwidthSpecification =
  | "scott"
  | number
  | readonly ["scott" | number, "scott" | number];

export interface FitKDEVoteCalibrationOptions {
  nBins?: number;
  bandwidth?: KDEBandwidthSpecification;
}

function resolveBandwidth(
  samples: readonly number[],
  specification: "scott" | number,
  label: string,
): number {
  if (specification === "scott") {
    if (samples.length < 2) {
      throw new Error(
        `Scott bandwidth for ${label} needs at least two samples; ` +
          "supply an explicit bandwidth instead.",
      );
    }
    let mean = 0;
    for (const sample of samples) mean += sample;
    mean /= samples.length;
    let squared = 0;
    for (const sample of samples) squared += (sample - mean) ** 2;
    const standardDeviation = Math.sqrt(squared / (samples.length - 1));
    if (!Number.isFinite(standardDeviation) || standardDeviation <= 0) {
      throw new Error(
        `Scott bandwidth is undefined for constant ${label} logits; ` +
          "supply an explicit positive bandwidth instead.",
      );
    }
    return standardDeviation * samples.length ** (-1 / 5);
  }
  if (typeof specification === "string") {
    throw new Error("bandwidth must be a positive number, pair, or 'scott'.");
  }
  const value = pythonFloat(specification, "bandwidth");
  if (!Number.isFinite(value) || value <= 0) {
    throw new Error(`bandwidth must be finite and > 0; got ${specification}.`);
  }
  return value;
}

function roundHalfToEven(value: number): number {
  const lower = Math.floor(value);
  const fraction = value - lower;
  if (fraction < 0.5) return lower;
  if (fraction > 0.5) return lower + 1;
  return lower % 2 === 0 ? lower : lower + 1;
}

/** Fit class-conditional Gaussian KDEs and a quantile-bin calibrator. */
export function fitKdeVoteCalibration(
  scores: NumericInput,
  correct: unknown,
  options: FitKDEVoteCalibrationOptions = {},
): KDEVoteCalibration {
  const { nBins = 10, bandwidth = "scott" } = options;
  if (!Number.isInteger(nBins) || nBins < 1) {
    throw new Error(`n_bins must be an integer >= 1; got ${nBins}.`);
  }
  const scoreData = probabilityValues(scores, "scores");
  const correctData = flattenCorrect(correct);
  if (!sameShape(scoreData.shape, correctData.shape)) {
    throw new Error(
      `scores and correct must have the same shape; got (${scoreData.shape.join(", ")}) ` +
        `and (${correctData.shape.join(", ")}).`,
    );
  }
  const labels = correctData.values.map((value) => {
    if (value === true || value === 1) return true;
    if (value === false || value === 0) return false;
    throw new Error("correct must contain only boolean or 0/1 values.");
  });
  if (scoreData.values.length === 0) {
    throw new Error("need at least one calibration response.");
  }
  if (!labels.some(Boolean) || labels.every(Boolean)) {
    throw new Error("KDE calibration needs correct and incorrect responses.");
  }

  const logits = scoreData.values.map(logit);
  const correctLogits = logits.filter((_, index) => labels[index]);
  const incorrectLogits = logits.filter((_, index) => !labels[index]);
  let correctSpecification: "scott" | number;
  let incorrectSpecification: "scott" | number;
  if (Array.isArray(bandwidth)) {
    if (bandwidth.length !== 2) {
      throw new Error("a bandwidth sequence must be (correct, incorrect).");
    }
    [correctSpecification, incorrectSpecification] = bandwidth;
  } else {
    correctSpecification = bandwidth as "scott" | number;
    incorrectSpecification = bandwidth as "scott" | number;
  }

  const ordered = [...scoreData.values].sort((left, right) => left - right);
  const boundaries: number[] = [];
  const minimum = ordered[0]!;
  const maximum = ordered[ordered.length - 1]!;
  // Reproduce `np.quantile(scores, np.linspace(0, 1, n_bins + 1), method="nearest")`
  // operation for operation: NumPy forms each probability as `i * (1 / n_bins)`
  // and the virtual index as `(n - 1) * q`, then rounds half to even. Folding
  // those into `(n - 1) * i / n_bins` rounds differently whenever `i / n_bins`
  // is inexact (e.g. `15 * 0.30000000000000004 = 4.500000000000001` rounds to 5,
  // while an exact `4.5` rounds to 4).
  const step = 1 / nBins;
  for (let index = 1; index < nBins; index++) {
    const position = (ordered.length - 1) * (index * step);
    const boundary = ordered[roundHalfToEven(position)]!;
    if (
      boundary > minimum &&
      boundary < maximum &&
      !boundaries.includes(boundary)
    ) {
      boundaries.push(boundary);
    }
  }
  boundaries.sort((left, right) => left - right);
  const binEdges = [-Infinity, ...boundaries, Infinity];
  const binCorrect = new Array<number>(binEdges.length - 1).fill(0);
  const binCount = new Array<number>(binEdges.length - 1).fill(0);
  for (let index = 0; index < scoreData.values.length; index++) {
    const value = scoreData.values[index]!;
    let bin = 0;
    while (bin < boundaries.length && value >= boundaries[bin]!) bin++;
    binCount[bin]! += 1;
    if (labels[index]) binCorrect[bin]! += 1;
  }
  const binProbability = binCount.map((count, index) => {
    if (count === 0) {
      throw new Error("internal quantile construction produced an empty bin.");
    }
    return binCorrect[index]! / count;
  });

  return new KDEVoteCalibration({
    correctLogits,
    incorrectLogits,
    correctBandwidth: resolveBandwidth(
      correctLogits,
      correctSpecification,
      "correct-class",
    ),
    incorrectBandwidth: resolveBandwidth(
      incorrectLogits,
      incorrectSpecification,
      "incorrect-class",
    ),
    binEdges,
    binProbability,
  });
}

export interface KDEWeightedVoteOptions<
  ReturnIndex extends boolean = false,
  ReturnScore extends boolean = false,
> extends SelectionReturnOptions<ReturnIndex, ReturnScore> {}

/** Select answers using fitted non-parametric KDE vote weights. */
export function kdeWeightedVote<
  T,
  ReturnIndex extends boolean = false,
  ReturnScore extends boolean = false,
>(
  answersInput: AnswerInput<T>,
  scoresInput: ScoreInput,
  calibration: KDEVoteCalibration,
  options: KDEWeightedVoteOptions<ReturnIndex, ReturnScore> = {},
): PackedSelection<T, ReturnIndex, ReturnScore> {
  if (!(calibration instanceof KDEVoteCalibration)) {
    throw new Error("calibration must be a KDEVoteCalibration.");
  }
  const { answers, scores, single } = normalizePool(
    answersInput,
    scoresInput,
    true,
  );
  const selected: (T | null)[] = [];
  const indices: number[] = [];
  const selectedScores: number[] = [];

  for (let rowIndex = 0; rowIndex < answers.length; rowIndex++) {
    const row = answers[rowIndex]!;
    const scoreRow = scores![rowIndex]!;
    const part = validIndices(row);
    if (part.length === 0) {
      selected.push(null);
      indices.push(-1);
      selectedScores.push(NaN);
      continue;
    }
    const values = part.map((index) => scoreRow[index]!);
    probabilityValues(values, "valid scores");
    const groups = new Map<T, {
      local: number[];
      first: number;
      representative: number;
    }>();
    for (let local = 0; local < part.length; local++) {
      const index = part[local]!;
      const answer = row[index]!;
      const group = groups.get(answer);
      if (group === undefined) {
        groups.set(answer, { local: [local], first: index, representative: index });
      } else {
        group.local.push(local);
        if (scoreRow[index]! > scoreRow[group.representative]!) {
          group.representative = index;
        }
      }
    }

    let winner: T;
    if (groups.size === 1) {
      winner = groups.keys().next().value as T;
    } else {
      const densityRatio = calibration.logDensityRatio(values) as number[];
      const calibrated = calibration.calibratedProbability(values) as number[];
      let qHat = 0;
      for (const value of calibrated) qHat += value;
      qHat /= calibrated.length;
      const nAnswers = groups.size;
      let winnerSet = false;
      let bestPrimary = -Infinity;
      let bestSecondary = -Infinity;
      let bestFirst = Infinity;
      winner = groups.keys().next().value as T;
      for (const [answer, group] of groups) {
        let ratioSum = 0;
        for (const local of group.local) ratioSum += densityRatio[local]!;
        const primary =
          qHat === 1
            ? group.local.length
            : qHat === 0
              ? -group.local.length
              : ratioSum +
                group.local.length *
                  (Math.log(qHat) +
                    Math.log(nAnswers - 1) -
                    Math.log1p(-qHat));
        const secondary = qHat === 0 || qHat === 1 ? ratioSum : 0;
        if (
          !winnerSet ||
          primary > bestPrimary ||
          (primary === bestPrimary && secondary > bestSecondary) ||
          (primary === bestPrimary &&
            secondary === bestSecondary &&
            group.first < bestFirst)
        ) {
          winnerSet = true;
          winner = answer;
          bestPrimary = primary;
          bestSecondary = secondary;
          bestFirst = group.first;
        }
      }
    }
    const representative = groups.get(winner)!.representative;
    selected.push(winner);
    indices.push(representative);
    selectedScores.push(scoreRow[representative]!);
  }

  return packSelection(
    selected,
    indices,
    selectedScores,
    single,
    (options.returnIndex ?? false) as ReturnIndex,
    (options.returnScore ?? false) as ReturnScore,
  );
}
