/** Confidence-Guided Early Stopping (CGES) selection and stopping. */

import {
  type AnswerInput,
  normalizePool,
  packSelection,
  type PackedSelection,
  type ScoreInput,
  type SelectionReturnOptions,
  validIndices,
} from "./internal/base.js";
import {
  pythonComparableNumber,
  pythonTruthy,
} from "./internal/runtime.js";

/** Identity sentinel for a correct answer that has not appeared yet. */
class CGESOtherSentinel {
  private readonly brand = "CGES_OTHER";

  toString(): string {
    return this.brand;
  }
}

export type CGESOther = CGESOtherSentinel;

const cgesOther = new CGESOtherSentinel();
Object.freeze(cgesOther);
export const CGES_OTHER: CGESOther = cgesOther;

function validateBoolean(value: unknown, name: string): asserts value is boolean {
  if (typeof value !== "boolean") {
    throw new Error(`${name} must be a boolean; got ${String(value)}.`);
  }
}

function logSumExp(values: readonly number[]): number {
  let maximum = -Infinity;
  for (const value of values) if (value > maximum) maximum = value;
  if (maximum === -Infinity) return -Infinity;
  let sum = 0;
  for (const value of values) sum += Math.exp(value - maximum);
  return maximum + Math.log(sum);
}

function rowCgesPosterior<T>(
  answers: readonly T[],
  scores: readonly number[],
): Map<T | CGESOther, number> {
  const part = validIndices(answers);
  if (part.length === 0) return new Map([[CGES_OTHER, 1]]);

  const concrete: T[] = [];
  const seen = new Set<T>();
  for (const index of part) {
    const answer = answers[index]!;
    if ((answer as unknown) === CGES_OTHER) {
      throw new Error("CGES_OTHER is reserved and cannot be an observed answer.");
    }
    if (!seen.has(answer)) {
      seen.add(answer);
      concrete.push(answer);
    }
  }

  const supportSize = concrete.length + 1;
  const mismatch = new Map<number, number>();
  let base = 0;
  for (const index of part) {
    const confidence = scores[index]!;
    if (!Number.isFinite(confidence) || !(confidence > 0 && confidence < 1)) {
      throw new Error(
        "CGES requires every valid candidate score to be finite and " +
          `strictly in (0, 1); got ${confidence}.`,
      );
    }
    const value = Math.log1p(-confidence) - Math.log(supportSize - 1);
    mismatch.set(index, value);
    base += value;
  }

  const logScores = new Map<T | CGESOther, number>();
  for (const answer of concrete) logScores.set(answer, base);
  logScores.set(CGES_OTHER, base);
  for (const index of part) {
    const answer = answers[index]!;
    const current = logScores.get(answer)!;
    logScores.set(
      answer,
      current + Math.log(scores[index]!) - mismatch.get(index)!,
    );
  }
  const normalizer = logSumExp([...logScores.values()]);
  const posterior = new Map<T | CGESOther, number>();
  for (const [answer, logScore] of logScores) {
    posterior.set(answer, Math.exp(logScore - normalizer));
  }
  return posterior;
}

export interface CGESVoteOptions<
  ReturnIndex extends boolean = false,
  ReturnScore extends boolean = false,
> extends SelectionReturnOptions<ReturnIndex, ReturnScore> {
  allowOther?: boolean;
}

/** Select the observed answer (or optional OTHER bucket) with largest CGES score. */
export function cgesVote<
  T,
  ReturnIndex extends boolean = false,
  ReturnScore extends boolean = false,
>(
  answersInput: AnswerInput<T>,
  scoresInput: ScoreInput,
  options: CGESVoteOptions<ReturnIndex, ReturnScore> = {},
): PackedSelection<T | CGESOther, ReturnIndex, ReturnScore> {
  const allowOther =
    options.allowOther === undefined ? false : options.allowOther;
  validateBoolean(allowOther, "allow_other");
  const { answers, scores, single } = normalizePool(
    answersInput,
    scoresInput,
    true,
  );
  const selected: (T | CGESOther | null)[] = [];
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
    const posterior = rowCgesPosterior(row, scoreRow);
    let winnerSet = false;
    let winner!: T | CGESOther;
    let winnerProbability = -Infinity;
    for (const [answer, probability] of posterior) {
      if (!allowOther && answer === CGES_OTHER) continue;
      if (!winnerSet || probability > winnerProbability) {
        winnerSet = true;
        winner = answer;
        winnerProbability = probability;
      }
    }
    selected.push(winner);
    if (winner === CGES_OTHER) {
      indices.push(-1);
      selectedScores.push(NaN);
      continue;
    }
    let representative = -1;
    let representativeScore = -Infinity;
    for (const index of part) {
      if (row[index] === winner && scoreRow[index]! > representativeScore) {
        representative = index;
        representativeScore = scoreRow[index]!;
      }
    }
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

export interface CGESStopOptions<ReturnProb extends boolean = false> {
  threshold?: number;
  includeOther?: boolean;
  minSamples?: number;
  returnProb?: ReturnProb;
}

/** Stop a one-dimensional CGES sampling stream once a hypothesis crosses threshold. */
export function cgesStop<T, ReturnProb extends boolean = false>(
  answersInput: AnswerInput<T>,
  scoresInput: ScoreInput,
  options: CGESStopOptions<ReturnProb> = {},
): ReturnProb extends true ? [boolean, number] : boolean {
  const {
    threshold = 0.95,
    includeOther = false,
    minSamples = 1,
  } = options;
  const boundary = pythonComparableNumber(threshold, "threshold");
  if (!(boundary > 0 && boundary < 1)) {
    throw new Error(`threshold must be in (0, 1); got ${threshold}.`);
  }
  validateBoolean(includeOther, "include_other");
  if (!Number.isInteger(minSamples) || minSamples < 1) {
    throw new Error(`min_samples must be an integer >= 1; got ${minSamples}.`);
  }
  const { answers, scores, single } = normalizePool(
    answersInput,
    scoresInput,
    true,
  );
  if (!single) {
    throw new Error("cges_stop expects one 1D sampling stream, not a batch.");
  }
  const part = validIndices(answers[0]!);
  if (part.length === 0) {
    return (pythonTruthy(options.returnProb) ? [false, 0] : false) as ReturnProb extends true
      ? [boolean, number]
      : boolean;
  }
  const posterior = rowCgesPosterior(answers[0]!, scores![0]!);
  let probability = 0;
  for (const [answer, value] of posterior) {
    if (!includeOther && answer === CGES_OTHER) continue;
    if (value > probability) probability = value;
  }
  const stop = part.length >= minSamples && probability >= boundary;
  return (pythonTruthy(options.returnProb) ? [stop, probability] : stop) as ReturnProb extends true
    ? [boolean, number]
    : boolean;
}
