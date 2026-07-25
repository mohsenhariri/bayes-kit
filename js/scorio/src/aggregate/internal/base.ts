/** Shared input normalization and result packing for answer selection rules. */

import { pythonTruthy } from "./runtime.js";

/** One question's answers, or a batch in `(M, N)` layout. */
export type AnswerInput<T> =
  | readonly T[]
  | readonly (readonly T[])[];

/** One question's numeric scores, or a batch in `(M, N)` layout. */
export type ScoreInput =
  | readonly number[]
  | readonly (readonly number[])[];

/** A scalar for one question and an array for a batch. */
export type ScalarOrBatch<T> = T | T[];

/** Selected labels use `null` when a row contains no valid answer. */
export type Selection<T> = ScalarOrBatch<T | null>;
export type SelectionIndex = ScalarOrBatch<number>;
export type SelectionScore = ScalarOrBatch<number>;

/** Return type controlled by `returnIndex` and `returnScore`. */
export type PackedSelection<
  T,
  ReturnIndex extends boolean,
  ReturnScore extends boolean,
> = ReturnIndex extends true
  ? ReturnScore extends true
    ? [Selection<T>, SelectionIndex, SelectionScore]
    : [Selection<T>, SelectionIndex]
  : ReturnScore extends true
    ? [Selection<T>, SelectionScore]
    : Selection<T>;

export interface SelectionReturnOptions<
  ReturnIndex extends boolean = false,
  ReturnScore extends boolean = false,
> {
  returnIndex?: ReturnIndex;
  returnScore?: ReturnScore;
}

/**
 * Candidate-retention setting for {@link resolveKeepCount}.
 *
 * Numeric values reproduce Python's runtime convention: integer values are
 * counts and non-integers in `(0, 1]` are fractions. JavaScript cannot
 * distinguish the values `1` and `1.0`, so `{ fraction: 1 }` explicitly
 * represents Python's `1.0` (keep all), while numeric `1` is a count (keep one).
 */
export type Keep =
  | number
  | Readonly<{ count: number }>
  | Readonly<{ fraction: number }>;

export interface NormalizedPool<T> {
  answers: T[][];
  scores: number[][] | null;
  single: boolean;
}

/** Whether an answer label is usable by an aggregation rule. */
export function isValidAnswer<T>(answer: T): boolean {
  if (answer === null || answer === undefined) return false;
  if (typeof answer === "string") return answer !== "";
  return !(typeof answer === "number" && Number.isNaN(answer));
}

function asRows<T>(input: AnswerInput<T>, name: string): {
  rows: T[][];
  single: boolean;
} {
  if (!Array.isArray(input)) {
    throw new Error(`${name} must be a 1D (N,) or 2D (M, N) array.`);
  }

  const values = input as readonly unknown[];
  if (values.length === 0) return { rows: [[]], single: true };

  const nested = Array.isArray(values[0]);
  if (nested) {
    if (!values.every(Array.isArray)) {
      throw new Error(`${name} must be a rectangular 2D array.`);
    }
    const rows = (values as readonly (readonly T[])[]).map((row) => [...row]);
    if (rows.some((row) => row.some(Array.isArray))) {
      throw new Error(`${name} must be a 1D (N,) or 2D (M, N) array.`);
    }
    const width = rows[0]!.length;
    if (rows.some((row) => row.length !== width)) {
      throw new Error(`${name} must be a rectangular 2D array.`);
    }
    return { rows, single: false };
  }

  if (values.some(Array.isArray)) {
    throw new Error(`${name} must be a 1D (N,) or 2D (M, N) array.`);
  }
  return { rows: [[...(input as readonly T[])]], single: true };
}

function asScoreRows(input: ScoreInput): number[][] {
  const { rows } = asRows(input, "scores");
  return rows.map((row) =>
    row.map((value) => {
      if (typeof value !== "number") {
        throw new Error(`scores must contain numbers; got ${String(value)}.`);
      }
      return value;
    }),
  );
}

/** Coerce answer/score inputs to a common rectangular `(M, N)` layout. */
export function normalizePool<T>(
  answersInput: AnswerInput<T>,
  scoresInput?: ScoreInput | null,
  requireScores = false,
): NormalizedPool<T> {
  const { rows: answers, single } = asRows(answersInput, "answers");
  const n = answers[0]?.length ?? 0;
  if (n === 0) {
    throw new Error("need at least one candidate per question (N >= 1).");
  }

  if (scoresInput === null || scoresInput === undefined) {
    if (requireScores) {
      throw new Error("scores are required for this selection rule.");
    }
    return { answers, scores: null, single };
  }

  const scores = asScoreRows(scoresInput);
  const sameShape =
    scores.length === answers.length &&
    scores.every((row, i) => row.length === answers[i]!.length);
  if (!sameShape) {
    const aShape = `(${answers.length}, ${n})`;
    const sWidth = scores[0]?.length ?? 0;
    const sShape = `(${scores.length}, ${sWidth})`;
    throw new Error(
      `answers and scores must have the same shape; got ${aShape} and ${sShape}.`,
    );
  }
  return { answers, scores, single };
}

export function validIndices<T>(row: readonly T[]): number[] {
  const out: number[] = [];
  for (let i = 0; i < row.length; i++) {
    if (isValidAnswer(row[i]!)) out.push(i);
  }
  return out;
}

/** Stable descending score order, with lower candidate index breaking ties. */
export function scoreOrder(
  indices: readonly number[],
  scores: readonly number[],
): number[] {
  return [...indices].sort((a, b) => {
    const delta = scores[b]! - scores[a]!;
    return Number.isNaN(delta) || delta === 0 ? a - b : delta;
  });
}

/** Default MoB resample size `floor(sqrt(n))`, with a minimum of one. */
export function defaultResampleSize(n: number): number {
  return Math.max(1, Math.floor(Math.sqrt(n)));
}

/** Resolve a fraction/count retention setting to an integer in `[1, n]`. */
export function resolveKeepCount(keep: Keep, n: number): number {
  if (typeof keep === "boolean") {
    throw new Error(
      "keep must be a float in (0, 1] or an integer count >= 1; got a boolean.",
    );
  }

  let mode: "count" | "fraction";
  let value: number;
  if (typeof keep === "number") {
    value = keep;
    mode = Number.isInteger(value) && value >= 1 ? "count" : "fraction";
  } else if (keep !== null && "count" in keep) {
    mode = "count";
    value = keep.count;
  } else if (keep !== null && "fraction" in keep) {
    mode = "fraction";
    value = keep.fraction;
  } else {
    throw new Error("keep must be a fraction in (0, 1] or an integer count >= 1.");
  }

  if (mode === "count") {
    if (!Number.isInteger(value) || value < 1) {
      throw new Error(`keep count must be an integer >= 1; got ${value}.`);
    }
    return Math.min(value, n);
  }
  if (!(value > 0 && value <= 1) || !Number.isFinite(value)) {
    throw new Error(`keep fraction must be in (0, 1]; got ${value}.`);
  }
  return Math.max(1, Math.ceil(value * n - 1e-9));
}

/** Assemble selection, representative index, and representative score. */
export function packSelection<
  T,
  ReturnIndex extends boolean,
  ReturnScore extends boolean,
>(
  selected: (T | null)[],
  indices: number[],
  scores: number[],
  single: boolean,
  returnIndex: ReturnIndex,
  returnScore: ReturnScore,
): PackedSelection<T, ReturnIndex, ReturnScore> {
  const selection = (single ? selected[0]! : selected) as Selection<T>;
  const index = (single ? indices[0]! : indices) as SelectionIndex;
  const score = (single ? scores[0]! : scores) as SelectionScore;
  const includeIndex = pythonTruthy(returnIndex);
  const includeScore = pythonTruthy(returnScore);
  if (includeIndex && includeScore) {
    return [selection, index, score] as PackedSelection<
      T,
      ReturnIndex,
      ReturnScore
    >;
  }
  if (includeIndex) {
    return [selection, index] as PackedSelection<T, ReturnIndex, ReturnScore>;
  }
  if (includeScore) {
    return [selection, score] as PackedSelection<T, ReturnIndex, ReturnScore>;
  }
  return selection as PackedSelection<T, ReturnIndex, ReturnScore>;
}
