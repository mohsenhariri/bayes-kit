/** Vote-based answer aggregation for fixed candidate pools. */

import {
  type AnswerInput,
  type Keep,
  isValidAnswer,
  normalizePool,
  packSelection,
  type PackedSelection,
  resolveKeepCount,
  scoreOrder,
  type ScoreInput,
  type SelectionReturnOptions,
  validIndices,
} from "./internal/base.js";
import {
  pythonComparableNumber,
  pythonTruthy,
} from "./internal/runtime.js";

export interface MajorityVoteOptions<ReturnIndex extends boolean = false> {
  returnIndex?: ReturnIndex;
}

function rowMajority<T>(answers: readonly T[]): [T | null, number] {
  const groups = new Map<T, { count: number; first: number }>();
  for (let index = 0; index < answers.length; index++) {
    const answer = answers[index]!;
    if (!isValidAnswer(answer)) continue;
    const group = groups.get(answer);
    if (group === undefined) groups.set(answer, { count: 1, first: index });
    else group.count += 1;
  }
  if (groups.size === 0) return [null, -1];

  let winnerSet = false;
  let winner!: T;
  let winnerCount = -1;
  let winnerFirst = Infinity;
  for (const [answer, group] of groups) {
    if (
      !winnerSet ||
      group.count > winnerCount ||
      (group.count === winnerCount && group.first < winnerFirst)
    ) {
      winnerSet = true;
      winner = answer;
      winnerCount = group.count;
      winnerFirst = group.first;
    }
  }
  return [winner, winnerFirst];
}

/** Plain plurality/majority vote, with ties broken by earliest appearance. */
export function majorityVote<T, ReturnIndex extends boolean = false>(
  answersInput: AnswerInput<T>,
  options: MajorityVoteOptions<ReturnIndex> = {},
): PackedSelection<T, ReturnIndex, false> {
  const { answers, single } = normalizePool(answersInput);
  const selected: (T | null)[] = [];
  const indices: number[] = [];
  for (const row of answers) {
    const [answer, index] = rowMajority(row);
    selected.push(answer);
    indices.push(index);
  }
  return packSelection(
    selected,
    indices,
    new Array<number>(indices.length).fill(NaN),
    single,
    (options.returnIndex ?? false) as ReturnIndex,
    false,
  );
}

export type WeightedVoteAggregate = "sum" | "mean";

export interface WeightedMajorityVoteOptions<
  ReturnIndex extends boolean = false,
  ReturnScore extends boolean = false,
> extends SelectionReturnOptions<ReturnIndex, ReturnScore> {
  aggregate?: WeightedVoteAggregate;
}

interface WeightedGroup {
  sum: number;
  count: number;
  first: number;
  bestScore: number;
  bestIndex: number;
}

function rowWeighted<T>(
  answers: readonly T[],
  scores: readonly number[],
  aggregate: WeightedVoteAggregate,
): [T | null, number] {
  const groups = new Map<T, WeightedGroup>();
  for (let index = 0; index < answers.length; index++) {
    const answer = answers[index]!;
    if (!isValidAnswer(answer)) continue;
    const score = scores[index]!;
    const group = groups.get(answer);
    if (group === undefined) {
      groups.set(answer, {
        sum: score,
        count: 1,
        first: index,
        bestScore: score,
        bestIndex: index,
      });
    } else {
      group.sum += score;
      group.count += 1;
      if (score > group.bestScore) {
        group.bestScore = score;
        group.bestIndex = index;
      }
    }
  }
  if (groups.size === 0) return [null, -1];

  let winnerSet = false;
  let winner!: T;
  let winnerGroup!: WeightedGroup;
  let winnerWeight = -Infinity;
  for (const [answer, group] of groups) {
    const weight = aggregate === "mean" ? group.sum / group.count : group.sum;
    if (
      !winnerSet ||
      weight > winnerWeight ||
      (weight === winnerWeight && group.first < winnerGroup.first)
    ) {
      winnerSet = true;
      winner = answer;
      winnerGroup = group;
      winnerWeight = weight;
    }
  }
  return [winner, winnerGroup.bestIndex];
}

function numericPlurality<T>(
  answers: readonly T[],
  part: readonly number[],
  weightOf: (index: number) => number,
  scoreOf: (index: number) => number,
): [T | null, number] {
  if (part.length === 0) return [null, -1];
  const total = new Map<T, number>();
  const first = new Map<T, number>();
  const representative = new Map<T, number>();
  for (const index of part) {
    const answer = answers[index]!;
    if (!total.has(answer)) {
      total.set(answer, 0);
      first.set(answer, index);
      representative.set(answer, index);
    }
    total.set(answer, total.get(answer)! + weightOf(index));
    if (scoreOf(index) > scoreOf(representative.get(answer)!)) {
      representative.set(answer, index);
    }
  }

  let winnerSet = false;
  let winner!: T;
  let winnerTotal = -Infinity;
  let winnerFirst = Infinity;
  for (const [answer, weight] of total) {
    const firstIndex = first.get(answer)!;
    if (
      !winnerSet ||
      weight > winnerTotal ||
      (weight === winnerTotal && firstIndex < winnerFirst)
    ) {
      winnerSet = true;
      winner = answer;
      winnerTotal = weight;
      winnerFirst = firstIndex;
    }
  }
  return [winner, representative.get(winner)!];
}

function integerPlurality<T>(
  answers: readonly T[],
  part: readonly number[],
  weights: ReadonlyMap<number, bigint>,
  scores: readonly number[],
): [T | null, number] {
  if (part.length === 0) return [null, -1];
  const total = new Map<T, bigint>();
  const first = new Map<T, number>();
  const representative = new Map<T, number>();
  for (const index of part) {
    const answer = answers[index]!;
    if (!total.has(answer)) {
      total.set(answer, 0n);
      first.set(answer, index);
      representative.set(answer, index);
    }
    total.set(answer, total.get(answer)! + weights.get(index)!);
    if (scores[index]! > scores[representative.get(answer)!]!) {
      representative.set(answer, index);
    }
  }

  let winnerSet = false;
  let winner!: T;
  let winnerTotal = 0n;
  let winnerFirst = Infinity;
  for (const [answer, weight] of total) {
    const firstIndex = first.get(answer)!;
    if (
      !winnerSet ||
      weight > winnerTotal ||
      (weight === winnerTotal && firstIndex < winnerFirst)
    ) {
      winnerSet = true;
      winner = answer;
      winnerTotal = weight;
      winnerFirst = firstIndex;
    }
  }
  return [winner, representative.get(winner)!];
}

function runScoreVote<
  T,
  ReturnIndex extends boolean,
  ReturnScore extends boolean,
>(
  answers: readonly (readonly T[])[],
  scores: readonly (readonly number[])[],
  single: boolean,
  selector: (
    row: readonly T[],
    scoreRow: readonly number[],
  ) => [T | null, number],
  returnIndex: ReturnIndex,
  returnScore: ReturnScore,
): PackedSelection<T, ReturnIndex, ReturnScore> {
  const selected: (T | null)[] = [];
  const indices: number[] = [];
  const selectedScores: number[] = [];
  for (let i = 0; i < answers.length; i++) {
    const [answer, index] = selector(answers[i]!, scores[i]!);
    selected.push(answer);
    indices.push(index);
    selectedScores.push(index >= 0 ? scores[i]![index]! : NaN);
  }
  return packSelection(
    selected,
    indices,
    selectedScores,
    single,
    returnIndex,
    returnScore,
  );
}

/** Select the answer group with maximum summed or mean raw score. */
export function weightedMajorityVote<
  T,
  ReturnIndex extends boolean = false,
  ReturnScore extends boolean = false,
>(
  answersInput: AnswerInput<T>,
  scoresInput: ScoreInput,
  options: WeightedMajorityVoteOptions<ReturnIndex, ReturnScore> = {},
): PackedSelection<T, ReturnIndex, ReturnScore> {
  const aggregate = options.aggregate === undefined ? "sum" : options.aggregate;
  if (aggregate !== "sum" && aggregate !== "mean") {
    throw new Error(
      `aggregate must be 'sum' or 'mean'; got ${String(aggregate)}.`,
    );
  }
  const { answers, scores, single } = normalizePool(
    answersInput,
    scoresInput,
    true,
  );
  return runScoreVote(
    answers,
    scores!,
    single,
    (row, scoreRow) => rowWeighted(row, scoreRow, aggregate),
    (options.returnIndex ?? false) as ReturnIndex,
    (options.returnScore ?? false) as ReturnScore,
  );
}

export interface SoftmaxWeightedVoteOptions<
  ReturnIndex extends boolean = false,
  ReturnScore extends boolean = false,
> extends SelectionReturnOptions<ReturnIndex, ReturnScore> {
  temperature?: number;
}

/** Temperature-softmax weighted vote (CISC). */
export function softmaxWeightedVote<
  T,
  ReturnIndex extends boolean = false,
  ReturnScore extends boolean = false,
>(
  answersInput: AnswerInput<T>,
  scoresInput: ScoreInput,
  options: SoftmaxWeightedVoteOptions<ReturnIndex, ReturnScore> = {},
): PackedSelection<T, ReturnIndex, ReturnScore> {
  const requested =
    options.temperature === undefined ? 1 : options.temperature;
  const temperature = pythonComparableNumber(requested, "temperature");
  if (!(temperature > 0)) {
    throw new Error(`temperature must be > 0; got ${requested}.`);
  }
  const { answers, scores, single } = normalizePool(
    answersInput,
    scoresInput,
    true,
  );
  return runScoreVote(
    answers,
    scores!,
    single,
    (row, scoreRow) => {
      const part = validIndices(row);
      if (part.length === 0) return [null, -1];
      let scoreMax = scoreRow[part[0]!]!;
      for (let i = 1; i < part.length; i++) {
        const score = scoreRow[part[i]!]!;
        if (score > scoreMax) scoreMax = score;
      }
      return numericPlurality(
        row,
        part,
        (index) => Math.exp((scoreRow[index]! - scoreMax) / temperature),
        (index) => scoreRow[index]!,
      );
    },
    (options.returnIndex ?? false) as ReturnIndex,
    (options.returnScore ?? false) as ReturnScore,
  );
}

export interface RankWeightedVoteOptions<
  ReturnIndex extends boolean = false,
  ReturnScore extends boolean = false,
> extends SelectionReturnOptions<ReturnIndex, ReturnScore> {
  p?: number;
}

/** Borda/rank-weighted vote using exact integer weights when `p` is integral. */
export function rankWeightedVote<
  T,
  ReturnIndex extends boolean = false,
  ReturnScore extends boolean = false,
>(
  answersInput: AnswerInput<T>,
  scoresInput: ScoreInput,
  options: RankWeightedVoteOptions<ReturnIndex, ReturnScore> = {},
): PackedSelection<T, ReturnIndex, ReturnScore> {
  const requested = options.p === undefined ? 1 : options.p;
  const p = pythonComparableNumber(requested, "p");
  if (!Number.isFinite(p) || p < 0) {
    throw new Error(`p must be a finite non-negative number; got ${requested}.`);
  }
  const { answers, scores, single } = normalizePool(
    answersInput,
    scoresInput,
    true,
  );
  return runScoreVote(
    answers,
    scores!,
    single,
    (row, scoreRow) => {
      const part = validIndices(row);
      if (part.length === 0) return [null, -1];
      const order = scoreOrder(part, scoreRow);
      const n = part.length;
      if (Number.isInteger(p)) {
        const weights = new Map<number, bigint>();
        for (let rank = 0; rank < order.length; rank++) {
          weights.set(order[rank]!, BigInt(n - rank) ** BigInt(p));
        }
        return integerPlurality(row, part, weights, scoreRow);
      }
      const weights = new Map<number, number>();
      for (let rank = 0; rank < order.length; rank++) {
        weights.set(order[rank]!, ((n - rank) / n) ** p);
      }
      return numericPlurality(
        row,
        part,
        (index) => weights.get(index)!,
        (index) => scoreRow[index]!,
      );
    },
    (options.returnIndex ?? false) as ReturnIndex,
    (options.returnScore ?? false) as ReturnScore,
  );
}

export type LogitTransform = "logit" | "linear";

export interface LogitWeightedVoteOptions<
  ReturnIndex extends boolean = false,
  ReturnScore extends boolean = false,
> extends SelectionReturnOptions<ReturnIndex, ReturnScore> {
  threshold?: number;
  transform?: LogitTransform;
}

/** Threshold-shifted log-odds (or linear) weighted vote. */
export function logitWeightedVote<
  T,
  ReturnIndex extends boolean = false,
  ReturnScore extends boolean = false,
>(
  answersInput: AnswerInput<T>,
  scoresInput: ScoreInput,
  options: LogitWeightedVoteOptions<ReturnIndex, ReturnScore> = {},
): PackedSelection<T, ReturnIndex, ReturnScore> {
  const { threshold = 0.5, transform = "logit" } = options;
  if (transform !== "logit" && transform !== "linear") {
    throw new Error(
      `transform must be 'logit' or 'linear'; got ${String(transform)}.`,
    );
  }
  const { answers, scores, single } = normalizePool(
    answersInput,
    scoresInput,
    true,
  );
  let logitThreshold: number | undefined;
  if (transform === "logit") {
    logitThreshold = pythonComparableNumber(threshold, "threshold");
    if (!(logitThreshold > 0 && logitThreshold < 1)) {
      throw new Error(
        `threshold must be in (0, 1) for transform='logit'; got ${threshold}.`,
      );
    }
    for (let rowIndex = 0; rowIndex < answers.length; rowIndex++) {
      for (const index of validIndices(answers[rowIndex]!)) {
        const score = scores![rowIndex]![index]!;
        if (!(score > 0 && score < 1)) {
          throw new Error(
            "transform='logit' requires every valid score in (0, 1); " +
              `got ${score}. Use transform='linear' for unbounded scores.`,
          );
        }
      }
    }
  }

  return runScoreVote(
    answers,
    scores!,
    single,
    (row, scoreRow) => {
      const part = validIndices(row);
      if (part.length === 0) return [null, -1];
      const boundary =
        transform === "logit"
          ? logitThreshold!
          : pythonComparableNumber(threshold, "threshold");
      const thresholdLogit =
        transform === "logit" ? Math.log(boundary / (1 - boundary)) : 0;
      return numericPlurality(
        row,
        part,
        (index) => {
          const score = scoreRow[index]!;
          return transform === "logit"
            ? Math.log(score / (1 - score)) - thresholdLogit
            : score - boundary;
        },
        (index) => scoreRow[index]!,
      );
    },
    (options.returnIndex ?? false) as ReturnIndex,
    (options.returnScore ?? false) as ReturnScore,
  );
}

export interface FilteredVoteOptions<
  ReturnIndex extends boolean = false,
  ReturnScore extends boolean = false,
> extends SelectionReturnOptions<ReturnIndex, ReturnScore> {
  keep?: Keep;
  weighted?: boolean;
}

/** Vote after retaining only the top-scoring fraction/count of valid answers. */
export function filteredVote<
  T,
  ReturnIndex extends boolean = false,
  ReturnScore extends boolean = false,
>(
  answersInput: AnswerInput<T>,
  scoresInput: ScoreInput,
  options: FilteredVoteOptions<ReturnIndex, ReturnScore> = {},
): PackedSelection<T, ReturnIndex, ReturnScore> {
  const { keep = 0.5, weighted = true } = options;
  const { answers, scores, single } = normalizePool(
    answersInput,
    scoresInput,
    true,
  );
  resolveKeepCount(keep, answers[0]!.length);
  return runScoreVote(
    answers,
    scores!,
    single,
    (row, scoreRow) => {
      const valid = validIndices(row);
      if (valid.length === 0) return [null, -1];
      const count = resolveKeepCount(keep, valid.length);
      const kept = scoreOrder(valid, scoreRow).slice(0, count);
      return numericPlurality(
        row,
        kept,
        (index) => (pythonTruthy(weighted) ? scoreRow[index]! : 1),
        (index) => scoreRow[index]!,
      );
    },
    (options.returnIndex ?? false) as ReturnIndex,
    (options.returnScore ?? false) as ReturnScore,
  );
}
