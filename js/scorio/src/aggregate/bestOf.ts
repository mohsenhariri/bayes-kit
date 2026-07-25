/** Reward-based answer selection: Best-of-N, MoB, and Best-of-Majority. */

import {
  type AnswerInput,
  defaultResampleSize,
  isValidAnswer,
  normalizePool,
  packSelection,
  type PackedSelection,
  scoreOrder,
  type ScoreInput,
  type SelectionReturnOptions,
  validIndices,
} from "./internal/base.js";
import { pythonComparableNumber } from "./internal/runtime.js";

export interface BestOfNOptions<
  ReturnIndex extends boolean = false,
  ReturnScore extends boolean = false,
> extends SelectionReturnOptions<ReturnIndex, ReturnScore> {}

function rowBestOfN<T>(
  answers: readonly T[],
  scores: readonly number[],
): [T | null, number] {
  let bestIndex = -1;
  let bestScore = -Infinity;
  for (let i = 0; i < answers.length; i++) {
    if (!isValidAnswer(answers[i]!)) continue;
    const score = scores[i]!;
    if (score > bestScore) {
      bestScore = score;
      bestIndex = i;
    }
  }
  return bestIndex < 0 ? [null, -1] : [answers[bestIndex]!, bestIndex];
}

function runRows<T, ReturnIndex extends boolean, ReturnScore extends boolean>(
  answers: readonly (readonly T[])[],
  scores: readonly (readonly number[])[],
  single: boolean,
  rowSelector: (
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
    const [answer, index] = rowSelector(answers[i]!, scores[i]!);
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

/** Select the valid candidate with the highest score (ties: lowest index). */
export function bestOfN<
  T,
  ReturnIndex extends boolean = false,
  ReturnScore extends boolean = false,
>(
  answersInput: AnswerInput<T>,
  scoresInput: ScoreInput,
  options: BestOfNOptions<ReturnIndex, ReturnScore> = {},
): PackedSelection<T, ReturnIndex, ReturnScore> {
  const { answers, scores, single } = normalizePool(
    answersInput,
    scoresInput,
    true,
  );
  return runRows(
    answers,
    scores!,
    single,
    rowBestOfN,
    (options.returnIndex ?? false) as ReturnIndex,
    (options.returnScore ?? false) as ReturnScore,
  );
}

export interface MajorityOfTheBestsOptions<
  ReturnIndex extends boolean = false,
  ReturnScore extends boolean = false,
> extends SelectionReturnOptions<ReturnIndex, ReturnScore> {
  m?: number | null;
}

function rowMajorityOfTheBests<T>(
  answers: readonly T[],
  scores: readonly number[],
  m: number | null | undefined,
): [T | null, number] {
  const valid = validIndices(answers);
  if (valid.length === 0) return [null, -1];
  if (valid.length === 1) return [answers[valid[0]!]!, valid[0]!];

  const n = valid.length;
  const requested = m === null || m === undefined ? defaultResampleSize(n) : Math.trunc(m);
  const size = Math.min(Math.max(requested, 1), n);
  const order = scoreOrder(valid, scores);

  // Python uses arbitrary-precision integer mass `(n-t)^m-(n-t-1)^m` so
  // genuine group ties remain exact. BigInt provides the same property in JS.
  const weights = new Map<T, bigint>();
  const representative = new Map<T, number>();
  for (let rank = 0; rank < order.length; rank++) {
    const index = order[rank]!;
    const answer = answers[index]!;
    const upper = BigInt(n - rank) ** BigInt(size);
    const lower = BigInt(n - rank - 1) ** BigInt(size);
    weights.set(answer, (weights.get(answer) ?? 0n) + upper - lower);
    if (!representative.has(answer)) representative.set(answer, index);
  }

  const first = new Map<T, number>();
  for (const index of valid) {
    const answer = answers[index]!;
    if (!first.has(answer)) first.set(answer, index);
  }

  let winner: T | undefined;
  let winnerWeight = 0n;
  let winnerFirst = Infinity;
  for (const [answer, weight] of weights) {
    const firstIndex = first.get(answer)!;
    if (
      winner === undefined ||
      weight > winnerWeight ||
      (weight === winnerWeight && firstIndex < winnerFirst)
    ) {
      winner = answer;
      winnerWeight = weight;
      winnerFirst = firstIndex;
    }
  }
  return [winner!, representative.get(winner!)!];
}

/** Exact closed-form mode of the bootstrapped Best-of-N distribution. */
export function majorityOfTheBests<
  T,
  ReturnIndex extends boolean = false,
  ReturnScore extends boolean = false,
>(
  answersInput: AnswerInput<T>,
  scoresInput: ScoreInput,
  options: MajorityOfTheBestsOptions<ReturnIndex, ReturnScore> = {},
): PackedSelection<T, ReturnIndex, ReturnScore> {
  const { m = null } = options;
  if (m !== null && m !== undefined) {
    if (!Number.isInteger(m) || m < 1) {
      throw new Error(`m must be a positive integer or null; got ${m}.`);
    }
  }
  const { answers, scores, single } = normalizePool(
    answersInput,
    scoresInput,
    true,
  );
  return runRows(
    answers,
    scores!,
    single,
    (row, scoreRow) => rowMajorityOfTheBests(row, scoreRow, m),
    (options.returnIndex ?? false) as ReturnIndex,
    (options.returnScore ?? false) as ReturnScore,
  );
}

/** Short alias for {@link majorityOfTheBests}. */
export const mob = majorityOfTheBests;

export type BestOfMajorityAggregate = "mean" | "sum" | "max";

export interface BestOfMajorityOptions<
  ReturnIndex extends boolean = false,
  ReturnScore extends boolean = false,
> extends SelectionReturnOptions<ReturnIndex, ReturnScore> {
  alpha?: number;
  aggregate?: BestOfMajorityAggregate;
}

interface AnswerGroup {
  indices: number[];
  first: number;
  representative: number;
}

function rowBestOfMajority<T>(
  answers: readonly T[],
  scores: readonly number[],
  alpha: number,
  aggregate: BestOfMajorityAggregate,
): [T | null, number] {
  const valid = validIndices(answers);
  if (valid.length === 0) return [null, -1];

  const groups = new Map<T, AnswerGroup>();
  for (const index of valid) {
    const answer = answers[index]!;
    const group = groups.get(answer);
    if (group === undefined) {
      groups.set(answer, { indices: [index], first: index, representative: index });
    } else {
      group.indices.push(index);
      if (scores[index]! > scores[group.representative]!) {
        group.representative = index;
      }
    }
  }

  let eligible = [...groups.entries()].filter(
    ([, group]) => group.indices.length / valid.length >= alpha,
  );
  if (eligible.length === 0) eligible = [...groups.entries()];

  const reward = (group: AnswerGroup): number => {
    let sum = 0;
    let max = -Infinity;
    for (const index of group.indices) {
      const score = scores[index]!;
      sum += score;
      if (score > max) max = score;
    }
    if (aggregate === "sum") return sum;
    if (aggregate === "mean") return sum / group.indices.length;
    return max;
  };

  let [winner, winnerGroup] = eligible[0]!;
  let winnerReward = reward(winnerGroup);
  for (let i = 1; i < eligible.length; i++) {
    const [answer, group] = eligible[i]!;
    const candidateReward = reward(group);
    if (
      candidateReward > winnerReward ||
      (candidateReward === winnerReward && group.first < winnerGroup.first)
    ) {
      winner = answer;
      winnerGroup = group;
      winnerReward = candidateReward;
    }
  }
  return [winner, winnerGroup.representative];
}

/** Highest aggregated-reward answer among groups passing a frequency gate. */
export function bestOfMajority<
  T,
  ReturnIndex extends boolean = false,
  ReturnScore extends boolean = false,
>(
  answersInput: AnswerInput<T>,
  scoresInput: ScoreInput,
  options: BestOfMajorityOptions<ReturnIndex, ReturnScore> = {},
): PackedSelection<T, ReturnIndex, ReturnScore> {
  const { alpha = 0, aggregate = "mean" } = options;
  const frequency = pythonComparableNumber(alpha, "alpha");
  if (!(frequency >= 0 && frequency <= 1)) {
    throw new Error(`alpha must be in [0, 1]; got ${alpha}.`);
  }
  if (aggregate !== "mean" && aggregate !== "sum" && aggregate !== "max") {
    throw new Error(
      `aggregate must be 'mean', 'sum', or 'max'; got ${String(aggregate)}.`,
    );
  }
  const { answers, scores, single } = normalizePool(
    answersInput,
    scoresInput,
    true,
  );
  return runRows(
    answers,
    scores!,
    single,
    (row, scoreRow) => rowBestOfMajority(row, scoreRow, frequency, aggregate),
    (options.returnIndex ?? false) as ReturnIndex,
    (options.returnScore ?? false) as ReturnScore,
  );
}
