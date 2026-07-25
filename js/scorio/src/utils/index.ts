/**
 * Ranking utilities — TypeScript port of `scorio.utils`.
 *
 * Public functions are exported in camelCase and under snake_case aliases that
 * match Python. Combinatorial hashes use `bigint` once a result exceeds
 * JavaScript's safe-integer range; smaller results remain ordinary numbers.
 */

import { normPpf } from "../rank/internal/special.js";
import { rankdata } from "../rank/internal/rankScores.js";
import {
  kendallTau,
  spearmanR,
  weightedKendallTau,
  type Correlation,
} from "./internal/stats.js";

export type { Correlation } from "./internal/stats.js";
export type Integer = number | bigint;
export type CiTieMethod = "zscore_adjacent" | "ci_overlap_adjacent";
export type CompareRankingsMethod = "kendall" | "spearman" | "weighted_kendall" | "all";
type SingleComparisonMethod = Exclude<CompareRankingsMethod, "all">;

export interface RankScoresOptions {
  tol?: number;
  sigmas?: ArrayLike<number>;
  confidence?: number;
  ciTieMethod?: CiTieMethod;
}

export interface RankScoresResult {
  competition: number[];
  competition_max: number[];
  dense: number[];
  avg: number[];
  competition_ci?: number[];
  competition_max_ci?: number[];
  dense_ci?: number[];
  avg_ci?: number[];
}

export interface RankingComparison {
  kendalltau: Correlation;
  spearmanr: Correlation;
  weighted_kendalltau: Correlation;
  fraction_mismatched: number;
  max_disp: number;
}

function asNumericVector(values: ArrayLike<number>, name: string): number[] {
  if (values === null || values === undefined || typeof values.length !== "number") {
    throw new TypeError(`${name} must be a 1D numeric sequence.`);
  }
  const out = Array.from(values);
  if (out.some((value) => typeof value !== "number")) {
    throw new TypeError(`${name} must be a 1D numeric sequence.`);
  }
  return out;
}

function descendingOrder(values: readonly number[]): number[] {
  const order = Array.from({ length: values.length }, (_, i) => i);
  order.sort((a, b) => {
    const va = values[a]!;
    const vb = values[b]!;
    if (Number.isNaN(va)) return Number.isNaN(vb) ? a - b : 1;
    if (Number.isNaN(vb)) return -1;
    return va > vb ? -1 : va < vb ? 1 : a - b;
  });
  return order;
}

function ranksFromSortedGroups(
  order: readonly number[],
  grouped: readonly number[],
): RankScoresResult {
  // scipy.stats.rankdata uses nan_policy="propagate" by default.
  if (grouped.some(Number.isNaN)) {
    const nanRanks = (): number[] => new Array<number>(order.length).fill(NaN);
    return {
      competition: nanRanks(),
      competition_max: nanRanks(),
      dense: nanRanks(),
      avg: nanRanks(),
    };
  }
  const negated = grouped.map((value) => -value);
  const scatter = (method: "min" | "max" | "dense" | "average"): number[] => {
    const sortedRanks = rankdata(negated, method);
    const ranks = new Array<number>(order.length);
    for (let i = 0; i < order.length; i++) ranks[order[i]!] = sortedRanks[i]!;
    return ranks;
  };
  return {
    competition: scatter("min"),
    competition_max: scatter("max"),
    dense: scatter("dense"),
    avg: scatter("average"),
  };
}

function normalizeRankScoreArguments(
  optionsOrTol: RankScoresOptions | number | undefined,
  positionalSigmas: ArrayLike<number> | undefined,
  positionalConfidence: number | undefined,
  positionalTieMethod: CiTieMethod | undefined,
): Required<Pick<RankScoresOptions, "tol" | "confidence" | "ciTieMethod">> &
  Pick<RankScoresOptions, "sigmas"> {
  if (typeof optionsOrTol === "object" && optionsOrTol !== null) {
    return {
      tol: optionsOrTol.tol ?? 1e-12,
      sigmas: optionsOrTol.sigmas,
      confidence: optionsOrTol.confidence ?? 0.95,
      ciTieMethod: optionsOrTol.ciTieMethod ?? "zscore_adjacent",
    };
  }
  return {
    tol: optionsOrTol ?? 1e-12,
    sigmas: positionalSigmas,
    confidence: positionalConfidence ?? 0.95,
    ciTieMethod: positionalTieMethod ?? "zscore_adjacent",
  };
}

/**
 * Convert higher-is-better scores to all four Python rank conventions.
 *
 * The options-object form is idiomatic TypeScript. A positional form is also
 * accepted so the snake_case alias can mirror Python's argument order.
 */
export function rankScores(
  scoresInput: ArrayLike<number>,
  options?: RankScoresOptions,
): RankScoresResult;
export function rankScores(
  scoresInput: ArrayLike<number>,
  tol?: number,
  sigmas?: ArrayLike<number>,
  confidence?: number,
  ciTieMethod?: CiTieMethod,
): RankScoresResult;
export function rankScores(
  scoresInput: ArrayLike<number>,
  optionsOrTol?: RankScoresOptions | number,
  positionalSigmas?: ArrayLike<number>,
  positionalConfidence?: number,
  positionalTieMethod?: CiTieMethod,
): RankScoresResult {
  const scores = asNumericVector(scoresInput, "scores_in_id_order");
  const { tol, sigmas: sigmasInput, confidence, ciTieMethod } =
    normalizeRankScoreArguments(
      optionsOrTol,
      positionalSigmas,
      positionalConfidence,
      positionalTieMethod,
    );
  const order = descendingOrder(scores);
  const grouped = order.map((index) => scores[index]!);
  for (let i = 1; i < grouped.length; i++) {
    if (Math.abs(grouped[i]! - grouped[i - 1]!) <= tol) grouped[i] = grouped[i - 1]!;
  }
  const result = ranksFromSortedGroups(order, grouped);
  if (sigmasInput === undefined) return result;

  const sigmas = asNumericVector(sigmasInput, "sigmas_in_id_order");
  if (sigmas.length !== scores.length) {
    throw new Error("sigmas_in_id_order must have the same length as scores.");
  }
  const sortedMus = order.map((index) => scores[index]!);
  const sortedSigmas = order.map((index) => sigmas[index]!);
  const ciGrouped = [...grouped];

  if (ciTieMethod === "zscore_adjacent") {
    const threshold = normPpf(confidence);
    for (let i = 1; i < ciGrouped.length; i++) {
      if (Math.abs(ciGrouped[i]! - ciGrouped[i - 1]!) <= tol) {
        ciGrouped[i] = ciGrouped[i - 1]!;
        continue;
      }
      const denominator = Math.sqrt(sortedSigmas[i - 1]! ** 2 + sortedSigmas[i]! ** 2);
      if (denominator === 0) continue;
      const z = Math.abs(sortedMus[i - 1]! - sortedMus[i]!) / denominator;
      if (z < threshold) ciGrouped[i] = ciGrouped[i - 1]!;
    }
  } else if (ciTieMethod === "ci_overlap_adjacent") {
    const z = normPpf(0.5 + confidence / 2);
    for (let i = 1; i < ciGrouped.length; i++) {
      if (Math.abs(ciGrouped[i]! - ciGrouped[i - 1]!) <= tol) {
        ciGrouped[i] = ciGrouped[i - 1]!;
        continue;
      }
      const previousLo = sortedMus[i - 1]! - z * sortedSigmas[i - 1]!;
      const currentHi = sortedMus[i]! + z * sortedSigmas[i]!;
      if (previousLo <= currentHi) ciGrouped[i] = ciGrouped[i - 1]!;
    }
  } else {
    throw new Error("Unknown ci_tie_method.");
  }

  const ciRanks = ranksFromSortedGroups(order, ciGrouped);
  result.competition_ci = ciRanks.competition;
  result.competition_max_ci = ciRanks.competition_max;
  result.dense_ci = ciRanks.dense;
  result.avg_ci = ciRanks.avg;
  return result;
}

/** Compare two rankings using the same statistics and return shape as Python. */
export function compareRankings(
  rankedListA: ArrayLike<number>,
  rankedListB: ArrayLike<number>,
  method?: "all",
): RankingComparison;
export function compareRankings(
  rankedListA: ArrayLike<number>,
  rankedListB: ArrayLike<number>,
  method: SingleComparisonMethod,
): Correlation;
export function compareRankings(
  rankedListA: ArrayLike<number>,
  rankedListB: ArrayLike<number>,
  method: CompareRankingsMethod,
): RankingComparison | Correlation;
export function compareRankings(
  rankedListA: ArrayLike<number>,
  rankedListB: ArrayLike<number>,
  method: CompareRankingsMethod = "all",
): RankingComparison | Correlation {
  const allowed = new Set<CompareRankingsMethod>([
    "kendall",
    "spearman",
    "weighted_kendall",
    "all",
  ]);
  if (!allowed.has(method)) {
    throw new Error(
      `method must be one of ['all', 'kendall', 'spearman', 'weighted_kendall']; got '${method}'`,
    );
  }
  const a = asNumericVector(rankedListA, "ranked lists");
  const b = asNumericVector(rankedListB, "ranked lists");
  if (a.length === 0 || a.length !== b.length) {
    throw new Error("Ranked lists must have the same non-zero length.");
  }
  if (![...a, ...b].every(Number.isFinite)) {
    throw new Error("ranked lists must not contain NaN or inf.");
  }

  const kendall = kendallTau(a, b);
  const spearman = spearmanR(a, b);
  const weighted = weightedKendallTau(a, b);
  if (method === "kendall") return kendall;
  if (method === "spearman") return spearman;
  if (method === "weighted_kendall") return weighted;

  let mismatched = 0;
  let maxDifference = 0;
  for (let i = 0; i < a.length; i++) {
    const difference = Math.abs(b[i]! - a[i]!);
    if (difference !== 0) mismatched += 1;
    if (difference > maxDifference) maxDifference = difference;
  }
  return {
    kendalltau: kendall,
    spearmanr: spearman,
    weighted_kendalltau: weighted,
    fraction_mismatched: mismatched / a.length,
    max_disp: a.length > 1 ? maxDifference / (a.length - 1) : 0,
  };
}

const MAX_SAFE_BIGINT = BigInt(Number.MAX_SAFE_INTEGER);

function validateSize(name: string, value: number): void {
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new TypeError(`${name} must be a nonnegative safe integer.`);
  }
}

function asBigInt(value: Integer, name: string): bigint {
  if (typeof value === "bigint") return value;
  if (!Number.isSafeInteger(value)) {
    throw new TypeError(`${name} must be a safe integer or bigint.`);
  }
  return BigInt(value);
}

function publicInteger(value: bigint): Integer {
  return value <= MAX_SAFE_BIGINT ? Number(value) : value;
}

function factorial(n: number): bigint {
  let value = 1n;
  for (let i = 2n; i <= BigInt(n); i++) value *= i;
  return value;
}

function binomial(n: number, k: number): bigint {
  if (k < 0 || n < 0 || k > n) return 0n;
  const kk = Math.min(k, n - k);
  let value = 1n;
  for (let i = 1; i <= kk; i++) {
    value = (value * BigInt(n - kk + i)) / BigInt(i);
  }
  return value;
}

function orderedBellBigInt(n: number): bigint[] {
  const values = new Array<bigint>(n + 1).fill(0n);
  values[0] = 1n;
  for (let m = 1; m <= n; m++) {
    let sum = 0n;
    for (let k = 1; k <= m; k++) sum += binomial(m, k) * values[m - k]!;
    values[m] = sum;
  }
  return values;
}

/** Compute the ordered Bell/Fubini numbers F[0..n]. */
export function orderedBell(n: number): Integer[] {
  validateSize("n", n);
  return orderedBellBigInt(n).map(publicInteger);
}

/** Lexicographic rank of a sorted k-combination from `0..n-1`. */
export function combRankLex(indices: readonly number[], n: number, k: number): Integer {
  validateSize("n", n);
  validateSize("k", k);
  let rank = 0n;
  let previous = -1;
  for (let position = 0; position < k; position++) {
    const end = indices[position]!;
    const remaining = k - position - 1;
    for (let x = previous + 1; x < end; x++) rank += binomial(n - 1 - x, remaining);
    previous = end;
  }
  return publicInteger(rank);
}

/** Inverse of {@link combRankLex}. */
export function combUnrankLex(rankInput: Integer, n: number, k: number): number[] {
  validateSize("n", n);
  validateSize("k", k);
  if (k === 0) return [];
  let rank = asBigInt(rankInput, "Combination rank");
  const total = binomial(n, k);
  if (rank < 0n || rank >= total) throw new Error("Combination rank out of range.");
  const combination: number[] = [];
  let x = 0;
  for (let position = 0; position < k; position++) {
    const remaining = k - position - 1;
    while (true) {
      const count = n - 1 - x >= remaining ? binomial(n - 1 - x, remaining) : 0n;
      if (rank < count) {
        combination.push(x);
        x += 1;
        break;
      }
      rank -= count;
      x += 1;
    }
  }
  return combination;
}

/** Convert a ranking to canonical ordered tie blocks. */
export function blocksFromRankList(
  rankList: ArrayLike<number>,
  tol = 1e-12,
): number[][] {
  const ranks = asNumericVector(rankList, "rank_list");
  const order = Array.from({ length: ranks.length }, (_, i) => i);
  order.sort((a, b) => {
    const ra = ranks[a]!;
    const rb = ranks[b]!;
    if (Number.isNaN(ra)) return Number.isNaN(rb) ? a - b : 1;
    if (Number.isNaN(rb)) return -1;
    return ra !== rb ? ra - rb : a - b;
  });
  if (order.length === 0) return [];
  const blocks: number[][] = [[order[0]!]];
  for (let i = 1; i < order.length; i++) {
    const id = order[i]!;
    const previous = order[i - 1]!;
    if (Math.abs(ranks[id]! - ranks[previous]!) <= tol) {
      blocks[blocks.length - 1]!.push(id);
    }
    else blocks.push([id]);
  }
  for (const block of blocks) block.sort((a, b) => a - b);
  return blocks;
}

/** Convert a permutation of `0..n-1` to its Lehmer code. */
export function lehmerHash(rankedList: readonly number[]): Integer {
  const permutation = Array.from(rankedList);
  const n = permutation.length;
  if (permutation.some((value) => !Number.isInteger(value))) {
    throw new TypeError("ranked_list must be a permutation of integers 0..n-1.");
  }
  if (
    new Set(permutation).size !== n ||
    permutation.some((value) => value < 0 || value >= n)
  ) {
    throw new Error("ranked_list must be a permutation of 0..n-1 with no ties.");
  }
  const factorials = new Array<bigint>(n).fill(1n);
  for (let i = 1; i < n; i++) factorials[i] = factorials[i - 1]! * BigInt(i);
  let hash = 0n;
  for (let i = 0; i < n; i++) {
    let inversions = 0;
    for (let j = i + 1; j < n; j++) if (permutation[j]! < permutation[i]!) inversions++;
    hash += BigInt(inversions) * factorials[n - 1 - i]!;
  }
  return publicInteger(hash);
}

/** Convert a Lehmer code back to a permutation. */
export function lehmerUnhash(hashInput: Integer, n: number): number[] {
  validateSize("n", n);
  let hash = asBigInt(hashInput, "hash_value");
  const maximum = factorial(n);
  if (hash < 0n || hash >= maximum) {
    throw new Error(
      `hash_value must be in range 0..${n}!-1 = ${maximum - 1n}; got ${hashInput}`,
    );
  }
  const factorials = new Array<bigint>(n).fill(1n);
  for (let i = 1; i < n; i++) factorials[i] = factorials[i - 1]! * BigInt(i);
  const available = Array.from({ length: n }, (_, i) => i);
  const result: number[] = [];
  for (let i = 0; i < n; i++) {
    const divisor = factorials[n - 1 - i]!;
    const index = Number(hash / divisor);
    hash %= divisor;
    result.push(available.splice(index, 1)[0]!);
  }
  return result;
}

/** Perfect collision-free hash for rankings with ties. */
export function rankingHash(rankList: ArrayLike<number>, tol = 1e-12): Integer {
  const ranks = asNumericVector(rankList, "rank_list");
  const blocks = blocksFromRankList(ranks, tol);
  const fubini = orderedBellBigInt(ranks.length);
  let remaining = Array.from({ length: ranks.length }, (_, i) => i);
  let hash = 0n;
  for (const block of blocks) {
    const m = remaining.length;
    const k = block.length;
    for (let size = 1; size < k; size++) {
      hash += binomial(m, size) * fubini[m - size]!;
    }
    const positions = new Map(remaining.map((value, index) => [value, index]));
    const indices = block.map((value) => positions.get(value)!);
    hash += asBigInt(combRankLex(indices, m, k), "subset rank") * fubini[m - k]!;
    const chosen = new Set(block);
    remaining = remaining.filter((value) => !chosen.has(value));
  }
  return publicInteger(hash);
}

/** Reconstruct a competition-format ranking from a collision-free hash. */
export function unhashRanking(hashInput: Integer, n: number): number[] {
  validateSize("n", n);
  let hash = asBigInt(hashInput, "h");
  const fubini = orderedBellBigInt(n);
  if (hash < 0n || hash >= fubini[n]!) {
    throw new Error(`h out of range for n=${n}. Must be 0..${fubini[n]! - 1n}.`);
  }
  let remaining = Array.from({ length: n }, (_, i) => i);
  const ranking = new Array<number>(n).fill(0);
  let currentRank = 1;
  while (remaining.length > 0) {
    const m = remaining.length;
    let offset = 0n;
    let groupSize = 0;
    for (let k = 1; k <= m; k++) {
      const count = binomial(m, k) * fubini[m - k]!;
      if (hash < offset + count) {
        hash -= offset;
        groupSize = k;
        break;
      }
      offset += count;
    }
    if (groupSize === 0) throw new Error("Unhashing failed.");
    const suffixCount = fubini[m - groupSize]!;
    const subsetRank = hash / suffixCount;
    hash %= suffixCount;
    const indices = combUnrankLex(subsetRank, m, groupSize);
    const group = indices.map((index) => remaining[index]!);
    for (const item of group) ranking[item] = currentRank;
    const chosen = new Set(group);
    remaining = remaining.filter((item) => !chosen.has(item));
    currentRank += groupSize;
  }
  return ranking;
}

// Python-compatible aliases.
export {
  rankScores as rank_scores,
  compareRankings as compare_rankings,
  lehmerHash as lehmer_hash,
  lehmerUnhash as lehmer_unhash,
  orderedBell as ordered_bell,
  combRankLex as comb_rank_lex,
  combUnrankLex as comb_unrank_lex,
  blocksFromRankList as blocks_from_rank_list,
  rankingHash as ranking_hash,
  unhashRanking as unhash_ranking,
};
