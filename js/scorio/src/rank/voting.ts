/**
 * Voting-based ranking methods. Port of `scorio/rank/voting.py`.
 *
 * Each question is treated as a voter over models, based on its per-question
 * correct counts `k[l][m] = Σ_n R[l][m][n]`. Social-choice rules turn those
 * per-question preferences into model scores.
 *
 * `kemeny_young` is solved here by exact enumeration of the `L!` linear orders
 * (the linear-ordering problem) rather than the reference's MILP; for the small
 * `L` typical of model ranking this is exact and the tie-aware mode reproduces
 * the forced-order DAG over all optimal Kemeny solutions.
 */

import {
  rankdata,
  rankScores,
  type RankDataMethod,
} from "./internal/rankScores.js";
import {
  perQuestionCorrectCounts,
  shape3,
  validateInput,
  zeros2,
  type TensorInput,
} from "./internal/tensor.js";
import type { BaseRankOptions, RankResult } from "./internal/result.js";

type TiePolicy = "ignore" | "half";

function pairwisePreferenceCounts(k: number[][], tiePolicy: TiePolicy): number[][] {
  const L = k.length;
  const M = k[0]!.length;
  if (tiePolicy !== "ignore" && tiePolicy !== "half") {
    throw new Error("tie_policy must be one of {'ignore','half'}");
  }
  const P = zeros2(L, L);
  for (let i = 0; i < L; i++) {
    for (let j = i + 1; j < L; j++) {
      let iOverJ = 0;
      let jOverI = 0;
      for (let m = 0; m < M; m++) {
        if (k[i]![m]! > k[j]![m]!) iOverJ += 1;
        else if (k[j]![m]! > k[i]![m]!) jOverI += 1;
      }
      if (tiePolicy === "half") {
        const ties = M - iOverJ - jOverI;
        iOverJ += 0.5 * ties;
        jOverI += 0.5 * ties;
      }
      P[i]![j] = iOverJ;
      P[j]![i] = jOverI;
    }
  }
  return P;
}

/** Tie-aware level scores from a DAG `adj[i][j]=true` ⇒ `i` ranks above `j`. */
function topologicalLevelScores(adj: boolean[][]): number[] {
  const L = adj.length;
  const remaining = new Array<boolean>(L).fill(true);
  const indeg = new Array<number>(L).fill(0);
  for (let i = 0; i < L; i++)
    for (let j = 0; j < L; j++) if (adj[i]![j]) indeg[j]! += 1;

  const scores = new Array<number>(L).fill(0);
  let current = L;
  let remainingCount = L;
  while (remainingCount > 0) {
    const nodes: number[] = [];
    for (let i = 0; i < L; i++) if (remaining[i] && indeg[i] === 0) nodes.push(i);
    if (nodes.length === 0) {
      for (let i = 0; i < L; i++) if (remaining[i]) scores[i] = current;
      break;
    }
    for (const u of nodes) scores[u] = current;
    current -= 1;
    for (const u of nodes) {
      remaining[u] = false;
      remainingCount -= 1;
      for (let v = 0; v < L; v++) if (adj[u]![v]) indeg[v]! -= 1;
    }
  }
  return scores;
}

/** Rank models with Borda count on per-question rankings. */
export function borda(R: TensorInput, options: BaseRankOptions = {}): RankResult {
  const method = options.method ?? "competition";
  const k = perQuestionCorrectCounts(validateInput(R));
  const L = k.length;
  const M = k[0]!.length;
  const scores = new Array<number>(L).fill(0);
  for (let m = 0; m < M; m++) {
    const col = k.map((row) => -row[m]!);
    const r = rankdata(col, "average");
    for (let l = 0; l < L; l++) scores[l]! += L - r[l]!;
  }
  return { ranking: rankScores(scores, method), scores };
}

/** Rank models with Copeland pairwise-majority scores. */
export function copeland(R: TensorInput, options: BaseRankOptions = {}): RankResult {
  const method = options.method ?? "competition";
  const k = perQuestionCorrectCounts(validateInput(R));
  const L = k.length;
  const M = k[0]!.length;
  const scores = new Array<number>(L).fill(0);
  for (let i = 0; i < L; i++) {
    for (let j = i + 1; j < L; j++) {
      let iOverJ = 0;
      let jOverI = 0;
      for (let m = 0; m < M; m++) {
        if (k[i]![m]! > k[j]![m]!) iOverJ += 1;
        else if (k[j]![m]! > k[i]![m]!) jOverI += 1;
      }
      if (iOverJ > jOverI) {
        scores[i]! += 1;
        scores[j]! -= 1;
      } else if (jOverI > iOverJ) {
        scores[i]! -= 1;
        scores[j]! += 1;
      }
    }
  }
  return { ranking: rankScores(scores, method), scores };
}

/** Rank models by pairwise question-level win rate. */
export function winRate(R: TensorInput, options: BaseRankOptions = {}): RankResult {
  const method = options.method ?? "competition";
  const k = perQuestionCorrectCounts(validateInput(R));
  const L = k.length;
  const M = k[0]!.length;
  const wins = zeros2(L, L);
  for (let i = 0; i < L; i++) {
    for (let j = i + 1; j < L; j++) {
      let iOverJ = 0;
      let jOverI = 0;
      for (let m = 0; m < M; m++) {
        if (k[i]![m]! > k[j]![m]!) iOverJ += 1;
        else if (k[j]![m]! > k[i]![m]!) jOverI += 1;
      }
      wins[i]![j] = iOverJ;
      wins[j]![i] = jOverI;
    }
  }
  const scores = new Array<number>(L).fill(0.5);
  for (let i = 0; i < L; i++) {
    let totalWins = 0;
    let totalComparisons = 0;
    for (let j = 0; j < L; j++) {
      totalWins += wins[i]![j]!;
      totalComparisons += wins[i]![j]! + wins[j]![i]!;
    }
    if (totalComparisons > 0) scores[i] = totalWins / totalComparisons;
  }
  return { ranking: rankScores(scores, method), scores };
}

/** Options for {@link minimax}. */
export interface MinimaxOptions extends BaseRankOptions {
  /** `"margin"` (default) or `"winning_votes"`. */
  variant?: "margin" | "winning_votes";
  /** Per-question tie handling: `"half"` (default) or `"ignore"`. */
  tiePolicy?: TiePolicy;
}

/** Rank models with the Minimax (Simpson-Kramer) Condorcet rule. */
export function minimax(R: TensorInput, options: MinimaxOptions = {}): RankResult {
  const method = options.method ?? "competition";
  const variant = options.variant === undefined ? "margin" : options.variant;
  if (variant !== "margin" && variant !== "winning_votes") {
    throw new Error("variant must be one of {'margin','winning_votes'}");
  }
  const k = perQuestionCorrectCounts(validateInput(R));
  const P = pairwisePreferenceCounts(
    k,
    options.tiePolicy === undefined ? "half" : options.tiePolicy,
  );
  const L = P.length;
  const scores = new Array<number>(L).fill(0);
  for (let i = 0; i < L; i++) {
    let worst = 0;
    for (let j = 0; j < L; j++) {
      if (i === j) continue;
      const marginJI = P[j]![i]! - P[i]![j]!;
      if (marginJI > 0) {
        const defeat = variant === "margin" ? marginJI : P[j]![i]!;
        if (defeat > worst) worst = defeat;
      }
    }
    scores[i] = -worst;
  }
  return { ranking: rankScores(scores, method), scores };
}

/** Options for {@link schulze}. */
export interface SchulzeOptions extends BaseRankOptions {
  tiePolicy?: TiePolicy;
}

/** Rank models with the Schulze beatpath Condorcet method. */
export function schulze(R: TensorInput, options: SchulzeOptions = {}): RankResult {
  const method = options.method ?? "competition";
  const k = perQuestionCorrectCounts(validateInput(R));
  const P = pairwisePreferenceCounts(k, options.tiePolicy ?? "half");
  const L = P.length;
  const p = zeros2(L, L);
  for (let i = 0; i < L; i++)
    for (let j = 0; j < L; j++)
      if (i !== j && P[i]![j]! > P[j]![i]!) p[i]![j] = P[i]![j]!;

  for (let i = 0; i < L; i++) {
    for (let j = 0; j < L; j++) {
      if (i === j) continue;
      for (let kk = 0; kk < L; kk++) {
        if (i === kk || j === kk) continue;
        p[j]![kk] = Math.max(p[j]![kk]!, Math.min(p[j]![i]!, p[i]![kk]!));
      }
    }
  }
  const beats: boolean[][] = Array.from({ length: L }, (_, i) =>
    Array.from({ length: L }, (_, j) => p[i]![j]! > p[j]![i]!),
  );
  const scores = topologicalLevelScores(beats);
  return { ranking: rankScores(scores, method), scores };
}

/** Options for {@link rankedPairs}. */
export interface RankedPairsOptions extends BaseRankOptions {
  /** Primary edge-strength key: `"margin"` (default) or `"winning_votes"`. */
  strength?: "margin" | "winning_votes";
  tiePolicy?: TiePolicy;
}

/** Rank models with the Ranked Pairs (Tideman) Condorcet method. */
export function rankedPairs(
  R: TensorInput,
  options: RankedPairsOptions = {},
): RankResult {
  const method = options.method ?? "competition";
  const strength = options.strength ?? "margin";
  if (strength !== "margin" && strength !== "winning_votes") {
    throw new Error("strength must be one of {'margin','winning_votes'}");
  }
  const k = perQuestionCorrectCounts(validateInput(R));
  const P = pairwisePreferenceCounts(k, options.tiePolicy ?? "half");
  const L = P.length;

  const victories: { primary: number; wv: number; winner: number; loser: number }[] =
    [];
  for (let i = 0; i < L; i++) {
    for (let j = i + 1; j < L; j++) {
      const margin = P[i]![j]! - P[j]![i]!;
      if (margin === 0) continue;
      const winner = margin > 0 ? i : j;
      const loser = margin > 0 ? j : i;
      const m = Math.abs(margin);
      const wv = P[winner]![loser]!;
      victories.push({ primary: strength === "margin" ? m : wv, wv, winner, loser });
    }
  }
  victories.sort(
    (a, b) =>
      b.primary - a.primary || b.wv - a.wv || a.winner - b.winner || a.loser - b.loser,
  );

  const locked: boolean[][] = Array.from({ length: L }, () =>
    new Array<boolean>(L).fill(false),
  );
  const hasPath = (src: number, dst: number): boolean => {
    const stack = [src];
    const seen = new Set<number>([src]);
    while (stack.length) {
      const u = stack.pop()!;
      if (u === dst) return true;
      for (let v = 0; v < L; v++) {
        if (locked[u]![v] && !seen.has(v)) {
          seen.add(v);
          stack.push(v);
        }
      }
    }
    return false;
  };
  for (const { winner, loser } of victories) {
    if (hasPath(loser, winner)) continue;
    locked[winner]![loser] = true;
  }
  const scores = topologicalLevelScores(locked);
  return { ranking: rankScores(scores, method), scores };
}

/** Options for {@link kemenyYoung}. */
export interface KemenyYoungOptions extends BaseRankOptions {
  tiePolicy?: TiePolicy;
  /**
   * Tie-aware preorder over all optimal Kemeny orders. Default `true`.
   * When false and several total orders are optimal, this port returns its
   * first deterministically enumerated optimum. SciPy/HiGHS may select a
   * different, equally optimal labelled order from the same optimal face.
   */
  tieAware?: boolean | null;
  /** Optional positive exact-solver time limit in seconds. */
  timeLimit?: number | null;
}

function* permutations(n: number): Generator<number[]> {
  const arr = Array.from({ length: n }, (_, i) => i);
  const c = new Array<number>(n).fill(0);
  yield arr.slice();
  let i = 0;
  while (i < n) {
    if (c[i]! < i) {
      const swap = i % 2 === 0 ? 0 : c[i]!;
      const tmp = arr[swap]!;
      arr[swap] = arr[i]!;
      arr[i] = tmp;
      yield arr.slice();
      c[i]! += 1;
      i = 0;
    } else {
      c[i] = 0;
      i += 1;
    }
  }
}

/**
 * Rank models with Kemeny-Young rank aggregation (exact, by enumeration).
 * Non-unique single-order solutions (`tieAware: false`) are solver-selection
 * dependent; use the default tie-aware preorder for label-invariant parity.
 */
export function kemenyYoung(
  R: TensorInput,
  options: KemenyYoungOptions = {},
): RankResult {
  const method = options.method ?? "competition";
  const tieAware =
    options.tieAware === undefined ? true : Boolean(options.tieAware);
  const timeLimit = options.timeLimit;
  if (
    timeLimit != null &&
    (!Number.isFinite(timeLimit) || timeLimit <= 0)
  ) {
    throw new Error("time_limit must be a positive finite scalar.");
  }
  const k = perQuestionCorrectCounts(validateInput(R));
  const P = pairwisePreferenceCounts(
    k,
    options.tiePolicy === undefined ? "half" : options.tiePolicy,
  );
  const L = P.length;

  const objectiveOf = (perm: readonly number[]): number => {
    let s = 0;
    for (let p = 0; p < L; p++)
      for (let q = p + 1; q < L; q++) s += P[perm[p]!]![perm[q]!]!;
    return s;
  };

  let best = -Infinity;
  const optimal: number[][] = [];
  const startedAt = Date.now();
  for (const perm of permutations(L)) {
    if (timeLimit != null && Date.now() - startedAt > timeLimit * 1000) {
      throw new Error(
        "Kemeny-Young exact enumeration did not prove an optimal solution within time_limit.",
      );
    }
    const obj = objectiveOf(perm);
    if (obj > best + 1e-9) {
      best = obj;
      optimal.length = 0;
      optimal.push(perm.slice());
    } else if (Math.abs(obj - best) <= 1e-9) {
      optimal.push(perm.slice());
    }
  }

  if (!tieAware) {
    // Number of opponents ranked below each model in the first optimal order.
    const order = optimal[0]!;
    const pos = new Array<number>(L).fill(0);
    for (let p = 0; p < L; p++) pos[order[p]!] = p;
    const scores = pos.map((p) => L - 1 - p);
    return { ranking: rankScores(scores, method), scores };
  }

  // Forced order: a above b in every optimal permutation.
  const forced: boolean[][] = Array.from({ length: L }, () =>
    new Array<boolean>(L).fill(false),
  );
  for (let a = 0; a < L; a++) {
    for (let b = 0; b < L; b++) {
      if (a === b) continue;
      let allAbove = true;
      for (const perm of optimal) {
        if (perm.indexOf(a) > perm.indexOf(b)) {
          allAbove = false;
          break;
        }
      }
      if (allAbove) forced[a]![b] = true;
    }
  }
  const scores = topologicalLevelScores(forced);
  return { ranking: rankScores(scores, method), scores };
}

/** Options for {@link nanson} and {@link baldwin}. */
export interface EliminationOptions extends BaseRankOptions {
  /** Tie rule for per-question Borda ranks among active models. Default `"average"`. */
  rankTies?: RankDataMethod;
}

function bordaElimination(
  R: TensorInput,
  method: BaseRankOptions["method"],
  rankTies: RankDataMethod,
  eliminate: (bordaSub: number[]) => boolean[] | null,
): RankResult {
  if (!(["min", "max", "dense", "average", "ordinal"] as const).includes(rankTies)) {
    throw new Error(`Unknown rankdata method: ${rankTies}`);
  }
  const k = perQuestionCorrectCounts(validateInput(R));
  const L = k.length;
  const M = k[0]!.length;
  const alive = new Array<boolean>(L).fill(true);
  const survival = new Array<number>(L).fill(0);
  let round = 0;

  const aliveCount = () => alive.reduce((s, a) => s + (a ? 1 : 0), 0);
  while (aliveCount() > 1) {
    const idx: number[] = [];
    for (let l = 0; l < L; l++) if (alive[l]) idx.push(l);
    const bordaSub = new Array<number>(idx.length).fill(0);
    for (let m = 0; m < M; m++) {
      const col = idx.map((l) => -k[l]![m]!);
      const r = rankdata(col, rankTies);
      for (let t = 0; t < idx.length; t++) bordaSub[t]! += idx.length - r[t]!;
    }
    const toElim = eliminate(bordaSub);
    if (toElim === null) break;
    for (let t = 0; t < idx.length; t++) {
      if (toElim[t]) {
        survival[idx[t]!] = round;
        alive[idx[t]!] = false;
      }
    }
    round += 1;
  }
  for (let l = 0; l < L; l++) if (alive[l]) survival[l] = round;
  return { ranking: rankScores(survival, method ?? "competition"), scores: survival };
}

/** Rank models with Nanson's Borda-elimination rule (eliminate at or below mean). */
export function nanson(R: TensorInput, options: EliminationOptions = {}): RankResult {
  return bordaElimination(R, options.method, options.rankTies ?? "average", (bordaSub) => {
    const mean = bordaSub.reduce((s, v) => s + v, 0) / bordaSub.length;
    const toElim = bordaSub.map((v) => v <= mean);
    return toElim.every((x) => x) ? null : toElim;
  });
}

/** Rank models with Baldwin's Borda-elimination rule (eliminate the minimum). */
export function baldwin(R: TensorInput, options: EliminationOptions = {}): RankResult {
  return bordaElimination(R, options.method, options.rankTies ?? "average", (bordaSub) => {
    const min = Math.min(...bordaSub);
    const toElim = bordaSub.map((v) => v === min);
    return toElim.every((x) => x) ? null : toElim;
  });
}

/** Rank models with Majority Judgment on per-question grade counts. */
export function majorityJudgment(
  R: TensorInput,
  options: BaseRankOptions = {},
): RankResult {
  const method = options.method ?? "competition";
  const tensor = validateInput(R);
  const k = perQuestionCorrectCounts(tensor);
  const [, , N] = shape3(tensor);
  const L = k.length;
  const M = k[0]!.length;

  const counts: number[][] = Array.from({ length: L }, () =>
    new Array<number>(N + 1).fill(0),
  );
  for (let i = 0; i < L; i++) for (let m = 0; m < M; m++) counts[i]![k[i]![m]!]! += 1;

  const lowerMedianGrade = (hist: number[], total: number): number => {
    const target = Math.floor((total - 1) / 2);
    let cum = 0;
    for (let g = 0; g < hist.length; g++) {
      cum += hist[g]!;
      if (cum > target) return g;
    }
    return hist.length - 1;
  };

  const compare = (i: number, j: number): number => {
    const hi = counts[i]!.slice();
    const hj = counts[j]!.slice();
    let ti = M;
    let tj = M;
    while (ti > 0 && tj > 0) {
      const gi = lowerMedianGrade(hi, ti);
      const gj = lowerMedianGrade(hj, tj);
      if (gi !== gj) return gi > gj ? -1 : 1;
      hi[gi]! -= 1;
      hj[gj]! -= 1;
      ti -= 1;
      tj -= 1;
    }
    return 0;
  };

  const order = Array.from({ length: L }, (_, i) => i).sort(compare);
  const scores = new Array<number>(L).fill(0);
  let current = L;
  let start = 0;
  while (start < L) {
    let end = start + 1;
    while (end < L && compare(order[start]!, order[end]!) === 0) end += 1;
    for (let t = start; t < end; t++) scores[order[t]!] = current;
    current -= 1;
    start = end;
  }
  return { ranking: rankScores(scores, method), scores };
}
