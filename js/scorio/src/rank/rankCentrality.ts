/**
 * Rank Centrality (Negahban, Oh & Shah, 2017).
 * Port of `scorio/rank/rank_centrality.py`.
 *
 * Builds a row-stochastic random walk that prefers moving from a model to the
 * models that beat it, and ranks by the chain's stationary distribution.
 */

import { matTVec, l1Diff } from "./internal/linalg.js";
import { rankScores } from "./internal/rankScores.js";
import { validatePositiveFloat, validatePositiveInt } from "./internal/validate.js";
import {
  buildPairwiseCounts,
  buildPairwiseWins,
  shape3,
  validateInput,
  zeros2,
  type TensorInput,
} from "./internal/tensor.js";
import type { BaseRankOptions, RankResult } from "./internal/result.js";

const sum = (a: readonly number[]): number => a.reduce((s, v) => s + v, 0);

/** Options for {@link rankCentrality}. */
export interface RankCentralityOptions extends BaseRankOptions {
  /** `"half"` (default) splits ties; `"ignore"` uses only decisive comparisons. */
  tieHandling?: "ignore" | "half";
  /** Pseudocount added to every directed win count. Default `0`. */
  smoothing?: number;
  /** Teleportation probability in `[0, 1)`. Default `0`. */
  teleport?: number;
  maxIter?: number;
  tol?: number;
}

function isConnected(adj: boolean[][]): boolean {
  const n = adj.length;
  if (n === 0) return true;
  const seen = new Array<boolean>(n).fill(false);
  const stack = [0];
  seen[0] = true;
  while (stack.length) {
    const i = stack.pop()!;
    for (let j = 0; j < n; j++) {
      if (adj[i]![j] && !seen[j]) {
        seen[j] = true;
        stack.push(j);
      }
    }
  }
  return seen.every((x) => x);
}

function stationaryPower(P: number[][], maxIter: number, tol: number): number[] {
  const n = P.length;
  if (n === 0) return [];
  let pi = new Array<number>(n).fill(1 / n);
  for (let it = 0; it < maxIter; it++) {
    const piNew = matTVec(P, pi);
    const s = sum(piNew);
    if (s <= 0) return new Array<number>(n).fill(1 / n);
    for (let j = 0; j < n; j++) piNew[j]! /= s;
    if (l1Diff(piNew, pi) < tol) return piNew;
    pi = piNew;
  }
  return pi;
}

/** Rank models with Rank Centrality. */
export function rankCentrality(
  R: TensorInput,
  options: RankCentralityOptions = {},
): RankResult {
  const method = options.method ?? "competition";
  const tieHandling = options.tieHandling ?? "half";
  const smoothing = options.smoothing ?? 0;
  const teleport = options.teleport ?? 0;
  const maxIter = validatePositiveInt("max_iter", options.maxIter ?? 10000);
  const tol = validatePositiveFloat("tol", options.tol ?? 1e-12);
  if (tieHandling !== "ignore" && tieHandling !== "half") {
    throw new Error('tie_handling must be "ignore" or "half"');
  }
  if (!Number.isFinite(smoothing) || smoothing < 0) throw new Error("smoothing must be >= 0");
  if (!Number.isFinite(teleport) || !(teleport >= 0 && teleport < 1)) {
    throw new Error("teleport must be in [0, 1)");
  }

  const tensor = validateInput(R);
  const [L] = shape3(tensor);

  let wins: number[][];
  if (tieHandling === "ignore") {
    wins = buildPairwiseWins(tensor);
  } else {
    const counts = buildPairwiseCounts(tensor);
    wins = counts.wins.map((row, i) => row.map((v, j) => v + 0.5 * counts.ties[i]![j]!));
  }

  const winsS = wins.map((row) => row.map((v) => v + smoothing));
  const denom = zeros2(L, L);
  for (let i = 0; i < L; i++)
    for (let j = 0; j < L; j++) denom[i]![j] = winsS[i]![j]! + winsS[j]![i]!;

  const adj: boolean[][] = Array.from({ length: L }, (_, i) =>
    Array.from({ length: L }, (_, j) => denom[i]![j]! > 0 && i !== j),
  );
  let dMax = 0;
  for (let i = 0; i < L; i++) {
    let deg = 0;
    for (let j = 0; j < L; j++) if (adj[i]![j]) deg += 1;
    dMax = Math.max(dMax, deg);
  }

  if (dMax === 0) {
    const scores = new Array<number>(L).fill(1 / L);
    return { ranking: rankScores(scores, method), scores };
  }
  if (teleport === 0 && smoothing === 0 && tieHandling === "ignore" && !isConnected(adj)) {
    throw new Error(
      "Rank Centrality requires a connected comparison graph; " +
        "use teleport>0, smoothing>0, or tie_handling='half'.",
    );
  }

  // P[i][j] = (1/d_max) · p̂_{j,i} on edges; P[i][i] = 1 - Σ_{j≠i} P[i][j].
  const P = zeros2(L, L);
  for (let i = 0; i < L; i++) {
    for (let j = 0; j < L; j++) {
      if (!adj[i]![j]) continue;
      const pJi = winsS[j]![i]! / denom[i]![j]!;
      P[i]![j] = pJi / dMax;
    }
  }
  for (let i = 0; i < L; i++) {
    let rowSum = 0;
    for (let j = 0; j < L; j++) if (j !== i) rowSum += P[i]![j]!;
    P[i]![i] = 1 - rowSum;
  }

  if (teleport > 0) {
    for (let i = 0; i < L; i++)
      for (let j = 0; j < L; j++)
        P[i]![j] = (1 - teleport) * P[i]![j]! + teleport * (1 / L);
  }

  const scores = stationaryPower(P, maxIter, tol);
  return { ranking: rankScores(scores, method), scores };
}
