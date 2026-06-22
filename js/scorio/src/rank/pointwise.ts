/**
 * Pointwise ranking methods. Port of `scorio/rank/pointwise.py`.
 */

import { rankScores } from "./internal/rankScores.js";
import { clip } from "./internal/special.js";
import {
  validateInput,
  shape3,
  type TensorInput,
} from "./internal/tensor.js";
import type { BaseRankOptions, RankResult } from "./internal/result.js";

/** Options for {@link inverseDifficulty}. */
export interface InverseDifficultyOptions extends BaseRankOptions {
  /** Two-sided clipping interval `[a, b]` on global solve rates (`0 < a < b <= 1`). */
  clipRange?: readonly [number, number];
}

/**
 * Rank models by inverse-difficulty-weighted per-question accuracy.
 *
 * Each question is weighted by the reciprocal of its clipped global solve rate,
 * upweighting hard questions; a model's score is the weighted average of its
 * per-question accuracies.
 */
export function inverseDifficulty(
  R: TensorInput,
  options: InverseDifficultyOptions = {},
): RankResult {
  const method = options.method ?? "competition";
  const clipRange = options.clipRange ?? [0.01, 0.99];
  const tensor = validateInput(R);
  const [L, M, N] = shape3(tensor);

  if (clipRange.length !== 2) {
    throw new Error("clip_range must be a length-2 tuple (low, high).");
  }
  const low = Number(clipRange[0]);
  const high = Number(clipRange[1]);
  if (!Number.isFinite(low) || !Number.isFinite(high)) {
    throw new Error("clip_range values must be finite.");
  }
  if (!(low > 0 && low < high && high <= 1)) {
    throw new Error("clip_range must satisfy 0 < low < high <= 1.");
  }

  // Global difficulty per question: mean over all models and trials.
  const weights = new Array<number>(M).fill(0);
  for (let m = 0; m < M; m++) {
    let s = 0;
    for (let l = 0; l < L; l++) for (let n = 0; n < N; n++) s += tensor[l]![m]![n]!;
    weights[m] = clip(s / (L * N), low, high);
  }
  let totalWeight = 0;
  for (let m = 0; m < M; m++) {
    weights[m] = 1 / weights[m]!;
    totalWeight += weights[m]!;
  }
  if (!Number.isFinite(totalWeight) || totalWeight <= 0) {
    throw new Error(
      "inverse-difficulty weights are not finite; choose a stricter clip_range.",
    );
  }
  for (let m = 0; m < M; m++) weights[m]! /= totalWeight;

  const scores = new Array<number>(L).fill(0);
  for (let l = 0; l < L; l++) {
    let s = 0;
    for (let m = 0; m < M; m++) {
      let acc = 0;
      for (let n = 0; n < N; n++) acc += tensor[l]![m]![n]!;
      s += (acc / N) * weights[m]!;
    }
    scores[l] = s;
  }

  return { ranking: rankScores(scores, method), scores };
}
