/**
 * Shared return shape and option types for ranking methods.
 *
 * Unlike the Python API's `return_scores` flag, every TypeScript ranking method
 * returns both the ranking and the raw method scores. `ranking` is the primary
 * output (1 = best); `scores` are the underlying values that induced it.
 */

import type { RankMethod } from "./rankScores.js";

export type { RankMethod } from "./rankScores.js";

/** Result of a ranking method: ranks (1 = best) plus the raw scores. */
export interface RankResult {
  /** Rank of each model, shape `(L,)`, where 1 is best. */
  ranking: number[];
  /** Raw method scores, shape `(L,)`, where larger is better. */
  scores: number[];
}

/** Options shared by every ranking method. */
export interface BaseRankOptions {
  /** Tie-handling convention. Default `"competition"`. */
  method?: RankMethod;
}
