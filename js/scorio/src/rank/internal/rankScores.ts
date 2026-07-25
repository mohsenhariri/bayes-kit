/**
 * Score-to-rank conversion.
 *
 * Port of `scorio.utils.rank_scores` and the slice of `scipy.stats.rankdata`
 * it relies on. Higher score ⇒ better (lower) rank. Four tie conventions are
 * supported, matching the Python API:
 *
 * - `competition`      → rankdata "min"     (1, 2, 2, 4)
 * - `competition_max`  → rankdata "max"     (1, 3, 3, 4)
 * - `dense`            → rankdata "dense"   (1, 2, 2, 3)
 * - `avg`              → rankdata "average" (1, 2.5, 2.5, 4)
 */

/** Tie-handling convention passed to ranking methods. */
export type RankMethod = "competition" | "competition_max" | "dense" | "avg";

/** Low-level `scipy.stats.rankdata` tie rule (ascending; rank 1 = smallest). */
export type RankDataMethod = "min" | "max" | "dense" | "average" | "ordinal";

const METHOD_TO_RANKDATA: Record<RankMethod, RankDataMethod> = {
  competition: "min",
  competition_max: "max",
  dense: "dense",
  avg: "average",
};

/**
 * `scipy.stats.rankdata(values, method)` — ascending ranks (rank 1 = smallest),
 * ties resolved per `method`. Equal values are detected by exact equality.
 */
export function rankdata(
  values: readonly number[],
  method: RankDataMethod = "average",
): number[] {
  const n = values.length;
  const order = Array.from({ length: n }, (_, i) => i);
  // Stable ascending sort by value.
  order.sort((a, b) => {
    const va = values[a]!;
    const vb = values[b]!;
    return va < vb ? -1 : va > vb ? 1 : a - b;
  });

  const ranks = new Array<number>(n).fill(0);
  // "ordinal": every element gets a distinct rank 1..n in (stable) sorted order.
  if (method === "ordinal") {
    for (let p = 0; p < n; p++) ranks[order[p]!] = p + 1;
    return ranks;
  }
  let i = 0;
  let dense = 0;
  while (i < n) {
    let j = i;
    const v = values[order[i]!]!;
    while (j + 1 < n && values[order[j + 1]!]! === v) j += 1;
    // Group spans sorted positions [i, j] (0-based) → 1-based positions [i+1, j+1].
    dense += 1;
    for (let p = i; p <= j; p++) {
      const idx = order[p]!;
      switch (method) {
        case "min":
          ranks[idx] = i + 1;
          break;
        case "max":
          ranks[idx] = j + 1;
          break;
        case "dense":
          ranks[idx] = dense;
          break;
        case "average":
          ranks[idx] = (i + 1 + (j + 1)) / 2;
          break;
      }
    }
    i = j + 1;
  }
  return ranks;
}

/**
 * Convert scores (higher is better) to ranks under all four conventions.
 * Mirrors `rank_scores`: scores within `tol` of an adjacent (sorted) score are
 * collapsed so they share a rank.
 */
export function rankScoresAll(
  scores: readonly number[],
  tol = 1e-12,
): Record<RankMethod, number[]> {
  const n = scores.length;
  // Descending order; stable so ties keep ascending index order (immaterial to
  // the result since tied scores share a rank).
  const order = Array.from({ length: n }, (_, i) => i);
  order.sort((a, b) => {
    const va = scores[a]!;
    const vb = scores[b]!;
    return vb < va ? -1 : vb > va ? 1 : a - b;
  });

  const grouped = order.map((i) => scores[i]!);
  for (let i = 1; i < n; i++) {
    if (Math.abs(grouped[i]! - grouped[i - 1]!) <= tol) {
      grouped[i] = grouped[i - 1]!;
    }
  }
  const neg = grouped.map((v) => -v);

  const scatter = (method: RankDataMethod): number[] => {
    const sorted = rankdata(neg, method);
    const out = new Array<number>(n).fill(0);
    for (let idx = 0; idx < n; idx++) out[order[idx]!] = sorted[idx]!;
    return out;
  };

  return {
    competition: scatter("min"),
    competition_max: scatter("max"),
    dense: scatter("dense"),
    avg: scatter("average"),
  };
}

/** Convenience: ranks for a single convention. */
export function rankScores(
  scores: readonly number[],
  method: RankMethod = "competition",
  tol = 1e-12,
): number[] {
  return rankScoresAll(scores, tol)[asRankMethod(method)];
}

/** Validate and normalize a user-supplied ranking method string. */
export function asRankMethod(method: string): RankMethod {
  if (
    method === "competition" ||
    method === "competition_max" ||
    method === "dense" ||
    method === "avg"
  ) {
    return method;
  }
  throw new Error(
    `method must be one of "competition", "competition_max", "dense", "avg"; got ${method}`,
  );
}

/** Map a {@link RankMethod} to its low-level rankdata tie rule. */
export function rankDataMethodFor(method: RankMethod): RankDataMethod {
  return METHOD_TO_RANKDATA[method];
}
