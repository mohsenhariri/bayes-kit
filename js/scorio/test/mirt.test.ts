import { describe, expect, it } from "vitest";

import { mirt } from "../src/rank/index.js";

/**
 * Behavioral tests for compensatory MIRT. Unlike the unidimensional IRT
 * methods (whose scalar ability is uniquely identified and is checked against
 * Python golden fixtures), MIRT's latent space has rotational freedom, so it is
 * validated behaviorally — smoke, valid ranking/score structure, and recovery
 * of a dominance ordering — mirroring how the Julia port tests its IRT family.
 */

// Deterministic, dominance-ordered tensor: model 0 dominates model 3 on every
// (item, trial), so the reference composite must rank 0 best and 3 worst.
function dominanceOrderedR(): number[][][] {
  const L = 4;
  const M = 12;
  const N = 8;
  let seed = 42;
  const rnd = (): number => {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    return seed / 0x7fffffff;
  };
  const R: number[][][] = [];
  for (let l = 0; l < L; l++) {
    const thr = 0.8 - 0.2 * l;
    const mat: number[][] = [];
    for (let m = 0; m < M; m++) {
      const row: number[] = [];
      for (let n = 0; n < N; n++) row.push(rnd() < thr ? 1 : 0);
      mat.push(row);
    }
    R.push(mat);
  }
  for (let l = L - 2; l >= 0; l--)
    for (let m = 0; m < M; m++)
      for (let n = 0; n < N; n++) R[l]![m]![n] = Math.max(R[l]![m]![n]!, R[l + 1]![m]![n]!);
  return R;
}

const R = dominanceOrderedR();
const BASE = { nQuadrature: 7, emIter: 25, maxIter: 40, tol: 1e-3 } as const;

function assertValidRanking(ranking: number[], L: number): void {
  expect(ranking).toHaveLength(L);
  expect(ranking.every((r) => Number.isFinite(r))).toBe(true);
  expect(Math.min(...ranking)).toBeCloseTo(1, 10);
  expect(ranking.every((r) => r >= 1 && r <= L)).toBe(true);
}

function assertRankingScoreConsistency(ranking: number[], scores: number[]): void {
  const L = ranking.length;
  const eps = 1e-12;
  for (let i = 0; i < L; i++)
    for (let j = 0; j < L; j++) {
      if (scores[i]! > scores[j]! + eps) expect(ranking[i]!).toBeLessThanOrEqual(ranking[j]! + eps);
      else if (scores[i]! < scores[j]! - eps)
        expect(ranking[i]!).toBeGreaterThanOrEqual(ranking[j]! - eps);
    }
}

describe("mirt (compensatory multidimensional IRT)", () => {
  it("2PL recovers the dominance ordering", () => {
    const { ranking, scores } = mirt(R, { nFactors: 2, ...BASE });
    assertValidRanking(ranking, 4);
    assertRankingScoreConsistency(ranking, scores);
    expect(ranking[0]).toBe(Math.min(...ranking));
    expect(ranking[3]).toBe(Math.max(...ranking));
  });

  it("single factor reduces cleanly", () => {
    const { ranking, scores } = mirt(R, { nFactors: 1, nQuadrature: 11, emIter: 25, maxIter: 40, tol: 1e-3 });
    assertValidRanking(ranking, 4);
    assertRankingScoreConsistency(ranking, scores);
    expect(ranking[0]).toBe(Math.min(...ranking));
    expect(ranking[3]).toBe(Math.max(...ranking));
  });

  it("3PL with fixed guessing is a valid ranking", () => {
    const { ranking, scores } = mirt(R, { nFactors: 2, model: "3pl", fixGuessing: 0.2, ...BASE });
    assertValidRanking(ranking, 4);
    assertRankingScoreConsistency(ranking, scores);
  });

  it("3PL with estimated guessing is a valid ranking", () => {
    const { ranking, scores } = mirt(R, { nFactors: 2, model: "3pl", ...BASE });
    assertValidRanking(ranking, 4);
    assertRankingScoreConsistency(ranking, scores);
  });

  it("validates options", () => {
    // @ts-expect-error invalid model
    expect(() => mirt(R, { model: "4pl" })).toThrow(/model must be/);
    expect(() => mirt(R, { model: "2pl", fixGuessing: 0.2 })).toThrow(/fixGuessing is only valid/);
    expect(() => mirt(R, { nFactors: 12, nQuadrature: 15 })).toThrow(/Product quadrature grid/);
    expect(() => mirt(R, { nFactors: 13, nQuadrature: 2 })).toThrow(/cannot exceed number of questions/);
  });
});
