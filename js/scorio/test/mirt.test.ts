import { describe, expect, it } from "vitest";

import { mirt } from "../src/rank/index.js";
import fixtures from "./fixtures/rank.json";

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
const ORDERED = fixtures.shared_inputs.ordered_binary_small_R as number[][][];
const BASE = { nQuadrature: 7, emIter: 25, maxIter: 40, tol: 1e-3 } as const;

function expectNumbersClose(
  actual: readonly number[],
  expected: readonly number[],
  tolerance = 1e-10,
): void {
  expect(actual).toHaveLength(expected.length);
  for (let index = 0; index < actual.length; index++) {
    expect(Math.abs(actual[index]! - expected[index]!)).toBeLessThanOrEqual(
      tolerance,
    );
  }
}

function expectRowsClose(
  actual: readonly (readonly number[])[],
  expected: readonly (readonly number[])[],
  tolerance = 1e-10,
): void {
  expect(actual).toHaveLength(expected.length);
  for (let index = 0; index < actual.length; index++) {
    expectNumbersClose(actual[index]!, expected[index]!, tolerance);
  }
}

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

  it("returns Python-compatible multidimensional item parameters", () => {
    const twoPl = mirt(R, {
      nFactors: 2,
      model: "2pl",
      returnItemParams: true,
      ...BASE,
    });
    expect(twoPl.itemParams!.difficulty).toHaveLength(12);
    expect(twoPl.itemParams!.discrimination).toHaveLength(12);
    expect(twoPl.itemParams!.slopes).toHaveLength(12);
    expect(twoPl.itemParams!.slopes.every((row) => row.length === 2)).toBe(true);
    expect(twoPl.itemParams!.abilities).toHaveLength(4);
    expect(twoPl.itemParams!.ability_sd).toHaveLength(4);
    expect(twoPl.itemParams!.guessing).toBeUndefined();
    expect(Object.keys(twoPl.itemParams!).sort()).toEqual([
      "abilities",
      "ability_sd",
      "difficulty",
      "discrimination",
      "intercept",
      "slopes",
    ]);

    const threePl = mirt(R, {
      nFactors: 2,
      model: "3pl",
      returnItemParams: true,
      ...BASE,
    });
    expect(threePl.itemParams!.guessing).toHaveLength(12);
    expect(
      threePl.itemParams!.guessing!.every(
        (value) => value >= 0 && value <= 0.5,
      ),
    ).toBe(true);
    expect(
      threePl.itemParams!.slopes.every((_, factor) =>
        factor >= 2 ||
        threePl.itemParams!.slopes.reduce(
          (total, row) => total + row[factor]!,
          0,
        ) >= -1e-9,
      ),
    ).toBe(true);
    expect(Object.keys(threePl.itemParams!).sort()).toEqual([
      "abilities",
      "ability_sd",
      "difficulty",
      "discrimination",
      "guessing",
      "intercept",
      "slopes",
    ]);
  });

  it("restores every MIRT output under model and item permutations", () => {
    const modelOrder = [2, 0, 3, 1];
    const itemOrder = [7, 2, 9, 0, 5, 1, 8, 4, 6, 3];
    const byModelInput = modelOrder.map((model) => ORDERED[model]!);
    const byItemInput = ORDERED.map((model) =>
      itemOrder.map((item) => model[item]!),
    );
    const configurations = [
      { model: "2pl" as const },
      { model: "3pl" as const, fixGuessing: 0.2 },
      { model: "3pl" as const },
    ];

    for (const configuration of configurations) {
      const options = {
        nFactors: 2,
        returnItemParams: true,
        ...BASE,
        ...configuration,
      };
      const base = mirt(ORDERED, options);
      const byModel = mirt(byModelInput, options);
      const byItem = mirt(byItemInput, options);
      const baseParams = base.itemParams!;
      const modelParams = byModel.itemParams!;
      const itemParams = byItem.itemParams!;

      expect(byModel.ranking).toEqual(
        modelOrder.map((model) => base.ranking[model]!),
      );
      expect(byItem.ranking).toEqual(base.ranking);
      expectNumbersClose(
        byModel.scores,
        modelOrder.map((model) => base.scores[model]!),
      );
      expectNumbersClose(byItem.scores, base.scores);
      expectRowsClose(
        modelParams.abilities,
        modelOrder.map((model) => baseParams.abilities[model]!),
      );
      expectRowsClose(
        modelParams.ability_sd,
        modelOrder.map((model) => baseParams.ability_sd[model]!),
      );
      expectRowsClose(itemParams.abilities, baseParams.abilities);
      expectRowsClose(itemParams.ability_sd, baseParams.ability_sd);

      expectNumbersClose(modelParams.difficulty, baseParams.difficulty);
      expectNumbersClose(
        itemParams.difficulty,
        itemOrder.map((item) => baseParams.difficulty[item]!),
      );
      expectNumbersClose(modelParams.discrimination, baseParams.discrimination);
      expectNumbersClose(
        itemParams.discrimination,
        itemOrder.map((item) => baseParams.discrimination[item]!),
      );
      expectRowsClose(modelParams.slopes, baseParams.slopes);
      expectRowsClose(
        itemParams.slopes,
        itemOrder.map((item) => baseParams.slopes[item]!),
      );
      expectNumbersClose(modelParams.intercept, baseParams.intercept);
      expectNumbersClose(
        itemParams.intercept,
        itemOrder.map((item) => baseParams.intercept[item]!),
      );
      if (baseParams.guessing !== undefined) {
        expectNumbersClose(modelParams.guessing!, baseParams.guessing);
        expectNumbersClose(
          itemParams.guessing!,
          itemOrder.map((item) => baseParams.guessing![item]!),
        );
      }
    }
  });

  it("validates options", () => {
    // @ts-expect-error invalid model
    expect(() => mirt(R, { model: "4pl" })).toThrow(/model must be/);
    expect(() => mirt(R, { model: "2pl", fixGuessing: 0.2 })).toThrow(/fixGuessing is only valid/);
    expect(() => mirt(R, { nFactors: 12, nQuadrature: 15 })).toThrow(/Product quadrature grid/);
    expect(() => mirt(R, { nFactors: 13, nQuadrature: 2 })).toThrow(/cannot exceed number of questions/);
    expect(
      mirt(ORDERED, {
        nFactors: 2,
        model: " 2PL " as "2pl",
        ...BASE,
      }),
    ).toEqual(mirt(ORDERED, { nFactors: 2, model: "2pl", ...BASE }));
    expect(() =>
      mirt(ORDERED, {
        nFactors: 2,
        model: "4pl" as "2pl",
        guessingUpper: 0,
        ...BASE,
      }),
    ).toThrow("guessing_upper must be in (0, 1) and finite");
    expect(() =>
      mirt(ORDERED, {
        nFactors: 2,
        model: "2pl",
        fixGuessing: 0.2,
        guessingUpper: 0,
        ...BASE,
      }),
    ).toThrow("guessing_upper must be in (0, 1) and finite");
  });
});
