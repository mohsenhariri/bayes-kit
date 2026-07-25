import { describe, expect, it } from "vitest";

import * as scorio from "../src/index.js";
import * as rank from "../src/rank/index.js";
import * as utils from "../src/utils/index.js";
import fixtures from "./fixtures/utils.json";

interface RankScoreCase {
  name: string;
  scores: number[];
  kwargs: {
    tol?: number;
    sigmas_in_id_order?: number[];
    confidence?: number;
    ci_tie_method?: utils.CiTieMethod;
  };
  expected: utils.RankScoresResult;
}

interface ComparisonCase {
  name: string;
  a: number[];
  b: number[];
  expected: {
    kendalltau: [number | null, number | null];
    spearmanr: [number | null, number | null];
    weighted_kendalltau: [number | null, number | null];
    fraction_mismatched: number;
    max_disp: number;
  };
}

interface UtilsFixtures {
  public_api: { rank: string[]; utils: string[] };
  rank_scores: RankScoreCase[];
  comparisons: ComparisonCase[];
  combinatorial: {
    ordered_bell_17: string[];
    lehmer: { n: number; permutation: number[]; hash: string }[];
    weak_rankings: { n: number; hash: string; ranking: number[] }[];
    combinations: { n: number; k: number; indices: number[]; rank: string }[];
    large: { lehmer_reverse_n19: string; ranking_all_tied_n17: string };
    blocks: { ranks: number[]; tol: number; expected: number[][] }[];
  };
}

const fx = fixtures as unknown as UtilsFixtures;

function expectPythonFloat(actual: number, expected: number | null): void {
  if (expected === null) {
    expect(Number.isNaN(actual)).toBe(true);
    return;
  }
  // Python/SciPy and JS use the same formulas but can differ in their final
  // floating-point rounding. This threshold is near machine precision.
  expect(Math.abs(actual - expected)).toBeLessThanOrEqual(
    2e-14 * Math.max(1, Math.abs(expected)),
  );
}

describe("rankScores parity with Python", () => {
  for (const c of fx.rank_scores) {
    it(c.name, () => {
      const result = utils.rankScores(c.scores, {
        tol: c.kwargs.tol,
        sigmas: c.kwargs.sigmas_in_id_order,
        confidence: c.kwargs.confidence,
        ciTieMethod: c.kwargs.ci_tie_method,
      });
      expect(result).toEqual(c.expected);
    });
  }

  it("supports Python's positional argument order through the snake_case alias", () => {
    const result = utils.rank_scores(
      [10, 9.5, 5],
      1e-12,
      [1, 1, 0.1],
      0.95,
      "ci_overlap_adjacent",
    );
    expect(result).toEqual(fx.rank_scores.find((c) => c.name === "overlap_ties")!.expected);
  });

  it("matches scipy rankdata NaN propagation", () => {
    const result = utils.rankScores([1, NaN, 2], { sigmas: [0.1, 0.1, 0.1] });
    for (const ranks of Object.values(result)) {
      expect(ranks.every(Number.isNaN)).toBe(true);
    }
  });

  it("validates sigma length and tie method like Python", () => {
    expect(() => utils.rankScores([1, 2, 3], { sigmas: [0.1, 0.2] })).toThrow(
      /same length/,
    );
    expect(() =>
      utils.rankScores([1, 2], {
        sigmas: [0.1, 0.1],
        ciTieMethod: "bad" as utils.CiTieMethod,
      }),
    ).toThrow(/Unknown ci_tie_method/);
  });
});

describe("compareRankings parity with scipy", () => {
  for (const c of fx.comparisons) {
    it(c.name, () => {
      const result = utils.compareRankings(c.a, c.b) as utils.RankingComparison;
      for (const key of ["kendalltau", "spearmanr", "weighted_kendalltau"] as const) {
        expectPythonFloat(result[key][0], c.expected[key][0]);
        expectPythonFloat(result[key][1], c.expected[key][1]);
      }
      expect(result.fraction_mismatched).toBe(c.expected.fraction_mismatched);
      expect(result.max_disp).toBe(c.expected.max_disp);

      expect(utils.compareRankings(c.a, c.b, "kendall")).toEqual(result.kendalltau);
      expect(utils.compareRankings(c.a, c.b, "spearman")).toEqual(result.spearmanr);
      expect(utils.compareRankings(c.a, c.b, "weighted_kendall")).toEqual(
        result.weighted_kendalltau,
      );
    });
  }

  it("matches Python validation", () => {
    expect(() => utils.compareRankings([], [])).toThrow(/same non-zero length/);
    expect(() => utils.compareRankings([1, 2], [1])).toThrow(/same non-zero length/);
    expect(() => utils.compareRankings([1, NaN], [1, 2])).toThrow(/NaN or inf/);
    expect(() =>
      utils.compareRankings([1, 2], [1, 2], "bad" as never),
    ).toThrow(/method must be one of/);
  });
});

describe("combinatorial utilities parity with Python", () => {
  it("matches ordered Bell numbers through the bigint boundary", () => {
    expect(utils.orderedBell(17).map(String)).toEqual(fx.combinatorial.ordered_bell_17);
  });

  it("exhaustively matches Lehmer hashes and inverse hashes through n=6", () => {
    for (const c of fx.combinatorial.lehmer) {
      expect(String(utils.lehmerHash(c.permutation))).toBe(c.hash);
      expect(utils.lehmerUnhash(BigInt(c.hash), c.n)).toEqual(c.permutation);
    }
  });

  it("exhaustively matches weak-ranking hashes and inverse hashes through n=5", () => {
    for (const c of fx.combinatorial.weak_rankings) {
      expect(utils.unhashRanking(BigInt(c.hash), c.n)).toEqual(c.ranking);
      expect(String(utils.rankingHash(c.ranking))).toBe(c.hash);
    }
  });

  it("exhaustively matches combination rank/unrank through n=8", () => {
    for (const c of fx.combinatorial.combinations) {
      expect(String(utils.combRankLex(c.indices, c.n, c.k))).toBe(c.rank);
      expect(utils.combUnrankLex(BigInt(c.rank), c.n, c.k)).toEqual(c.indices);
    }
  });

  it("matches canonical tie blocks", () => {
    for (const c of fx.combinatorial.blocks) {
      expect(utils.blocksFromRankList(c.ranks, c.tol)).toEqual(c.expected);
    }
  });

  it("stays collision-free beyond Number.MAX_SAFE_INTEGER", () => {
    const reverse19 = Array.from({ length: 19 }, (_, i) => 18 - i);
    const lehmer = utils.lehmerHash(reverse19);
    expect(typeof lehmer).toBe("bigint");
    expect(String(lehmer)).toBe(fx.combinatorial.large.lehmer_reverse_n19);
    expect(utils.lehmerUnhash(lehmer, 19)).toEqual(reverse19);

    const tied = utils.rankingHash(new Array<number>(17).fill(1));
    expect(typeof tied).toBe("bigint");
    expect(String(tied)).toBe(fx.combinatorial.large.ranking_all_tied_n17);
    expect(utils.unhashRanking(tied, 17)).toEqual(new Array<number>(17).fill(1));
  });

  it("matches Python validation errors", () => {
    expect(() => utils.lehmerHash([0.5, 1.5])).toThrow(/permutation of integers/);
    expect(() => utils.lehmerHash([0, 0, 1])).toThrow(/permutation of 0..n-1/);
    expect(() => utils.lehmerUnhash(6, 3)).toThrow(/must be in range/);
    expect(() => utils.unhashRanking(13, 3)).toThrow(/h out of range/);
    expect(() => utils.combUnrankLex(10, 4, 2)).toThrow(/out of range/);
  });
});

describe("utils public API", () => {
  it("exports every Python rank/utils __all__ runtime name", () => {
    for (const name of fx.public_api.rank) {
      expect(typeof rank[name as keyof typeof rank], `rank.${name}`).not.toBe("undefined");
    }
    for (const name of fx.public_api.utils) {
      expect(typeof utils[name as keyof typeof utils], `utils.${name}`).toBe("function");
    }
  });

  it("exports every public helper and the root utils namespace", () => {
    const publicHelpers = [
      "ordered_bell",
      "comb_rank_lex",
      "comb_unrank_lex",
      "blocks_from_rank_list",
    ];
    for (const name of publicHelpers) {
      expect(typeof utils[name as keyof typeof utils]).toBe("function");
    }
    expect(scorio.utils.rank_scores).toBe(utils.rank_scores);
  });
});
