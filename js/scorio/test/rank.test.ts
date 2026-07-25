import { describe, expect, it } from "vitest";

import * as rank from "../src/rank/index.js";
import type { RankResult, TensorInput } from "../src/rank/index.js";
import fixtures from "./fixtures/rank.json";

interface Fixture {
  shared_inputs: { R0_shared: number[][]; distinct_R: number[][][] };
  cases: {
    name: string;
    input: number[][] | number[][][];
    ranking?: number[];
    scores?: number[];
    kind: "exact" | "loose" | "structural" | "error";
    error?: string;
  }[];
  option_cases: { name: string; ranking: number[]; scores: number[] }[];
}

const fx = fixtures as unknown as Fixture;

const R0_SHARED = fx.shared_inputs.R0_shared;

// Map a fixture method name (suffix stripped) to a call on the input tensor.
const METHODS: Record<string, (R: TensorInput) => RankResult> = {
  avg: (R) => rank.avg(R),
  bayes: (R) => rank.bayes(R, { w: [0, 0.5, 1], R0: R0_SHARED }),
  pass_at_k: (R) => rank.passAtK(R, 2),
  pass_hat_k: (R) => rank.passHatK(R, 2),
  g_pass_at_k_tau: (R) => rank.gPassAtKTau(R, 2, 0.7),
  mg_pass_at_k: (R) => rank.mgPassAtK(R, 2),
  inverse_difficulty: (R) => rank.inverseDifficulty(R),
  elo: (R) => rank.elo(R),
  glicko: (R) => rank.glicko(R),
  trueskill: (R) => rank.trueskill(R),
  bradley_terry: (R) => rank.bradleyTerry(R, { maxIter: 80 }),
  bradley_terry_map: (R) => rank.bradleyTerryMap(R, { prior: 1, maxIter: 80 }),
  bradley_terry_davidson: (R) => rank.bradleyTerryDavidson(R, { maxIter: 80 }),
  bradley_terry_davidson_map: (R) =>
    rank.bradleyTerryDavidsonMap(R, { prior: 1, maxIter: 80 }),
  rao_kupper: (R) => rank.raoKupper(R, { tieStrength: 1.1, maxIter: 80 }),
  rao_kupper_map: (R) =>
    rank.raoKupperMap(R, { tieStrength: 1.1, prior: 1, maxIter: 80 }),
  thompson: (R) => rank.thompson(R, { nSamples: 700, seed: 7 }),
  bayesian_mcmc: (R) => rank.bayesianMcmc(R, { nSamples: 400, burnin: 100, seed: 7 }),
  borda: (R) => rank.borda(R),
  copeland: (R) => rank.copeland(R),
  win_rate: (R) => rank.winRate(R),
  minimax: (R) => rank.minimax(R),
  schulze: (R) => rank.schulze(R),
  ranked_pairs: (R) => rank.rankedPairs(R),
  kemeny_young: (R) => rank.kemenyYoung(R),
  nanson: (R) => rank.nanson(R),
  baldwin: (R) => rank.baldwin(R),
  majority_judgment: (R) => rank.majorityJudgment(R),
  rasch: (R) => rank.rasch(R, { maxIter: 60 }),
  rasch_map: (R) => rank.raschMap(R, { prior: 1, maxIter: 60 }),
  rasch_2pl: (R) => rank.rasch2pl(R, { maxIter: 60 }),
  rasch_2pl_map: (R) => rank.rasch2plMap(R, { prior: 1, maxIter: 60 }),
  rasch_3pl: (R) => rank.rasch3pl(R, { maxIter: 50, fixGuessing: 0.2 }),
  rasch_3pl_map: (R) => rank.rasch3plMap(R, { prior: 1, maxIter: 50, fixGuessing: 0.2 }),
  rasch_mml: (R) => rank.raschMml(R, { maxIter: 10, emIter: 6, nQuadrature: 9 }),
  rasch_mml_credible: (R) =>
    rank.raschMmlCredible(R, { quantile: 0.1, maxIter: 10, emIter: 6, nQuadrature: 9 }),
  dynamic_irt: (R) => rank.dynamicIrt(R, { variant: "linear", maxIter: 60 }),
  pagerank: (R) => rank.pagerank(R),
  spectral: (R) => rank.spectral(R),
  alpharank: (R) => rank.alpharank(R, { populationSize: 20, maxIter: 10000 }),
  nash: (R) => rank.nash(R),
  rank_centrality: (R) => rank.rankCentrality(R),
  serial_rank: (R) => rank.serialRank(R),
  hodge_rank: (R) => rank.hodgeRank(R),
  plackett_luce: (R) => rank.plackettLuce(R, { maxIter: 80 }),
  plackett_luce_map: (R) => rank.plackettLuceMap(R, { prior: 1, maxIter: 80 }),
  davidson_luce: (R) => rank.davidsonLuce(R, { maxIter: 80 }),
  davidson_luce_map: (R) => rank.davidsonLuceMap(R, { prior: 1, maxIter: 80 }),
  bradley_terry_luce: (R) => rank.bradleyTerryLuce(R, { maxIter: 80 }),
  bradley_terry_luce_map: (R) => rank.bradleyTerryLuceMap(R, { prior: 1, maxIter: 80 }),
};

function allClose(a: number, b: number, rtol: number, atol: number): boolean {
  return Math.abs(a - b) <= atol + rtol * Math.abs(b);
}

function assertValidRanking(ranking: number[], L: number): void {
  expect(ranking).toHaveLength(L);
  expect(ranking.every((r) => Number.isFinite(r))).toBe(true);
  expect(Math.min(...ranking)).toBeCloseTo(1, 10);
  expect(ranking.every((r) => r >= 1 && r <= L)).toBe(true);
}

/**
 * Assert two rankings agree on every pair of models the reference scored as
 * *genuinely* different. Pairs whose reference scores are within tolerance are
 * treated as tied and skipped, since the order within a tie is broken by
 * implementation-specific floating-point noise (e.g. Rasch ability depends only
 * on a model's raw total score, so equal-total models are exactly tied).
 */
function assertRankingAgreesUpToTies(
  mine: number[],
  refRanking: number[],
  refScores: number[],
): void {
  const L = mine.length;
  for (let i = 0; i < L; i++) {
    for (let j = i + 1; j < L; j++) {
      const tied =
        Math.abs(refScores[i]! - refScores[j]!) <=
        1e-3 + 1e-2 * Math.max(Math.abs(refScores[i]!), Math.abs(refScores[j]!));
      if (tied) continue;
      expect(
        Math.sign(mine[i]! - mine[j]!),
        `pair (${i},${j}): mine=${mine} ref=${refRanking}`,
      ).toBe(Math.sign(refRanking[i]! - refRanking[j]!));
    }
  }
}

function assertRankingScoreConsistency(ranking: number[], scores: number[]): void {
  const L = ranking.length;
  const eps = 1e-9;
  for (let i = 0; i < L; i++) {
    for (let j = 0; j < L; j++) {
      if (scores[i]! > scores[j]! + eps) expect(ranking[i]!).toBeLessThanOrEqual(ranking[j]! + eps);
      else if (scores[i]! < scores[j]! - eps) expect(ranking[i]!).toBeGreaterThanOrEqual(ranking[j]! - eps);
    }
  }
}

describe("rank fixtures vs Python reference", () => {
  for (const c of fx.cases) {
    const key = c.name.split("@")[0]!;
    const dataset = c.name.split("@")[1]!;
    const fn = METHODS[key];
    it(c.name, () => {
      expect(fn, `no dispatcher for ${key}`).toBeDefined();
      if (c.kind === "error") {
        expect(() => fn!(c.input as TensorInput)).toThrow();
        return;
      }
      const { ranking, scores } = fn!(c.input as TensorInput);
      const referenceRanking = c.ranking!;
      const referenceScores = c.scores!;
      const L = referenceRanking.length;

      assertValidRanking(ranking, L);
      assertRankingScoreConsistency(ranking, scores);

      if (c.kind === "exact") {
        expect(ranking).toEqual(referenceRanking);
        for (let i = 0; i < L; i++) {
          expect(
            allClose(scores[i]!, referenceScores[i]!, 1e-6, 1e-8),
            `scores[${i}] ${scores[i]} vs ${referenceScores[i]}`,
          ).toBe(true);
        }
      } else if (c.kind === "loose") {
        // Optimizer-based: ranking must agree with the reference on every
        // genuinely-separated pair (score magnitudes are not compared, and ties
        // broken by optimizer noise are ignored).
        assertRankingAgreesUpToTies(ranking, referenceRanking, referenceScores);
      } else {
        // structural: stochastic / non-unique. On the strongly separated D1
        // dataset all methods still agree on the ranking.
        if (dataset === "D1") expect(ranking).toEqual(referenceRanking);
      }
    });
  }
});

describe("rank Python runtime coercion parity", () => {
  const runtimeR = [
    [1, 1, 0, 1],
    [1, 0, 1, 0],
    [0, 0, 1, 0],
  ];

  it("coerces pairwise float and string options like Python", () => {
    expect(
      rank.elo(runtimeR, {
        K: "1_6" as never,
        initialRating: "1400" as never,
      }),
    ).toEqual(rank.elo(runtimeR, { K: 16, initialRating: 1400 }));

    expect(
      rank.trueskill(runtimeR, {
        muInitial: "25" as never,
        sigmaInitial: "8" as never,
        beta: true as never,
        tau: false as never,
        drawMargin: "0" as never,
      }),
    ).toEqual(
      rank.trueskill(runtimeR, {
        muInitial: 25,
        sigmaInitial: 8,
        beta: 1,
        tau: 0,
        drawMargin: 0,
      }),
    );

    expect(
      rank.glicko(runtimeR, {
        initialRating: "1500" as never,
        initialRd: "350" as never,
        rdMax: "350" as never,
        c: true as never,
      }),
    ).toEqual(
      rank.glicko(runtimeR, {
        initialRating: 1500,
        initialRd: 350,
        rdMax: 350,
        c: 1,
      }),
    );
  });

  it("preserves pairwise explicit-null and validation-order behavior", () => {
    expect(() => rank.trueskill(runtimeR, { tau: null as never })).toThrow(
      "tau must be a nonnegative finite scalar",
    );
    expect(() => rank.glicko(runtimeR, { initialRd: null as never })).toThrow(
      "initial_rd must be > 0 and finite",
    );
    expect(() =>
      rank.elo([[1]] as never, { K: null as never }),
    ).toThrow("Need at least 2 models");
  });

  it("coerces Bayesian float parameters while retaining strict integer options", () => {
    expect(
      rank.thompson(runtimeR, {
        nSamples: 30,
        priorAlpha: "2" as never,
        priorBeta: true as never,
        seed: true as never,
      }),
    ).toEqual(
      rank.thompson(runtimeR, {
        nSamples: 30,
        priorAlpha: 2,
        priorBeta: 1,
        seed: 1,
      }),
    );

    expect(
      rank.bayesianMcmc(runtimeR, {
        nSamples: 20,
        burnin: 5,
        priorVar: "2" as never,
        seed: false as never,
      }),
    ).toEqual(
      rank.bayesianMcmc(runtimeR, {
        nSamples: 20,
        burnin: 5,
        priorVar: 2,
        seed: 0,
      }),
    );

    expect(() => rank.thompson(runtimeR, { nSamples: true as never })).toThrow(
      "n_samples must be an integer",
    );
    expect(() => rank.thompson(runtimeR, { seed: "42" as never })).toThrow(
      "seed must be a nonnegative integer or null",
    );
    expect(() => rank.thompson(runtimeR, { priorAlpha: null as never })).toThrow(
      "prior_alpha must be > 0 and finite",
    );
    expect(() =>
      rank.thompson([[1]] as never, { nSamples: null as never }),
    ).toThrow("Need at least 2 models");

    // NumPy accepts None as a request for fresh seed entropy.
    expect(
      rank.thompson(runtimeR, { nSamples: 2, seed: null as never }).ranking,
    ).toHaveLength(3);
  });

  it("coerces graph scalars and teleport vectors through Python float semantics", () => {
    expect(
      rank.pagerank(runtimeR, {
        damping: "8_5e-2" as never,
        tol: "1e-6" as never,
        teleport: ["1", false, true] as never,
      }),
    ).toEqual(
      rank.pagerank(runtimeR, {
        damping: 0.85,
        tol: 1e-6,
        teleport: [1, 0, 1],
      }),
    );

    expect(
      rank.alpharank(runtimeR, {
        alpha: false as never,
        populationSize: 2,
        maxIter: 10,
        tol: true as never,
      }),
    ).toEqual(
      rank.alpharank(runtimeR, {
        alpha: 0,
        populationSize: 2,
        maxIter: 10,
        tol: 1,
      }),
    );
    expect(rank.nash(runtimeR, { temperature: true as never })).toEqual(
      rank.nash(runtimeR, { temperature: 1 }),
    );

    expect(() =>
      rank.pagerank(runtimeR, { damping: "0x1" as never }),
    ).toThrow("damping must be in (0, 1)");
    expect(() =>
      rank.pagerank(runtimeR, { damping: null as never }),
    ).toThrow("damping must be in (0, 1)");
    expect(() =>
      rank.spectral(runtimeR, { maxIter: null as never }),
    ).toThrow("max_iter must be an integer");
  });

  it("matches Nash's split validation order around tensor validation", () => {
    expect(() =>
      rank.nash([[1]] as never, { nIter: null as never, solver: "bad" }),
    ).toThrow("n_iter must be an integer");
    expect(() => rank.nash([[1]] as never, { solver: "bad" })).toThrow(
      "Need at least 2 models",
    );
  });

  it("matches Rank Centrality's float coercion and bool-as-int exception", () => {
    expect(
      rank.rankCentrality(runtimeR, {
        smoothing: "0.5" as never,
        teleport: false as never,
        maxIter: true as never,
        tol: true as never,
      }),
    ).toEqual(
      rank.rankCentrality(runtimeR, {
        smoothing: 0.5,
        teleport: 0,
        maxIter: 1,
        tol: 1,
      }),
    );

    expect(() =>
      rank.rankCentrality(runtimeR, { maxIter: false as never }),
    ).toThrow("max_iter must be >= 1");
    expect(() =>
      rank.rankCentrality(runtimeR, { smoothing: null as never }),
    ).toThrow("smoothing must be >= 0");
    expect(() =>
      rank.rankCentrality(runtimeR, { tieHandling: null as never }),
    ).toThrow('tie_handling must be "ignore" or "half"');
    expect(() =>
      rank.rankCentrality([[1]] as never, { smoothing: null as never }),
    ).toThrow("Need at least 2 models");
  });

  it("coerces Hodge log-odds epsilon only when that option is used", () => {
    expect(
      rank.hodgeRank(runtimeR, {
        pairwiseStat: "log_odds",
        epsilon: "0.5" as never,
      }),
    ).toEqual(
      rank.hodgeRank(runtimeR, {
        pairwiseStat: "log_odds",
        epsilon: 0.5,
      }),
    );
    expect(
      rank.hodgeRank(runtimeR, {
        pairwiseStat: "log_odds",
        epsilon: true as never,
      }),
    ).toEqual(
      rank.hodgeRank(runtimeR, { pairwiseStat: "log_odds", epsilon: 1 }),
    );

    expect(rank.hodgeRank(runtimeR, { epsilon: null as never })).toEqual(
      rank.hodgeRank(runtimeR),
    );
    expect(() =>
      rank.hodgeRank(runtimeR, {
        pairwiseStat: "log_odds",
        epsilon: null as never,
      }),
    ).toThrow("epsilon must be > 0 for log-odds smoothing");
    expect(() =>
      rank.hodgeRank([[1]] as never, { pairwiseStat: null as never }),
    ).toThrow("Need at least 2 models");
  });

  it("uses Python truthiness for optional extra-result flags", () => {
    expect(
      rank.glicko(runtimeR, { returnDeviation: "yes" as never }).deviation,
    ).toBeDefined();
    expect(
      rank.nash(runtimeR, { returnEquilibrium: "yes" as never }).equilibrium,
    ).toBeDefined();
    expect(
      rank.hodgeRank(runtimeR, { returnDiagnostics: "yes" as never })
        .diagnostics,
    ).toBeDefined();
    expect(
      rank.glicko(runtimeR, { returnDeviation: Number.NaN as never }).deviation,
    ).toBeDefined();
    expect(
      rank.nash(runtimeR, { returnEquilibrium: [] as never }).equilibrium,
    ).toBeUndefined();
    expect(
      rank.hodgeRank(runtimeR, { returnDiagnostics: "" as never })
        .diagnostics,
    ).toBeUndefined();
  });

  it("does not treat explicit null methods as omitted", () => {
    const calls = [
      () => rank.elo(runtimeR, { method: null as never }),
      () =>
        rank.thompson(runtimeR, { nSamples: 2, method: null as never }),
      () => rank.pagerank(runtimeR, { method: null as never }),
      () => rank.rankCentrality(runtimeR, { method: null as never }),
      () => rank.hodgeRank(runtimeR, { method: null as never }),
    ];
    for (const call of calls) expect(call).toThrow("method must be one of");
  });
});

describe("rank non-default option paths vs Python reference", () => {
  const D = fx.shared_inputs.distinct_R;
  // Each option case is a deterministic non-default call; ranking and scores
  // must both match the Python reference exactly.
  const OPTIONS: Record<string, () => RankResult> = {
    borda_dense: () => rank.borda(D, { method: "dense" }),
    minimax_wv: () => rank.minimax(D, { variant: "winning_votes" }),
    schulze_ignore: () => rank.schulze(D, { tiePolicy: "ignore" }),
    nash_eq: () => rank.nash(D, { scoreType: "equilibrium" }),
    nash_adv: () => rank.nash(D, { scoreType: "advantage_vs_equilibrium" }),
    glicko_c30: () => rank.glicko(D, { c: 30 }),
    elo_draw_k16: () => rank.elo(D, { tieHandling: "draw", K: 16 }),
    trueskill_draw: () => rank.trueskill(D, { tieHandling: "draw", drawMargin: 0.1 }),
    hodge_decisive: () => rank.hodgeRank(D, { weightMethod: "decisive" }),
    hodge_logodds: () => rank.hodgeRank(D, { pairwiseStat: "log_odds" }),
    rc_teleport: () => rank.rankCentrality(D, { tieHandling: "half", teleport: 0.1 }),
    rc_ignore_smooth: () => rank.rankCentrality(D, { tieHandling: "ignore", smoothing: 1 }),
    serial_sign: () => rank.serialRank(D, { comparison: "sign" }),
    bayes_q05_dense: () => rank.bayes(D, { quantile: 0.05, method: "dense" }),
    kemeny_not_tieaware: () => rank.kemenyYoung(D, { tieAware: false }),
    rp_wv: () => rank.rankedPairs(D, { strength: "winning_votes" }),
    nanson_min: () => rank.nanson(D, { rankTies: "min" }),
    pagerank_damp: () => rank.pagerank(D, { damping: 0.6 }),
    bt_dense: () => rank.bradleyTerry(D, { method: "dense" }),
    pl_avg: () => rank.plackettLuce(D, { method: "avg" }),
  };
  // BT/PL scores are optimizer-sensitive: only their ranking is checked.
  const RANKING_ONLY = new Set(["bt_dense", "pl_avg"]);

  for (const c of fx.option_cases) {
    it(c.name, () => {
      const build = OPTIONS[c.name];
      expect(build, `no dispatcher for ${c.name}`).toBeDefined();
      const { ranking, scores } = build!();
      assertValidRanking(ranking, c.ranking.length);
      assertRankingScoreConsistency(ranking, scores);
      if (RANKING_ONLY.has(c.name)) {
        assertRankingAgreesUpToTies(ranking, c.ranking, c.scores);
      } else {
        expect(ranking).toEqual(c.ranking);
        for (let i = 0; i < scores.length; i++) {
          expect(
            allClose(scores[i]!, c.scores[i]!, 1e-6, 1e-8),
            `scores[${i}] ${scores[i]} vs ${c.scores[i]}`,
          ).toBe(true);
        }
      }
    });
  }
});

function expectArrayClose(
  actual: readonly number[],
  expected: readonly number[],
  digits = 10,
): void {
  expect(actual).toHaveLength(expected.length);
  for (let index = 0; index < expected.length; index++) {
    expect(actual[index]).toBeCloseTo(expected[index]!, digits);
  }
}

describe("rank contracts from the current repository Python implementation", () => {
  it("applies all Elo matches from a common event snapshot", () => {
    const R = [[[1]], [[1]], [[0]]];
    const result = rank.elo(R);
    expect(result.ranking).toEqual([1, 1, 3]);
    expectArrayClose(result.scores, [1516, 1516, 1468], 12);

    const permutation = [2, 0, 1];
    const permuted = rank.elo(permutation.map((index) => R[index]!));
    const restored = new Array<number>(3);
    permutation.forEach((original, current) => {
      restored[original] = permuted.scores[current]!;
    });
    expectArrayClose(restored, result.scores, 12);
  });

  it("combines TrueSkill sites simultaneously and handles zero-width draws", () => {
    const result = rank.trueskill([[[1]], [[1]], [[0]]]);
    expect(result.ranking).toEqual([1, 1, 3]);
    expectArrayClose(
      result.scores,
      [29.2052208700336, 29.2052208700336, 18.296572145785756],
      10,
    );

    const decisive = rank.trueskill([[1], [0]], {
      tieHandling: "draw",
      drawMargin: 0,
      tau: 0,
    });
    const draw = rank.trueskill([[1, 1], [0, 1]], {
      tieHandling: "draw",
      drawMargin: 0,
      tau: 0,
    });
    expect(Math.abs(draw.scores[0]! - draw.scores[1]!)).toBeLessThan(
      Math.abs(decisive.scores[0]! - decisive.scores[1]!),
    );
  });

  it("keeps TrueSkill corrections finite in extreme normal tails", () => {
    const sigma = 25 / 3;
    const beta = 25 / 6;
    const c = Math.sqrt(2 * beta ** 2 + 2 * sigma ** 2);
    const scale = 1e10;
    const decisive = rank.trueskill([[1], [0]], {
      drawMargin: c * scale,
      tau: 0,
    });
    const expectedShift = (sigma ** 2 / c) * scale;
    expect(decisive.scores.every(Number.isFinite)).toBe(true);
    expect((decisive.scores[0]! - 25) / expectedShift).toBeCloseTo(1, 8);
    expect((25 - decisive.scores[1]!) / expectedShift).toBeCloseTo(1, 8);
    expect(
      rank.trueskill([[1], [1]], {
        tieHandling: "draw",
        drawMargin: c * scale,
        tau: 0,
      }).scores,
    ).toEqual([25, 25]);
  });

  it("returns Glicko deviations on request and validates finite controls", () => {
    const result = rank.glicko([[1, 1, 1], [0, 0, 1]], {
      returnDeviation: true,
    });
    expect(result.ranking).toEqual([1, 2]);
    expectArrayClose(result.scores, [1621.4601323625304, 1378.5398676374698], 9);
    expectArrayClose(result.deviation!, [243.23157217833932, 243.23157217833935], 9);
    expect(() => rank.glicko([[1], [0]], { c: Number.NaN })).toThrow(
      "c must be >= 0 and finite",
    );
    expect(() => rank.glicko([[1], [0]], { rdMax: Infinity })).toThrow(
      "rd_max must be > 0 and finite",
    );
  });

  it("uses Python's PageRank and Keener spectral operators", () => {
    const R = [[1, 1, 1], [0, 0, 1], [0, 0, 0]];
    const page = rank.pagerank(R);
    expect(page.ranking).toEqual([1, 2, 3]);
    expectArrayClose(
      page.scores,
      [0.7140481572914308, 0.19544196554873383, 0.0905098771598356],
      10,
    );
    const spectral = rank.spectral(R);
    expect(spectral.ranking).toEqual([1, 2, 3]);
    expectArrayClose(spectral.scores, [19 / 42, 13 / 42, 10 / 42], 10);
  });

  it("returns the label-invariant Nash equilibrium over accuracy maximizers", () => {
    const R = [[1, 1, 1, 0], [1, 1, 0, 1], [0, 0, 0, 1]];
    const result = rank.nash(R, {
      scoreType: "equilibrium",
      returnEquilibrium: true,
    });
    expect(result.ranking).toEqual([1, 1, 3]);
    expect(result.scores).toEqual([0.5, 0.5, 0]);
    expect(result.equilibrium).toEqual([0.5, 0.5, 0]);
    expect(() => rank.nash(R, { solver: "bad" })).toThrow('solver must be "lp"');
    expect(() => rank.nash(R, { temperature: 0 })).toThrow(
      "temperature must be a positive finite scalar",
    );
  });

  it("requires directed support for unregularized decisive Rank Centrality", () => {
    const oneWayChain = [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 1]];
    expect(() => rank.rankCentrality(oneWayChain, { tieHandling: "ignore" })).toThrow(
      "strongly connected",
    );
    expect(() =>
      rank.rankCentrality([[0, 0], [0, 0], [0, 0]], { tieHandling: "ignore" }),
    ).toThrow("strongly connected");
    expect(
      rank.rankCentrality(oneWayChain, { tieHandling: "ignore", smoothing: 1e-3 })
        .ranking,
    ).toEqual([3, 2, 1]);
    expect(
      rank.rankCentrality(oneWayChain, { tieHandling: "ignore", teleport: 0.05 })
        .ranking,
    ).toEqual([3, 2, 1]);
  });

  it("provides HodgeRank residual diagnostics", () => {
    const result = rank.hodgeRank([[1, 1, 1], [0, 0, 1], [0, 0, 0]], {
      returnDiagnostics: true,
    });
    expect(result.ranking).toEqual([1, 2, 3]);
    expectArrayClose(result.scores, [5 / 9, -1 / 9, -4 / 9], 10);
    expect(Number.isFinite(result.diagnostics!.residualL2)).toBe(true);
    expect(Number.isFinite(result.diagnostics!.relativeResidualL2)).toBe(true);
  });

  it("uses Nanson's at-or-below-mean elimination and validates Kemeny limits", () => {
    const grades = [
      [3, 2, 0, 2, 1],
      [0, 1, 3, 0, 0],
      [2, 0, 2, 1, 3],
      [1, 3, 1, 3, 2],
    ];
    const R = grades.map((model) =>
      model.map((grade) => Array.from({ length: 3 }, (_, trial) => (trial < grade ? 1 : 0))),
    );
    const result = rank.nanson(R);
    expect(result.scores).toEqual([1, 0, 1, 2]);
    expect(result.ranking).toEqual([2, 4, 2, 1]);
    expect(() => rank.kemenyYoung([[1], [0]], { timeLimit: 0 })).toThrow(
      "time_limit must be a positive finite scalar",
    );
  });

  it("matches Python's explicit-null option semantics", () => {
    const R = [[1, 1], [0, 1]];

    expect(() =>
      rank.elo(R, { K: null as unknown as number }),
    ).toThrow("K must be a positive finite scalar");
    expect(() =>
      rank.minimax(R, {
        variant: null as unknown as "margin",
      }),
    ).toThrow("variant must be one of");
    expect(() =>
      rank.hodgeRank(R, {
        pairwiseStat: null as unknown as "binary",
      }),
    ).toThrow('pairwise_stat must be one of: "binary", "log_odds"');

    expect(rank.pagerank(R, { teleport: null })).toEqual(rank.pagerank(R));
    expect(rank.kemenyYoung(R, { timeLimit: null })).toEqual(
      rank.kemenyYoung(R),
    );

    const equal = [[0, 0], [0, 0], [0, 0]];
    expect(rank.kemenyYoung(equal, { tieAware: null })).toEqual(
      rank.kemenyYoung(equal, { tieAware: false }),
    );
  });

  it("returns a deterministic optimal Kemeny order when the optimum is non-unique", () => {
    const equal = [[0, 0], [0, 0], [0, 0]];
    const result = rank.kemenyYoung(equal, { tieAware: false });

    // Every pairwise preference count is one, so every total order has the
    // exact optimum objective 3. The selected labelled order is deliberately
    // deterministic but need not be the same vertex chosen by SciPy/HiGHS.
    expect(result.ranking).toEqual([1, 2, 3]);
    expect(result.scores).toEqual([2, 1, 0]);
    const order = [0, 1, 2].sort(
      (left, right) => result.scores[right]! - result.scores[left]!,
    );
    let selectedObjective = 0;
    for (let first = 0; first < order.length; first++) {
      for (let second = first + 1; second < order.length; second++) {
        const higher = order[first]!;
        const lower = order[second]!;
        for (let question = 0; question < equal[0]!.length; question++) {
          selectedObjective +=
            equal[higher]![question]! === equal[lower]![question]! ? 0.5 : 0;
        }
      }
    }
    expect(selectedObjective).toBe(3);
  });
});

describe("IRT contracts from the current repository Python implementation", () => {
  const ordered = fx.cases.find((testCase) => testCase.name === "rasch_2pl@D1")!
    .input as number[][][];

  it("rejects ranking-ambiguous nonconvex 2PL fits", () => {
    const ambiguous = [
      [1, 1, 0, 0, 1, 0],
      [1, 0, 1, 0, 0, 1],
      [0, 1, 1, 0, 1, 0],
      [0, 0, 0, 1, 1, 1],
    ];
    expect(() => rank.rasch2pl(ambiguous, { maxIter: 500 })).toThrow(
      "multiple equally good nonconvex solutions",
    );
  });

  it("rejects infinite and quasi-separated joint IRT estimates", () => {
    const extremePerson = [
      [1, 1, 1, 1],
      [0, 1, 0, 1],
      [0, 0, 0, 0],
    ];
    const extremeItem = [
      [1, 1, 0],
      [1, 0, 1],
      [1, 0, 0],
    ];
    const quasiSeparated = [
      [0, 0, 0, 1],
      [0, 0, 1, 0],
      [0, 1, 1, 1],
      [1, 0, 1, 1],
    ];
    const mleCalls = [
      (R: TensorInput) => rank.rasch(R),
      (R: TensorInput) => rank.rasch2pl(R, { maxIter: 300 }),
      (R: TensorInput) =>
        rank.rasch3pl(R, { maxIter: 500, fixGuessing: 0.2 }),
    ];
    for (const call of mleCalls) {
      expect(() => call(extremePerson)).toThrow("no finite ability MLE");
      expect(() => call(extremeItem)).toThrow("no finite item-parameter estimate");
      expect(() => call(quasiSeparated)).toThrow("completely or quasi-separated");
    }

    const uniformCalls = [
      () => rank.raschMap(extremePerson, { prior: new rank.UniformPrior() }),
      () =>
        rank.rasch2plMap(extremePerson, {
          prior: new rank.UniformPrior(),
          maxIter: 300,
        }),
      () =>
        rank.rasch3plMap(extremePerson, {
          prior: new rank.UniformPrior(),
          maxIter: 500,
          fixGuessing: 0.2,
        }),
    ];
    for (const call of uniformCalls) {
      expect(call).toThrow("no finite ability MLE");
    }
  });

  it("is model- and item-permutation equivariant for nonconvex IRT", () => {
    const modelOrder = [2, 0, 3, 1];
    const itemOrder = [7, 2, 9, 0, 5, 1, 8, 4, 6, 3];
    const permutedModels = modelOrder.map((model) => ordered[model]!);
    const permutedItems = ordered.map((model) =>
      itemOrder.map((item) => model[item]!),
    );
    const calls = [
      (R: TensorInput) => rank.rasch2pl(R, { maxIter: 500 }),
      (R: TensorInput) => rank.rasch2plMap(R, { maxIter: 500 }),
      (R: TensorInput) =>
        rank.rasch3pl(R, { maxIter: 500, fixGuessing: 0.2 }),
      (R: TensorInput) =>
        rank.rasch3plMap(R, { maxIter: 500, fixGuessing: 0.2 }),
    ];
    for (const call of calls) {
      const base = call(ordered);
      const byModel = call(permutedModels);
      const byItem = call(permutedItems);
      expect(byModel.ranking).toEqual(modelOrder.map((model) => base.ranking[model]!));
      expect(byItem.ranking).toEqual(base.ranking);
      for (let index = 0; index < modelOrder.length; index++) {
        expect(
          Math.abs(byModel.scores[index]! - base.scores[modelOrder[index]!]!),
        ).toBeLessThanOrEqual(1e-6);
        expect(Math.abs(byItem.scores[index]! - base.scores[index]!)).toBeLessThanOrEqual(
          1e-6,
        );
      }
    }
  });

  it("keeps asymmetric 2PL scores invariant to item order", () => {
    const R = [
      [1, 0, 0, 0, 0, 1, 1],
      [0, 1, 1, 0, 0, 0, 0],
      [0, 0, 1, 0, 1, 0, 0],
      [1, 1, 1, 0, 0, 0, 0],
      [1, 0, 0, 1, 0, 0, 0],
    ];
    const itemOrder = [6, 2, 3, 0, 5, 1, 4];
    const base = rank.rasch2pl(R, { maxIter: 500 });
    const permuted = rank.rasch2pl(
      R.map((row) => itemOrder.map((item) => row[item]!)),
      { maxIter: 500 },
    );
    expect(permuted.ranking).toEqual(base.ranking);
    for (let index = 0; index < base.scores.length; index++) {
      expect(Math.abs(permuted.scores[index]! - base.scores[index]!)).toBeLessThanOrEqual(
        1e-5,
      );
    }
  });

  it("exposes Python-compatible MML item keys and extended boundary MLEs", () => {
    for (const [value, difficulty] of [
      [1, -Infinity],
      [0, Infinity],
    ] as const) {
      const result = rank.raschMml(
        Array.from({ length: 8 }, () => [[value, value, value]]),
        { maxIter: 5, emIter: 3, nQuadrature: 9, returnItemParams: true },
      );
      expect(result.ranking).toEqual(new Array<number>(8).fill(1));
      expect(result.scores.every((score) => Math.abs(score) < 1e-14)).toBe(true);
      expect(result.itemParams!.difficulty[0]).toBe(difficulty);
      expect(result.itemParams!.ability_sd).toHaveLength(8);
      expect(Object.keys(result.itemParams!).sort()).toEqual([
        "ability_sd",
        "difficulty",
      ]);
    }
  });

  it("returns fitted item parameters for joint and longitudinal IRT", () => {
    const rasch = rank.rasch(ordered, {
      maxIter: 500,
      returnItemParams: true,
    });
    expect(rasch.itemParams!.difficulty).toHaveLength(ordered[0]!.length);
    expect(Object.keys(rasch.itemParams!)).toEqual(["difficulty"]);

    const twoPl = rank.rasch2pl(ordered, {
      maxIter: 500,
      returnItemParams: true,
    });
    expect(twoPl.itemParams!.difficulty).toHaveLength(ordered[0]!.length);
    expect(twoPl.itemParams!.discrimination).toHaveLength(ordered[0]!.length);
    expect(Object.keys(twoPl.itemParams!).sort()).toEqual([
      "difficulty",
      "discrimination",
    ]);

    const threePl = rank.rasch3pl(ordered, {
      maxIter: 500,
      fixGuessing: 0.2,
      returnItemParams: true,
    });
    expect(threePl.itemParams!.difficulty).toHaveLength(ordered[0]!.length);
    expect(threePl.itemParams!.discrimination).toHaveLength(ordered[0]!.length);
    expect(threePl.itemParams!.guessing).toEqual(
      new Array<number>(ordered[0]!.length).fill(0.2),
    );
    expect(Object.keys(threePl.itemParams!).sort()).toEqual([
      "difficulty",
      "discrimination",
      "guessing",
    ]);

    const growth = rank.dynamicIrt(ordered, {
      variant: "growth",
      assumeTimeAxis: true,
      scoreTarget: "gain",
      timePoints: null,
      maxIter: 500,
      returnItemParams: true,
    });
    const omittedTime = rank.dynamicIrt(ordered, {
      variant: "growth",
      assumeTimeAxis: true,
      scoreTarget: "gain",
      maxIter: 500,
    });
    expectArrayClose(growth.scores, omittedTime.scores, 10);
    expect(growth.itemParams!.ability_path).toHaveLength(ordered.length);
    expect(growth.itemParams!.time_points).toEqual([0, 0.25, 0.5, 0.75, 1]);
    expectArrayClose(growth.scores, growth.itemParams!.slope!, 8);
    expect(Object.keys(growth.itemParams!).sort()).toEqual([
      "ability_path",
      "baseline",
      "difficulty",
      "slope",
      "time_points",
    ]);

    const state = rank.dynamicIrt(ordered, {
      variant: "state_space",
      assumeTimeAxis: true,
      scoreTarget: "gain",
      maxIter: 500,
      returnItemParams: true,
    });
    expect(state.itemParams!.ability_path).toHaveLength(ordered.length);
    expect(state.itemParams!.time_points).toEqual([0, 0.25, 0.5, 0.75, 1]);
    expectArrayClose(state.scores, state.itemParams!.gain!, 8);
    expect(Object.keys(state.itemParams!).sort()).toEqual([
      "ability_path",
      "difficulty",
      "gain",
      "time_points",
    ]);
  });

  it("matches Python's explicit-null and credible-MML option behavior", () => {
    const valid = [
      [1, 1, 0, 1],
      [1, 0, 1, 0],
      [0, 0, 1, 0],
    ];
    expect(() => rank.dynamicIrt(valid, { variant: null } as never)).toThrow(
      "Unknown variant",
    );
    expect(() => rank.dynamicIrt(valid, { scoreTarget: null } as never)).toThrow(
      "score_target must be",
    );
    expect(() => rank.rasch(valid, { maxIter: null } as never)).toThrow(
      "max_iter must be an integer",
    );
    expect(() => rank.raschMap(valid, { prior: null } as never)).toThrow(
      "prior must be",
    );
    expect(() =>
      rank.raschMmlCredible(valid, { quantile: "0.1" } as never),
    ).toThrow("quantile must be a finite scalar");
    expect(() =>
      rank.raschMmlCredible(valid, { returnItemParams: true } as never),
    ).toThrow("does not accept returnItemParams");
  });

  it("rejects unidentified longitudinal and MIRT settings", () => {
    const R = [
      [
        [1, 1],
        [1, 1],
      ],
      [
        [1, 0],
        [0, 1],
      ],
      [
        [0, 1],
        [1, 0],
      ],
    ];
    expect(() =>
      rank.dynamicIrt(R, { variant: "growth", assumeTimeAxis: true }),
    ).toThrow("no finite ability MLE");
    expect(() =>
      rank.dynamicIrt(R.slice(1), {
        variant: "growth",
        assumeTimeAxis: true,
        slopeReg: 0,
      }),
    ).toThrow("slope_reg must be positive");
    expect(() =>
      rank.dynamicIrt(R.slice(1), {
        variant: "state_space",
        assumeTimeAxis: true,
        stateReg: 0,
      }),
    ).toThrow("state_reg must be positive");
    expect(() =>
      rank.dynamicIrt(
        R.map((model) => model.map((item) => item.slice(0, 1))),
        { variant: "growth", assumeTimeAxis: true },
      ),
    ).toThrow("at least two time points");

    const boundaryBase = [
      [1, 1, 0],
      [1, 0, 1],
      [1, 0, 0],
    ];
    const boundary = boundaryBase.map((row) =>
      row.map((value) => [value, value]),
    );
    expect(() =>
      rank.dynamicIrt(boundary, {
        variant: "growth",
        assumeTimeAxis: true,
      }),
    ).toThrow("no finite item-parameter estimate");
    expect(() => rank.mirt(boundary, { nFactors: 1, nQuadrature: 7 })).toThrow(
      "no finite item-parameter estimate",
    );
  });
});

describe("probabilistic rank Python runtime coercion parity", () => {
  const symmetric = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ];
  const validIrt = [
    [1, 1, 0, 1],
    [1, 0, 1, 0],
    [0, 0, 1, 0],
  ];

  it("matches eval-ranking optional None and scalar coercion behavior", () => {
    expect(rank.bayes(symmetric, { quantile: "0.1" as never })).toEqual(
      rank.bayes(symmetric, { quantile: 0.1 }),
    );
    expect(rank.bayes(symmetric, { quantile: null })).toEqual(
      rank.bayes(symmetric),
    );
    expect(rank.bayes(symmetric, { w: null, R0: null })).toEqual(
      rank.bayes(symmetric),
    );
    expect(rank.passAtK(symmetric, true as never)).toEqual(
      rank.passAtK(symmetric, 1),
    );
    expect(rank.gPassAtKTau(symmetric, true as never, false as never)).toEqual(
      rank.gPassAtKTau(symmetric, 1, 0),
    );
    expect(() => rank.passAtK(symmetric, "1" as never)).toThrow();
    expect(() => rank.gPassAtKTau(symmetric, 1, "0.5" as never)).toThrow();
  });

  it("coerces Rao-Kupper floats but explicitly rejects boolean tie strength", () => {
    expect(
      rank.raoKupper(symmetric, { tieStrength: "1.1" as never }),
    ).toEqual(rank.raoKupper(symmetric, { tieStrength: 1.1 }));
    expect(() =>
      rank.raoKupper(symmetric, { tieStrength: true as never }),
    ).toThrow(TypeError);
    expect(() =>
      rank.raoKupper(symmetric, { tieStrength: null as never }),
    ).toThrow();
    expect(() =>
      rank.bradleyTerryMap(symmetric, { prior: true as never }),
    ).toThrow(TypeError);
  });

  it("matches listwise float, integer, prior, and semantic-null behavior", () => {
    expect(rank.plackettLuce(symmetric, { tol: "1e-8" as never })).toEqual(
      rank.plackettLuce(symmetric, { tol: 1e-8 }),
    );
    expect(rank.davidsonLuce(symmetric, { maxTieOrder: null })).toEqual(
      rank.davidsonLuce(symmetric),
    );
    expect(() =>
      rank.plackettLuce(symmetric, { maxIter: null as never }),
    ).toThrow(TypeError);
    expect(() =>
      rank.plackettLuceMap(symmetric, { prior: true as never }),
    ).toThrow(TypeError);
  });

  it("uses IRT's Python float coercion and bool-as-variance exception", () => {
    expect(rank.raschMap(validIrt, { prior: true as never })).toEqual(
      rank.raschMap(validIrt, { prior: 1 }),
    );
    expect(() => rank.raschMap(validIrt, { prior: false as never })).toThrow();
    expect(
      rank.rasch2plMap(validIrt, {
        regDiscrimination: "0.01" as never,
        maxIter: 500,
      }),
    ).toEqual(
      rank.rasch2plMap(validIrt, {
        regDiscrimination: 0.01,
        maxIter: 500,
      }),
    );

    const ordered = fx.cases.find(
      (testCase) => testCase.name === "rasch_2pl@D1",
    )!.input as number[][][];
    const numericTime = rank.dynamicIrt(ordered, {
      variant: "growth",
      assumeTimeAxis: true,
      scoreTarget: "gain",
      timePoints: [0, 0.25, 0.5, 0.75, 1],
      maxIter: 500,
    });
    const coercedTime = rank.dynamicIrt(ordered, {
      variant: "growth",
      assumeTimeAxis: "yes" as never,
      scoreTarget: "gain",
      timePoints: [false, "0.25", "0.5", "0.75", true] as never,
      maxIter: 500,
      returnItemParams: "yes" as never,
    });
    expect(coercedTime.scores).toEqual(numericTime.scores);
    expect(coercedTime.itemParams).toBeDefined();
  });

  it("preserves each Python family's validation precedence and null methods", () => {
    expect(() => rank.rasch([[1]] as never, { maxIter: null as never })).toThrow(
      "max_iter must be an integer",
    );
    expect(() =>
      rank.bradleyTerry([[1]] as never, { maxIter: null as never }),
    ).toThrow("Need at least 2 models");
    expect(() =>
      rank.plackettLuce([[1]] as never, { maxIter: null as never }),
    ).toThrow("Need at least 2 models");
    expect(() => rank.avg(symmetric, { method: null as never })).toThrow();
  });
});
