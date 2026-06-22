import { describe, expect, it } from "vitest";

import * as rank from "../src/rank/index.js";
import type { RankResult, TensorInput } from "../src/rank/index.js";
import fixtures from "./fixtures/rank.json";

interface Fixture {
  shared_inputs: { R0_shared: number[][]; distinct_R: number[][][] };
  cases: {
    name: string;
    input: number[][] | number[][][];
    ranking: number[];
    scores: number[];
    kind: "exact" | "loose" | "structural";
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
      const { ranking, scores } = fn!(c.input as TensorInput);
      const L = c.ranking.length;

      assertValidRanking(ranking, L);
      assertRankingScoreConsistency(ranking, scores);

      if (c.kind === "exact") {
        expect(ranking).toEqual(c.ranking);
        for (let i = 0; i < L; i++) {
          expect(
            allClose(scores[i]!, c.scores[i]!, 1e-6, 1e-8),
            `scores[${i}] ${scores[i]} vs ${c.scores[i]}`,
          ).toBe(true);
        }
      } else if (c.kind === "loose") {
        // Optimizer-based: ranking must agree with the reference on every
        // genuinely-separated pair (score magnitudes are not compared, and ties
        // broken by optimizer noise are ignored).
        assertRankingAgreesUpToTies(ranking, c.ranking, c.scores);
      } else {
        // structural: stochastic / non-unique. On the strongly separated D1
        // dataset all methods still agree on the ranking.
        if (dataset === "D1") expect(ranking).toEqual(c.ranking);
      }
    });
  }
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
