import { describe, expect, it } from "vitest";

import * as sinf from "../src/sinf/index.js";
import fixtures from "./fixtures/sinf.json";

// `console` is available at runtime (Node/vitest); declare it since the tsconfig
// `lib` (ES2020, no DOM/node types) does not provide the type.
declare const console: { log(...args: unknown[]): void };

// The fixture JSON serializes non-finite floats as tokens so it stays valid
// JSON; `num` reverses that. Bulk arrays (CS bound paths) are always finite.
function num(v: unknown): number {
  if (v === "inf") return Infinity;
  if (v === "-inf") return -Infinity;
  if (v === "nan") return NaN;
  return v as number;
}

const fx = fixtures as unknown as {
  streams: {
    name: string;
    x: number[];
    confseq: Record<
      "betting" | "hoeffding" | "asymp",
      { lo: number[]; hi: number[]; final: { lo: number; hi: number } }
    >;
    fixed_ci: { lo: number[]; hi: number[] };
  }[];
  panel_single: any;
  panel_pair_indep: any;
  panel_pair_decisive: any;
  tensor: any;
  tensor_sep: any;
  select_best: any;
  votes: any[];
  counts_from_answers: { answers: (string | null)[]; labels: string[]; counts: number[] };
  legacy: any;
};

// Largest absolute discrepancy observed on any CS bound, reported at the end.
let maxCsDiff = 0;
const RTOL = 1e-6;
const ATOL = 1e-6;

function close(a: number, b: unknown, label: string, trackCs = false): void {
  const bb = num(b);
  if (!Number.isFinite(bb) || !Number.isFinite(a)) {
    // Non-finite (e.g. z = Infinity, or NaN): require exact identity.
    expect(Object.is(a, bb), `${label}: ${a} vs ${bb}`).toBe(true);
    return;
  }
  const diff = Math.abs(a - bb);
  if (trackCs && diff > maxCsDiff) maxCsDiff = diff;
  expect(diff <= ATOL + RTOL * Math.abs(bb), `${label}: ${a} vs ${bb} (|d|=${diff})`).toBe(true);
}

function closeArr(a: number[], b: unknown[], label: string, trackCs = false): void {
  expect(a.length, `${label} length`).toBe(b.length);
  for (let i = 0; i < b.length; i++) close(a[i]!, b[i], `${label}[${i}]`, trackCs);
}

// --------------------------------------------------------------------------- #
// Confidence sequences on bounded streams (the numerical crux).
// --------------------------------------------------------------------------- #
describe("confseq bounds vs Python reference (<=1e-6)", () => {
  for (const s of fx.streams) {
    for (const method of ["betting", "hoeffding", "asymp"] as const) {
      it(`confseqMeanPath ${method} @ ${s.name}`, () => {
        const { lo, hi } = sinf.confseqMeanPath(s.x, { method });
        closeArr(lo, s.confseq[method].lo, `${s.name}/${method}/lo`, true);
        closeArr(hi, s.confseq[method].hi, `${s.name}/${method}/hi`, true);
        const fin = sinf.confseqMean(s.x, { method });
        close(fin.lo, s.confseq[method].final.lo, `${s.name}/${method}/final.lo`, true);
        close(fin.hi, s.confseq[method].final.hi, `${s.name}/${method}/final.hi`, true);
      });
    }
    it(`fixedCiPath @ ${s.name}`, () => {
      const { lo, hi } = sinf.fixedCiPath(s.x);
      closeArr(lo, s.fixed_ci.lo, `${s.name}/fixed/lo`, true);
      closeArr(hi, s.fixed_ci.hi, `${s.name}/fixed/hi`, true);
    });
  }
});

// --------------------------------------------------------------------------- #
// Panel -> stream helpers and single-model score CS.
// --------------------------------------------------------------------------- #
describe("panel helpers + single-model score CS", () => {
  const p = fx.panel_single;
  it("trialScores / questionScores / streamFromTensor", () => {
    closeArr(sinf.trialScores(p.R), p.trial_scores, "trialScores");
    closeArr(sinf.questionScores(p.R), p.question_scores, "questionScores");
    closeArr(sinf.streamFromTensor(p.R, "trials"), p.stream_trials, "streamTrials");
    closeArr(sinf.streamFromTensor(p.R, "questions"), p.stream_questions, "streamQuestions");
  });
  it("scoreConfseqPath / scoreConfseq", () => {
    const path = sinf.scoreConfseqPath(p.R);
    closeArr(path.lo, p.score_confseq_path.lo, "scorePath/lo", true);
    closeArr(path.hi, p.score_confseq_path.hi, "scorePath/hi", true);
    const fin = sinf.scoreConfseq(p.R);
    close(fin.lo, p.score_confseq.lo, "score/lo", true);
    close(fin.hi, p.score_confseq.hi, "score/hi", true);
  });
  it("precisionStop (met + unmet) — discrete stop exact, bounds close", () => {
    const met = sinf.precisionStop(p.R, 0.25);
    expect(met.stopped).toBe(p.precision_stop_met.stopped);
    expect(met.n).toBe(p.precision_stop_met.n);
    close(met.lo, p.precision_stop_met.lo, "met/lo", true);
    close(met.hi, p.precision_stop_met.hi, "met/hi", true);

    const unmet = sinf.precisionStop(p.R, 0.001);
    expect(unmet.stopped).toBe(p.precision_stop_unmet.stopped);
    expect(unmet.n).toBe(p.precision_stop_unmet.n);
    close(unmet.lo, p.precision_stop_unmet.lo, "unmet/lo", true);
    close(unmet.hi, p.precision_stop_unmet.hi, "unmet/hi", true);
  });
});

// --------------------------------------------------------------------------- #
// Paired two-model comparison.
// --------------------------------------------------------------------------- #
describe("paired comparison", () => {
  it("independent pair: diffs, CS path, decideBetter=continue", () => {
    const q = fx.panel_pair_indep;
    closeArr(sinf.pairedTrialDiffs(q.RA, q.RB), q.paired_trial_diffs, "pairedDiffs");
    const path = sinf.comparePairedPath(q.RA, q.RB);
    closeArr(path.lo, q.compare_paired_path.lo, "cmpPath/lo", true);
    closeArr(path.hi, q.compare_paired_path.hi, "cmpPath/hi", true);
    const fin = sinf.comparePaired(q.RA, q.RB);
    close(fin.lo, q.compare_paired.lo, "cmp/lo", true);
    close(fin.hi, q.compare_paired.hi, "cmp/hi", true);
    const d = sinf.decideBetter(q.RA, q.RB);
    expect(d.decision).toBe(q.decide_better.decision);
    expect(d.n).toBe(q.decide_better.n);
    close(d.lo, q.decide_better.lo, "decideIndep/lo", true);
    close(d.hi, q.decide_better.hi, "decideIndep/hi", true);
  });
  it("decisive pair: decideBetter=A with matching first-crossing n", () => {
    const q = fx.panel_pair_decisive;
    const d = sinf.decideBetter(q.RA, q.RB);
    expect(d.decision).toBe(q.decide_better.decision); // "A"
    expect(d.n).toBe(q.decide_better.n); // exact first-crossing trial
    close(d.lo, q.decide_better.lo, "decideDec/lo", true);
    close(d.hi, q.decide_better.hi, "decideDec/hi", true);
    const fin = sinf.comparePaired(q.RA, q.RB);
    close(fin.lo, q.compare_paired.lo, "cmpDec/lo", true);
    close(fin.hi, q.compare_paired.hi, "cmpDec/hi", true);
  });
});

// --------------------------------------------------------------------------- #
// Multi-model ranking.
// --------------------------------------------------------------------------- #
describe("multi-model ranking", () => {
  function checkTop1(res: any, ref: any, label: string): void {
    expect(res.stop, `${label}/stop`).toBe(ref.stop);
    expect(res.leader, `${label}/leader`).toBe(ref.leader);
    expect(res.ambiguous, `${label}/ambiguous`).toEqual(ref.ambiguous);
    close(res.margin, ref.margin, `${label}/margin`, true);
  }
  it("random tensor: empiricalScores, top1 (pairs/leader/none), full ranking, allocation", () => {
    const t = fx.tensor;
    closeArr(sinf.empiricalScores(t.R), t.empirical_scores, "empScores");
    checkTop1(sinf.shouldStopTop1Av(t.R), t.top1_pairs, "top1_pairs");
    checkTop1(sinf.shouldStopTop1Av(t.R, { correction: "leader" }), t.top1_leader, "top1_leader");
    checkTop1(sinf.shouldStopTop1Av(t.R, { correction: "none" }), t.top1_none, "top1_none");

    const fr = sinf.shouldStopFullRanking(t.R);
    expect(fr.stop).toBe(t.full_ranking.stop);
    expect(fr.ranking).toEqual(t.full_ranking.ranking);
    expect(fr.unresolved).toEqual(t.full_ranking.unresolved);

    const al = sinf.suggestNextAllocationStratified(t.R);
    expect(al.leader).toBe(t.allocation.leader);
    expect(al.competitor).toBe(t.allocation.competitor);
    closeArr(al.questionPriority, t.allocation.question_priority, "questionPriority");
  });
  it("separated tensor: leader, ranking order, priorities", () => {
    const t = fx.tensor_sep;
    closeArr(sinf.empiricalScores(t.R), t.empirical_scores, "sep/empScores");
    const top1 = sinf.shouldStopTop1Av(t.R);
    expect(top1.leader).toBe(t.top1_pairs.leader);
    expect(top1.stop).toBe(t.top1_pairs.stop);
    expect(top1.ambiguous).toEqual(t.top1_pairs.ambiguous);
    const fr = sinf.shouldStopFullRanking(t.R);
    expect(fr.ranking).toEqual(t.full_ranking.ranking);
    expect(fr.unresolved).toEqual(t.full_ranking.unresolved);
  });
  it("selectBestFixedBudget: SELECTED BEST + spent match (RNG parity not expected)", () => {
    const sb = fx.select_best;
    const res = sinf.selectBestFixedBudget(sb.R, sb.budget, { seed: sb.seed });
    // `best` and the shape-determined `spent` are RNG-independent on a
    // clearly-separated tensor; the exact subsampling is not compared.
    expect(res.best).toBe(sb.best);
    expect(res.spent).toBe(sb.spent);
    expect(res.rounds).toEqual(sb.rounds);
  });
});

// --------------------------------------------------------------------------- #
// Inference-time voting (Beta-Binomial mixture martingale + adaptive baseline).
// --------------------------------------------------------------------------- #
describe("inference-time voting", () => {
  it("countsFromAnswers ignores nullish/empty labels", () => {
    const { labels, counts } = sinf.countsFromAnswers(fx.counts_from_answers.answers);
    expect(labels).toEqual(fx.counts_from_answers.labels);
    expect(counts).toEqual(fx.counts_from_answers.counts);
  });
  for (let i = 0; i < 5; i++) {
    it(`votes[${i}] shouldStopSampling + adaptiveConsistencyStop`, () => {
      const v = fx.votes[i];
      const s = sinf.shouldStopSampling(v.counts);
      expect(s.stop).toBe(v.should_stop_sampling.stop);
      expect(s.mode).toBe(v.should_stop_sampling.mode);
      expect(s.runnerUp).toBe(v.should_stop_sampling.runner_up);
      expect(s.nTotal).toBe(v.should_stop_sampling.n_total);
      expect(s.nTop2).toBe(v.should_stop_sampling.n_top2);
      close(s.martingale, v.should_stop_sampling.martingale, `votes[${i}]/martingale`);
      close(s.pValue, v.should_stop_sampling.p_value, `votes[${i}]/pValue`);

      const s01 = sinf.shouldStopSampling(v.counts, { alpha: 0.01 });
      expect(s01.stop).toBe(v.should_stop_sampling_a01.stop);

      const a = sinf.adaptiveConsistencyStop(v.counts);
      expect(a.stop).toBe(v.adaptive.stop);
      expect(a.mode).toBe(v.adaptive.mode);
      expect(a.nTotal).toBe(v.adaptive.n_total);
      close(a.posterior, v.adaptive.posterior, `votes[${i}]/posterior`);
      const a80 = sinf.adaptiveConsistencyStop(v.counts, { thresh: 0.8 });
      expect(a80.stop).toBe(v.adaptive_t80.stop);
    });
  }
});

// --------------------------------------------------------------------------- #
// Fixed-look legacy (mu, sigma) API.
// --------------------------------------------------------------------------- #
describe("fixed-look legacy API", () => {
  const L = fx.legacy;
  it("rankingConfidence / pairwiseConfidence (+ degenerate branches)", () => {
    const rc = sinf.rankingConfidence(L.mus[0], L.sigmas[0], L.mus[1], L.sigmas[1]);
    close(rc.rho, L.ranking_confidence.rho, "rc/rho");
    close(rc.z, L.ranking_confidence.z, "rc/z");

    const pc = sinf.pairwiseConfidence(L.mus[0], L.sigmas[0], L.mus[1], L.sigmas[1], 0.0005);
    close(pc.rho, L.pairwise_cov.rho, "pc/rho");
    close(pc.z, L.pairwise_cov.z, "pc/z");
    const pc0 = sinf.pairwiseConfidence(L.mus[0], L.sigmas[0], L.mus[1], L.sigmas[1]);
    close(pc0.rho, L.pairwise_indep.rho, "pc0/rho");
    close(pc0.z, L.pairwise_indep.z, "pc0/z");

    const tie = sinf.pairwiseConfidence(0.8, 0, 0.8, 0);
    close(tie.rho, L.pairwise_tie.rho, "tie/rho");
    close(tie.z, L.pairwise_tie.z, "tie/z"); // Infinity
    const certain = sinf.rankingConfidence(0.8, 0, 0.7, 0);
    close(certain.rho, L.ranking_conf_certain.rho, "certain/rho");
    close(certain.z, L.ranking_conf_certain.z, "certain/z"); // Infinity
  });
  it("ciFromMuSigma (+ clip) / shouldStop", () => {
    const ci = sinf.ciFromMuSigma(0.7, 0.05, { confidence: 0.9 });
    close(ci.lo, L.ci_90.lo, "ci/lo");
    close(ci.hi, L.ci_90.hi, "ci/hi");
    const cic = sinf.ciFromMuSigma(0.97, 0.05, { confidence: 0.9, clip: [0, 1] });
    close(cic.lo, L.ci_90_clip.lo, "ciClip/lo");
    close(cic.hi, L.ci_90_clip.hi, "ciClip/hi");

    expect(sinf.shouldStop(0.01, { confidence: 0.95, maxHalfWidth: 0.02 })).toBe(L.should_stop_half);
    expect(sinf.shouldStop(0.02, { confidence: 0.95, maxHalfWidth: 0.02 })).toBe(L.should_stop_half_false);
    expect(sinf.shouldStop(0.02, { confidence: 0.95, maxCiWidth: 0.1 })).toBe(L.should_stop_ci);
  });
  it("shouldStopTop1 / suggestNextAllocation (ci_overlap + zscore)", () => {
    const ci = sinf.shouldStopTop1(L.mus, L.sigmas, { method: "ci_overlap" });
    expect(ci.stop).toBe(L.top1_ci_overlap.stop);
    expect(ci.leader).toBe(L.top1_ci_overlap.leader);
    expect(ci.ambiguous).toEqual(L.top1_ci_overlap.ambiguous);
    const zs = sinf.shouldStopTop1(L.mus, L.sigmas, { method: "zscore" });
    expect(zs.stop).toBe(L.top1_zscore.stop);
    expect(zs.leader).toBe(L.top1_zscore.leader);
    expect(zs.ambiguous).toEqual(L.top1_zscore.ambiguous);

    const aci = sinf.suggestNextAllocation(L.mus, L.sigmas, { method: "ci_overlap" });
    expect([aci.leader, aci.competitor]).toEqual(L.alloc_ci_overlap);
    const azs = sinf.suggestNextAllocation(L.mus, L.sigmas, { method: "zscore" });
    expect([azs.leader, azs.competitor]).toEqual(L.alloc_zscore);
  });
});

describe("CS bound parity summary", () => {
  it("max abs diff across all CS bounds is < 1e-6", () => {
    // eslint-disable-next-line no-console
    console.log(`\n[sinf] max abs diff on CS bounds vs Python: ${maxCsDiff.toExponential(3)}`);
    expect(maxCsDiff).toBeLessThan(1e-6);
  });
});
