import { describe, expect, it } from "vitest";

import { agg, aggregate as aggregateNamespace } from "../src/index.js";
import * as aggregate from "../src/aggregate/index.js";
import fixtures from "./fixtures/aggregate.json";

const fx = fixtures as any;

function decode(value: any): any {
  if (Array.isArray(value)) return value.map(decode);
  if (value !== null && typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value).map(([key, item]) => [key, decode(item)]),
    );
  }
  if (value === "nan") return NaN;
  if (value === "inf") return Infinity;
  if (value === "-inf") return -Infinity;
  return value;
}

function equivalent(actual: any, expectedInput: any, path = "value"): void {
  const expected = decode(expectedInput);
  if (Array.isArray(expected)) {
    expect(Array.isArray(actual), `${path} is array`).toBe(true);
    expect(actual.length, `${path} length`).toBe(expected.length);
    for (let i = 0; i < expected.length; i++) {
      equivalent(actual[i], expected[i], `${path}[${i}]`);
    }
    return;
  }
  if (typeof expected === "number") {
    if (!Number.isFinite(expected)) {
      expect(Object.is(actual, expected), `${path}: ${actual} vs ${expected}`).toBe(
        true,
      );
    } else {
      expect(actual, path).toBeCloseTo(expected, 12);
    }
    return;
  }
  expect(actual, path).toEqual(expected);
}

describe("aggregate exports", () => {
  it("exposes identical aggregate/agg namespaces and direct subpath API", () => {
    expect(aggregateNamespace).toBe(agg);
    expect(aggregateNamespace.bestOfN).toBe(aggregate.bestOfN);
    expect(aggregateNamespace.confidence.selfCertainty).toBe(
      aggregate.selfCertainty,
    );
    expect(aggregate.mob).toBe(aggregate.majorityOfTheBests);
  });

  it("exports Python-compatible snake_case aliases", () => {
    expect(aggregate.mean_logprob).toBe(aggregate.meanLogprob);
    expect(aggregate.best_of_n).toBe(aggregate.bestOfN);
    expect(aggregate.majority_of_the_bests).toBe(
      aggregate.majorityOfTheBests,
    );
    expect(aggregate.weighted_majority_vote).toBe(
      aggregate.weightedMajorityVote,
    );
    expect(aggregate.fit_kde_vote_calibration).toBe(
      aggregate.fitKdeVoteCalibration,
    );
    expect(aggregate.kde_weighted_vote).toBe(aggregate.kdeWeightedVote);
    expect(aggregate.cges_vote).toBe(aggregate.cgesVote);
    expect(aggregate.cges_stop).toBe(aggregate.cgesStop);
    expect(aggregate.adaptive_consistency_dirichlet_stop).toBe(
      aggregate.adaptiveConsistencyDirichletStop,
    );
    expect(aggregate.adaptive_consistency_crp_stop).toBe(
      aggregate.adaptiveConsistencyCrpStop,
    );
    expect(aggregate.deepconf_online_stop).toBe(aggregate.deepconfOnlineStop);
  });
});

describe("confidence signals match Python fixtures", () => {
  const c = fx.confidence;

  it("chosen-token likelihood, perplexity, and PiCSAR", () => {
    equivalent(aggregate.meanLogprob(c.logprobs), c.mean_logprob);
    equivalent(aggregate.sequenceLogprob(c.logprobs), c.sequence_logprob);
    equivalent(aggregate.perplexity(c.logprobs), c.perplexity);
    equivalent(aggregate.picsar(c.logprobs), c.picsar);
    equivalent(
      aggregate.picsar(c.logprobs, { answerStart: 3 }),
      c.picsar_split,
    );
    equivalent(
      aggregate.picsar(c.logprobs, {
        answerStart: 3,
        normalizeReasoning: true,
      }),
      c.picsar_normalized,
    );
    equivalent(aggregate.picsar(c.logprobs, { answerStart: 0 }), c.picsar_zero);
    equivalent(
      aggregate.picsar(c.logprobs, { answerStart: c.logprobs.length }),
      c.picsar_end,
    );
  });

  for (const reducer of ["mean", "min", "max"] as const) {
    it(`distribution signals with ${reducer} reduction`, () => {
      equivalent(
        aggregate.selfCertainty(c.topk, { aggregate: reducer }),
        c.self_certainty[reducer],
      );
      equivalent(
        aggregate.tokenEntropy(c.topk, { aggregate: reducer }),
        c.token_entropy[reducer],
      );
      equivalent(
        aggregate.varentropy(c.topk, { aggregate: reducer }),
        c.varentropy[reducer],
      );
      equivalent(
        aggregate.maxSoftmaxProbability(c.topk, { aggregate: reducer }),
        c.max_softmax_probability[reducer],
      );
      equivalent(
        aggregate.logprobMargin(c.topk, { aggregate: reducer }),
        c.logprob_margin[reducer],
      );
    });
  }

  it("probability margin, per-token confidence, and every DeepConf mode", () => {
    equivalent(
      aggregate.logprobMargin(c.topk, { useProb: true }),
      c.prob_margin,
    );
    equivalent(aggregate.tokenConfidence(c.topk), c.token_confidence);
    equivalent(aggregate.deepconfConfidence(c.topk), c.deepconf.mean);
    equivalent(
      aggregate.deepconfConfidence(c.topk, { mode: "tail", tailTokens: 3 }),
      c.deepconf.tail,
    );
    equivalent(
      aggregate.deepconfConfidence(c.topk, {
        mode: "lowest_group",
        window: 3,
      }),
      c.deepconf.lowest_group,
    );
    equivalent(
      aggregate.deepconfConfidence(c.topk, {
        mode: "bottom_group",
        window: 3,
        bottomQuantile: 0.4,
      }),
      c.deepconf.bottom_group,
    );
  });

  it("supports a bare top-k row and genuinely ragged top-k traces", () => {
    equivalent(aggregate.tokenEntropy(c.topk[0]), c.bare_row_entropy);
    equivalent(aggregate.tokenEntropy(c.ragged), c.ragged_entropy);
    equivalent(aggregate.selfCertainty(c.ragged), c.ragged_self_certainty);
    equivalent(aggregate.logprobMargin([[-0.4]]), c.single_candidate_margin);
  });

  it("matches NumPy scalar and rectangular reshape coercion", () => {
    expect(aggregate.meanLogprob(-0.4)).toBe(-0.4);
    expect(aggregate.sequenceLogprob([[[-0.1, -0.2], [-0.3, -0.4]]])).toBeCloseTo(
      -1,
      14,
    );
  });

  it("validates empty, non-finite, ragged, reducer, and split inputs", () => {
    expect(() => aggregate.meanLogprob([])).toThrow(/at least one token/);
    expect(() => aggregate.meanLogprob([0, -Infinity])).toThrow(/finite/);
    expect(() => aggregate.meanLogprob([[0], [1, 2]])).toThrow(/rectangular/);
    expect(() => aggregate.selfCertainty([])).toThrow(/at least one token/);
    expect(() => aggregate.tokenEntropy([[0, NaN]])).toThrow(/finite/);
    expect(() =>
      aggregate.selfCertainty([[0, -1]], { aggregate: "median" as any }),
    ).toThrow(/aggregate/);
    expect(() => aggregate.picsar([-0.1], { answerStart: 2 })).toThrow(
      /answer_start/,
    );
    expect(() =>
      aggregate.deepconfConfidence([[0, -1]], { mode: "p50" as any }),
    ).toThrow(/mode/);
    expect(() =>
      aggregate.deepconfConfidence([[0, -1]], {
        mode: "bottom_group",
        bottomQuantile: 0,
      }),
    ).toThrow(/bottom_quantile/);
  });

  it("keeps extreme finite top-k log probabilities stable in log space", () => {
    const extreme = c.extreme;
    equivalent(aggregate.tokenEntropy(extreme.topk), extreme.token_entropy);
    equivalent(aggregate.varentropy(extreme.topk), extreme.varentropy);
    equivalent(aggregate.selfCertainty(extreme.topk), extreme.self_certainty);
  });
});

describe("PRM aggregation matches Python fixtures", () => {
  for (const method of ["last", "min", "mean", "prod", "max"] as const) {
    it(method, () => {
      equivalent(
        aggregate.prmAggregate(fx.prm.steps, { method }),
        fx.prm.outputs[method],
      );
    });
  }

  it("accepts scalar/rectangular inputs and validates failures", () => {
    expect(aggregate.prmAggregate(0.7)).toBe(0.7);
    expect(
      aggregate.prmAggregate([[[0.9, 0.8]], [[0.7, 0.6]]], { method: "min" }),
    ).toBe(0.6);
    expect(() => aggregate.prmAggregate([])).toThrow(/non-empty/);
    expect(() => aggregate.prmAggregate([0.5, Infinity])).toThrow(/finite/);
    expect(() =>
      aggregate.prmAggregate([0.5], { method: "median" as any }),
    ).toThrow(/method/);
  });
});

describe("selection rules match Python fixtures", () => {
  const s = fx.selection;
  const answers = s.answers as (string | null)[];
  const scores = s.scores as number[];

  it("majority, Best-of-N, weighted votes, and representative extras", () => {
    equivalent(
      aggregate.majorityVote(answers, { returnIndex: true }),
      s.majority_vote,
    );
    equivalent(
      aggregate.bestOfN(answers, scores, {
        returnIndex: true,
        returnScore: true,
      }),
      s.best_of_n,
    );
    equivalent(
      aggregate.weightedMajorityVote(answers, scores, {
        aggregate: "sum",
        returnIndex: true,
        returnScore: true,
      }),
      s.weighted_sum,
    );
    equivalent(
      aggregate.weightedMajorityVote(answers, scores, {
        aggregate: "mean",
        returnIndex: true,
        returnScore: true,
      }),
      s.weighted_mean,
    );
  });

  it("Majority-of-the-Bests and Best-of-Majority variants", () => {
    equivalent(
      aggregate.majorityOfTheBests(answers, scores, {
        returnIndex: true,
        returnScore: true,
      }),
      s.mob_default,
    );
    equivalent(
      aggregate.majorityOfTheBests(answers, scores, {
        m: 3,
        returnIndex: true,
        returnScore: true,
      }),
      s.mob_m3,
    );
    equivalent(
      aggregate.bestOfMajority(answers, scores, {
        alpha: 0.4,
        aggregate: "mean",
        returnIndex: true,
        returnScore: true,
      }),
      s.best_of_majority,
    );
    for (const method of ["sum", "max"] as const) {
      equivalent(
        aggregate.bestOfMajority(answers, scores, {
          aggregate: method,
          returnIndex: true,
          returnScore: true,
        }),
        s[`best_of_majority_${method}`],
      );
    }
  });

  it("softmax, rank, logit/linear, and filtered votes", () => {
    equivalent(
      aggregate.softmaxWeightedVote(answers, scores, {
        temperature: 0.7,
        returnIndex: true,
        returnScore: true,
      }),
      s.softmax,
    );
    equivalent(
      aggregate.softmaxWeightedVote(answers, scores, {
        temperature: Infinity,
        returnIndex: true,
        returnScore: true,
      }),
      s.softmax_infinite,
    );
    for (const [p, key] of [
      [0, "rank_p0"],
      [1, "rank_p1"],
      [1.7, "rank_fractional"],
    ] as const) {
      equivalent(
        aggregate.rankWeightedVote(answers, scores, {
          p,
          returnIndex: true,
          returnScore: true,
        }),
        s[key],
      );
    }
    equivalent(
      aggregate.logitWeightedVote(answers, scores, {
        returnIndex: true,
        returnScore: true,
      }),
      s.logit,
    );
    equivalent(
      aggregate.logitWeightedVote(answers, scores, {
        threshold: 0.2,
        transform: "linear",
        returnIndex: true,
        returnScore: true,
      }),
      s.linear,
    );
    equivalent(
      aggregate.filteredVote(answers, scores, {
        keep: 0.5,
        returnIndex: true,
        returnScore: true,
      }),
      s.filtered_fraction,
    );
    equivalent(
      aggregate.filteredVote(answers, scores, {
        keep: 3,
        weighted: false,
        returnIndex: true,
        returnScore: true,
      }),
      s.filtered_count,
    );
    // Explicit fraction disambiguates Python's float 1.0 from integer 1.
    equivalent(
      aggregate.filteredVote(answers, scores, {
        keep: { fraction: 1 },
        weighted: false,
        returnIndex: true,
        returnScore: true,
      }),
      s.filtered_all,
    );
  });

  it("preserves batch result shapes and no-valid sentinels", () => {
    const b = fx.batch;
    const batchAnswers = decode(b.answers);
    equivalent(
      aggregate.majorityVote(batchAnswers, { returnIndex: true }),
      b.majority_vote,
    );
    equivalent(
      aggregate.bestOfN(batchAnswers, b.scores, {
        returnIndex: true,
        returnScore: true,
      }),
      b.best_of_n,
    );
    equivalent(
      aggregate.weightedMajorityVote(batchAnswers, b.scores, {
        returnIndex: true,
        returnScore: true,
      }),
      b.weighted,
    );
    equivalent(
      aggregate.majorityOfTheBests(batchAnswers, b.scores, {
        returnIndex: true,
        returnScore: true,
      }),
      b.mob,
    );
    equivalent(
      aggregate.filteredVote(batchAnswers, b.scores, {
        keep: 0.67,
        returnIndex: true,
        returnScore: true,
      }),
      b.filtered,
    );
  });

  it("uses exact integer MoB/Borda weights and deterministic ties", () => {
    const e = fx.exact;
    expect(
      aggregate.majorityOfTheBests(e.mob_tie_answers, e.mob_tie_scores),
    ).toBe(e.mob_tie);
    expect(aggregate.rankWeightedVote(e.rank_answers, e.rank_scores, { p: 3 })).toBe(
      e.rank_integer,
    );

    const manyAnswers = Array.from({ length: 200 }, (_, i) =>
      i % 3 === 0 ? "A" : "B",
    );
    const manyScores = Array.from({ length: 200 }, (_, i) => i);
    expect(aggregate.rankWeightedVote(manyAnswers, manyScores, { p: 300 })).toBe(
      aggregate.bestOfN(manyAnswers, manyScores),
    );
    expect(aggregate.majorityVote(["B", "A", "A", "B"])).toBe("B");
    equivalent(
      aggregate.filteredVote(["A", "B", "B", "A"], [0.1, 0.3, 0.8, 0.9], {
        keep: { fraction: 1 },
        weighted: false,
        returnIndex: true,
      }),
      e.filtered_sorted_tie,
    );
  });

  it("represents Python integer-vs-float keep semantics explicitly", () => {
    const a = ["A", "A", "B"];
    const q = [0.1, 0.2, 0.9];
    expect(aggregate.filteredVote(a, q, { keep: 1 })).toBe("B");
    expect(
      aggregate.filteredVote(a, q, {
        keep: { fraction: 1 },
        weighted: false,
      }),
    ).toBe("A");
  });

  it("ignores invalid answers and returns null/-1/NaN when all are invalid", () => {
    const result = aggregate.bestOfN([null, "", NaN, undefined], [1, 2, 3, 4], {
      returnIndex: true,
      returnScore: true,
    });
    expect(result[0]).toBeNull();
    expect(result[1]).toBe(-1);
    expect(Number.isNaN(result[2] as number)).toBe(true);
    expect(aggregate.majorityVote([null, "", NaN])).toBeNull();
    expect(
      aggregate.logitWeightedVote(["A", null], [0.7, 99]),
    ).toBe("A"); // invalid candidate's out-of-range score is never validated
  });

  it("packs score-only results and keys scalar-vs-batch shape from answers", () => {
    expect(
      aggregate.bestOfN(["A", "B"], [0.1, 0.9], { returnScore: true }),
    ).toEqual(["B", 0.9]);
    expect(
      aggregate.weightedMajorityVote(["A", "A", "B"], [0.3, 0.4, 0.5], {
        returnScore: true,
      }),
    ).toEqual(["A", 0.4]);
    expect(
      aggregate.bestOfN([["A", "B"]], [0.1, 0.9], {
        returnIndex: true,
        returnScore: true,
      }),
    ).toEqual([["B"], [1], [0.9]]);
  });

  it("validates pool shape and method-specific parameters", () => {
    expect(() => aggregate.majorityVote([])).toThrow(/at least one candidate/);
    expect(() => aggregate.majorityVote([["A"], ["B", "C"]])).toThrow(
      /rectangular/,
    );
    expect(() => aggregate.majorityVote([[[]]] as any)).toThrow(/1D|2D/);
    expect(() => aggregate.bestOfN([["A", "B"]], [[0.1]])).toThrow(
      /same shape/,
    );
    expect(() => aggregate.bestOfN(["A"], undefined as any)).toThrow(/scores/);
    expect(() =>
      aggregate.weightedMajorityVote(["A"], [1], { aggregate: "median" as any }),
    ).toThrow(/aggregate/);
    expect(() =>
      aggregate.majorityOfTheBests(["A"], [1], { m: 0 }),
    ).toThrow(/m/);
    expect(() =>
      aggregate.majorityOfTheBests(["A", "B"], [0.8, 0.2], { m: 1.5 }),
    ).toThrow(/integer/);
    expect(() =>
      aggregate.bestOfMajority(["A"], [1], { alpha: 1.5 }),
    ).toThrow(/alpha/);
    expect(() =>
      aggregate.softmaxWeightedVote(["A"], [1], { temperature: 0 }),
    ).toThrow(/temperature/);
    expect(() =>
      aggregate.rankWeightedVote(["A"], [1], { p: Infinity }),
    ).toThrow(/finite/);
    expect(() =>
      aggregate.logitWeightedVote(["A", "B"], [0.5, 1]),
    ).toThrow(/\(0, 1\)/);
    expect(() =>
      aggregate.logitWeightedVote(["A"], [0.5], { transform: "sqrt" as any }),
    ).toThrow(/transform/);
    expect(() => aggregate.filteredVote(["A"], [1], { keep: 1.5 })).toThrow(
      /keep/,
    );
  });
});

describe("KDE-calibrated voting matches Python", () => {
  function constantCalibration(probability: number): aggregate.KDEVoteCalibration {
    const samples = [Math.log(0.3 / 0.7), Math.log(0.7 / 0.3)];
    return new aggregate.KDEVoteCalibration({
      correctLogits: samples,
      incorrectLogits: samples,
      correctBandwidth: 0.5,
      incorrectBandwidth: 0.5,
      binEdges: [-Infinity, Infinity],
      binProbability: [probability],
    });
  }

  it("fits class KDEs, nearest-quantile bins, and defensive state", () => {
    const scores = [0.8, 0.9, 0.1, 0.2];
    const calibration = aggregate.fitKdeVoteCalibration(scores, [1, 1, 0, 0], {
      nBins: 2,
      bandwidth: 0.5,
    });
    equivalent(calibration.correctLogits, [Math.log(4), Math.log(9)]);
    equivalent(calibration.incorrectLogits, [-Math.log(9), -Math.log(4)]);
    expect(calibration.nBins).toBe(2);
    expect(calibration.n_bins).toBe(2);
    expect(calibration.correct_logits).toBe(calibration.correctLogits);
    expect(calibration.bin_edges).toBe(calibration.binEdges);
    expect(calibration.binEdges).toEqual([-Infinity, 0.8, Infinity]);
    expect(calibration.binProbability).toEqual([0, 1]);
    expect(calibration.calibratedProbability([0.2, 0.7, 0.8, 0.95])).toEqual([
      0,
      0,
      1,
      1,
    ]);
    scores[0] = 0.5;
    expect(calibration.correctLogits[0]).toBeCloseTo(Math.log(4), 14);
    expect(Object.isFrozen(calibration.correctLogits)).toBe(true);
    expect(Object.isFrozen(calibration)).toBe(true);
    equivalent(calibration.correctLogits, fx.calibration.correct_logits);
    equivalent(calibration.incorrectLogits, fx.calibration.incorrect_logits);
    equivalent(calibration.binEdges, fx.calibration.bin_edges);
    equivalent(calibration.binProbability, fx.calibration.bin_probability);
  });

  it("rounds nearest-quantile bins on NumPy's inexact probabilities", () => {
    // `i / n_bins` is inexact for several `i` here, so folding NumPy's
    // `(n - 1) * q` into `(n - 1) * i / n_bins` would round two boundaries onto
    // the neighbouring order statistic and shift the fitted bins.
    const fitted = aggregate.fitKdeVoteCalibration(
      fx.calibration.inexact_quantile_scores,
      fx.calibration.inexact_quantile_labels,
      { nBins: 10, bandwidth: 0.5 },
    );
    equivalent(fitted.binEdges, fx.calibration.inexact_quantile_edges);
    equivalent(fitted.binProbability, fx.calibration.inexact_quantile_probability);
    equivalent(
      fitted.calibratedProbability([0.16, 0.26, 0.31, 0.56, 0.61]),
      fx.calibration.inexact_quantile_calibrated,
    );
  });

  it("evaluates density/reliability weights and exact boundary limits", () => {
    const calibration = aggregate.fitKdeVoteCalibration(
      [0.7, 0.8, 0.2, 0.3],
      [1, 1, 0, 0],
      { nBins: 1, bandwidth: 0.4 },
    );
    const ratio = calibration.logDensityRatio([0.65]) as number[];
    expect(ratio[0]).toBeGreaterThan(0);
    expect(calibration.weights([0.4, 0.7], { nAnswers: 3 })).toHaveLength(2);
    expect(constantCalibration(0).weights([0.4], { nAnswers: 2 })[0]).toBe(
      -Infinity,
    );
    expect(constantCalibration(1).weights([0.4], { nAnswers: 2 })[0]).toBe(
      Infinity,
    );
  });

  it("selects by KDE weight and preserves representative metadata", () => {
    const fitted = aggregate.fitKdeVoteCalibration(
      [0.8, 0.9, 0.1, 0.2],
      [1, 1, 0, 0],
      { nBins: 2, bandwidth: 0.5 },
    );
    expect(aggregate.kdeWeightedVote(["A", "A", "B"], [0.2, 0.2, 0.8], fitted)).toBe(
      fx.calibration.vote,
    );
    expect(
      aggregate.kdeWeightedVote(
        [
          ["A", "A", "B"],
          ["X", "Y", "Y"],
        ],
        [
          [0.4, 0.7, 0.6],
          [0.8, 0.5, 0.6],
        ],
        constantCalibration(0.6),
        { returnIndex: true, returnScore: true },
      ),
    ).toEqual([["A", "Y"], [1, 2], [0.7, 0.6]]);
    expect(
      aggregate.kdeWeightedVote(["A", null, "B"], [0.4, 9, 0.6], constantCalibration(0.5)),
    ).toBe("A");
  });

  it("validates fit state, response scores, and configuration", () => {
    expect(() =>
      aggregate.fitKdeVoteCalibration([0.8, 0.2], [1], { bandwidth: 0.5 }),
    ).toThrow(/same shape/);
    expect(() =>
      aggregate.fitKdeVoteCalibration([0.8, 0.2], [1, 2], { bandwidth: 0.5 }),
    ).toThrow(/boolean or 0\/1/);
    expect(() =>
      aggregate.fitKdeVoteCalibration([0.8, 0.9], [1, 1], { bandwidth: 0.5 }),
    ).toThrow(/correct and incorrect/);
    expect(() =>
      aggregate.fitKdeVoteCalibration([0.8, 0.2], [1, 0], { nBins: 1.5 }),
    ).toThrow(/n_bins/);
    expect(() =>
      aggregate.fitKdeVoteCalibration([0.8, 0.8, 0.2, 0.3], [1, 1, 0, 0]),
    ).toThrow(/constant correct-class/);
    expect(() =>
      aggregate.kdeWeightedVote(["A", "B"], [0.5, 1], constantCalibration(0.5)),
    ).toThrow(/strictly in \(0, 1\)/);
    expect(() => constantCalibration(0.5).weights([], { nAnswers: 2 })).toThrow(
      /nonempty 1D/,
    );
  });

  it("collapses repeated-score quantile boundaries", () => {
    const calibration = aggregate.fitKdeVoteCalibration(
      [0.2, 0.2, 0.2, 0.8, 0.8, 0.8],
      [0, 0, 0, 1, 1, 1],
      { nBins: 20, bandwidth: 0.5 },
    );
    expect(calibration.nBins).toBe(1);
    expect(calibration.binProbability).toEqual([0.5]);
  });
});

describe("CGES selection and stopping match Python", () => {
  it("selects observed answers by default and exposes the OTHER sentinel", () => {
    expect(aggregate.cgesVote(["A"], [0.1])).toBe("A");
    expect(aggregate.cgesVote(["A"], [0.1], { allowOther: true })).toBe(
      aggregate.CGES_OTHER,
    );
    expect(String(aggregate.CGES_OTHER)).toBe("CGES_OTHER");
    expect(
      aggregate.cgesVote(["A", "A", "B"], [0.7, 0.9, 0.6], {
        returnIndex: true,
        returnScore: true,
      }),
    ).toEqual(fx.cges.vote);
  });

  it("returns correct stop probabilities, minimum-sample behavior, and sentinels", () => {
    equivalent(
      aggregate.cgesStop(["A"], [0.9], { threshold: 0.8, returnProb: true }),
      fx.cges.stop,
    );
    expect(
      aggregate.cgesStop(["A"], [0.1], { threshold: 0.8, returnProb: true }),
    ).toEqual([false, expect.closeTo(0.1, 14)]);
    expect(
      aggregate.cgesStop(["A"], [0.1], {
        threshold: 0.8,
        includeOther: true,
        returnProb: true,
      }),
    ).toEqual([true, expect.closeTo(0.9, 14)]);
    expect(aggregate.cgesStop([null, "A"], [0, 0.9], { minSamples: 2 })).toBe(
      false,
    );
    expect(aggregate.cgesStop([null, ""], [0, 1], { returnProb: true })).toEqual([
      false,
      0,
    ]);
  });

  it("is stable in log space and validates reserved/options/probabilities", () => {
    const answers = [...new Array<string>(1000).fill("A"), ...new Array<string>(1000).fill("B")];
    const scores = new Array<number>(2000).fill(0.99);
    const [, probability] = aggregate.cgesStop(answers, scores, { returnProb: true });
    expect(Number.isFinite(probability)).toBe(true);
    expect(() => aggregate.cgesVote([aggregate.CGES_OTHER], [0.8])).toThrow(
      /reserved/,
    );
    expect(() => aggregate.cgesVote(["A"], [1])).toThrow(/strictly in \(0, 1\)/);
    expect(() => aggregate.cgesVote(["A"], [0.8], { allowOther: 1 as any })).toThrow(
      /allow_other/,
    );
    expect(() => aggregate.cgesStop(["A"], [0.8], { minSamples: 1.5 })).toThrow(
      /min_samples/,
    );
    expect(() => aggregate.cgesStop([["A"], ["B"]], [[0.8], [0.7]])).toThrow(
      /1D sampling stream/,
    );
  });
});

describe("online stopping matches Python fixtures", () => {
  const o = fx.online;

  it("warmup quantiles and strict first-window crossing", () => {
    equivalent(
      aggregate.deepconfStopThreshold(o.warmup, { keep: 0.2 }),
      o.threshold_keep_02,
    );
    equivalent(
      aggregate.deepconfStopThreshold(o.warmup, { keep: 1 }),
      o.threshold_keep_all,
    );
    expect(aggregate.deepconfOnlineStop(o.topk, 2, { window: 3 })).toBe(
      o.token_stop,
    );
    expect(aggregate.deepconfOnlineStop(o.topk, 0.5, { window: 3 })).toBe(
      o.token_no_stop,
    );
    // The rule is strictly `< threshold`, not `<= threshold`.
    expect(aggregate.deepconfOnlineStop(o.topk, 1, { window: 3 })).toBe(
      o.token_equal_no_stop,
    );
    expect(
      aggregate.deepconfOnlineStop(
        [[-4, -6], [-4, -6], [-4, -6], [0, -2], [0, -2], [0, -2]],
        2,
        { window: 3 },
      ),
    ).toBe(o.token_later_stop);
  });

  it("Adaptive-Consistency posterior and ESC decisions", () => {
    equivalent(
      aggregate.adaptiveConsistencyStop(["A", "A", "A", "A", "A", "A", "A", "A", "B", "B"], {
        returnProb: true,
      }),
      o.adaptive_dominant,
    );
    equivalent(
      aggregate.adaptiveConsistencyStop(["A", "B", "A", "B"], {
        returnProb: true,
      }),
      o.adaptive_tie,
    );
    equivalent(
      aggregate.adaptiveConsistencyStop([null, ""], { returnProb: true }),
      o.adaptive_invalid,
    );
    const [largeA, largeB] = o.adaptive_large_near_counts as [number, number];
    const [, largeProbability] = aggregate.adaptiveConsistencyStop(
      [
        ...new Array<string>(largeA).fill("A"),
        ...new Array<string>(largeB).fill("B"),
      ],
      { returnProb: true },
    );
    expect(
      Math.abs(largeProbability - o.adaptive_large_near[1]),
      "large near-tie posterior vs scipy.special.betainc",
    ).toBeLessThan(1e-9);
    expect(
      aggregate.adaptiveConsistencyStop(["A", "A", "B", "B"], {
        threshold: 0.5000000000000005,
      }),
    ).toBe(false);
    expect(aggregate.escStop(["A", "A", "A"])).toBe(o.esc_true);
    expect(aggregate.escStop(["A", "B", "A"])).toBe(o.esc_false);
    expect(aggregate.escStop(["A", null, "A"])).toBe(o.esc_invalid);
    expect(aggregate.escStop([])).toBe(false);
  });

  it("clamps long windows and validates online parameters", () => {
    const short = [
      [0, -2],
      [-1, -3],
    ];
    expect(aggregate.deepconfOnlineStop(short, 2, { window: 999 })).toBe(1);
    expect(aggregate.deepconfOnlineStop(short, 1, { window: 999 })).toBeNull();
    expect(() => aggregate.adaptiveConsistencyStop(["A"], { threshold: 1 })).toThrow(
      /threshold/,
    );
    expect(() => aggregate.deepconfStopThreshold([], { keep: 0.1 })).toThrow(
      /warmup/,
    );
    expect(() => aggregate.deepconfStopThreshold([1], { keep: 0 })).toThrow(
      /keep/,
    );
    expect(() => aggregate.deepconfOnlineStop(short, 1, { window: 0 })).toThrow(
      /window/,
    );
  });
});

describe("full Adaptive-Consistency variants match Python", () => {
  it("uses every observed category in the Dirichlet posterior", () => {
    const result = aggregate.adaptiveConsistencyDirichletStop(
      [...new Array<string>(5).fill("A"), ...new Array<string>(2).fill("B"), "C"],
      { returnProb: true },
    );
    equivalent(result, fx.online.dirichlet_three);

    const large = aggregate.adaptiveConsistencyDirichletStop(
      [
        ...new Array<string>(1000).fill("A"),
        ...new Array<string>(900).fill("B"),
        "C",
      ],
      { returnProb: true },
    );
    expect(large[0]).toBe(fx.online.dirichlet_large[0]);
    expect(large[1]).toBeCloseTo(fx.online.dirichlet_large[1], 11);
    const veryLarge = aggregate.adaptiveConsistencyDirichletStop(
      [
        ...new Array<string>(100_000).fill("A"),
        ...new Array<string>(99_900).fill("B"),
        "C",
      ],
      { returnProb: true },
    );
    expect(veryLarge[0]).toBe(fx.online.dirichlet_very_large[0]);
    expect(veryLarge[1]).toBeCloseTo(fx.online.dirichlet_very_large[1], 11);
    expect(
      aggregate.adaptiveConsistencyDirichletStop(
        [
          ...new Array<string>(1000).fill("A"),
          ...new Array<string>(1000).fill("B"),
          ...new Array<string>(1000).fill("C"),
        ],
        { returnProb: true },
      )[1],
    ).toBe(fx.online.dirichlet_symmetric[1]);
  });

  it("resolves the endpoint layer a dominant leader creates", () => {
    // A leader far ahead of every rival concentrates the Dirichlet integrand's
    // variation near one endpoint; the quadrature has to spend its panels there
    // rather than spreading them evenly over [0, 1].
    const expand = (counts: number[]): string[] =>
      counts.flatMap((count, index) => new Array<string>(count).fill(`L${index}`));
    const peaked = aggregate.adaptiveConsistencyDirichletStop(
      expand(fx.online.dirichlet_peaked_counts),
      { returnProb: true },
    );
    expect(peaked[0]).toBe(fx.online.dirichlet_peaked[0]);
    expect(peaked[1]).toBeCloseTo(fx.online.dirichlet_peaked[1], 11);
    const peakedLarge = aggregate.adaptiveConsistencyDirichletStop(
      expand(fx.online.dirichlet_peaked_large_counts),
      { returnProb: true },
    );
    expect(peakedLarge[0]).toBe(fx.online.dirichlet_peaked_large[0]);
    expect(peakedLarge[1]).toBeCloseTo(fx.online.dirichlet_peaked_large[1], 11);
  });

  it("keeps the near-tie binomial tail accurate across magnitudes", () => {
    const counts = fx.online.adaptive_near_tie_counts as [number, number][];
    counts.forEach(([first, second], index) => {
      const stream = [
        ...new Array<string>(first).fill("L0"),
        ...new Array<string>(second).fill("L1"),
      ];
      const result = aggregate.adaptiveConsistencyStop(stream, { returnProb: true });
      expect(result[0]).toBe(fx.online.adaptive_near_tie[index][0]);
      expect(result[1]).toBeCloseTo(fx.online.adaptive_near_tie[index][1], 13);
    });
  });

  it("matches the top-two method for one/two categories and consumes generators once", () => {
    const answers = [...new Array<string>(5).fill("A"), ...new Array<string>(2).fill("B")];
    expect(
      aggregate.adaptiveConsistencyDirichletStop(answers, { returnProb: true }),
    ).toEqual(aggregate.adaptiveConsistencyStop(answers, { returnProb: true }));
    function* stream(): Generator<string> {
      yield* answers;
    }
    expect(
      aggregate.adaptiveConsistencyDirichletStop(stream(), { returnProb: true }),
    ).toEqual(aggregate.adaptiveConsistencyStop(answers, { returnProb: true }));
    expect(
      aggregate.adaptiveConsistencyDirichletStop([null, ""], { returnProb: true }),
    ).toEqual([false, 0]);
  });

  it("runs a deterministic finite-horizon CRP comparator", () => {
    const options = {
      horizon: 12,
      nAlpha: 20,
      nSimulations: 200,
      seed: 7,
      returnProb: true as const,
    };
    const dominant = aggregate.adaptiveConsistencyCrpStop(
      [...new Array<string>(7).fill("A"), "B"],
      options,
    );
    expect(dominant[0]).toBe(true);
    expect(dominant[0]).toBe(fx.online.crp_dominant[0]);
    expect(dominant[1]).toBeGreaterThanOrEqual(0.95);
    const tie = aggregate.adaptiveConsistencyCrpStop(["A", "B", "A", "B"], options);
    expect(tie[0]).toBe(false);
    expect(tie[0]).toBe(fx.online.crp_tie[0]);
    expect(tie[1]).toBeGreaterThan(0.4);
    expect(tie[1]).toBeLessThan(0.6);
    expect(
      aggregate.adaptiveConsistencyCrpStop(["A", "B", "A"], {
        horizon: 3,
        returnProb: true,
      }),
    ).toEqual([true, 1]);
    expect(aggregate.adaptiveConsistencyCrpStop([null, ""], { returnProb: true })).toEqual([
      false,
      0,
    ]);
  });

  it("validates Dirichlet and CRP parameters", () => {
    expect(() =>
      aggregate.adaptiveConsistencyDirichletStop(["A", "B", "C"], {
        threshold: 1,
      }),
    ).toThrow(/threshold/);
    expect(() =>
      aggregate.adaptiveConsistencyCrpStop(["A"], { horizon: 1.5 }),
    ).toThrow(/horizon/);
    expect(() =>
      aggregate.adaptiveConsistencyCrpStop(["A"], { nSimulations: 0 }),
    ).toThrow(/n_simulations/);
    expect(() => aggregate.adaptiveConsistencyCrpStop(["A"], { seed: -1 })).toThrow(
      /seed/,
    );
  });
});

describe("aggregate runtime options follow Python coercion and null semantics", () => {
  const topk = [
    [0, -2],
    [-1, -3],
    [-2, -4],
  ];

  it("does not replace explicit null for strict enum, scalar, or boolean options", () => {
    expect(() =>
      aggregate.selfCertainty(topk, { aggregate: null as any }),
    ).toThrow(/aggregate/);
    expect(() =>
      aggregate.prmAggregate([0.5], { method: null as any }),
    ).toThrow(/method/);
    expect(() =>
      aggregate.weightedMajorityVote(["A"], [0.5], {
        aggregate: null as any,
      }),
    ).toThrow(/aggregate/);
    expect(() =>
      aggregate.softmaxWeightedVote(["A"], [0.5], {
        temperature: null as any,
      }),
    ).toThrow(/temperature/);
    expect(() =>
      aggregate.rankWeightedVote(["A"], [0.5], { p: null as any }),
    ).toThrow(/p/);
    expect(() =>
      aggregate.bestOfMajority(["A"], [0.5], { alpha: null as any }),
    ).toThrow(/alpha/);
    expect(() =>
      aggregate.deepconfConfidence(topk, {
        mode: "bottom_group",
        bottomQuantile: null as any,
      }),
    ).toThrow(/bottom_quantile/);
    expect(() =>
      aggregate.deepconfConfidence(topk, {
        mode: "lowest_group",
        window: null as any,
      }),
    ).toThrow(/window/);
    expect(() =>
      aggregate.cgesVote(["A"], [0.8], { allowOther: null as any }),
    ).toThrow(/allow_other/);
    expect(() =>
      aggregate.cgesStop(["A"], [0.8], { includeOther: null as any }),
    ).toThrow(/include_other/);
    expect(() =>
      aggregate.adaptiveConsistencyDirichletStop(["A"], {
        threshold: null as any,
      }),
    ).toThrow(/threshold/);
    expect(() =>
      aggregate.deepconfStopThreshold([1], { keep: null as any }),
    ).toThrow(/keep/);
    expect(() =>
      aggregate.deepconfOnlineStop(topk, 2, { window: null as any }),
    ).toThrow(/window/);
    const calibrationState = {
      correctLogits: [-1, 1],
      incorrectLogits: [-2, 2],
      correctBandwidth: 0.5,
      incorrectBandwidth: 0.5,
      binEdges: [-Infinity, Infinity],
      binProbability: [0.5],
    };
    expect(() =>
      new aggregate.KDEVoteCalibration({
        ...calibrationState,
        kernel: null as any,
      }),
    ).toThrow(/gaussian/);
    expect(() =>
      new aggregate.KDEVoteCalibration({
        ...calibrationState,
        binning: null as any,
      }),
    ).toThrow(/quantile/);
  });

  it("retains the three Python options for which None is meaningful", () => {
    expect(aggregate.picsar([-0.1, -0.2], { answerStart: null })).toBeCloseTo(
      -0.3,
      14,
    );
    expect(
      aggregate.majorityOfTheBests(["A", "B", "A"], [0.1, 0.9, 0.8], {
        m: null,
      }),
    ).toBe(
      aggregate.majorityOfTheBests(["A", "B", "A"], [0.1, 0.9, 0.8]),
    );
    expect(
      aggregate.adaptiveConsistencyCrpStop(["A"], {
        horizon: 1,
        seed: null,
        returnProb: true,
      }),
    ).toEqual([true, 1]);
  });

  it("rejects string coercion for Python parameters that use direct comparisons", () => {
    expect(() =>
      aggregate.bestOfMajority(["A"], [0.5], { alpha: "0.5" as any }),
    ).toThrow(/alpha/);
    expect(() =>
      aggregate.softmaxWeightedVote(["A"], [0.5], {
        temperature: "1" as any,
      }),
    ).toThrow(/temperature/);
    expect(() =>
      aggregate.rankWeightedVote(["A"], [0.5], { p: "1" as any }),
    ).toThrow(/p/);
    expect(() =>
      aggregate.logitWeightedVote(["A"], [0.5], {
        threshold: "0.5" as any,
      }),
    ).toThrow(/threshold/);
    expect(() =>
      aggregate.cgesStop(["A"], [0.8], { threshold: "0.5" as any }),
    ).toThrow(/threshold/);
    expect(() =>
      aggregate.adaptiveConsistencyStop(["A"], {
        threshold: "0.5" as any,
      }),
    ).toThrow(/threshold/);
    expect(() =>
      aggregate.deepconfStopThreshold([1, 2], { keep: "0.5" as any }),
    ).toThrow(/keep/);
    expect(() =>
      aggregate.deepconfOnlineStop(topk, "2" as any, { window: 2 }),
    ).toThrow(/threshold/);
  });

  it("accepts Python bool-as-int behavior for comparison and index options", () => {
    expect(
      aggregate.picsar([-0.2, -0.4, -0.5], { answerStart: true as any }),
    ).toBe(
      aggregate.picsar([-0.2, -0.4, -0.5], { answerStart: 1 }),
    );
    expect(
      aggregate.softmaxWeightedVote(["A", "B"], [0.2, 0.8], {
        temperature: true as any,
      }),
    ).toBe(
      aggregate.softmaxWeightedVote(["A", "B"], [0.2, 0.8], {
        temperature: 1,
      }),
    );
    expect(
      aggregate.rankWeightedVote(["A", "A", "B"], [0.1, 0.2, 0.9], {
        p: false as any,
      }),
    ).toBe(
      aggregate.rankWeightedVote(["A", "A", "B"], [0.1, 0.2, 0.9], {
        p: 0,
      }),
    );
    expect(
      aggregate.deepconfStopThreshold([1, 2, 3], { keep: true as any }),
    ).toBe(1);
    expect(
      aggregate.deepconfConfidence(topk, {
        mode: "bottom_group",
        window: 2,
        bottomQuantile: true as any,
      }),
    ).toBe(
      aggregate.deepconfConfidence(topk, {
        mode: "bottom_group",
        window: 2,
        bottomQuantile: 1,
      }),
    );
  });

  it("uses Python int conversion for DeepConf window and tail lengths", () => {
    const tailTwo = aggregate.deepconfConfidence(topk, {
      mode: "tail",
      tailTokens: 2,
    });
    expect(
      aggregate.deepconfConfidence(topk, {
        mode: "tail",
        tailTokens: "2" as any,
      }),
    ).toBe(tailTwo);
    expect(
      aggregate.deepconfConfidence(topk, {
        mode: "tail",
        tailTokens: 2.9,
      }),
    ).toBe(tailTwo);
    expect(
      aggregate.deepconfConfidence(topk, {
        mode: "lowest_group",
        window: "2" as any,
      }),
    ).toBe(
      aggregate.deepconfConfidence(topk, {
        mode: "lowest_group",
        window: 2,
      }),
    );
    expect(aggregate.deepconfOnlineStop(topk, 2, { window: "2" as any })).toBe(
      aggregate.deepconfOnlineStop(topk, 2, { window: 2 }),
    );
    expect(() =>
      aggregate.deepconfConfidence(topk, {
        mode: "tail",
        tailTokens: "2.0" as any,
      }),
    ).toThrow(/tail_tokens/);
  });

  it("mirrors Python float parsing where KDE state explicitly calls float", () => {
    const calibration = new aggregate.KDEVoteCalibration({
      correctLogits: [-1, 1],
      incorrectLogits: [-2, 2],
      correctBandwidth: "5e-1" as any,
      incorrectBandwidth: true as any,
      binEdges: [-Infinity, Infinity],
      binProbability: [0.5],
    });
    expect(calibration.correctBandwidth).toBe(0.5);
    expect(calibration.incorrectBandwidth).toBe(1);
    expect(() =>
      new aggregate.KDEVoteCalibration({
        correctLogits: [-1, 1],
        incorrectLogits: [-2, 2],
        correctBandwidth: "0x10" as any,
        incorrectBandwidth: 1,
        binEdges: [-Infinity, Infinity],
        binProbability: [0.5],
      }),
    ).toThrow(/correct_bandwidth/);
    expect(() =>
      new aggregate.KDEVoteCalibration({
        correctLogits: [-1, 1],
        incorrectLogits: [-2, 2],
        correctBandwidth: "nan" as any,
        incorrectBandwidth: 1,
        binEdges: [-Infinity, Infinity],
        binProbability: [0.5],
      }),
    ).toThrow(/finite/);

    expect(
      aggregate.fitKdeVoteCalibration([0.8, 0.9, 0.1, 0.2], [1, 1, 0, 0], {
        nBins: 2,
        bandwidth: true as any,
      }).correctBandwidth,
    ).toBe(1);
    // `_resolve_bandwidth` reserves strings for the literal `"scott"` before
    // it reaches float conversion, exactly as the Python implementation does.
    expect(() =>
      aggregate.fitKdeVoteCalibration([0.8, 0.9, 0.1, 0.2], [1, 1, 0, 0], {
        nBins: 2,
        bandwidth: "0.5" as any,
      }),
    ).toThrow(/bandwidth/);
  });

  it("uses Python truthiness for loose boolean and return-shape flags", () => {
    expect(
      aggregate.picsar([-0.2, -0.4, -0.5], {
        answerStart: 2,
        normalizeReasoning: NaN as any,
      }),
    ).toBe(
      aggregate.picsar([-0.2, -0.4, -0.5], {
        answerStart: 2,
        normalizeReasoning: true,
      }),
    );
    expect(
      aggregate.logprobMargin([[0, -1]], { useProb: NaN as any }),
    ).toBe(aggregate.logprobMargin([[0, -1]], { useProb: true }));
    expect(
      aggregate.bestOfN(["A", "B"], [0.1, 0.9], {
        returnIndex: NaN as any,
      }),
    ).toEqual(["B", 1]);
    expect(
      aggregate.bestOfN(["A", "B"], [0.1, 0.9], {
        returnIndex: {} as any,
      }),
    ).toBe("B");
    equivalent(
      aggregate.adaptiveConsistencyStop(["A"], { returnProb: NaN as any }),
      [false, 0.75],
    );
    expect(
      aggregate.adaptiveConsistencyStop(["A"], { returnProb: {} as any }),
    ).toBe(false);
  });
});
