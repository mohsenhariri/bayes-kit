import { describe, expect, it } from "vitest";

import * as evalApi from "../src/eval/index.js";
import { bayes, bayesCi } from "../src/eval/bayes.js";
import { avg, avgCi } from "../src/eval/avg.js";
import {
  passAtK,
  passHatK,
  passAtKCi,
  passHatKCi,
} from "../src/eval/passAtK.js";

const round = (x: number, d: number) => Number(x.toFixed(d));

const Rbin = [
  [0, 1, 1, 0, 1],
  [1, 1, 0, 1, 1],
];
const Rmc = [
  [0, 1, 2, 2, 1],
  [1, 1, 0, 2, 2],
];
const w = [0.0, 0.5, 1.0];
const R0 = [
  [0, 2],
  [1, 2],
];

describe("bayes", () => {
  it("matches docstring values with prior", () => {
    const [mu, sigma] = bayes(Rmc, w, R0);
    expect(round(mu, 6)).toBe(0.575);
    expect(round(sigma, 6)).toBe(0.084275);
  });
  it("matches docstring values without prior", () => {
    const [mu, sigma] = bayes(Rmc, w);
    expect(round(mu, 6)).toBe(0.5625);
    expect(round(sigma, 6)).toBe(0.091998);
  });
  it("bayesCi matches docstring", () => {
    const [mu, sigma, lo, hi] = bayesCi(Rbin, undefined, undefined, 0.95, [
      0.0, 1.0,
    ]);
    expect(round(mu, 6)).toBe(0.642857);
    expect(round(sigma, 6)).toBe(0.118451);
    expect(round(lo, 4)).toBe(0.4107);
    expect(round(hi, 4)).toBe(0.875);
  });
});

describe("avg", () => {
  it("binary matches docstring", () => {
    const [a, sigma] = avg(Rbin);
    expect(round(a, 6)).toBe(0.7);
    expect(round(sigma, 6)).toBe(0.165831);
  });
  it("weighted matches docstring", () => {
    const [a, sigma] = avg(Rmc, w);
    expect(round(a, 6)).toBe(0.6);
    expect(round(sigma, 6)).toBe(0.147196);
  });
  it("avgCi weighted matches docstring", () => {
    const [a, sigma, lo, hi] = avgCi(Rmc, w, 0.95);
    expect(round(a, 4)).toBe(0.6);
    expect(round(sigma, 4)).toBe(0.1472);
    expect(round(lo, 4)).toBe(0.3115);
    expect(round(hi, 4)).toBe(0.8885);
  });
});

describe("pass family", () => {
  it("passAtK matches docstring", () => {
    expect(round(passAtK(Rbin, 1), 6)).toBe(0.7);
    expect(round(passAtK(Rbin, 2), 6)).toBe(0.95);
  });
  it("passHatK matches docstring", () => {
    expect(round(passHatK(Rbin, 1), 6)).toBe(0.7);
    expect(round(passHatK(Rbin, 2), 6)).toBe(0.45);
  });
  it("passAtKCi matches docstring", () => {
    let [mu, sigma, lo, hi] = passAtKCi(Rbin, 1);
    expect([round(mu, 6), round(sigma, 6), round(lo, 4), round(hi, 4)]).toEqual([
      0.642857, 0.118451, 0.4107, 0.875,
    ]);
    [mu, sigma, lo, hi] = passAtKCi(Rbin, 2);
    expect([round(mu, 6), round(sigma, 6), round(lo, 4), round(hi, 4)]).toEqual([
      0.839286, 0.097263, 0.6487, 1.0,
    ]);
  });
  it("passHatKCi matches docstring", () => {
    const [mu, sigma, lo, hi] = passHatKCi(Rbin, 2);
    expect([round(mu, 6), round(sigma, 6), round(lo, 4), round(hi, 4)]).toEqual([
      0.446429, 0.146167, 0.1599, 0.7329,
    ]);
  });
});

const largeExtremeCase = () => {
  const N = 2000;
  const k = 1000;
  const R = [Array<number>(N).fill(1), Array<number>(N).fill(0)];
  const unanimousWeights = Array<number>(k).fill(0.0);
  unanimousWeights[k - 1] = 1.0;
  const mgWeights = Array<number>(k).fill(0.0);
  for (let i = Math.ceil(k / 2); i < k; i++) mgWeights[i] = 2.0 / k;
  return { k, R, unanimousWeights, mgWeights };
};

describe("Python eval parity regressions", () => {
  it("keeps Python's hypergeometric point metrics finite at N=2000, k=1000", () => {
    const { k, R } = largeExtremeCase();
    const scores = [
      evalApi.passAtK(R, k),
      evalApi.passHatK(R, k),
      evalApi.gPassAtK(R, k),
      evalApi.gPassAtKTau(R, k, 0.5),
      evalApi.mgPassAtK(R, k),
      evalApi.majAtK(R, k),
      evalApi.geomDsAtK(R, k),
      evalApi.geoSpectrumAtK(R, k, 1.0),
    ];

    for (const score of scores) {
      expect(Number.isFinite(score)).toBe(true);
      expect(score).toBeCloseTo(0.5, 10);
    }
  });

  it("matches Python generalized-pass posterior values at k=1000", () => {
    const { k, R } = largeExtremeCase();
    const tau = evalApi.gPassAtKTauCi(R, k, 0.5);
    const mg = evalApi.mgPassAtKCi(R, k);
    const majority = evalApi.majAtKCi(R, k);

    expect(tau[0]).toBeCloseTo(0.49999999999909595, 10);
    expect(tau.slice(1)).toEqual([0, tau[0], tau[0]]);
    expect(mg[0]).toBeCloseTo(0.4995004994995972, 10);
    expect(mg.slice(1)).toEqual([0, mg[0], mg[0]]);
    expect(majority[0]).toBeCloseTo(0.49999999999909595, 10);
    expect(majority.slice(1)).toEqual([0, majority[0], majority[0]]);
  });

  it("matches Python spectrum and geometric posterior values at k=1000", () => {
    const { k, R, unanimousWeights, mgWeights } = largeExtremeCase();
    const unanimous = evalApi.thresholdSpectrumAtKCi(R, k, unanimousWeights);
    const mg = evalApi.thresholdSpectrumAtKCi(R, k, mgWeights);
    const questionwise = evalApi.geomAtKCi(R, k);
    const dataset = evalApi.geomDsAtKCi(R, k);
    const spectrum = evalApi.geoSpectrumAtKCi(R, k);
    const star = evalApi.geoSpectrumStarAtKCi(R, k);

    expect(unanimous[0]).toBeCloseTo(0.33338887037628834, 10);
    expect(unanimous[1]).toBeCloseTo(0.1178265814634313, 10);
    expect(mg[0]).toBeCloseTo(0.4995004994995975, 10);
    expect(mg.slice(1)).toEqual([0, mg[0], mg[0]]);
    expect(questionwise[0]).toBeCloseTo(0.40828229840166247, 10);
    expect(questionwise[1]).toBeCloseTo(0.07214774062254051, 10);
    expect(dataset[0]).toBeCloseTo(0.47142415242063146, 10);
    expect(dataset[1]).toBeCloseTo(0.0931431091385532, 10);
    expect(spectrum[0]).toBeCloseTo(0.5770377736500748, 10);
    expect(spectrum[1]).toBeCloseTo(0.050997039520504835, 10);
    expect(star).toEqual(spectrum);
  });

  it("reshapes a flat R0 to M rows for Bayes and Max@k", () => {
    const R = [
      [0, 1, 2],
      [2, 1, 0],
    ];
    const w = [0.0, 0.5, 1.0];
    const flatR0 = [0, 2, 1, 2];
    const nestedR0 = [
      [0, 2],
      [1, 2],
    ];

    expect(evalApi.bayes(R, w, flatR0)).toEqual(evalApi.bayes(R, w, nestedR0));
    expect(evalApi.bayesCi(R, w, flatR0)).toEqual(evalApi.bayesCi(R, w, nestedR0));
    expect(evalApi.maxAtKCi(R, 2, w, flatR0)).toEqual(
      evalApi.maxAtKCi(R, 2, w, nestedR0),
    );
    expect(evalApi.bayes(R, w, [])).toEqual(evalApi.bayes(R, w, [[], []]));
  });

  it("accepts null bounds as Python's explicit unbounded interval", () => {
    const R = [
      [1, 1, 1],
      [1, 1, 1],
    ];
    const pass = evalApi.passAtKCi(R, 1, 0.95, null);
    const spectrum = evalApi.geoSpectrumAtKCi(
      R,
      2,
      0.5,
      undefined,
      undefined,
      0.95,
      null,
    );

    expect(pass[3]).toBeGreaterThan(1.0);
    const bayesResult = evalApi.bayesCi(R, undefined, undefined, 0.95, null);
    pass.forEach((value, index) =>
      expect(value).toBeCloseTo(bayesResult[index]!, 12),
    );
    expect(spectrum[3]).toBeGreaterThan(1.0);
  });

  it("treats explicit null optionals as Python None", () => {
    expect(evalApi.avg(Rbin, null)).toEqual(evalApi.avg(Rbin));
    expect(evalApi.bayes(Rbin, null, null)).toEqual(evalApi.bayes(Rbin));
    expect(evalApi.maxAtK(Rbin, 2, null)).toBe(evalApi.maxAtK(Rbin, 2));
    expect(evalApi.maxAtKCi(Rbin, 2, null, null)).toEqual(
      evalApi.maxAtKCi(Rbin, 2),
    );
    expect(evalApi.geoSpectrumAtK(Rbin, 2, 0.5, null, null)).toBe(
      evalApi.geoSpectrumAtK(Rbin, 2),
    );
    expect(evalApi.geoSpectrumAtKCi(Rbin, 2, 0.5, null, null)).toEqual(
      evalApi.geoSpectrumAtKCi(Rbin, 2),
    );
  });

  it("mirrors NumPy integer coercion for boolean and integer-string outcomes", () => {
    const runtimeInput = [
      [false, "1", "0_0", true],
      ["1", false, true, "1"],
    ] as never;
    expect(evalApi.passAtK(runtimeInput, 2)).toBe(
      evalApi.passAtK(
        [
          [0, 1, 0, 1],
          [1, 0, 1, 1],
        ],
        2,
      ),
    );
    expect(() => evalApi.passAtK([["1.0"]] as never, 1)).toThrow(
      /integer-like/,
    );
  });

  it("matches Python's finite-float and generalized-k coercions", () => {
    const binary = [
      [0, 1, 0, 1, 0],
      [1, 0, 1, 0, 1],
    ];
    const fractional = binary.map((row) => row.map((value) => value + 0.8));
    expect(evalApi.passAtK(fractional, 1)).toBe(evalApi.passAtK(binary, 1));

    const ci = evalApi.passAtKCi(binary, 2.5);
    expect(ci[0]).toBeCloseTo(0.7762237762237763, 12);
    expect(ci[1]).toBeCloseTo(0.12026162842209105, 12);
    expect(Number.isNaN(evalApi.passAtK(binary, 2.5))).toBe(true);
    expect(
      evalApi.geomAtK(
        [
          [0, 1, 1],
          [1, 0, 1],
        ],
        1.5,
      ),
    ).toBeCloseTo(0.7071067811865476, 12);
  });

  it("exports exactly Python's snake_case eval surface plus camelCase aliases", () => {
    const pythonPublicNames = [
      "auc_at_k",
      "auc_at_k_ci",
      "avg",
      "avg_ci",
      "bayes",
      "bayes_ci",
      "g_pass_at_k",
      "g_pass_at_k_ci",
      "g_pass_at_k_tau",
      "g_pass_at_k_tau_ci",
      "geo_spectrum_at_k",
      "geo_spectrum_at_k_ci",
      "geo_spectrum_star_at_k",
      "geo_spectrum_star_at_k_ci",
      "geom_at_k",
      "geom_at_k_ci",
      "geom_ds_at_k",
      "geom_ds_at_k_ci",
      "maj_at_k",
      "maj_at_k_ci",
      "max_at_k",
      "max_at_k_ci",
      "mg_pass_at_k",
      "mg_pass_at_k_ci",
      "pass_at_k",
      "pass_at_k_ci",
      "pass_hat_k",
      "pass_hat_k_ci",
      "threshold_spectrum_at_k",
      "threshold_spectrum_at_k_ci",
      "unanimous_at_k",
      "unanimous_at_k_ci",
    ];
    const snakeCaseExports = Object.keys(evalApi)
      .filter((name) => name === "avg" || name === "bayes" || name.includes("_"))
      .sort();

    expect(snakeCaseExports).toEqual(pythonPublicNames);
    expect("normalCredibleInterval" in evalApi).toBe(false);
    expect("normal_credible_interval" in evalApi).toBe(false);
  });
});
