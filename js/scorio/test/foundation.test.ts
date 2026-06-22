import { describe, expect, it } from "vitest";

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
