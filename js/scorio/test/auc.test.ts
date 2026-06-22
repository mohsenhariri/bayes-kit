import { describe, expect, it } from "vitest";

import { aucAtK, aucAtKCi } from "../src/eval/auc.js";

const round = (x: number, d: number) => Number(x.toFixed(d));

const R = [
  [0, 1, 1, 0, 1],
  [1, 1, 0, 1, 1],
];

describe("aucAtK", () => {
  // Doctest values from auc.py.
  it("matches docstring values", () => {
    expect(round(aucAtK(R, 1), 6)).toBe(0.7);
    expect(round(aucAtK(R, 2), 6)).toBe(0.825);
    expect(round(aucAtK(R, 3), 6)).toBe(0.9);
  });

  // Reference values generated from the Python implementation.
  it("matches Python for higher k", () => {
    expect(round(aucAtK(R, 4), 6)).toBe(0.933333);
    expect(round(aucAtK(R, 5), 6)).toBe(0.95);
  });
});

describe("aucAtKCi", () => {
  // Reference values generated from the Python implementation.
  it("matches Python for k=1 (delegates to passAtKCi)", () => {
    const [mu, sigma, lo, hi] = aucAtKCi(R, 1);
    expect(round(mu, 9)).toBe(0.642857143);
    expect(round(sigma, 9)).toBe(0.118450885);
    // ndtri uses Acklam's approximation (~1e-9 rel error), so the interval
    // endpoints can differ from scipy in the last digit; assert at 8 places.
    expect(round(lo, 8)).toBe(0.41069767);
    expect(round(hi, 8)).toBe(0.87501661);
  });

  it("matches Python for k=2", () => {
    const [mu, sigma, lo, hi] = aucAtKCi(R, 2);
    expect(round(mu, 9)).toBe(0.741071429);
    expect(round(sigma, 9)).toBe(0.106770185);
    expect(round(lo, 9)).toBe(0.531805711);
    expect(round(hi, 9)).toBe(0.950337146);
  });

  it("matches Python for k=3", () => {
    const [mu, sigma, lo, hi] = aucAtKCi(R, 3);
    expect(round(mu, 9)).toBe(0.80952381);
    expect(round(sigma, 9)).toBe(0.095060373);
    expect(round(lo, 9)).toBe(0.623208903);
    expect(round(hi, 9)).toBe(0.995838716);
  });

  it("matches Python for k=5 (hi clipped to bound)", () => {
    const [mu, sigma, lo, hi] = aucAtKCi(R, 5);
    expect(round(mu, 9)).toBe(0.878787879);
    expect(round(sigma, 9)).toBe(0.074400784);
    expect(round(lo, 9)).toBe(0.732965022);
    expect(round(hi, 9)).toBe(1.0);
  });

  it("matches Python with confidence=0.90", () => {
    const [mu, sigma, lo, hi] = aucAtKCi(R, 2, 0.9);
    expect(round(mu, 9)).toBe(0.741071429);
    expect(round(sigma, 9)).toBe(0.106770185);
    expect(round(lo, 9)).toBe(0.565450102);
    expect(round(hi, 9)).toBe(0.916692755);
  });

  it("matches Python with custom prior", () => {
    const [mu, sigma, lo, hi] = aucAtKCi(R, 3, 0.95, [0.0, 1.0], 2.0, 0.5);
    expect(round(mu, 9)).toBe(0.873477812);
    expect(round(sigma, 9)).toBe(0.073275648);
    expect(round(lo, 9)).toBe(0.729860181);
    expect(round(hi, 9)).toBe(1.0);
  });
});
