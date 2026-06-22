import { describe, expect, it } from "vitest";

import { majAtK, majAtKCi } from "../src/eval/maj.js";

const round = (x: number, d: number) => Number(x.toFixed(d));

const R = [
  [0, 1, 1, 0, 1],
  [1, 1, 0, 1, 1],
];

describe("majAtK", () => {
  it("matches the doctest values", () => {
    expect(round(majAtK(R, 1), 6)).toBe(0.7);
    expect(round(majAtK(R, 2), 6)).toBe(0.45);
    expect(round(majAtK(R, 3), 6)).toBe(0.85);
  });
});

describe("majAtKCi", () => {
  it("matches the doctest values for k=2", () => {
    const [mu, sigma, lo, hi] = majAtKCi(R, 2);
    expect(round(mu, 6)).toBe(0.446429);
    expect(round(sigma, 6)).toBe(0.146167);
    expect(round(lo, 4)).toBe(0.1599);
    expect(round(hi, 4)).toBe(0.7329);
  });

  it("matches the doctest values for k=3", () => {
    const [mu, sigma, lo, hi] = majAtKCi(R, 3);
    expect(round(mu, 6)).toBe(0.684524);
    expect(round(sigma, 6)).toBe(0.151958);
    expect(round(lo, 4)).toBe(0.3867);
    expect(round(hi, 4)).toBe(0.9824);
  });
});
