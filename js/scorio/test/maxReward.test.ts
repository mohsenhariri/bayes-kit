import { describe, it, expect } from "vitest";
import { maxAtK, maxAtKCi } from "../src/eval/maxReward.js";
import { passAtK } from "../src/eval/passAtK.js";

const round = (x: number, d: number) => Number(x.toFixed(d));

describe("maxAtK (doctests)", () => {
  it("binary reduces to Pass@k", () => {
    const R = [
      [0, 1, 1, 0, 1],
      [1, 1, 0, 1, 1],
    ];
    expect(round(maxAtK(R, 2), 6)).toBe(0.95);
  });

  it("weighted categorical rewards", () => {
    const R = [
      [0, 1, 2, 2, 1],
      [1, 1, 0, 2, 2],
    ];
    const w = [0.0, 0.5, 1.0];
    expect(round(maxAtK(R, 2, w), 6)).toBe(0.85);
  });
});

describe("maxAtKCi (doctests)", () => {
  it("binary", () => {
    const R = [
      [0, 1, 1, 0, 1],
      [1, 1, 0, 1, 1],
    ];
    const [mu, sigma, lo, hi] = maxAtKCi(R, 2);
    expect(round(mu, 6)).toBe(0.839286);
    expect(round(sigma, 6)).toBe(0.097263);
    expect(round(lo, 4)).toBe(0.6487);
    expect(round(hi, 4)).toBe(1.0);
  });

  it("weighted categorical rewards", () => {
    const R = [
      [0, 1, 2, 2, 1],
      [1, 1, 0, 2, 2],
    ];
    const w = [0.0, 0.5, 1.0];
    const [mu, sigma, lo, hi] = maxAtKCi(R, 2, w);
    expect(round(mu, 6)).toBe(0.75);
    expect(round(sigma, 6)).toBe(0.08812);
    expect(round(lo, 4)).toBe(0.5773);
    expect(round(hi, 4)).toBe(0.9227);
  });
});

describe("maxAtKCi (Python-generated reference values for branches without doctests)", () => {
  const R = [
    [0, 1, 2, 2, 1],
    [1, 1, 0, 2, 2],
  ];
  const w = [0.0, 0.5, 1.0];

  it("with R0 prior and custom confidence (k=2)", () => {
    const R0 = [
      [0, 2],
      [1, 1],
    ];
    const [mu, sigma, lo, hi] = maxAtKCi(R, 2, w, R0, 0.9);
    expect(round(mu, 10)).toBe(0.7363636364);
    expect(round(sigma, 10)).toBe(0.0801797884);
    // lo/hi depend on the ndtri approximation (rel err ~1e-9); compare loosely.
    expect(lo).toBeCloseTo(0.6044796205, 7);
    expect(hi).toBeCloseTo(0.8682476522, 7);
  });

  it("with explicit bounds and confidence=0.99 (k=3)", () => {
    const [mu, sigma, lo, hi] = maxAtKCi(R, 3, w, undefined, 0.99, [0.0, 1.0]);
    expect(round(mu, 10)).toBe(0.8375);
    expect(round(sigma, 10)).toBe(0.0781056211);
    expect(lo).toBeCloseTo(0.6363132523, 7);
    expect(round(hi, 10)).toBe(1.0);
  });

  it("k=1 reduces to bayesCi", () => {
    const [mu, sigma, lo, hi] = maxAtKCi(R, 1, w);
    expect(round(mu, 10)).toBe(0.5625);
    expect(round(sigma, 10)).toBe(0.091997509);
    expect(lo).toBeCloseTo(0.3821881956, 7);
    expect(hi).toBeCloseTo(0.7428118044, 7);
  });
});

describe("pass@k parity (w=[0,1] => maxAtK equals passAtK)", () => {
  const R = [
    [0, 1, 1, 0, 1],
    [1, 1, 0, 1, 1],
    [0, 0, 1, 1, 0],
  ];
  for (const k of [1, 2]) {
    it(`k=${k}`, () => {
      expect(maxAtK(R, k)).toBeCloseTo(passAtK(R, k), 12);
    });
  }
});
