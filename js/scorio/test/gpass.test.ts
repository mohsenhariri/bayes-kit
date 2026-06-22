import { describe, expect, it } from "vitest";

import {
  gPassAtK,
  gPassAtKTau,
  mgPassAtK,
  gPassAtKCi,
  gPassAtKTauCi,
  mgPassAtKCi,
} from "../src/eval/gpass.js";

const round = (x: number, d: number) => Number(x.toFixed(d));

const R = [
  [0, 1, 1, 0, 1],
  [1, 1, 0, 1, 1],
];

describe("gPassAtK", () => {
  it("matches doctest values (alias of pass^k... actually pass_hat_k)", () => {
    // Doctest: round(g_pass_at_k(R, 1), 6) -> 0.7
    expect(round(gPassAtK(R, 1), 6)).toBe(0.7);
    // Doctest: round(g_pass_at_k(R, 2), 6) -> 0.45
    expect(round(gPassAtK(R, 2), 6)).toBe(0.45);
  });
});

describe("gPassAtKTau", () => {
  it("matches doctest values", () => {
    // Doctest: round(g_pass_at_k_tau(R, 2, 0.5), 6) -> 0.95
    expect(round(gPassAtKTau(R, 2, 0.5), 6)).toBe(0.95);
    // Doctest: round(g_pass_at_k_tau(R, 2, 1.0), 6) -> 0.45
    expect(round(gPassAtKTau(R, 2, 1.0), 6)).toBe(0.45);
  });

  it("matches Python reference for extra cases", () => {
    // python: eval.g_pass_at_k_tau(R, 3, 0.5) -> 0.85
    expect(round(gPassAtKTau(R, 3, 0.5), 6)).toBe(0.85);
  });
});

describe("mgPassAtK", () => {
  it("matches doctest values", () => {
    // Doctest: round(mg_pass_at_k(R, 2), 6) -> 0.45
    expect(round(mgPassAtK(R, 2), 6)).toBe(0.45);
    // Doctest: round(mg_pass_at_k(R, 3), 6) -> 0.166667
    expect(round(mgPassAtK(R, 3), 6)).toBe(0.166667);
  });
});

describe("gPassAtKCi", () => {
  it("matches Python reference values", () => {
    // python: eval.g_pass_at_k_ci(R, 2)
    const [mu, sigma, lo, hi] = gPassAtKCi(R, 2);
    expect(round(mu, 6)).toBe(0.446429);
    expect(round(sigma, 6)).toBe(0.146167);
    expect(round(lo, 6)).toBe(0.159946);
    expect(round(hi, 6)).toBe(0.732911);
  });
});

describe("gPassAtKTauCi", () => {
  it("matches Python reference values", () => {
    // python: eval.g_pass_at_k_tau_ci(R, 2, 0.5)
    const [mu, sigma, lo, hi] = gPassAtKTauCi(R, 2, 0.5);
    expect(round(mu, 6)).toBe(0.839286);
    expect(round(sigma, 6)).toBe(0.097263);
    expect(round(lo, 6)).toBe(0.648654);
    expect(round(hi, 6)).toBe(1.0);
  });

  it("matches Python reference for an interior tau (k=3)", () => {
    // python: eval.g_pass_at_k_tau_ci(R, 3, 0.5)
    const [mu, sigma, lo, hi] = gPassAtKTauCi(R, 3, 0.5);
    expect(round(mu, 6)).toBe(0.684524);
    expect(round(sigma, 6)).toBe(0.151958);
    expect(round(lo, 6)).toBe(0.386692);
    expect(round(hi, 6)).toBe(0.982356);
  });
});

describe("mgPassAtKCi", () => {
  it("matches Python reference values", () => {
    // python: eval.mg_pass_at_k_ci(R, 3)
    const [mu, sigma, lo, hi] = mgPassAtKCi(R, 3);
    expect(round(mu, 6)).toBe(0.218254);
    expect(round(sigma, 6)).toBe(0.098816);
    expect(round(lo, 6)).toBe(0.024578);
    expect(round(hi, 6)).toBe(0.41193);
  });
});
