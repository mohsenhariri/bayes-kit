import { describe, expect, it } from "vitest";

import {
  CauchyPrior,
  CustomPrior,
  EmpiricalPrior,
  GaussianPrior,
  LaplacePrior,
  Prior,
  UniformPrior,
} from "../src/rank/index.js";
import fixtures from "./fixtures/utils.json";

interface PriorFixtures {
  R0: number[][];
  theta: number[];
  empirical_prior_mean: number[];
  penalties: Record<string, number>;
}

const fx = (fixtures as { priors: PriorFixtures }).priors;

describe("Prior runtime API", () => {
  it("exports an abstract runtime base class", () => {
    expect(typeof Prior).toBe("function");
    expect(() => new (Prior as unknown as new () => Prior)()).toThrow(TypeError);
  });

  it("makes every concrete prior an instance of Prior", () => {
    const priors = [
      new GaussianPrior(),
      new LaplacePrior(),
      new CauchyPrior(),
      new UniformPrior(),
      new CustomPrior((theta) => theta.reduce((sum, value) => sum + value * value, 0)),
      new EmpiricalPrior(fx.R0),
    ];
    for (const prior of priors) expect(prior).toBeInstanceOf(Prior);
  });

  it("matches Python penalty results", () => {
    const priors: Record<string, Prior> = {
      gaussian: new GaussianPrior(0.2, 1.5),
      laplace: new LaplacePrior(0.2, 1.5),
      cauchy: new CauchyPrior(0.2, 1.5),
      uniform: new UniformPrior(),
      custom: new CustomPrior((theta) =>
        theta.reduce((sum, value) => sum + Math.abs(value), 0),
      ),
      empirical: new EmpiricalPrior(fx.R0, 1.5),
    };
    for (const [name, prior] of Object.entries(priors)) {
      expect(prior.penalty(fx.theta)).toBeCloseTo(fx.penalties[name]!, 14);
    }
  });

  it("exposes Python-compatible EmpiricalPrior state", () => {
    const prior = new EmpiricalPrior(fx.R0, 1.5);
    for (let i = 0; i < prior.prior_mean.length; i++) {
      expect(prior.prior_mean[i]).toBeCloseTo(fx.empirical_prior_mean[i]!, 14);
    }
    expect(prior.prior_mean).toBe(prior.priorMean);
    expect(prior.R0).toEqual(fx.R0.map((row) => row.map((value) => [value])));
  });

  it("matches Python scalar validation", () => {
    expect(() => new GaussianPrior(NaN)).toThrow(/Mean must be finite/);
    expect(() => new GaussianPrior(0, Infinity)).toThrow(/Variance must be finite/);
    expect(() => new LaplacePrior(NaN)).toThrow(/Location must be finite/);
    expect(() => new CauchyPrior(0, NaN)).toThrow(/Scale must be finite/);
    expect(() => new EmpiricalPrior([[0, 1]], Infinity)).toThrow(
      /Variance must be finite/,
    );
    for (const eps of [0, -0.1, 0.5, 1, NaN]) {
      expect(() => new EmpiricalPrior([[0, 1]], 1, eps)).toThrow(/eps/);
    }
  });

  it("validates empirical outcomes and parameter length", () => {
    expect(() => new EmpiricalPrior([[], []])).toThrow(/non-empty/);
    expect(() => new EmpiricalPrior([[0, NaN], [1, 0]])).toThrow(/NaN or Inf/);
    expect(() => new EmpiricalPrior([[0, 2], [1, 0]])).toThrow(/binary/);
    expect(() => new EmpiricalPrior([[[[0]]]] as any)).toThrow(/2D.*3D/);
    const prior = new EmpiricalPrior([[0, 1], [1, 1]]);
    expect(() => prior.penalty([0])).toThrow(/must match number of models/);
  });
});
