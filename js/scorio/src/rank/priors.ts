/**
 * Prior penalties for MAP ranking estimators. Port of `scorio/rank/priors.py`.
 *
 * Several methods estimate latent log-strengths `theta` by minimizing a MAP
 * objective `NLL(theta) + penalty(theta)`, where `penalty` is the negative
 * log-prior (up to a constant). These classes implement that term.
 */

import { clip } from "./internal/special.js";
import type { TensorInput } from "./internal/tensor.js";

function finiteScalar(value: unknown, name: string): number {
  if (typeof value !== "number") {
    throw new TypeError(`${name} must be a finite scalar`);
  }
  if (!Number.isFinite(value)) {
    throw new Error(`${name} must be finite`);
  }
  return value;
}

/**
 * Abstract base class for prior penalties on a log-strength vector `theta`.
 *
 * This is a runtime export, matching Python's `scorio.rank.Prior`, as well as
 * a TypeScript type. Instantiating it directly is an error.
 */
export abstract class Prior {
  constructor() {
    if (new.target === Prior) {
      throw new TypeError("Prior is an abstract class and cannot be instantiated");
    }
  }

  /** Evaluate the prior penalty `P(theta)` (added to a negative log-likelihood). */
  abstract penalty(theta: readonly number[]): number;
}

/** Isotropic Gaussian prior: `P(theta) = Σ (theta_i - mean)² / (2 var)`. */
export class GaussianPrior extends Prior {
  readonly mean: number;
  readonly var: number;

  constructor(mean = 0, variance = 1) {
    super();
    mean = finiteScalar(mean, "Mean");
    variance = finiteScalar(variance, "Variance");
    if (!(variance > 0)) throw new Error("Variance must be positive");
    this.mean = mean;
    this.var = variance;
  }

  penalty(theta: readonly number[]): number {
    let s = 0;
    for (const t of theta) s += (t - this.mean) ** 2;
    return s / (2 * this.var);
  }
}

/** Laplace prior: `P(theta) = Σ |theta_i - loc| / scale`. */
export class LaplacePrior extends Prior {
  readonly loc: number;
  readonly scale: number;

  constructor(loc = 0, scale = 1) {
    super();
    loc = finiteScalar(loc, "Location");
    scale = finiteScalar(scale, "Scale");
    if (!(scale > 0)) throw new Error("Scale must be positive");
    this.loc = loc;
    this.scale = scale;
  }

  penalty(theta: readonly number[]): number {
    let s = 0;
    for (const t of theta) s += Math.abs(t - this.loc);
    return s / this.scale;
  }
}

/** Cauchy prior: `P(theta) = Σ log(1 + ((theta_i - loc)/scale)²)`. */
export class CauchyPrior extends Prior {
  readonly loc: number;
  readonly scale: number;

  constructor(loc = 0, scale = 1) {
    super();
    loc = finiteScalar(loc, "Location");
    scale = finiteScalar(scale, "Scale");
    if (!(scale > 0)) throw new Error("Scale must be positive");
    this.loc = loc;
    this.scale = scale;
  }

  penalty(theta: readonly number[]): number {
    let s = 0;
    for (const t of theta) {
      const z = (t - this.loc) / this.scale;
      s += Math.log1p(z * z);
    }
    return s;
  }
}

/** Improper uniform prior: `P(theta) = 0` (disables regularization). */
export class UniformPrior extends Prior {
  penalty(_theta: readonly number[]): number {
    return 0;
  }
}

/** User-defined prior penalty wrapper. */
export class CustomPrior extends Prior {
  private readonly fn: (theta: readonly number[]) => number;

  constructor(penaltyFn: (theta: readonly number[]) => number) {
    super();
    if (typeof penaltyFn !== "function") {
      throw new Error("penalty_fn must be callable");
    }
    this.fn = penaltyFn;
  }

  penalty(theta: readonly number[]): number {
    return this.fn(theta);
  }
}

/**
 * Empirical Gaussian prior built from a prior outcome tensor `R0` of shape
 * `(L, M, D)` or `(L, M)`. Prior means are the centered empirical logits.
 */
export class EmpiricalPrior extends Prior {
  readonly R0: number[][][];
  readonly var: number;
  readonly eps: number;
  readonly priorMean: number[];
  /** Snake-case alias matching Python's public instance attribute. */
  readonly prior_mean: number[];

  constructor(R0: TensorInput, variance = 1, eps = 1e-6) {
    super();
    variance = finiteScalar(variance, "Variance");
    if (!(variance > 0)) throw new Error("Variance must be positive");
    eps = finiteScalar(eps, "eps");
    if (!(eps > 0 && eps < 0.5)) {
      throw new Error("eps must be strictly between 0 and 0.5");
    }

    const raw = R0 as unknown;
    if (!Array.isArray(raw)) {
      throw new Error("R0 must be 2D (L, M) or 3D (L, M, D)");
    }
    const isScalar = (value: unknown): boolean =>
      typeof value === "number" || typeof value === "boolean";
    let rows: number[][][];
    const looks2d = raw.every(
      (row) => Array.isArray(row) && row.every((value) => isScalar(value)),
    );
    const looks3d = raw.every(
      (matrix) =>
        Array.isArray(matrix) &&
        matrix.every(
          (row) => Array.isArray(row) && row.every((value) => isScalar(value)),
        ),
    );
    if (looks2d) {
      rows = raw.map((row) =>
        (row as unknown[]).map((value) => [Number(value)]),
      );
    } else if (looks3d) {
      rows = raw.map((matrix) =>
        (matrix as unknown[][]).map((row) => row.map((value) => Number(value))),
      );
    } else {
      throw new Error("R0 must be 2D (L, M) or 3D (L, M, D)");
    }

    const L = rows.length;
    const M = rows[0]?.length ?? 0;
    const D = rows[0]?.[0]?.length ?? 0;
    if (L === 0 || M === 0 || D === 0) {
      throw new Error("R0 must be non-empty in every dimension");
    }
    for (const matrix of rows) {
      if (matrix.length !== M || matrix.some((row) => row.length !== D)) {
        throw new Error("R0 must be a rectangular 2D or 3D array");
      }
      for (const row of matrix) {
        for (const value of row) {
          if (!Number.isFinite(value)) {
            throw new Error("R0 must not contain NaN or Inf values");
          }
          if (value !== 0 && value !== 1) {
            throw new Error("R0 must contain only binary values (0 or 1)");
          }
        }
      }
    }
    const acc = rows.map((mat) => {
      let s = 0;
      let n = 0;
      for (const row of mat)
        for (const v of row) {
          s += v;
          n += 1;
        }
      return s / n;
    });
    const logits = acc.map((a) => {
      const c = clip(a, eps, 1 - eps);
      return Math.log(c / (1 - c));
    });
    const mean = logits.reduce((s, v) => s + v, 0) / L;
    this.priorMean = logits.map((v) => v - mean);
    this.prior_mean = this.priorMean;
    this.R0 = rows;
    this.var = variance;
    this.eps = eps;
  }

  penalty(theta: readonly number[]): number {
    if (theta.length !== this.priorMean.length) {
      throw new Error(
        `theta length (${theta.length}) must match number of models (${this.priorMean.length})`,
      );
    }
    let s = 0;
    for (let i = 0; i < theta.length; i++) s += (theta[i]! - this.priorMean[i]!) ** 2;
    return s / (2 * this.var);
  }
}

/**
 * Normalize a prior argument to a {@link Prior}. A numeric value is interpreted
 * as the variance of a zero-mean {@link GaussianPrior} (matching `_coerce_prior`).
 */
export function coercePrior(prior: Prior | number): Prior;
export function coercePrior(prior: unknown): Prior {
  if (typeof prior === "boolean") {
    throw new TypeError("prior must be a Prior object or positive finite float");
  }
  if (typeof prior === "number") {
    if (!Number.isFinite(prior) || prior <= 0) {
      throw new Error("prior must be a positive finite scalar variance");
    }
    return new GaussianPrior(0, prior);
  }
  if (prior instanceof Prior) return prior;
  throw new TypeError("prior must be a Prior object or float");
}
