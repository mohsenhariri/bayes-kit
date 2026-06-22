/**
 * Prior penalties for MAP ranking estimators. Port of `scorio/rank/priors.py`.
 *
 * Several methods estimate latent log-strengths `theta` by minimizing a MAP
 * objective `NLL(theta) + penalty(theta)`, where `penalty` is the negative
 * log-prior (up to a constant). These classes implement that term.
 */

import { clip } from "./internal/special.js";
import type { TensorInput } from "./internal/tensor.js";

/** Common interface for prior penalties on a log-strength vector `theta`. */
export interface Prior {
  /** Evaluate the prior penalty `P(theta)` (added to a negative log-likelihood). */
  penalty(theta: readonly number[]): number;
}

/** Isotropic Gaussian prior: `P(theta) = Σ (theta_i - mean)² / (2 var)`. */
export class GaussianPrior implements Prior {
  readonly mean: number;
  readonly var: number;

  constructor(mean = 0, variance = 1) {
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
export class LaplacePrior implements Prior {
  readonly loc: number;
  readonly scale: number;

  constructor(loc = 0, scale = 1) {
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
export class CauchyPrior implements Prior {
  readonly loc: number;
  readonly scale: number;

  constructor(loc = 0, scale = 1) {
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
export class UniformPrior implements Prior {
  penalty(_theta: readonly number[]): number {
    return 0;
  }
}

/** User-defined prior penalty wrapper. */
export class CustomPrior implements Prior {
  private readonly fn: (theta: readonly number[]) => number;

  constructor(penaltyFn: (theta: readonly number[]) => number) {
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
export class EmpiricalPrior implements Prior {
  readonly var: number;
  readonly eps: number;
  readonly priorMean: number[];

  constructor(R0: TensorInput, variance = 1, eps = 1e-6) {
    if (!(variance > 0)) throw new Error("Variance must be positive");
    // Mirror the Python reference: only check dimensionality (2-D → D=1), then
    // use empirical accuracies directly. No binary/range/`L>=2` validation — any
    // finite numeric `R0` is accepted and clipped before the logit transform.
    const raw = R0 as readonly unknown[];
    const first = raw[0];
    let rows: number[][][];
    if (Array.isArray(first) && Array.isArray((first as unknown[])[0])) {
      rows = R0 as number[][][];
    } else if (Array.isArray(first)) {
      rows = (R0 as number[][]).map((row) => row.map((v) => [v]));
    } else {
      throw new Error("R0 must be 2D (L, M) or 3D (L, M, D)");
    }
    const L = rows.length;
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
export function coercePrior(prior: Prior | number): Prior {
  if (typeof prior === "number") {
    if (!Number.isFinite(prior) || prior <= 0) {
      throw new Error("prior must be a positive finite scalar variance");
    }
    return new GaussianPrior(0, prior);
  }
  if (prior && typeof (prior as Prior).penalty === "function") return prior;
  throw new Error("prior must be a Prior object or float");
}
