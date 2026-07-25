/** Process-reward-model step-score aggregation. */

import {
  asFiniteVector,
  type NumericInput,
} from "./internal/numeric.js";

export type PrmAggregateMethod = "last" | "min" | "mean" | "prod" | "max";

export interface PrmAggregateOptions {
  method?: PrmAggregateMethod;
}

function asStepScores(stepScores: NumericInput): number[] {
  return asFiniteVector(
    stepScores,
    "step_scores",
    "step_scores must be non-empty (L >= 1).",
  );
}

/** Reduce a trace's per-step PRM scores to one reward. */
export function prmAggregate(
  stepScores: NumericInput,
  options: PrmAggregateOptions = {},
): number {
  const method = options.method === undefined ? "last" : options.method;
  if (!(["last", "min", "mean", "prod", "max"] as const).includes(method)) {
    throw new Error(
      `method must be one of last, min, mean, prod, max; got ${String(method)}.`,
    );
  }
  const values = asStepScores(stepScores);
  if (method === "last") return values[values.length - 1]!;

  let total = method === "prod" ? 1 : 0;
  let minimum = Infinity;
  let maximum = -Infinity;
  for (const value of values) {
    total = method === "prod" ? total * value : total + value;
    if (value < minimum) minimum = value;
    if (value > maximum) maximum = value;
  }
  if (method === "min") return minimum;
  if (method === "max") return maximum;
  if (method === "mean") return total / values.length;
  return total;
}
