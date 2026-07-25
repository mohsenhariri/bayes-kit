/**
 * Evaluation-metric ranking methods. Port of `scorio/rank/eval_ranking.py`.
 *
 * Each method maps a model's responses to a scalar score using the ported
 * `scorio/eval` metric of the same name, then converts scores to ranks.
 */

import {
  avg as evalAvg,
  bayes as evalBayes,
  passAtK as evalPassAtK,
  passHatK as evalPassHatK,
  gPassAtKTau as evalGPassAtKTau,
  mgPassAtK as evalMgPassAtK,
} from "../eval/index.js";
import { rankScores } from "./internal/rankScores.js";
import { normPpf } from "./internal/special.js";
import {
  validateInput,
  shape3,
  type TensorInput,
} from "./internal/tensor.js";
import type { BaseRankOptions, RankResult } from "./internal/result.js";

const defaultIfUndefined = <T>(value: T | undefined, fallback: T): T =>
  value === undefined ? fallback : value;

function coercePythonFloat(value: unknown, name: string): number {
  if (typeof value === "number") return value;
  if (typeof value === "boolean") return value ? 1 : 0;
  if (typeof value === "string" && value.trim().length > 0) {
    const normalized = value.trim().replace(/(?<=\d)_(?=\d)/g, "");
    if (/^[+-]?(?:inf(?:inity)?)$/i.test(normalized)) {
      return normalized.startsWith("-") ? -Infinity : Infinity;
    }
    if (/^[+-]?nan$/i.test(normalized)) return NaN;
    if (/^[+-]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?)$/i.test(normalized)) {
      return Number(normalized);
    }
  }
  throw new TypeError(`${name} must be convertible to a float`);
}

/** Python's comparison-based pass-family checks accept numbers and booleans. */
function comparableNumber(value: unknown, name: string): number {
  if (typeof value === "boolean") return value ? 1 : 0;
  if (typeof value === "number") return value;
  throw new TypeError(`${name} must be a real scalar`);
}

/** Rank models by mean accuracy over all questions and trials (`eval.avg`). */
export function avg(R: TensorInput, options: BaseRankOptions = {}): RankResult {
  const method = defaultIfUndefined(options.method, "competition");
  const tensor = validateInput(R);
  const [L] = shape3(tensor);
  const scores = Array.from({ length: L }, (_, l) => evalAvg(tensor[l]!)[0]);
  return { ranking: rankScores(scores, method), scores };
}

/** Options for {@link bayes}. */
export interface BayesRankOptions extends BaseRankOptions {
  /** Category-to-score weight vector of length `C+1`. Required for non-binary `R`. */
  w?: readonly number[] | null;
  /** Prior outcomes: shared `(M, D)` or per-model `(L, M, D)`. */
  R0?: TensorInput | null;
  /** Posterior quantile `q` in `(0, 1)`; if set, rank by `mu + Φ⁻¹(q)·sigma`. */
  quantile?: number | null;
}

function priorOutcome(value: unknown): number {
  if (typeof value === "boolean") return value ? 1 : 0;
  if (typeof value !== "number" || !Number.isFinite(value) || !Number.isInteger(value)) {
    throw new Error("R0 must contain real, finite integer-valued outcomes");
  }
  return value;
}

/** Rank models with Bayes@N posterior statistics (`eval.bayes`). */
export function bayes(R: TensorInput, options: BayesRankOptions = {}): RankResult {
  const method = defaultIfUndefined(options.method, "competition");
  const tensor = validateInput(R, false);
  const [L, M] = shape3(tensor);

  let quantile: number | undefined;
  if (options.quantile !== undefined && options.quantile !== null) {
    quantile = coercePythonFloat(options.quantile, "quantile");
    if (!Number.isFinite(quantile) || !(quantile > 0 && quantile < 1)) {
      throw new Error(`quantile must be in (0, 1); got ${options.quantile}`);
    }
  }

  let r0Shared: number[][] | undefined;
  let r0PerModel: number[][][] | undefined;
  if (options.R0 !== undefined && options.R0 !== null) {
    const raw = options.R0 as unknown;
    if (!Array.isArray(raw)) {
      throw new Error("R0 must be shape (M, D) or (L, M, D)");
    }
    const is2d = raw.every(
      (row) =>
        Array.isArray(row) &&
        row.every((value) => !Array.isArray(value)),
    );
    const is3d = raw.every(
      (matrix) =>
        Array.isArray(matrix) &&
        matrix.every(
          (row) =>
            Array.isArray(row) &&
            row.every((value) => !Array.isArray(value)),
        ),
    );
    if (is2d) {
      const mat = raw.map((row) => (row as unknown[]).map(priorOutcome));
      const D = mat[0]?.length ?? 0;
      if (mat.some((row) => row.length !== D)) {
        throw new Error("R0 must be a rectangular array");
      }
      if (mat.length !== M) {
        throw new Error(`Shared R0 must have shape (M=${M}, D)`);
      }
      r0Shared = mat;
    } else if (is3d) {
      const t = raw.map((matrix) =>
        (matrix as unknown[][]).map((row) => row.map(priorOutcome)),
      );
      const d = t[0]?.[0]?.length ?? 0;
      if (
        t.some(
          (matrix) =>
            matrix.length !== (t[0]?.length ?? 0) ||
            matrix.some((row) => row.length !== d),
        )
      ) {
        throw new Error("R0 must be a rectangular array");
      }
      if (t.length !== L || (t[0]?.length ?? 0) !== M) {
        throw new Error(`Model-specific R0 must have shape (L=${L}, M=${M}, D)`);
      }
      r0PerModel = t;
    } else {
      throw new Error("R0 must be shape (M, D) or (L, M, D)");
    }
  }

  const z = quantile !== undefined ? normPpf(quantile) : undefined;
  const scores = new Array<number>(L).fill(0);
  for (let l = 0; l < L; l++) {
    const modelR0 = r0PerModel ? r0PerModel[l]! : r0Shared;
    const [mu, sigma] = evalBayes(tensor[l]!, options.w, modelR0);
    scores[l] = z !== undefined ? mu + z * sigma : mu;
  }
  return { ranking: rankScores(scores, method), scores };
}

/** Rank models by Pass@k (`eval.pass_at_k`). */
export function passAtK(
  R: TensorInput,
  k: number,
  options: BaseRankOptions = {},
): RankResult {
  const method = defaultIfUndefined(options.method, "competition");
  const tensor = validateInput(R);
  const normalizedK = comparableNumber(k, "k");
  const [L] = shape3(tensor);
  const scores = Array.from({ length: L }, (_, l) =>
    evalPassAtK(tensor[l]!, normalizedK),
  );
  return { ranking: rankScores(scores, method), scores };
}

/** Rank models by Pass-hat@k / G-Pass@k (`eval.pass_hat_k`). */
export function passHatK(
  R: TensorInput,
  k: number,
  options: BaseRankOptions = {},
): RankResult {
  const method = defaultIfUndefined(options.method, "competition");
  const tensor = validateInput(R);
  const normalizedK = comparableNumber(k, "k");
  const [L] = shape3(tensor);
  const scores = Array.from({ length: L }, (_, l) =>
    evalPassHatK(tensor[l]!, normalizedK),
  );
  return { ranking: rankScores(scores, method), scores };
}

/** Rank models by generalized G-Pass@k_tau (`eval.g_pass_at_k_tau`). */
export function gPassAtKTau(
  R: TensorInput,
  k: number,
  tau: number,
  options: BaseRankOptions = {},
): RankResult {
  const method = defaultIfUndefined(options.method, "competition");
  const tensor = validateInput(R);
  const normalizedTau = comparableNumber(tau, "tau");
  const normalizedK = comparableNumber(k, "k");
  const [L] = shape3(tensor);
  const scores = Array.from({ length: L }, (_, l) =>
    evalGPassAtKTau(tensor[l]!, normalizedK, normalizedTau),
  );
  return { ranking: rankScores(scores, method), scores };
}

/** Rank models by mG-Pass@k (`eval.mg_pass_at_k`). */
export function mgPassAtK(
  R: TensorInput,
  k: number,
  options: BaseRankOptions = {},
): RankResult {
  const method = defaultIfUndefined(options.method, "competition");
  const tensor = validateInput(R);
  const normalizedK = comparableNumber(k, "k");
  const [L] = shape3(tensor);
  const scores = Array.from({ length: L }, (_, l) =>
    evalMgPassAtK(tensor[l]!, normalizedK),
  );
  return { ranking: rankScores(scores, method), scores };
}
