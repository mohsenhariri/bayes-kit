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

/** Rank models by mean accuracy over all questions and trials (`eval.avg`). */
export function avg(R: TensorInput, options: BaseRankOptions = {}): RankResult {
  const method = options.method ?? "competition";
  const tensor = validateInput(R);
  const [L] = shape3(tensor);
  const scores = Array.from({ length: L }, (_, l) => evalAvg(tensor[l]!)[0]);
  return { ranking: rankScores(scores, method), scores };
}

/** Options for {@link bayes}. */
export interface BayesRankOptions extends BaseRankOptions {
  /** Category-to-score weight vector of length `C+1`. Required for non-binary `R`. */
  w?: readonly number[];
  /** Prior outcomes: shared `(M, D)` or per-model `(L, M, D)`. */
  R0?: TensorInput;
  /** Posterior quantile `q` in `[0, 1]`; if set, rank by `mu + Φ⁻¹(q)·sigma`. */
  quantile?: number;
}

/** Rank models with Bayes@N posterior statistics (`eval.bayes`). */
export function bayes(R: TensorInput, options: BayesRankOptions = {}): RankResult {
  const method = options.method ?? "competition";
  const tensor = validateInput(R, false);
  const [L, M] = shape3(tensor);

  const quantile = options.quantile;
  if (quantile !== undefined && !(quantile >= 0 && quantile <= 1)) {
    throw new Error(`quantile must be in [0, 1]; got ${quantile}`);
  }

  let r0Shared: number[][] | undefined;
  let r0PerModel: number[][][] | undefined;
  if (options.R0 !== undefined) {
    const raw = options.R0 as readonly unknown[];
    const first = raw[0];
    const is3d = Array.isArray(first) && Array.isArray((first as unknown[])[0]);
    if (is3d) {
      const t = options.R0 as number[][][];
      if (t.length !== L || t[0]!.length !== M) {
        throw new Error(`Model-specific R0 must have shape (L=${L}, M=${M}, D)`);
      }
      r0PerModel = t;
    } else if (Array.isArray(first)) {
      const mat = options.R0 as number[][];
      if (mat.length !== M) {
        throw new Error(`Shared R0 must have shape (M=${M}, D)`);
      }
      r0Shared = mat;
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
  const method = options.method ?? "competition";
  const tensor = validateInput(R);
  const [L] = shape3(tensor);
  const scores = Array.from({ length: L }, (_, l) => evalPassAtK(tensor[l]!, k));
  return { ranking: rankScores(scores, method), scores };
}

/** Rank models by Pass-hat@k / G-Pass@k (`eval.pass_hat_k`). */
export function passHatK(
  R: TensorInput,
  k: number,
  options: BaseRankOptions = {},
): RankResult {
  const method = options.method ?? "competition";
  const tensor = validateInput(R);
  const [L] = shape3(tensor);
  const scores = Array.from({ length: L }, (_, l) => evalPassHatK(tensor[l]!, k));
  return { ranking: rankScores(scores, method), scores };
}

/** Rank models by generalized G-Pass@k_tau (`eval.g_pass_at_k_tau`). */
export function gPassAtKTau(
  R: TensorInput,
  k: number,
  tau: number,
  options: BaseRankOptions = {},
): RankResult {
  const method = options.method ?? "competition";
  const tensor = validateInput(R);
  const [L] = shape3(tensor);
  const scores = Array.from({ length: L }, (_, l) =>
    evalGPassAtKTau(tensor[l]!, k, tau),
  );
  return { ranking: rankScores(scores, method), scores };
}

/** Rank models by mG-Pass@k (`eval.mg_pass_at_k`). */
export function mgPassAtK(
  R: TensorInput,
  k: number,
  options: BaseRankOptions = {},
): RankResult {
  const method = options.method ?? "competition";
  const tensor = validateInput(R);
  const [L] = shape3(tensor);
  const scores = Array.from({ length: L }, (_, l) => evalMgPassAtK(tensor[l]!, k));
  return { ranking: rankScores(scores, method), scores };
}
