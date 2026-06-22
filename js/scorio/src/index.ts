/**
 * Scorio — Bayesian evaluation toolkit for stochastic models.
 *
 * This entry point exposes the evaluation metrics under the `eval` namespace,
 * mirroring `import scorio; scorio.eval.bayes(...)` in Python:
 *
 * ```ts
 * import { eval as scorioEval } from "scorio";
 * const [mu, sigma] = scorioEval.bayes(R, w);
 * ```
 *
 * The metrics are also importable directly from `scorio/eval`.
 *
 * Ranking estimators live under the `rank` namespace (`scorio.rank.borda(...)`)
 * and are also importable directly from `scorio/rank`.
 */

export * as eval from "./eval/index.js";
export * as rank from "./rank/index.js";
