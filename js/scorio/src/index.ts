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
 *
 * Sequential-inference helpers live under the `sinf` namespace
 * (`scorio.sinf.confseqMean(...)`) and are also importable from `scorio/sinf`.
 *
 * Test-time-scaling aggregation lives under `aggregate` (with short alias
 * `agg`) and is also importable directly from `scorio/aggregate`.
 *
 * Ranking comparison and collision-free ranking hashes live under `utils`
 * and are also importable directly from `scorio/utils`.
 */

import * as aggregate from "./aggregate/index.js";

export * as eval from "./eval/index.js";
export * as rank from "./rank/index.js";
export * as sinf from "./sinf/index.js";
export * as utils from "./utils/index.js";
export { aggregate, aggregate as agg };
