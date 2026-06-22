/**
 * Scorio evaluation metrics and uncertainty estimators.
 *
 * TypeScript port of `scorio.eval`. Point estimators return a scalar score;
 * companion `*Ci` / `*_ci` functions return `[mu, sigma, lo, hi]`, where `mu`
 * is the estimate, `sigma` the posterior standard deviation, and `lo`/`hi`
 * a normal-approximation credible interval.
 *
 * Every metric is exported under two names: an idiomatic camelCase name
 * (`passAtK`) and a snake_case alias matching the Python/Julia API
 * (`pass_at_k`).
 */

import { bayes, bayesCi } from "./bayes.js";
import { avg, avgCi } from "./avg.js";
import { passAtK, passHatK, passAtKCi, passHatKCi } from "./passAtK.js";
import { majAtK, majAtKCi } from "./maj.js";
import { aucAtK, aucAtKCi } from "./auc.js";
import {
  gPassAtK,
  gPassAtKCi,
  gPassAtKTau,
  gPassAtKTauCi,
  mgPassAtK,
  mgPassAtKCi,
} from "./gpass.js";
import { maxAtK, maxAtKCi } from "./maxReward.js";
import {
  geomAtK,
  geomAtKCi,
  geomDsAtK,
  geomDsAtKCi,
  geoSpectrumAtK,
  geoSpectrumAtKCi,
  geoSpectrumStarAtK,
  geoSpectrumStarAtKCi,
  thresholdSpectrumAtK,
  thresholdSpectrumAtKCi,
} from "./geom.js";
import { normalCredibleInterval } from "./internal/ci.js";

export type { Matrix } from "./internal/validate.js";
export type { Bounds } from "./internal/ci.js";

// `unanimous@k` is a synonym for `Pass^k`, mirroring scorio.eval.
const unanimousAtK = passHatK;
const unanimousAtKCi = passHatKCi;

// ---------------------------------------------------------------------------
// Primary camelCase API
// ---------------------------------------------------------------------------
export {
  // Bayes@N
  bayes,
  bayesCi,
  // Avg@N
  avg,
  avgCi,
  // Pass family
  passAtK,
  passAtKCi,
  passHatK,
  passHatKCi,
  unanimousAtK,
  unanimousAtKCi,
  // Generalized pass family
  gPassAtK,
  gPassAtKCi,
  gPassAtKTau,
  gPassAtKTauCi,
  mgPassAtK,
  mgPassAtKCi,
  // Majority
  majAtK,
  majAtKCi,
  // AUC@K
  aucAtK,
  aucAtKCi,
  // Max-reward
  maxAtK,
  maxAtKCi,
  // Geometric / spectrum
  thresholdSpectrumAtK,
  thresholdSpectrumAtKCi,
  geomAtK,
  geomAtKCi,
  geomDsAtK,
  geomDsAtKCi,
  geoSpectrumAtK,
  geoSpectrumAtKCi,
  geoSpectrumStarAtK,
  geoSpectrumStarAtKCi,
  // Shared utility
  normalCredibleInterval,
};

// ---------------------------------------------------------------------------
// snake_case aliases (Python / Julia parity)
// ---------------------------------------------------------------------------
export {
  bayesCi as bayes_ci,
  avgCi as avg_ci,
  passAtK as pass_at_k,
  passAtKCi as pass_at_k_ci,
  passHatK as pass_hat_k,
  passHatKCi as pass_hat_k_ci,
  unanimousAtK as unanimous_at_k,
  unanimousAtKCi as unanimous_at_k_ci,
  gPassAtK as g_pass_at_k,
  gPassAtKCi as g_pass_at_k_ci,
  gPassAtKTau as g_pass_at_k_tau,
  gPassAtKTauCi as g_pass_at_k_tau_ci,
  mgPassAtK as mg_pass_at_k,
  mgPassAtKCi as mg_pass_at_k_ci,
  majAtK as maj_at_k,
  majAtKCi as maj_at_k_ci,
  aucAtK as auc_at_k,
  aucAtKCi as auc_at_k_ci,
  maxAtK as max_at_k,
  maxAtKCi as max_at_k_ci,
  thresholdSpectrumAtK as threshold_spectrum_at_k,
  thresholdSpectrumAtKCi as threshold_spectrum_at_k_ci,
  geomAtK as geom_at_k,
  geomAtKCi as geom_at_k_ci,
  geomDsAtK as geom_ds_at_k,
  geomDsAtKCi as geom_ds_at_k_ci,
  geoSpectrumAtK as geo_spectrum_at_k,
  geoSpectrumAtKCi as geo_spectrum_at_k_ci,
  geoSpectrumStarAtK as geo_spectrum_star_at_k,
  geoSpectrumStarAtKCi as geo_spectrum_star_at_k_ci,
  normalCredibleInterval as normal_credible_interval,
};
