# scorio

Bayesian evaluation toolkit for stochastic models — a TypeScript/JavaScript port of the [Scorio](https://github.com/mohsenhariri/scorio) `eval` APIs.

It computes point estimates **and** Bayesian uncertainty for the metrics commonly used to evaluate LLMs and other stochastic models under repeated sampling: Bayes@N, Avg@N, Pass@k / Pass^k, G-Pass@k, Maj@k, AUC@K, Max@k, and the geometric/spectrum blends.

- **Zero runtime dependencies** — pure TypeScript (special functions reimplemented from `scipy.special`).
- **Dual ESM + CommonJS** builds with full type declarations.
- **Numerically faithful** to the Python reference (verified against its published values).
- **Two naming styles**: idiomatic camelCase (`passAtK`) and snake_case aliases matching the Python/Julia API (`pass_at_k`).

> Scope: this package currently covers the **evaluation** APIs. The ranking APIs are not yet ported.

## Install

```sh
npm install scorio
```

## Usage

The outcome matrix `R` has shape `M × N` (M questions, N trials per question) with integer category entries in `{0,…,C}`. Binary metrics use entries in `{0,1}`. A 1-D array is treated as a single row.

```ts
import { eval as scorio } from "scorio";
// or: import { bayes, passAtK } from "scorio/eval";

// Multi-category outcomes with a rubric weight vector (length C+1)
const R = [
  [0, 1, 2, 2, 1],
  [1, 1, 0, 2, 2],
];
const w = [0.0, 0.5, 1.0]; // 0=incorrect, 1=partial, 2=correct
const R0 = [               // optional prior outcomes (M × D)
  [0, 2],
  [1, 2],
];

const [mu, sigma] = scorio.bayes(R, w, R0);
// mu ≈ 0.575, sigma ≈ 0.084275

const [a, sa] = scorio.avg(R, w);
// weighted average with Bayesian uncertainty

// Binary metrics
const B = [
  [0, 1, 1, 0, 1],
  [1, 1, 0, 1, 1],
];
scorio.passAtK(B, 2);   // 0.95
scorio.passHatK(B, 2);  // 0.45  (a.k.a. unanimousAtK / g_pass@k)
scorio.passAtKCi(B, 2); // [mu, sigma, lo, hi]
```

### Point estimators vs. credible intervals

Point estimators return a scalar score. Every metric has a companion `*Ci` function (and a `*_ci` alias) returning `[mu, sigma, lo, hi]`, where `mu` is the estimate, `sigma` the posterior standard deviation, and `lo`/`hi` a normal-approximation credible interval.

## API

| Family | Point estimator | Credible interval |
| --- | --- | --- |
| Bayes@N | `bayes` | `bayesCi` |
| Avg@N | `avg` | `avgCi` |
| Pass@k | `passAtK` | `passAtKCi` |
| Pass^k / unanimous | `passHatK`, `unanimousAtK` | `passHatKCi`, `unanimousAtKCi` |
| G-Pass@k | `gPassAtK`, `gPassAtKTau`, `mgPassAtK` | `gPassAtKCi`, `gPassAtKTauCi`, `mgPassAtKCi` |
| Majority | `majAtK` | `majAtKCi` |
| AUC@K | `aucAtK` | `aucAtKCi` |
| Max@k | `maxAtK` | `maxAtKCi` |
| Geometric / spectrum | `geomAtK`, `geomDsAtK`, `geoSpectrumAtK`, `geoSpectrumStarAtK`, `thresholdSpectrumAtK` | each with a `*Ci` variant |

Each camelCase name has a snake_case alias (`pass_at_k`, `g_pass_at_k_tau`, `geo_spectrum_at_k`, …) for parity with the Python and Julia packages.

## Development

```sh
npm install
npm test          # vitest golden tests (parity with the Python reference)
npm run build     # tsup -> dist/ (ESM + CJS + d.ts)
npm run typecheck
```

## License

MIT © Mohsen Hariri. See the repository root `LICENSE` and `CITATION.cff`.
