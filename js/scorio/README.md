# scorio

Bayesian evaluation toolkit for stochastic models — a TypeScript/JavaScript port of the [Scorio](https://github.com/mohsenhariri/scorio) `eval` APIs.

It provides two families of APIs:

- **`scorio/eval`** — point estimates **and** Bayesian uncertainty for metrics used to evaluate LLMs and other stochastic models under repeated sampling: Bayes@N, Avg@N, Pass@k / Pass^k, G-Pass@k, Maj@k, AUC@K, Max@k, and the geometric/spectrum blends.
- **`scorio/rank`** — 40+ ranking estimators that order multiple models from a binary (or categorical) response tensor: eval-metric, voting, pairwise-rating (Elo/Glicko/TrueSkill), Bradley-Terry / Plackett-Luce / Rao-Kupper, IRT (Rasch/2PL/3PL/MML), graph (PageRank/spectral/α-Rank/Nash), seriation, and Hodge-theoretic methods.

- **Zero runtime dependencies** — pure TypeScript (special functions, linear algebra, optimization, and an LP solver reimplemented from `scipy`/`numpy`).
- **Dual ESM + CommonJS** builds with full type declarations.
- **Numerically faithful** to the Python reference (verified against generated ground-truth fixtures).
- **Two naming styles**: idiomatic camelCase (`passAtK`, `bradleyTerry`) and snake_case aliases matching the Python/Julia API (`pass_at_k`, `bradley_terry`).

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

## Ranking (`scorio/rank`)

Ranking estimators take a response tensor `R` of shape `(L, M, N)` — `L` models, `M` questions, `N` trials — with binary entries (a 2-D `(L, M)` matrix is treated as `N = 1`). Each method returns `{ ranking, scores }`: `ranking[l]` is model `l`'s rank (1 = best) and `scores[l]` the raw method score (larger is better). The optional `method` selects the tie convention (`"competition"` by default; also `"competition_max"`, `"dense"`, `"avg"`).

```ts
import { rank } from "scorio";
// or: import { borda, bradleyTerry } from "scorio/rank";

// 2 models, 2 questions, 2 trials
const R = [
  [[1, 1], [1, 1]],
  [[0, 0], [0, 0]],
];

rank.borda(R).ranking;          // [1, 2]
rank.elo(R).scores;             // final Elo ratings
rank.bradleyTerry(R, { maxIter: 100 }).ranking;
rank.bayes(R, { quantile: 0.05 });   // conservative, uncertainty-aware
rank.raschMap(R, { prior: 1.0 });    // MAP IRT with a Gaussian prior

// snake_case aliases mirror the Python API
rank.pass_at_k(R, 2);
rank.rank_centrality(R);
```

| Family | Methods |
| --- | --- |
| Eval-metric | `avg`, `bayes`, `passAtK`, `passHatK`, `gPassAtKTau`, `mgPassAtK` |
| Pointwise | `inverseDifficulty` |
| Pairwise ratings | `elo`, `glicko`, `trueskill` |
| Bradley-Terry | `bradleyTerry`(`Map`), `bradleyTerryDavidson`(`Map`), `raoKupper`(`Map`) |
| Bayesian | `thompson`, `bayesianMcmc` |
| Voting | `borda`, `copeland`, `winRate`, `minimax`, `schulze`, `rankedPairs`, `kemenyYoung`, `nanson`, `baldwin`, `majorityJudgment` |
| IRT | `rasch`(`Map`), `rasch2pl`(`Map`), `rasch3pl`(`Map`), `raschMml`, `raschMmlCredible`, `dynamicIrt` |
| Graph | `pagerank`, `spectral`, `alpharank`, `nash`, `rankCentrality` |
| Seriation / Hodge | `serialRank`, `hodgeRank` |
| Plackett-Luce | `plackettLuce`(`Map`), `davidsonLuce`(`Map`), `bradleyTerryLuce`(`Map`) |
| Priors (for MAP) | `GaussianPrior`, `LaplacePrior`, `CauchyPrior`, `UniformPrior`, `CustomPrior`, `EmpiricalPrior` |

The MAP estimators accept a `prior` option — either a variance (interpreted as a zero-mean `GaussianPrior`) or a `Prior` instance. The Monte-Carlo methods (`thompson`, `bayesianMcmc`) are seeded and reproducible but, since they use a different RNG, are not bit-identical to the Python reference.

## Development

```sh
npm install
npm test          # vitest golden tests (parity with the Python reference)
npm run build     # tsup -> dist/ (ESM + CJS + d.ts)
npm run typecheck
```

## License

MIT © Mohsen Hariri. See the repository root `LICENSE` and `CITATION.cff`.
