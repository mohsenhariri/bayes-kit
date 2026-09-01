# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Scorio is a research toolkit for **Bayesian evaluation and ranking of LLMs** (especially reasoning models under test-time scaling / repeated sampling). It backs three papers (ICLR 2026, ACL 2026, an arXiv preprint on test-time scaling) and ships companion HuggingFace datasets of reasoning traces (`notebooks/datasets/{trace,lite,math,gpqa}`).

The repo contains **three parallel implementations** of the same API, kept in version lockstep via the root `VERSION` file and `scripts/sync_version.py`:

- `scorio/` — Python, the reference implementation (published to PyPI)
- `julia/Scorio.jl/` — Julia, mirrors the Python `eval`/`rank`/`aggregate`/`sinf`/`utils` API (registered in the Julia General Registry)
- `js/scorio/` — TypeScript, zero-dependency port (npm), currently centered on the `eval` APIs

When changing the public API or fixing a bug in one implementation, check whether the equivalent should also change in the others.

## Common commands

All commands below run from the repo root unless noted; see the root `Makefile` for the authoritative list.

### Python (`scorio/`)

```bash
pip install -e ".[dev]"          # install with dev deps (pytest, ruff, mypy, sphinx...)

make test                        # pytest tests/
pytest tests/eval                # a single test dir
pytest tests/rank -m "not slow"  # rank tests excluding slow ones (make test-rank-py)
pytest tests/rank -m slow        # slow rank tests only (make test-rank-py-slow)
pytest tests/eval/test_bayes.py::test_name -v   # a single test

make format-check                # ruff format --check + ruff check
make format                      # ruff format + ruff check --fix
make lint                        # ruff check + mypy

make py-docs-build               # Sphinx docs -> docs/_build/html
make py-docs-serve                # serve built docs on :4000
```

Pytest marker `slow` is registered in `pyproject.toml` for heavy ranking APIs — use it on any new test that's expensive (MCMC, MILP solvers, etc.).

### Julia (`julia/Scorio.jl/`)

```bash
make jl-install                  # Pkg.instantiate() for the package + its docs env
make jl-test                     # Pkg.test()
make jl-test-slow                # SCORIO_JL_RUN_SLOW=1 Pkg.test()
make test-eval-jl                # just the eval test file
make test-rank-jl                # just the rank test suite
make jl-docs-build / jl-docs-serve
```

### JavaScript / TypeScript (`js/scorio/`)

```bash
cd js/scorio && npm ci
make js-build                    # tsup -> dist/ (ESM + CJS + d.ts)
make js-test                     # vitest run
npx vitest run <file>            # a single test file (from js/scorio/)
make -C js typecheck             # tsc --noEmit
```

### Release

Version is sourced from the root `VERSION` file; `docs/changelog.rst` is the human-written release-notes source. See `CONTRIBUTING.md` for the full release checklist (`make sync-version`, `make pkg-check`, `make release-py`, `make release-jl`).

## Architecture

### Core data convention

Nearly every module operates on the same shapes, defined in the README and `scorio/eval/_inputs.py`:

- `R`: `M × N` integer matrix, outcomes for `M` questions over `N` trials, categories in `{0, ..., C}`
- `w`: length-`C+1` float vector of rubric weights mapping categories to scores
- `R0` (optional): `M × D` matrix of prior outcomes, same category set as `R`

Understanding this convention is required before reading `eval`, `rank`, or `aggregate` — they all consume/produce data in this shape, and ranking methods construct pairwise/setwise comparisons *from* `R` rather than taking a different input format.

### `scorio/eval/` — scalar evaluation metrics

`bayes.py` implements the Bayes@N posterior (mean, std) from `_posterior.py`, optionally combining prior outcomes `R0`. Other metrics (`avg`, `maj`, `pass_at_k`, `gpass`, `max_reward`, `auc`) are built on shared helpers in `_inputs.py` (shape/validation), `_count_score.py`, and `_categorical.py`.

### `scorio/rank/` — model ranking, built on `eval` and pairwise construction

`_base.py` and `_types.py` hold shared ranking infrastructure. Method families each live in their own file and wrap a distinct statistical model, documented with paper references in `scorio/rank/README.md`:

- `eval_ranking.py` — rank by an evaluation metric (`avg`, `bayes`, `pass_at_k`, `g_pass_at_k_tau`, ...)
- `bradley_terry.py` / `pairwise.py` — Bradley-Terry family, Elo/TrueSkill/Glicko
- `voting.py` — Borda, Copeland, Schulze, ranked pairs, Kemeny-Young, etc.
- `irt.py` — Rasch/2PL/3PL, MML, dynamic/longitudinal IRT, `mirt`
- `graph.py`, `rank_centrality.py`, `serial_rank.py`, `hodge_rank.py` — spectral/graph-theoretic methods
- `listwise.py` — Plackett-Luce, Davidson-Luce and related choice models
- `bayesian.py` — Thompson sampling estimator, Bradley-Terry MCMC
- `priors.py` — shared prior/regularization helpers (MAP variants)

Many of these methods raise rather than return a degenerate estimate (e.g. non-existent MLE under separated data, unproven MILP optimum) — see `scorio/rank/README.md` for the specific failure conditions before "fixing" a raised exception.

### `scorio/aggregate/` (aliased `scorio.agg`) — test-time-scaling answer selection

Picks a final answer among `N` sampled completions per question. `_base.py` defines the shared interface; `confidence.py` derives confidence signals from token log-probs (self-certainty, DeepConf, entropy), `prm.py` aggregates via process reward models, `vote.py`/`best_of.py` implement offline selection (majority vote, weighted vote, Best-of-N, Majority-of-the-Bests), `online.py` implements early-stopping rules, `calibration.py` and `cges.py` support calibration/scoring.

### `scorio/categorical/` — rubric-based evaluation pipeline

A higher-level pipeline over `eval.bayes`: `io.py` loads per-completion JSONL files, `thresholds.py` computes corpus-level thresholds, `schemas.py` defines rubric schemas, `evaluate.py` ties these together to classify completions and score models.

### `scorio/sinf/` — sequential inference helpers (adaptive stopping/allocation)

### Notebooks and datasets (`notebooks/`)

`notebooks/datasets/<trace|lite|math|gpqa>/` each follow the same four-notebook workflow against a companion HuggingFace dataset: load/explore (`<name>.ipynb`) → `eval.ipynb` → `rank.ipynb` → `aggregate.ipynb`, plus a dataset-specific `README.md`. `notebooks/01_bayes_eval.ipynb` is the top-level API walkthrough. `notebooks/datasets/README.md` compares the four datasets.

### `benchmarks/`

`eval_runtime.py` plus checked-in baseline JSON in `benchmarks/baselines/` track performance regressions in the NumPy eval kernels — update baselines deliberately, not to silence a regression.

### Test fixtures

`tests/data/generate_simulation_data.py` regenerates the outcome-matrix fixtures (`R_greedy.npz`, `R_top_p.npz`) that several tests load — regenerate via that script rather than hand-editing the `.npz` files.

### `citations/`

`aggregate.bib`, `bayes_at_N.bib`, `rank.bib` are the bibliography sources backing the citation blocks in `README.md` — update both together when a method's citation changes.

## Code style notes (Python)

- Ruff config in `pyproject.toml` intentionally allows math-conventional short/uppercase names (`R`, `w`, `N803`/`N806` ignored) — don't "fix" these.
- Google-style docstrings, type hints on public APIs (per `CONTRIBUTING.md`).

## CI

`.github/workflows/python-package.yml` runs on push/PR to `main`: `ruff format --check` + `ruff check` on `scorio/`, and `pytest` (`make test`) across Python 3.10–3.13. **mypy is not enforced in CI** — it's part of `make lint` locally only, so a mypy failure won't block a PR but should still be fixed.
