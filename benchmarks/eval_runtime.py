#!/usr/bin/env python3
"""Reproducible runtime benchmark for the public :mod:`scorio.eval` API.

The harness intentionally treats the installed evaluation implementation as a
black box.  It is suitable for recording a before-refactor baseline and for
comparing a later checkout against that baseline::

    python benchmarks/eval_runtime.py \
        --output benchmarks/baselines/eval_runtime_68fdf6d.json

    python benchmarks/eval_runtime.py \
        --compare benchmarks/baselines/eval_runtime_68fdf6d.json \
        --output /tmp/eval_runtime_after.json

Timings use ``perf_counter_ns``.  No benchmark asserts a performance threshold;
the output records raw samples, the median, and the 95th percentile.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter_ns
from typing import Any, Callable, Sequence

import numpy as np
import scipy

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import scorio  # noqa: E402
from scorio import eval as eval_api  # noqa: E402


SCHEMA_VERSION = 1
SEED = 20260814


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    """One named public-API workload."""

    name: str
    group: str
    dataset: str
    run: Callable[[], Any]
    default_repeats: int


def _git_output(*args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def _metadata() -> dict[str, Any]:
    status = _git_output("status", "--short")
    tracked_changes = _git_output("diff", "--name-only", "HEAD")
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
        },
        "packages": {
            "scorio": scorio.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "git": {
            "commit": _git_output("rev-parse", "HEAD"),
            "branch": _git_output("branch", "--show-current"),
            "tracked_changes": (
                [] if not tracked_changes else tracked_changes.splitlines()
            ),
            "status_short": [] if not status else status.splitlines(),
        },
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            )
            if os.environ.get(name) is not None
        },
    }


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _percentile_95(samples: Sequence[int]) -> float:
    return float(np.percentile(np.asarray(samples, dtype=np.float64), 95.0))


def _measure(
    case: BenchmarkCase,
    *,
    warmups: int,
    repeats: int,
) -> dict[str, Any]:
    last_result: Any = None
    for _ in range(warmups):
        last_result = case.run()

    samples: list[int] = []
    for _ in range(repeats):
        start = perf_counter_ns()
        last_result = case.run()
        samples.append(perf_counter_ns() - start)

    return {
        "group": case.group,
        "dataset": case.dataset,
        "warmups": warmups,
        "repeats": repeats,
        "samples_ns": samples,
        "median_ns": float(statistics.median(samples)),
        "p95_ns": _percentile_95(samples),
        "last_result": _json_value(last_result),
    }


def _build_cases(quick: bool) -> tuple[list[BenchmarkCase], dict[str, Any]]:
    rng = np.random.default_rng(SEED)

    if quick:
        point_shape = (20, 8)
        ci_shape = (8, 8)
        categorical_shape = (10, 8)
        prior_columns = 3
        point_k = 4
        ci_k = 4
        category_count = 3
        repeated_ks = (1, 2, 4, 8)
        cheap_repeats = composite_repeats = posterior_repeats = 1
    else:
        point_shape = (1000, 64)
        ci_shape = (100, 64)
        categorical_shape = (100, 64)
        prior_columns = 8
        point_k = 16
        ci_k = 12
        category_count = 5
        repeated_ks = (1, 2, 4, 8, 16, 32, 64)
        cheap_repeats = 25
        composite_repeats = 11
        posterior_repeats = 7

    binary_point = rng.integers(0, 2, size=point_shape, dtype=np.int64)
    binary_ci = rng.integers(0, 2, size=ci_shape, dtype=np.int64)
    categorical = rng.integers(
        0, category_count, size=categorical_shape, dtype=np.int64
    )
    categorical_prior = rng.integers(
        0,
        category_count,
        size=(categorical_shape[0], prior_columns),
        dtype=np.int64,
    )
    category_weights = np.linspace(0.0, 1.0, category_count, dtype=float)
    point_spectrum_weights = np.full(point_k, 1.0 / point_k, dtype=float)
    ci_spectrum_weights = np.full(ci_k, 1.0 / ci_k, dtype=float)

    cases = [
        BenchmarkCase(
            "pass_at_k",
            "point",
            "binary_point",
            lambda: eval_api.pass_at_k(binary_point, point_k),
            cheap_repeats,
        ),
        BenchmarkCase(
            "pass_hat_k",
            "point",
            "binary_point",
            lambda: eval_api.pass_hat_k(binary_point, point_k),
            cheap_repeats,
        ),
        BenchmarkCase(
            "g_pass_at_k_tau",
            "point",
            "binary_point",
            lambda: eval_api.g_pass_at_k_tau(binary_point, point_k, 0.5),
            cheap_repeats,
        ),
        BenchmarkCase(
            "maj_at_k",
            "point",
            "binary_point",
            lambda: eval_api.maj_at_k(binary_point, point_k),
            cheap_repeats,
        ),
        BenchmarkCase(
            "auc_at_k",
            "point",
            "binary_point",
            lambda: eval_api.auc_at_k(binary_point, point_k),
            composite_repeats,
        ),
        BenchmarkCase(
            "mg_pass_at_k",
            "composite",
            "binary_point",
            lambda: eval_api.mg_pass_at_k(binary_point, point_k),
            composite_repeats,
        ),
        BenchmarkCase(
            "geom_at_k",
            "composite",
            "binary_point",
            lambda: eval_api.geom_at_k(binary_point, point_k),
            composite_repeats,
        ),
        BenchmarkCase(
            "geom_ds_at_k",
            "composite",
            "binary_point",
            lambda: eval_api.geom_ds_at_k(binary_point, point_k),
            composite_repeats,
        ),
        BenchmarkCase(
            "threshold_spectrum_at_k",
            "composite",
            "binary_point",
            lambda: eval_api.threshold_spectrum_at_k(
                binary_point, point_k, point_spectrum_weights
            ),
            composite_repeats,
        ),
        BenchmarkCase(
            "geo_spectrum_at_k",
            "composite",
            "binary_point",
            lambda: eval_api.geo_spectrum_at_k(
                binary_point, point_k, weights=point_spectrum_weights
            ),
            composite_repeats,
        ),
        BenchmarkCase(
            "geo_spectrum_star_at_k",
            "composite",
            "binary_point",
            lambda: eval_api.geo_spectrum_star_at_k(binary_point, point_k),
            composite_repeats,
        ),
        BenchmarkCase(
            "pass_at_k_ci",
            "posterior",
            "binary_ci",
            lambda: eval_api.pass_at_k_ci(binary_ci, ci_k),
            posterior_repeats,
        ),
        BenchmarkCase(
            "g_pass_at_k_tau_ci",
            "posterior",
            "binary_ci",
            lambda: eval_api.g_pass_at_k_tau_ci(binary_ci, ci_k, 0.5),
            posterior_repeats,
        ),
        BenchmarkCase(
            "mg_pass_at_k_ci",
            "posterior",
            "binary_ci",
            lambda: eval_api.mg_pass_at_k_ci(binary_ci, ci_k),
            posterior_repeats,
        ),
        BenchmarkCase(
            "auc_at_k_ci",
            "posterior",
            "binary_ci",
            lambda: eval_api.auc_at_k_ci(binary_ci, ci_k),
            posterior_repeats,
        ),
        BenchmarkCase(
            "geom_at_k_ci",
            "posterior",
            "binary_ci",
            lambda: eval_api.geom_at_k_ci(binary_ci, ci_k),
            posterior_repeats,
        ),
        BenchmarkCase(
            "geom_ds_at_k_ci",
            "posterior",
            "binary_ci",
            lambda: eval_api.geom_ds_at_k_ci(binary_ci, ci_k),
            posterior_repeats,
        ),
        BenchmarkCase(
            "threshold_spectrum_at_k_ci",
            "posterior",
            "binary_ci",
            lambda: eval_api.threshold_spectrum_at_k_ci(
                binary_ci, ci_k, ci_spectrum_weights
            ),
            posterior_repeats,
        ),
        BenchmarkCase(
            "geo_spectrum_at_k_ci",
            "posterior",
            "binary_ci",
            lambda: eval_api.geo_spectrum_at_k_ci(
                binary_ci, ci_k, weights=ci_spectrum_weights
            ),
            posterior_repeats,
        ),
        BenchmarkCase(
            "bayes",
            "categorical",
            "categorical",
            lambda: eval_api.bayes(
                categorical, category_weights, categorical_prior
            ),
            cheap_repeats,
        ),
        BenchmarkCase(
            "avg",
            "categorical",
            "categorical",
            lambda: eval_api.avg(categorical, category_weights),
            cheap_repeats,
        ),
        BenchmarkCase(
            "max_at_k",
            "categorical",
            "categorical",
            lambda: eval_api.max_at_k(categorical, point_k, category_weights),
            composite_repeats,
        ),
        BenchmarkCase(
            "bayes_ci",
            "categorical",
            "categorical",
            lambda: eval_api.bayes_ci(
                categorical, category_weights, categorical_prior
            ),
            cheap_repeats,
        ),
        BenchmarkCase(
            "avg_ci",
            "categorical",
            "categorical",
            lambda: eval_api.avg_ci(categorical, category_weights),
            cheap_repeats,
        ),
        BenchmarkCase(
            "max_at_k_ci",
            "categorical",
            "categorical",
            lambda: eval_api.max_at_k_ci(
                categorical, point_k, category_weights, categorical_prior
            ),
            posterior_repeats,
        ),
        BenchmarkCase(
            "pass_at_k_repeated_k",
            "repeated_k",
            "binary_point",
            lambda: [eval_api.pass_at_k(binary_point, k) for k in repeated_ks],
            composite_repeats,
        ),
        BenchmarkCase(
            "geom_ds_at_k_repeated_k",
            "repeated_k",
            "binary_point",
            lambda: [eval_api.geom_ds_at_k(binary_point, k) for k in repeated_ks],
            composite_repeats,
        ),
        BenchmarkCase(
            "geo_spectrum_star_at_k_repeated_k",
            "repeated_k",
            "binary_point",
            lambda: [
                eval_api.geo_spectrum_star_at_k(binary_point, k)
                for k in repeated_ks
            ],
            posterior_repeats,
        ),
    ]

    datasets = {
        "binary_point": {
            "shape": list(binary_point.shape),
            "dtype": str(binary_point.dtype),
            "k": point_k,
        },
        "binary_ci": {
            "shape": list(binary_ci.shape),
            "dtype": str(binary_ci.dtype),
            "k": ci_k,
        },
        "categorical": {
            "shape": list(categorical.shape),
            "prior_shape": list(categorical_prior.shape),
            "dtype": str(categorical.dtype),
            "category_count": category_count,
            "k": point_k,
        },
        "repeated_ks": list(repeated_ks),
    }
    return cases, datasets


def _comparison(
    current: dict[str, Any], baseline_path: Path
) -> dict[str, Any]:
    with baseline_path.open("r", encoding="utf-8") as handle:
        baseline = json.load(handle)

    if baseline.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            "baseline schema version does not match this benchmark harness"
        )
    if baseline.get("mode") != current.get("mode"):
        raise ValueError("baseline and current runs must use the same benchmark mode")
    if baseline.get("datasets") != current.get("datasets"):
        raise ValueError("baseline and current runs use different benchmark datasets")

    current_results = current["results"]
    baseline_results = baseline.get("results", {})
    rows: dict[str, Any] = {}
    speedups: list[float] = []
    for name, current_case in current_results.items():
        baseline_case = baseline_results.get(name)
        if baseline_case is None:
            rows[name] = {"status": "missing_from_baseline"}
            continue

        baseline_median = float(baseline_case["median_ns"])
        current_median = float(current_case["median_ns"])
        baseline_p95 = float(baseline_case["p95_ns"])
        current_p95 = float(current_case["p95_ns"])
        median_speedup = baseline_median / current_median
        p95_speedup = baseline_p95 / current_p95
        speedups.append(median_speedup)
        rows[name] = {
            "status": "compared",
            "baseline_median_ns": baseline_median,
            "current_median_ns": current_median,
            "median_speedup": median_speedup,
            "p95_speedup": p95_speedup,
            "median_change_percent": 100.0
            * (current_median / baseline_median - 1.0),
        }

    missing_current = sorted(set(baseline_results) - set(current_results))
    geomean = (
        math.exp(sum(math.log(value) for value in speedups) / len(speedups))
        if speedups
        else None
    )
    return {
        "baseline_path": str(baseline_path.resolve()),
        "baseline_git_commit": baseline.get("metadata", {})
        .get("git", {})
        .get("commit"),
        "geometric_mean_median_speedup": geomean,
        "missing_from_current": missing_current,
        "cases": rows,
    }


def _print_comparison(comparison: dict[str, Any]) -> None:
    print("\nRuntime comparison (speedup = baseline / current):", file=sys.stderr)
    print(
        f"{'case':42s} {'baseline ms':>12s} {'current ms':>12s} {'speedup':>9s}",
        file=sys.stderr,
    )
    for name, row in comparison["cases"].items():
        if row["status"] != "compared":
            print(f"{name:42s} {row['status']}", file=sys.stderr)
            continue
        print(
            f"{name:42s} "
            f"{row['baseline_median_ns'] / 1e6:12.3f} "
            f"{row['current_median_ns'] / 1e6:12.3f} "
            f"{row['median_speedup']:9.3f}x",
            file=sys.stderr,
        )
    geomean = comparison["geometric_mean_median_speedup"]
    if geomean is not None:
        print(f"Geometric-mean median speedup: {geomean:.3f}x", file=sys.stderr)


def run_benchmarks(
    *,
    quick: bool,
    warmups_override: int | None,
    repeats_override: int | None,
) -> dict[str, Any]:
    cases, datasets = _build_cases(quick)
    default_warmups = 0 if quick else 2
    warmups = default_warmups if warmups_override is None else warmups_override

    results: dict[str, Any] = {}
    for case in cases:
        repeats = (
            case.default_repeats if repeats_override is None else repeats_override
        )
        results[case.name] = _measure(case, warmups=warmups, repeats=repeats)

    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark": "scorio.eval public API runtime",
        "mode": "quick" if quick else "standard",
        "seed": SEED,
        "timer": "time.perf_counter_ns",
        "metadata": _metadata(),
        "datasets": datasets,
        "results": results,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help="write JSON to this path; otherwise write JSON to stdout",
    )
    parser.add_argument(
        "--compare",
        type=Path,
        help="compare the new run against a saved baseline JSON",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="use tiny deterministic inputs for a smoke run",
    )
    parser.add_argument(
        "--warmups",
        type=int,
        help="override warm-up calls per case",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        help="override measured calls per case",
    )
    args = parser.parse_args(argv)
    if args.warmups is not None and args.warmups < 0:
        parser.error("--warmups must be >= 0")
    if args.repeats is not None and args.repeats < 1:
        parser.error("--repeats must be >= 1")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = run_benchmarks(
        quick=args.quick,
        warmups_override=args.warmups,
        repeats_override=args.repeats,
    )
    if args.compare is not None:
        report["comparison"] = _comparison(report, args.compare)
        _print_comparison(report["comparison"])

    rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is None:
        sys.stdout.write(rendered)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
