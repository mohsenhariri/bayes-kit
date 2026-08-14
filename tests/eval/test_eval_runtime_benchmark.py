import json
import subprocess
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
HARNESS = REPOSITORY_ROOT / "benchmarks" / "eval_runtime.py"


def _run_harness(output: Path, *extra_args: str) -> dict:
    subprocess.run(
        [
            sys.executable,
            str(HARNESS),
            "--quick",
            "--warmups",
            "0",
            "--repeats",
            "1",
            "--output",
            str(output),
            *extra_args,
        ],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return json.loads(output.read_text(encoding="utf-8"))


def test_eval_runtime_harness_schema_and_comparison(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.json"
    baseline = _run_harness(baseline_path)

    assert baseline["schema_version"] == 1
    assert baseline["mode"] == "quick"
    assert baseline["seed"] == 20260814
    assert baseline["timer"] == "time.perf_counter_ns"
    assert baseline["metadata"]["git"]["commit"]
    assert baseline["metadata"]["git"]["tracked_changes"] == []
    assert {case["group"] for case in baseline["results"].values()} == {
        "categorical",
        "composite",
        "point",
        "posterior",
        "repeated_k",
    }

    for case in baseline["results"].values():
        assert case["warmups"] == 0
        assert case["repeats"] == 1
        assert len(case["samples_ns"]) == 1
        assert case["median_ns"] >= 0
        assert case["p95_ns"] >= 0
        assert "last_result" in case

    current_path = tmp_path / "current.json"
    current = _run_harness(
        current_path,
        "--compare",
        str(baseline_path),
    )
    comparison = current["comparison"]
    assert comparison["baseline_git_commit"] == baseline["metadata"]["git"]["commit"]
    assert comparison["geometric_mean_median_speedup"] > 0
    assert set(comparison["cases"]) == set(current["results"])
    assert all(
        row["status"] == "compared" for row in comparison["cases"].values()
    )
