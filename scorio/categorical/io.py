"""Load raw per-completion JSONL inference files into a columnar dict.

Expected record format (the raw inference output):

    {
      "seed": 1261,          # → trial
      "data_id": 6,          # → problem
      "model": "...",
      "output": {
        "finish_reason": "stop",        # "length" → hit_max_len=1
        "num_completion_tokens": 22818  # → completion_length
      },
      "tokens": {
        "completion_avg_logprob": -0.09,   # → completion_avg_logprob (C1)
        "completion_sum_logprob": -2069.7, # → completion_sum_logprob (C3)
        "completion_ppl": 1.09,            # → completion_perplexity
        "prompt_sum_logprob": -484.9,      # → prompt_sum_logprob     (P3)
        "prompt_ppl": 70.38,               # → prompt_perplexity
        "completion_logprob_list": [...]   # → logprob_min / logprob_iqr / tail64_avg_logprob
      },
      "processed_results": {
        "is_correct": 0,
        "has_box": 1
      }
    }

The columns dict produced by :func:`load_records` is consumed by
:mod:`scorio.categorical.thresholds` and :mod:`scorio.categorical.schemas`.
"""

import gzip
import json
import logging
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from scorio.categorical._util import Columns, to_float

logger = logging.getLogger(__name__)

# Columns kept as object arrays (identity / labels) rather than coerced to float.
_NON_NUMERIC = {"model", "problem", "source_file"}

# ── PRM helper ───────────────────────────────────────────────────────


def _prm_summary(steps: list[float], prefix: str) -> dict:
    """Expand a PRM step-score list into flat scalar columns."""
    arr = np.asarray(steps, dtype=np.float64)
    return {
        f"{prefix}_mean": float(arr.mean()),
        f"{prefix}_min": float(arr.min()),
        f"{prefix}_max": float(arr.max()),
        f"{prefix}_std": float(arr.std()),
        f"{prefix}_last": float(arr[-1]),
        f"{prefix}_n_steps": len(arr),
    }


def _extract_row(rec: dict, source_file: str) -> dict:
    """Flatten one raw inference JSON record into a single-level dict."""
    out = rec.get("output") or {}
    tok = rec.get("tokens") or {}
    pr = rec.get("processed_results") or {}

    lp_list = [v for v in (tok.get("completion_logprob_list") or []) if v is not None]
    arr = (
        np.asarray(lp_list, dtype=np.float64)
        if lp_list
        else np.array([], dtype=np.float64)
    )

    return {
        "source_file": source_file,
        "model": rec.get("model"),
        "problem": rec.get("data_id"),
        "trial": rec.get("seed"),
        "is_correct": pr.get("is_correct"),
        "has_box": pr.get("has_box"),
        "hit_max_len": int(out.get("finish_reason") == "length"),
        "completion_length": out.get("num_completion_tokens"),
        "completion_perplexity": tok.get("completion_ppl"),
        "prompt_perplexity": tok.get("prompt_ppl"),
        "logprob_min": float(arr.min()) if arr.size else None,
        "logprob_iqr": float(np.percentile(arr, 75) - np.percentile(arr, 25))
        if arr.size
        else None,
        "tail64_avg_logprob": float(arr[-64:].mean()) if arr.size else None,
        "completion_avg_logprob": tok.get("completion_avg_logprob"),
        "completion_sum_logprob": tok.get("completion_sum_logprob"),
        "prompt_sum_logprob": tok.get("prompt_sum_logprob"),
        "acemath_orm": None,
        "skywork_orm": None,
        "verifier_pA": None,
    }


# ── worker result container ──────────────────────────────────────────


@dataclass
class FileResult:
    """Diagnostics returned by each worker process."""

    file_path: str
    rows: list[dict] = field(default_factory=list)
    n_records_parsed: int = 0
    n_lines_total: int = 0
    n_lines_skipped: int = 0
    n_lines_malformed: int = 0
    file_size_bytes: int = 0
    elapsed_secs: float = 0.0
    error: str | None = None


# ── JSONL reading ────────────────────────────────────────────────────


def _iter_jsonl(path: Path) -> tuple[list[dict], int, int, int]:
    """Read all valid JSON lines from a .jsonl or .jsonl.gz file.

    Returns:
        (records, total_lines, skipped_blank, malformed)
    """
    records: list[dict] = []
    total = skipped = malformed = 0

    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            total += 1
            raw = raw.strip()
            if not raw:
                skipped += 1
                continue
            try:
                records.append(json.loads(raw))
            except json.JSONDecodeError:
                malformed += 1

    return records, total, skipped, malformed


# ── worker (top-level so it is picklable) ────────────────────────────


def _process_file(file_path: Path) -> FileResult:
    """Runs in a subprocess: reads one file and returns rows + diagnostics."""
    result = FileResult(file_path=str(file_path))
    t0 = time.perf_counter()
    try:
        result.file_size_bytes = file_path.stat().st_size
        records, total, skipped, malformed = _iter_jsonl(file_path)
        result.n_lines_total = total
        result.n_lines_skipped = skipped
        result.n_lines_malformed = malformed
        result.n_records_parsed = len(records)
        result.rows = [_extract_row(rec, file_path.name) for rec in records]
    except Exception as exc:
        result.error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
    result.elapsed_secs = time.perf_counter() - t0
    return result


# ── public API ───────────────────────────────────────────────────────


def _rows_to_columns(rows: list[dict]) -> Columns:
    """Convert a list of flat row dicts into a columns dict of numpy arrays.

    Numeric columns become ``float64`` arrays (missing → ``NaN``); identity
    columns in :data:`_NON_NUMERIC` become ``object`` arrays.
    """
    if not rows:
        return {}

    keys: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                keys.append(k)

    n = len(rows)
    columns: Columns = {}
    for k in keys:
        if k in _NON_NUMERIC:
            columns[k] = np.array([r.get(k) for r in rows], dtype=object)
        else:
            columns[k] = np.fromiter(
                (to_float(r.get(k)) for r in rows), dtype=np.float64, count=n
            )
    return columns


def load_records(
    path: str | Path,
    workers: int | None = None,
) -> Columns:
    """Load all .jsonl (or .jsonl.gz) files under *path* into a columns dict.

    Args:
        path: Path to a single ``.jsonl`` / ``.jsonl.gz`` file, **or** a
              directory containing one or more such files (searched
              recursively).
        workers: Number of parallel workers.  Defaults to
                 ``os.cpu_count()`` (or 4 if unavailable).  Pass ``1``
                 to disable multiprocessing.

    Returns:
        A ``dict[str, np.ndarray]`` mapping column name to a length-N array
        (one entry per completion).  Numeric columns are ``float64`` (missing
        values, including optional reward-model columns, are ``NaN``); the
        ``model``, ``problem`` and ``source_file`` columns are ``object`` arrays.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If no ``.jsonl`` files are found under *path*.
    """
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"Path does not exist: {root}")

    if root.is_file():
        if not root.name.endswith((".jsonl", ".jsonl.gz")):
            raise ValueError(f"Expected a .jsonl or .jsonl.gz file, got: {root.name}")
        files = [root]
    else:
        files = sorted(root.rglob("*.jsonl")) + sorted(root.rglob("*.jsonl.gz"))
        if not files:
            raise ValueError(f"No .jsonl files found under {root}")

    n_files = len(files)
    n_workers = workers if workers is not None else (os.cpu_count() or 4)
    logger.info(
        "Loading %d file(s) from %s with %d worker(s)", n_files, root, n_workers
    )

    all_rows: list[dict] = []
    failed = 0
    t0 = time.perf_counter()

    if n_workers == 1 or n_files == 1:
        for fp in files:
            res = _process_file(fp)
            _handle_result(res, all_rows)
            if res.error:
                failed += 1
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_process_file, fp): fp for fp in files}
            for future in as_completed(futures):
                res: FileResult = future.result()
                _handle_result(res, all_rows)
                if res.error:
                    failed += 1

    elapsed = time.perf_counter() - t0
    logger.info(
        "Loaded %d rows from %d/%d file(s) in %.2fs",
        len(all_rows),
        n_files - failed,
        n_files,
        elapsed,
    )

    return _rows_to_columns(all_rows)


def _handle_result(res: FileResult, all_rows: list[dict]) -> None:
    """Log diagnostics for one FileResult and extend all_rows in place."""
    if res.error:
        logger.error("Failed %s:\n%s", Path(res.file_path).name, res.error)
        return
    all_rows.extend(res.rows)
    logger.debug(
        "%s — %d lines, %d parsed, %d blank, %d malformed (%.2fs)",
        Path(res.file_path).name,
        res.n_lines_total,
        res.n_records_parsed,
        res.n_lines_skipped,
        res.n_lines_malformed,
        res.elapsed_secs,
    )
    if res.n_lines_malformed:
        logger.warning(
            "%s: %d malformed JSON line(s) skipped",
            Path(res.file_path).name,
            res.n_lines_malformed,
        )


__all__ = ["FileResult", "load_records"]
