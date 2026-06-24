"""Internal helpers for the columnar (``dict[str, np.ndarray]``) data model.

The categorical pipeline represents a loaded corpus as a *columns dict*:
``{column_name: np.ndarray}`` where every array has the same length (one entry
per completion).  Numeric columns are ``float64`` arrays (missing values are
``NaN``); identity columns (``model``, ``problem``, ``source_file``) are
``object`` arrays.  These helpers replace the few pandas operations the pipeline
used to rely on, so the package depends only on numpy.
"""

from __future__ import annotations

import math

import numpy as np

Columns = dict


def to_float(v) -> float:
    """Coerce a value to float, returning ``NaN`` on failure (like ``to_numeric``)."""
    if v is None:
        return float("nan")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def is_missing(v) -> bool:
    """True if *v* is ``None`` or a NaN float."""
    return v is None or (isinstance(v, float) and math.isnan(v))


def num_rows(columns: Columns) -> int:
    """Number of rows (length of any column); 0 for an empty columns dict."""
    for arr in columns.values():
        return len(arr)
    return 0


def select_rows(columns: Columns, idxs) -> Columns:
    """Return a new columns dict containing only rows *idxs* (fancy-indexed)."""
    idx = np.asarray(idxs, dtype=int)
    return {k: v[idx] for k, v in columns.items()}


__all__ = ["Columns", "to_float", "is_missing", "num_rows", "select_rows"]
