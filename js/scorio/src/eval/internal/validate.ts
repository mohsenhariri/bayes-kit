/**
 * Input coercion and validation, mirroring `scorio/eval/utils.py`.
 */

/** Accepted outcome-matrix input: a 1-D row or a 2-D `M x N` matrix. */
export type Matrix = readonly number[] | readonly (readonly number[])[];

/**
 * Coerce input to a rectangular 2-D integer matrix.
 *
 * A 1-D array is treated as a single row, matching `_as_2d_int_matrix`.
 */
export function asMatrix(R: Matrix): number[][] {
  if (R.length === 0) return [[]];
  if (Array.isArray((R as readonly unknown[])[0])) {
    const rows = R as readonly (readonly number[])[];
    const ncols = rows[0]!.length;
    return rows.map((row) => {
      if (row.length !== ncols) {
        throw new Error("R must be a rectangular 2D array.");
      }
      return row.map(toInt);
    });
  }
  return [(R as readonly number[]).map(toInt)];
}

function toInt(x: number): number {
  // NumPy's `np.asarray(..., dtype=int)` truncates finite floating-point values
  // toward zero, accepts booleans, and parses integer strings before the
  // metric-specific range check. Mirror those runtime coercions even though
  // the typed JS surface intentionally advertises numeric matrices.
  const raw = x as unknown;
  let numeric: number;
  if (typeof raw === "boolean") {
    numeric = Number(raw);
  } else if (typeof raw === "string") {
    const stripped = raw.trim();
    if (!/^[+-]?\d(?:_?\d)*$/.test(stripped)) {
      throw new Error(`Outcome matrix entries must be integer-like; got ${raw}`);
    }
    numeric = Number(stripped.replace(/_/g, ""));
  } else if (typeof raw === "number") {
    numeric = raw;
  } else {
    throw new Error(`Outcome matrix entries must be numeric; got ${String(raw)}`);
  }
  if (!Number.isFinite(numeric)) {
    throw new Error(`Outcome matrix entries must be finite; got ${x}`);
  }
  const value = Math.trunc(numeric);
  return value === 0 ? 0 : value;
}

/**
 * Coerce an optional-prior input using Python's special 1-D `reshape(M, -1)`
 * rule. A nested 2-D input retains its explicit row count.
 */
export function asPriorMatrix(R0: Matrix, rowCount: number): number[][] {
  if (R0.length === 0) {
    return Array.from({ length: rowCount }, () => []);
  }
  if (Array.isArray((R0 as readonly unknown[])[0])) {
    return asMatrix(R0);
  }

  const flat = (R0 as readonly number[]).map(toInt);
  if (rowCount <= 0 || flat.length % rowCount !== 0) {
    throw new Error(
      `R0 with ${flat.length} entries cannot be reshaped to ${rowCount} rows.`,
    );
  }
  const columnCount = flat.length / rowCount;
  return Array.from({ length: rowCount }, (_, row) =>
    flat.slice(row * columnCount, (row + 1) * columnCount),
  );
}

/** Validate that every entry lies in the closed integer interval `[low, high]`. */
export function validateMatrixRange(
  R: readonly (readonly number[])[],
  low: number,
  high: number,
  name: string,
): void {
  for (const row of R) {
    for (const v of row) {
      if (v < low || v > high) {
        throw new Error(
          `Entries of ${name} must be integers in [${low}, ${high}].`,
        );
      }
    }
  }
}

/** Validate that every entry is binary (in `{0, 1}`). */
export function validateBinary(
  R: readonly (readonly number[])[],
  name = "R",
): void {
  validateMatrixRange(R, 0, 1, name);
}

/** Sum each row of the matrix. */
export function rowSums(R: readonly (readonly number[])[]): number[] {
  return R.map((row) => row.reduce((s, v) => s + v, 0));
}

/** Per-row counts of the values `0..length-1` (a row-wise bincount). */
export function rowBincount(
  R: readonly (readonly number[])[],
  length: number,
): number[][] {
  return R.map((row) => {
    const counts = new Array<number>(length).fill(0);
    for (const v of row) counts[v]! += 1;
    return counts;
  });
}
