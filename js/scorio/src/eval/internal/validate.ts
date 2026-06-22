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
  // Reject genuinely fractional values (e.g. probabilities like 0.8) rather
  // than silently truncating them. Integer-valued floats (1.0) pass, since
  // `Number.isInteger(1.0)` is true.
  if (!Number.isInteger(x)) {
    throw new Error(`Outcome matrix entries must be integers; got ${x}`);
  }
  return x;
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
