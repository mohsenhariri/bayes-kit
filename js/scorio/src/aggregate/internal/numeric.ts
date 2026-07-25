/** NumPy-like coercion for scalar or rectangular, arbitrarily nested numbers. */

export type NumericInput = number | readonly NumericInput[];

interface Flattened {
  values: number[];
  shape: number[];
}

function sameShape(left: readonly number[], right: readonly number[]): boolean {
  return (
    left.length === right.length &&
    left.every((dimension, index) => dimension === right[index])
  );
}

function flattenRectangular(input: NumericInput, name: string): Flattened {
  if (typeof input === "number") return { values: [input], shape: [] };
  if (!Array.isArray(input)) throw new Error(`${name} must contain numbers.`);
  if (input.length === 0) return { values: [], shape: [0] };

  const children = input.map((entry) => flattenRectangular(entry, name));
  const childShape = children[0]!.shape;
  if (children.some((child) => !sameShape(child.shape, childShape))) {
    throw new Error(`${name} must be a rectangular numeric array.`);
  }
  return {
    values: children.flatMap((child) => child.values),
    shape: [input.length, ...childShape],
  };
}

/** Equivalent to `np.asarray(input, dtype=float).reshape(-1)` plus finiteness. */
export function asFiniteVector(
  input: NumericInput,
  name: string,
  emptyMessage = "need at least one token (T >= 1).",
): number[] {
  const { values } = flattenRectangular(input, name);
  if (values.length === 0) throw new Error(emptyMessage);
  if (values.some((value) => !Number.isFinite(value))) {
    throw new Error(`${name} must all be finite.`);
  }
  return values;
}

