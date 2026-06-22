/**
 * Hyperparameter validation helpers, mirroring the `_validate_positive_int` /
 * `_validate_positive_float` guards used throughout `scorio.rank`.
 */

/** Validate an integer hyperparameter `value >= minValue`. */
export function validatePositiveInt(name: string, value: number, minValue = 1): number {
  if (typeof value !== "number" || !Number.isInteger(value)) {
    throw new TypeError(`${name} must be an integer, got ${value}`);
  }
  if (value < minValue) {
    throw new Error(`${name} must be >= ${minValue}, got ${value}`);
  }
  return value;
}

/** Validate a positive finite scalar hyperparameter. */
export function validatePositiveFloat(name: string, value: number): number {
  const v = Number(value);
  if (!Number.isFinite(v) || v <= 0) {
    throw new Error(`${name} must be a positive finite scalar, got ${value}`);
  }
  return v;
}
