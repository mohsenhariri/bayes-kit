/** Runtime helpers for mirroring Python's scalar argument conventions. */

/** Python-style truth testing for option values accepted dynamically at runtime. */
export function pythonTruthy(value: unknown): boolean {
  if (value === null || value === undefined || value === false) return false;
  if (typeof value === "number") return value !== 0;
  if (typeof value === "bigint") return value !== 0n;
  if (typeof value === "string" || Array.isArray(value)) return value.length !== 0;
  if (value instanceof Map || value instanceof Set) return value.size !== 0;
  if (ArrayBuffer.isView(value)) return value.byteLength !== 0;
  if (typeof value === "object") {
    const prototype = Object.getPrototypeOf(value);
    if (prototype === Object.prototype || prototype === null) {
      return Object.keys(value).length !== 0;
    }
  }
  return true;
}

/**
 * Accept the JS counterparts of Python real scalars used directly in numeric
 * comparisons. Python booleans are integers, while strings and `None` are not
 * implicitly converted by those comparisons.
 */
export function pythonComparableNumber(value: unknown, name: string): number {
  if (typeof value === "boolean") return value ? 1 : 0;
  if (typeof value === "number") return value;
  throw new Error(`${name} must be a number; got ${String(value)}.`);
}

/** Python `float(value)` for the scalar option forms used by aggregate APIs. */
const PYTHON_FLOAT_PATTERN =
  /^[+-]?(?:(?:(?:\d(?:_?\d)*(?:\.(?:\d(?:_?\d)*)?)?|\.\d(?:_?\d)*)(?:[eE][+-]?\d(?:_?\d)*)?)|inf(?:inity)?|nan)$/i;

export function pythonFloat(value: unknown, name: string): number {
  if (value === null || value === undefined) {
    throw new Error(`${name} must be a number; got ${String(value)}.`);
  }
  if (typeof value === "number") return value;
  if (typeof value === "boolean") return value ? 1 : 0;
  if (typeof value === "string") {
    const stripped = value.trim();
    if (!PYTHON_FLOAT_PATTERN.test(stripped)) {
      throw new Error(`${name} must be a number; got ${String(value)}.`);
    }
    if (/^[+-]?nan$/i.test(stripped)) return NaN;
    if (/^[+-]?inf(?:inity)?$/i.test(stripped)) {
      return stripped.startsWith("-") ? -Infinity : Infinity;
    }
    return Number(stripped.split("_").join(""));
  }
  throw new Error(`${name} must be a number; got ${String(value)}.`);
}

/** Python `int(value)` for DeepConf window and tail-length options. */
export function pythonInt(value: unknown, name: string): number {
  if (value === null || value === undefined) {
    throw new Error(`${name} must be an integer; got ${String(value)}.`);
  }
  if (typeof value === "boolean") return value ? 1 : 0;
  if (typeof value === "number") {
    if (!Number.isFinite(value)) {
      throw new Error(`${name} must be an integer; got ${String(value)}.`);
    }
    return Math.trunc(value);
  }
  if (typeof value === "string") {
    const stripped = value.trim();
    // Python permits underscores between decimal digits in integer strings.
    if (/^[+-]?\d(?:_?\d)*$/.test(stripped)) {
      const converted = Number(stripped.split("_").join(""));
      if (Number.isSafeInteger(converted)) return converted;
    }
  }
  throw new Error(`${name} must be an integer; got ${String(value)}.`);
}
