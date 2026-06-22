import { describe, expect, it } from "vitest";

import { gammaln } from "../src/eval/internal/math.js";
import { asMatrix } from "../src/eval/internal/validate.js";
import { passAtK } from "../src/eval/passAtK.js";
import { maxAtKCi } from "../src/eval/maxReward.js";

describe("review fixes", () => {
  it("toInt rejects fractional entries instead of truncating", () => {
    expect(() => asMatrix([[0, 0.8]])).toThrow(/must be integers/);
    expect(() => passAtK([[0, 0.8, 1]], 1)).toThrow(/must be integers/);
  });

  it("toInt still accepts integer-valued floats", () => {
    expect(asMatrix([[0.0, 1.0]])).toEqual([[0, 1]]);
    // Integer-valued floats compute identically to plain integers.
    expect(passAtK([[0.0, 1.0, 1.0]], 1)).toBe(passAtK([[0, 1, 1]], 1));
  });

  it("gammaln of a negative non-integer matches scipy (not NaN)", () => {
    // scipy.special.gammaln(-0.5) = 1.2655121234846454
    expect(gammaln(-0.5)).toBeCloseTo(1.2655121234846454, 10);
    expect(Number.isNaN(gammaln(-0.5))).toBe(false);
  });

  it("maxAtKCi rejects a non-integer k", () => {
    const R = [
      [0, 1, 2, 2, 1],
      [1, 1, 0, 2, 2],
    ];
    expect(() => maxAtKCi(R, 2.5, [0.0, 0.5, 1.0])).toThrow(/integer/);
  });
});
