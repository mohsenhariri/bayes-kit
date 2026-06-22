/**
 * Response-tensor coercion, validation, and pairwise summaries.
 *
 * Port of `scorio/rank/_base.py`. Ranking methods operate on a binary (or
 * categorical) response tensor `R` of shape `(L, M, N)`:
 *
 * - `L` = number of models, `M` = number of questions, `N` = trials per question.
 * - `R[l][m][n] = 1` means model `l` solved question `m` on trial `n`.
 *
 * A 2-D `(L, M)` matrix is promoted to `(L, M, 1)`.
 */

/** A 2-D `(L, M)` matrix or a 3-D `(L, M, N)` tensor of outcomes. */
export type TensorInput =
  | readonly (readonly number[])[]
  | readonly (readonly (readonly number[])[])[];

/** Validated dense 3-D response tensor, shape `[L][M][N]`. */
export type Tensor3 = number[][][];

function isNumber(x: unknown): x is number {
  return typeof x === "number";
}

/**
 * Validate and coerce input to a 3-D integer tensor `(L, M, N)`.
 *
 * Mirrors `_base.validate_input`. A 2-D matrix becomes `(L, M, 1)`. With
 * `binaryOnly` (the default), entries must be in `{0, 1}`; otherwise any
 * non-negative integer-valued outcome is allowed (matching the categorical
 * `bayes` path which passes `binary_only=False`).
 */
export function validateInput(R: TensorInput, binaryOnly = true): Tensor3 {
  if (!Array.isArray(R) || R.length === 0) {
    throw new Error(
      "Input R must be a 2D array of shape (L, M) or 3D array of shape (L, M, N)",
    );
  }
  const first = (R as readonly unknown[])[0];
  let tensor: Tensor3;

  if (Array.isArray(first) && first.length > 0 && Array.isArray(first[0])) {
    // 3-D input.
    tensor = (R as readonly (readonly (readonly number[])[])[]).map((mat) =>
      (mat as readonly (readonly number[])[]).map((row) => row.map(coerceInt)),
    );
  } else if (Array.isArray(first)) {
    // 2-D input -> promote to (L, M, 1).
    tensor = (R as readonly (readonly number[])[]).map((row) =>
      row.map((v) => [coerceInt(v)]),
    );
  } else {
    throw new Error(
      "Input R must be a 2D array of shape (L, M) or 3D array of shape (L, M, N)",
    );
  }

  const L = tensor.length;
  const M = tensor[0]!.length;
  const N = tensor[0]![0]!.length;

  // Rectangularity.
  for (const mat of tensor) {
    if (mat.length !== M) {
      throw new Error("Input R must be a rectangular (L, M, N) tensor.");
    }
    for (const row of mat) {
      if (row.length !== N) {
        throw new Error("Input R must be a rectangular (L, M, N) tensor.");
      }
    }
  }

  // Value validation.
  for (const mat of tensor) {
    for (const row of mat) {
      for (const v of row) {
        if (!Number.isFinite(v)) {
          throw new Error("Input R must not contain NaN or Inf values");
        }
        if (binaryOnly && v !== 0 && v !== 1) {
          throw new Error("Input R must contain only binary values (0 or 1)");
        }
        if (v < 0) {
          throw new Error("Input R must contain only non-negative outcomes");
        }
      }
    }
  }

  if (L < 2) throw new Error(`Need at least 2 models to rank, got L=${L}`);
  if (M < 1) throw new Error(`Need at least 1 question, got M=${M}`);
  if (N < 1) throw new Error(`Need at least 1 trial, got N=${N}`);

  return tensor;
}

function coerceInt(x: number): number {
  if (!isNumber(x) || !Number.isFinite(x)) {
    if (Number.isNaN(x) || x === Infinity || x === -Infinity) {
      throw new Error("Input R must not contain NaN or Inf values");
    }
    throw new Error("Input R must be numeric");
  }
  if (!Number.isInteger(x)) {
    throw new Error("Float inputs must be binary values (0.0 or 1.0).");
  }
  return x;
}

/** Shape `[L, M, N]` of a validated tensor. */
export function shape3(R: Tensor3): [number, number, number] {
  return [R.length, R[0]!.length, R[0]![0]!.length];
}

/**
 * Pairwise decisive-win matrix. `wins[i][j]` counts `(m, n)` where model `i`
 * answered correctly and model `j` did not. Mirrors `build_pairwise_wins`.
 */
export function buildPairwiseWins(R: Tensor3): number[][] {
  const [L, M, N] = shape3(R);
  const wins = zeros2(L, L);
  for (let i = 0; i < L; i++) {
    for (let j = i + 1; j < L; j++) {
      let iWins = 0;
      let jWins = 0;
      for (let m = 0; m < M; m++) {
        for (let n = 0; n < N; n++) {
          const ri = R[i]![m]![n]!;
          const rj = R[j]![m]![n]!;
          if (ri === 1 && rj === 0) iWins += 1;
          else if (rj === 1 && ri === 0) jWins += 1;
        }
      }
      wins[i]![j] = iWins;
      wins[j]![i] = jWins;
    }
  }
  return wins;
}

/**
 * Pairwise win and tie counts. `wins[i][j]` are decisive wins of `i` over `j`;
 * `ties[i][j] = ties[j][i]` count `(m, n)` where both outcomes are equal.
 * Mirrors `build_pairwise_counts`.
 */
export function buildPairwiseCounts(R: Tensor3): {
  wins: number[][];
  ties: number[][];
} {
  const [L, M, N] = shape3(R);
  const wins = zeros2(L, L);
  const ties = zeros2(L, L);
  for (let i = 0; i < L; i++) {
    for (let j = i + 1; j < L; j++) {
      let iWins = 0;
      let jWins = 0;
      let same = 0;
      for (let m = 0; m < M; m++) {
        for (let n = 0; n < N; n++) {
          const ri = R[i]![m]![n]!;
          const rj = R[j]![m]![n]!;
          if (ri === rj) same += 1;
          else if (ri === 1 && rj === 0) iWins += 1;
          else if (rj === 1 && ri === 0) jWins += 1;
        }
      }
      wins[i]![j] = iWins;
      wins[j]![i] = jWins;
      ties[i]![j] = same;
      ties[j]![i] = same;
    }
  }
  return { wins, ties };
}

/** Per-question correct counts `k[l][m] = sum_n R[l][m][n]`, shape `(L, M)`. */
export function perQuestionCorrectCounts(R: Tensor3): number[][] {
  const [L, M, N] = shape3(R);
  const k = zeros2(L, M);
  for (let l = 0; l < L; l++) {
    for (let m = 0; m < M; m++) {
      let s = 0;
      for (let n = 0; n < N; n++) s += R[l]![m]![n]!;
      k[l]![m] = s;
    }
  }
  return k;
}

/** Numerically stable logistic sigmoid with the reference's `[-30, 30]` clamp. */
export function sigmoid(x: number): number {
  const c = x < -30 ? -30 : x > 30 ? 30 : x;
  return 1 / (1 + Math.exp(-c));
}

/** Allocate an `r x c` matrix of zeros. */
export function zeros2(r: number, c: number): number[][] {
  return Array.from({ length: r }, () => new Array<number>(c).fill(0));
}
