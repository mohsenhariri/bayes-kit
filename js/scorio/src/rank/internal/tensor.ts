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

/** Outcome accepted by Python's rank validator (`bool` is binary). */
export type Outcome = number | boolean;

/** A 2-D `(L, M)` matrix or a 3-D `(L, M, N)` tensor of outcomes. */
export type TensorInput =
  | readonly (readonly Outcome[])[]
  | readonly (readonly (readonly Outcome[])[])[];

/** Validated dense 3-D response tensor, shape `[L][M][N]`. */
export type Tensor3 = number[][][];

function isOutcome(x: unknown): x is Outcome {
  return typeof x === "number" || typeof x === "boolean";
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
  if (!Array.isArray(R)) {
    throw new Error(
      "Input R must be a 2D array of shape (L, M) or 3D array of shape (L, M, N)",
    );
  }
  const raw = R as readonly unknown[];
  let tensor: Tensor3;

  // Empty 2-D rows are intentionally classified as 2-D, matching
  // `np.asarray([[], []]).ndim == 2`. A nested empty trial row is 3-D.
  const is2d = raw.every(
    (row) => Array.isArray(row) && row.every((value) => isOutcome(value)),
  );
  const is3d = raw.every(
    (matrix) =>
      Array.isArray(matrix) &&
      matrix.every(
        (row) => Array.isArray(row) && row.every((value) => isOutcome(value)),
      ),
  );

  if (is2d) {
    tensor = raw.map((row) =>
      (row as readonly Outcome[]).map((value) => [coerceOutcome(value, binaryOnly)]),
    );
  } else if (is3d) {
    tensor = raw.map((matrix) =>
      (matrix as readonly (readonly Outcome[])[]).map((row) =>
        row.map((value) => coerceOutcome(value, binaryOnly)),
      ),
    );
  } else {
    throw new Error(
      "Input R must be a 2D array of shape (L, M) or 3D array of shape (L, M, N)",
    );
  }

  const L = tensor.length;
  const M = tensor[0]?.length ?? 0;
  const N = tensor[0]?.[0]?.length ?? 0;

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

  if (L < 2) throw new Error(`Need at least 2 models to rank, got L=${L}`);
  if (M < 1) throw new Error(`Need at least 1 question, got M=${M}`);
  if (N < 1) throw new Error(`Need at least 1 trial, got N=${N}`);

  return tensor;
}

function coerceOutcome(value: Outcome, binaryOnly: boolean): number {
  if (typeof value === "boolean") return value ? 1 : 0;
  if (!Number.isFinite(value)) {
    throw new Error("Input R must not contain NaN or Inf values");
  }
  if (!Number.isInteger(value)) {
    throw new Error(
      "Float inputs must be binary values (0.0 or 1.0). Use integer values for multiclass outcomes.",
    );
  }
  if (binaryOnly && value !== 0 && value !== 1) {
    throw new Error("Input R must contain only binary values (0 or 1)");
  }
  return value;
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

/** Return whether every vertex is reachable in both graph directions. */
export function isStronglyConnected(adjacency: readonly (readonly unknown[])[]): boolean {
  const n = adjacency.length;
  if (adjacency.some((row) => !Array.isArray(row) || row.length !== n)) {
    throw new Error("adjacency must be a square matrix");
  }
  if (n <= 1) return true;

  const reachable = (transpose: boolean): boolean[] => {
    const seen = new Array<boolean>(n).fill(false);
    const stack = [0];
    seen[0] = true;
    while (stack.length > 0) {
      const vertex = stack.pop()!;
      for (let neighbour = 0; neighbour < n; neighbour++) {
        const edge = transpose
          ? Boolean(adjacency[neighbour]![vertex])
          : Boolean(adjacency[vertex]![neighbour]);
        if (edge && !seen[neighbour]) {
          seen[neighbour] = true;
          stack.push(neighbour);
        }
      }
    }
    return seen;
  };

  return reachable(false).every(Boolean) && reachable(true).every(Boolean);
}

function flattenValues(value: unknown, output: unknown[] = []): unknown[] {
  if (Array.isArray(value)) {
    for (const item of value) flattenValues(item, output);
  } else {
    output.push(value);
  }
  return output;
}

function valueKey(value: unknown): string {
  if (typeof value === "number") {
    if (Number.isNaN(value)) return "number:NaN";
    if (Object.is(value, -0)) return "number:0";
  }
  return `${typeof value}:${String(value)}`;
}

/** Average scores for model rows with identical sufficient statistics. */
export function averageEquivalentScores(
  scores: readonly number[],
  sufficientStatistics: readonly unknown[],
): number[] {
  if (!Array.isArray(scores)) {
    throw new Error("scores must be a one-dimensional array");
  }
  if (!Array.isArray(sufficientStatistics) || sufficientStatistics.length !== scores.length) {
    throw new Error("sufficient_statistics must have one row for every score");
  }
  const result = scores.slice();
  const groups = new Map<string, number[]>();
  for (let index = 0; index < scores.length; index++) {
    const key = flattenValues(sufficientStatistics[index]).map(valueKey).join("\u0000");
    const members = groups.get(key);
    if (members) members.push(index);
    else groups.set(key, [index]);
  }
  for (const members of groups.values()) {
    if (members.length < 2) continue;
    const groupMean = members.reduce((total, index) => total + scores[index]!, 0) /
      members.length;
    for (const index of members) result[index] = groupMean;
  }
  return result;
}

function compareTuples(first: readonly unknown[], second: readonly unknown[]): number {
  for (let index = 0; index < first.length; index++) {
    const a = valueKey(first[index]);
    const b = valueKey(second[index]);
    if (a < b) return -1;
    if (a > b) return 1;
  }
  return 0;
}

/**
 * Average exact model orbits under simultaneous model/observation
 * permutations, matching Python's `average_event_exchangeable_scores`.
 */
export function averageEventExchangeableScores(
  scores: readonly number[],
  observations: readonly unknown[],
): number[] {
  if (!Array.isArray(observations) || observations.length !== scores.length) {
    throw new Error("observations must have one row for every score");
  }
  const data = observations.map((row) => flattenValues(row));
  if (data.some((row) => row.length !== data[0]!.length)) {
    throw new Error("observations must be rectangular");
  }
  const L = scores.length;

  const projectionMatches = (sourceRows: number[], targetRows: number[]): boolean => {
    const nColumns = data[0]!.length;
    const sourceColumns = Array.from({ length: nColumns }, (_, column) =>
      sourceRows.map((row) => data[row]![column]),
    ).sort(compareTuples);
    const targetColumns = Array.from({ length: nColumns }, (_, column) =>
      targetRows.map((row) => data[row]![column]),
    ).sort(compareTuples);
    return sourceColumns.every((column, index) => compareTuples(column, targetColumns[index]!) === 0);
  };

  const rowSignatures = data.map((row) => row.map(valueKey).sort().join("\u0000"));
  const signatureSizes = new Map<string, number>();
  for (const signature of rowSignatures) {
    signatureSizes.set(signature, (signatureSizes.get(signature) ?? 0) + 1);
  }
  const parent = Array.from({ length: L }, (_, index) => index);
  const find = (start: number): number => {
    let index = start;
    while (parent[index] !== index) {
      parent[index] = parent[parent[index]!]!;
      index = parent[index]!;
    }
    return index;
  };
  const union = (first: number, second: number): void => {
    const firstRoot = find(first);
    const secondRoot = find(second);
    if (firstRoot !== secondRoot) parent[secondRoot] = firstRoot;
  };

  const findAutomorphism = (source: number, target: number): number[] | null => {
    if (rowSignatures[source] !== rowSignatures[target]) return null;
    const sourceRows = [source];
    const targetRows = [target];
    const usedSource = new Array<boolean>(L).fill(false);
    const usedTarget = new Array<boolean>(L).fill(false);
    usedSource[source] = true;
    usedTarget[target] = true;
    const mapping = new Array<number>(L).fill(-1);
    mapping[source] = target;
    const sourceOrder = Array.from({ length: L }, (_, index) => index)
      .filter((index) => index !== source)
      .sort(
        (first, second) =>
          signatureSizes.get(rowSignatures[first]!)! -
          signatureSizes.get(rowSignatures[second]!)!,
      );
    if (!projectionMatches(sourceRows, targetRows)) return null;

    const search = (): boolean => {
      if (sourceRows.length === L) return true;
      const nextSource = sourceOrder.find((index) => !usedSource[index]);
      if (nextSource === undefined) return true;
      const compatible: number[] = [];
      for (let candidateTarget = 0; candidateTarget < L; candidateTarget++) {
        if (usedTarget[candidateTarget]) continue;
        if (rowSignatures[nextSource] !== rowSignatures[candidateTarget]) continue;
        if (
          projectionMatches(
            [...sourceRows, nextSource],
            [...targetRows, candidateTarget],
          )
        ) {
          compatible.push(candidateTarget);
        }
      }
      if (compatible.length === 0) return false;
      usedSource[nextSource] = true;
      sourceRows.push(nextSource);
      for (const candidateTarget of compatible) {
        usedTarget[candidateTarget] = true;
        targetRows.push(candidateTarget);
        mapping[nextSource] = candidateTarget;
        if (search()) return true;
        mapping[nextSource] = -1;
        targetRows.pop();
        usedTarget[candidateTarget] = false;
      }
      sourceRows.pop();
      usedSource[nextSource] = false;
      return false;
    };
    return search() ? mapping : null;
  };

  for (let first = 0; first < L; first++) {
    for (let second = first + 1; second < L; second++) {
      if (find(first) === find(second)) continue;
      const automorphism = findAutomorphism(first, second);
      if (automorphism === null) continue;
      automorphism.forEach((target, source) => union(source, target));
    }
  }
  const groupStatistics = parent.map((_, index) => [find(index)]);
  return averageEquivalentScores(scores, groupStatistics);
}
