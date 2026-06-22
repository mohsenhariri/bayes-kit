/**
 * Small dense linear-algebra primitives (zero dependencies).
 *
 * The ranking estimators only need operations on small matrices: matrix-vector
 * products, a symmetric eigendecomposition (Jacobi), a symmetric pseudoinverse,
 * and Gauss-Hermite quadrature (built from the symmetric eigendecomposition via
 * Golub-Welsch). These stand in for the `numpy.linalg` / `numpy.polynomial`
 * calls in the Python reference.
 */

/** `A · v` for `A` shape `(r, c)` and `v` length `c`. */
export function matVec(A: readonly (readonly number[])[], v: readonly number[]): number[] {
  return A.map((row) => {
    let s = 0;
    for (let j = 0; j < row.length; j++) s += row[j]! * v[j]!;
    return s;
  });
}

/** `Aᵀ · v` for `A` shape `(r, c)` and `v` length `r` → result length `c`. */
export function matTVec(
  A: readonly (readonly number[])[],
  v: readonly number[],
): number[] {
  const r = A.length;
  const c = A[0]!.length;
  const out = new Array<number>(c).fill(0);
  for (let i = 0; i < r; i++) {
    const vi = v[i]!;
    const row = A[i]!;
    for (let j = 0; j < c; j++) out[j]! += row[j]! * vi;
  }
  return out;
}

/** Matrix product `A · B`. */
export function matMul(
  A: readonly (readonly number[])[],
  B: readonly (readonly number[])[],
): number[][] {
  const r = A.length;
  const k = B.length;
  const c = B[0]!.length;
  const out = Array.from({ length: r }, () => new Array<number>(c).fill(0));
  for (let i = 0; i < r; i++) {
    for (let t = 0; t < k; t++) {
      const a = A[i]![t]!;
      if (a === 0) continue;
      const brow = B[t]!;
      const orow = out[i]!;
      for (let j = 0; j < c; j++) orow[j]! += a * brow[j]!;
    }
  }
  return out;
}

/** Transpose. */
export function transpose(A: readonly (readonly number[])[]): number[][] {
  const r = A.length;
  const c = A[0]!.length;
  const out = Array.from({ length: c }, () => new Array<number>(r).fill(0));
  for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) out[j]![i] = A[i]![j]!;
  return out;
}

/** L1 norm of `a - b`. */
export function l1Diff(a: readonly number[], b: readonly number[]): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += Math.abs(a[i]! - b[i]!);
  return s;
}

export interface EigResult {
  /** Eigenvalues in ascending order. */
  values: number[];
  /** Eigenvectors as columns: `vectors[i][k]` is component `i` of eigenvector `k`. */
  vectors: number[][];
}

/**
 * Symmetric eigendecomposition via the cyclic Jacobi algorithm, returning
 * eigenvalues ascending (matching `numpy.linalg.eigh`) and orthonormal
 * eigenvectors as columns. Suitable for the small symmetric matrices used here.
 */
export function eigSymmetric(input: readonly (readonly number[])[]): EigResult {
  const n = input.length;
  // Work on a mutable copy.
  const a = input.map((row) => row.slice());
  const v: number[][] = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)),
  );

  if (n === 1) return { values: [a[0]![0]!], vectors: [[1]] };

  const maxSweeps = 100;
  for (let sweep = 0; sweep < maxSweeps; sweep++) {
    // Sum of squared off-diagonals.
    let off = 0;
    for (let p = 0; p < n; p++)
      for (let q = p + 1; q < n; q++) off += a[p]![q]! * a[p]![q]!;
    if (off < 1e-30) break;

    for (let p = 0; p < n; p++) {
      for (let q = p + 1; q < n; q++) {
        const apq = a[p]![q]!;
        if (Math.abs(apq) < 1e-300) continue;
        const app = a[p]![p]!;
        const aqq = a[q]![q]!;
        const theta = (aqq - app) / (2 * apq);
        const t =
          Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;

        // Rotate rows/cols p, q.
        for (let i = 0; i < n; i++) {
          const aip = a[i]![p]!;
          const aiq = a[i]![q]!;
          a[i]![p] = c * aip - s * aiq;
          a[i]![q] = s * aip + c * aiq;
        }
        for (let i = 0; i < n; i++) {
          const api = a[p]![i]!;
          const aqi = a[q]![i]!;
          a[p]![i] = c * api - s * aqi;
          a[q]![i] = s * api + c * aqi;
        }
        for (let i = 0; i < n; i++) {
          const vip = v[i]![p]!;
          const viq = v[i]![q]!;
          v[i]![p] = c * vip - s * viq;
          v[i]![q] = s * vip + c * viq;
        }
      }
    }
  }

  const values = a.map((row, i) => row[i]!);
  const idx = Array.from({ length: n }, (_, i) => i).sort(
    (x, y) => values[x]! - values[y]!,
  );
  const sortedValues = idx.map((i) => values[i]!);
  const sortedVectors = Array.from({ length: n }, (_, i) =>
    idx.map((k) => v[i]![k]!),
  );
  return { values: sortedValues, vectors: sortedVectors };
}

/**
 * Moore-Penrose pseudoinverse of a symmetric matrix via its eigendecomposition.
 * Eigenvalues with `|λ| <= rcond · max|λ|` are treated as zero.
 */
export function pinvSymmetric(
  A: readonly (readonly number[])[],
  rcond = 1e-12,
): number[][] {
  const n = A.length;
  const { values, vectors } = eigSymmetric(A);
  let maxAbs = 0;
  for (const w of values) maxAbs = Math.max(maxAbs, Math.abs(w));
  const cutoff = rcond * maxAbs;

  const out = Array.from({ length: n }, () => new Array<number>(n).fill(0));
  for (let k = 0; k < n; k++) {
    const w = values[k]!;
    if (Math.abs(w) <= cutoff) continue;
    const inv = 1 / w;
    for (let i = 0; i < n; i++) {
      const vik = vectors[i]![k]!;
      if (vik === 0) continue;
      for (let j = 0; j < n; j++) out[i]![j]! += inv * vik * vectors[j]![k]!;
    }
  }
  return out;
}

/**
 * Gauss-Hermite quadrature nodes and weights for the physicists' weight
 * `e^{-x²}` (matching `numpy.polynomial.hermite.hermgauss`). Computed via the
 * Golub-Welsch eigendecomposition of the Jacobi matrix.
 */
export function hermgauss(n: number): { nodes: number[]; weights: number[] } {
  // Symmetric tridiagonal Jacobi matrix: diagonal 0, off-diagonal sqrt(k/2).
  const J = Array.from({ length: n }, () => new Array<number>(n).fill(0));
  for (let k = 1; k < n; k++) {
    const b = Math.sqrt(k / 2);
    J[k]![k - 1] = b;
    J[k - 1]![k] = b;
  }
  const { values, vectors } = eigSymmetric(J);
  const mu0 = Math.sqrt(Math.PI); // ∫ e^{-x²} dx
  const nodes = values.slice();
  const weights = values.map((_, k) => {
    const first = vectors[0]![k]!;
    return mu0 * first * first;
  });
  return { nodes, weights };
}
