/**
 * A small two-phase primal simplex, used to solve the zero-sum meta-game
 * maximin in `nash`.
 *
 * The Python reference calls `scipy.optimize.linprog(method="highs")` on the
 * linear-ordering maximin LP. For an antisymmetric payoff matrix the game value
 * is zero and the optimal strategy can be non-unique, so this solver returns a
 * valid maximin strategy (a vertex of the optimal face) rather than promising
 * the same vertex HiGHS picks.
 */

/**
 * Solve `min cᵀx` s.t. `M x >= b`, `x >= 0`, with `b >= 0`, via a two-phase
 * simplex using Bland's rule (guaranteed to terminate). Returns the optimal
 * `x`, or `null` if infeasible.
 */
function simplexMinGE(
  M: readonly (readonly number[])[],
  b: readonly number[],
  c: readonly number[],
): number[] | null {
  const m = M.length;
  const n = c.length;
  // Variables: x (n), surplus s (m), artificial a (m). Total N = n + 2m.
  const N = n + 2 * m;
  const SURPLUS = n;
  const ARTIF = n + m;

  // Tableau rows: m constraints; columns 0..N-1 plus RHS at index N.
  const T: number[][] = [];
  for (let i = 0; i < m; i++) {
    const row = new Array<number>(N + 1).fill(0);
    for (let j = 0; j < n; j++) row[j] = M[i]![j]!;
    row[SURPLUS + i] = -1; // surplus
    row[ARTIF + i] = 1; // artificial
    row[N] = b[i]!;
    T.push(row);
  }
  const basis = Array.from({ length: m }, (_, i) => ARTIF + i);

  const pivot = (objCost: (j: number) => number): boolean => {
    // Returns true if optimal, false if it performed a pivot. Bland's rule.
    // Reduced costs relative to current basis.
    const reduced = new Array<number>(N).fill(0);
    // Compute objective row = cost - sum basis cost * row.
    const cb = basis.map((bi) => objCost(bi));
    for (let j = 0; j < N; j++) {
      let z = 0;
      for (let i = 0; i < m; i++) z += cb[i]! * T[i]![j]!;
      reduced[j] = objCost(j) - z;
    }
    // Entering: smallest index with reduced cost < -tol (Bland).
    let entering = -1;
    for (let j = 0; j < N; j++) {
      if (reduced[j]! < -1e-9) {
        entering = j;
        break;
      }
    }
    if (entering === -1) return true; // optimal

    // Ratio test (Bland: smallest basis index on ties).
    let leaving = -1;
    let bestRatio = Infinity;
    for (let i = 0; i < m; i++) {
      const a = T[i]![entering]!;
      if (a > 1e-12) {
        const ratio = T[i]![N]! / a;
        if (
          ratio < bestRatio - 1e-12 ||
          (Math.abs(ratio - bestRatio) <= 1e-12 &&
            (leaving === -1 || basis[i]! < basis[leaving]!))
        ) {
          bestRatio = ratio;
          leaving = i;
        }
      }
    }
    if (leaving === -1) return true; // unbounded; treat as done

    // Pivot on (leaving, entering).
    const prow = T[leaving]!;
    const pv = prow[entering]!;
    for (let j = 0; j <= N; j++) prow[j]! /= pv;
    for (let i = 0; i < m; i++) {
      if (i === leaving) continue;
      const f = T[i]![entering]!;
      if (f === 0) continue;
      const row = T[i]!;
      for (let j = 0; j <= N; j++) row[j]! -= f * prow[j]!;
    }
    basis[leaving] = entering;
    return false;
  };

  // Phase 1: minimize sum of artificials.
  const phase1Cost = (j: number) => (j >= ARTIF ? 1 : 0);
  for (let guard = 0; guard < 10000; guard++) {
    if (pivot(phase1Cost)) break;
  }
  // Feasibility check.
  let artSum = 0;
  for (let i = 0; i < m; i++) if (basis[i]! >= ARTIF) artSum += Math.abs(T[i]![N]!);
  if (artSum > 1e-6) return null;

  // Phase 2: minimize cᵀx. Forbid artificials from re-entering by costing them ∞.
  const phase2Cost = (j: number) => (j >= ARTIF ? 1e9 : j < n ? c[j]! : 0);
  for (let guard = 0; guard < 10000; guard++) {
    if (pivot(phase2Cost)) break;
  }

  const x = new Array<number>(n).fill(0);
  for (let i = 0; i < m; i++) {
    const bi = basis[i]!;
    if (bi < n) x[bi] = T[i]![N]!;
  }
  return x;
}

/**
 * Maximin mixed strategy `x` (row maximizer) for a zero-sum game with payoff
 * matrix `A`. Returns a probability vector of length `L`.
 */
export function solveMaximinStrategy(A: readonly (readonly number[])[]): number[] | null {
  const L = A.length;
  let minA = Infinity;
  for (const row of A) for (const v of row) if (v < minA) minA = v;
  const K = 1 - minA; // shift so all payoffs are >= 1 > 0
  // M = (A + K)ᵀ so that constraint j is Σ_i A'_{ij} p_i >= 1.
  const M: number[][] = Array.from({ length: L }, () => new Array<number>(L).fill(0));
  for (let i = 0; i < L; i++) {
    for (let j = 0; j < L; j++) M[j]![i] = A[i]![j]! + K;
  }
  const b = new Array<number>(L).fill(1);
  const c = new Array<number>(L).fill(1);
  const p = simplexMinGE(M, b, c);
  if (p === null) return null;
  let sum = 0;
  for (const v of p) sum += Math.max(0, v);
  if (!(sum > 0)) return null;
  return p.map((v) => Math.max(0, v) / sum);
}
