/**
 * Unconstrained minimizer — a limited-memory BFGS (L-BFGS) with an Armijo
 * backtracking line search and forward-difference gradients.
 *
 * The Python reference fits its Bradley-Terry / Plackett-Luce / IRT estimators
 * with `scipy.optimize.minimize(method="L-BFGS-B")` and no analytic gradient
 * (so SciPy itself uses 2-point finite differences). The objectives are smooth
 * and (for most variants) convex in the parameterization used, so any optimizer
 * that converges to the same stationary point reproduces the estimates. This
 * implementation mirrors SciPy's defaults closely enough for that purpose:
 * forward-difference step `√eps · max(1, |x|)`, gradient tolerance `1e-5`, and a
 * relative function-value tolerance `~2.2e-9`.
 */

const EPS = 2.220446049250313e-16;
const SQRT_EPS = Math.sqrt(EPS);

export interface MinimizeOptions {
  maxIter?: number;
  gtol?: number;
  ftol?: number;
  /** L-BFGS history size. */
  m?: number;
}

export interface MinimizeResult {
  x: number[];
  fun: number;
  iterations: number;
  success: boolean;
}

type Objective = (x: number[]) => number;

function dot(a: readonly number[], b: readonly number[]): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i]! * b[i]!;
  return s;
}

function maxAbs(a: readonly number[]): number {
  let m = 0;
  for (const v of a) m = Math.max(m, Math.abs(v));
  return m;
}

/** Forward-difference gradient, matching SciPy's default 2-point scheme. */
function numericalGradient(f: Objective, x: number[], fx: number): number[] {
  const n = x.length;
  const g = new Array<number>(n).fill(0);
  for (let i = 0; i < n; i++) {
    const xi = x[i]!;
    const h = SQRT_EPS * Math.max(1, Math.abs(xi));
    const saved = xi;
    x[i] = xi + h;
    const fh = f(x);
    x[i] = saved;
    g[i] = (fh - fx) / h;
  }
  return g;
}

/**
 * Minimize `f` starting from `x0`. Returns the best point found. `success` is
 * true when the gradient- or function-tolerance stopping criterion was met
 * before the iteration budget was exhausted.
 */
export function minimize(
  f: Objective,
  x0: readonly number[],
  options: MinimizeOptions = {},
): MinimizeResult {
  const maxIter = options.maxIter ?? 500;
  const gtol = options.gtol ?? 1e-5;
  const ftol = options.ftol ?? 2.220446049250313e-9;
  const m = options.m ?? 10;

  const n = x0.length;
  let x = x0.slice();
  let fx = f(x);
  let g = numericalGradient(f, x, fx);

  const sList: number[][] = [];
  const yList: number[][] = [];
  const rhoList: number[] = [];

  let success = false;
  let iter = 0;
  for (; iter < maxIter; iter++) {
    if (maxAbs(g) <= gtol) {
      success = true;
      break;
    }

    // Two-loop recursion to compute the search direction d = -H·g.
    const q = g.slice();
    const alphas: number[] = new Array(sList.length).fill(0);
    for (let i = sList.length - 1; i >= 0; i--) {
      const a = rhoList[i]! * dot(sList[i]!, q);
      alphas[i] = a;
      const yi = yList[i]!;
      for (let k = 0; k < n; k++) q[k]! -= a * yi[k]!;
    }
    let gamma = 1;
    if (sList.length > 0) {
      const sLast = sList[sList.length - 1]!;
      const yLast = yList[yList.length - 1]!;
      const yy = dot(yLast, yLast);
      if (yy > 0) gamma = dot(sLast, yLast) / yy;
    }
    const d = q.map((v) => v * gamma);
    for (let i = 0; i < sList.length; i++) {
      const beta = rhoList[i]! * dot(yList[i]!, d);
      const si = sList[i]!;
      const coef = alphas[i]! - beta;
      for (let k = 0; k < n; k++) d[k]! += coef * si[k]!;
    }
    for (let k = 0; k < n; k++) d[k] = -d[k]!;

    let slope = dot(g, d);
    if (slope >= 0) {
      // Not a descent direction; reset memory and use steepest descent.
      sList.length = 0;
      yList.length = 0;
      rhoList.length = 0;
      for (let k = 0; k < n; k++) d[k] = -g[k]!;
      slope = dot(g, d);
      if (slope >= 0) {
        success = true; // gradient effectively zero
        break;
      }
    }

    // Armijo backtracking line search.
    const c1 = 1e-4;
    let alpha = 1;
    const xNew = x.slice();
    let fNew = fx;
    let ok = false;
    for (let ls = 0; ls < 40; ls++) {
      for (let k = 0; k < n; k++) xNew[k] = x[k]! + alpha * d[k]!;
      fNew = f(xNew);
      if (Number.isFinite(fNew) && fNew <= fx + c1 * alpha * slope) {
        ok = true;
        break;
      }
      alpha *= 0.5;
    }
    if (!ok) {
      // Line search failed to make progress; stop.
      break;
    }

    const gNew = numericalGradient(f, xNew, fNew);
    const s = new Array<number>(n);
    const y = new Array<number>(n);
    for (let k = 0; k < n; k++) {
      s[k] = xNew[k]! - x[k]!;
      y[k] = gNew[k]! - g[k]!;
    }
    const sy = dot(s, y);
    if (sy > 1e-12) {
      sList.push(s);
      yList.push(y);
      rhoList.push(1 / sy);
      if (sList.length > m) {
        sList.shift();
        yList.shift();
        rhoList.shift();
      }
    }

    const fChange = Math.abs(fx - fNew);
    x = xNew.slice();
    fx = fNew;
    g = gNew;

    if (fChange <= ftol * Math.max(1, Math.abs(fx))) {
      success = true;
      break;
    }
  }

  return { x, fun: fx, iterations: iter, success };
}
