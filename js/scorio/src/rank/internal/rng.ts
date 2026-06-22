/**
 * A small seeded pseudo-random generator for the stochastic ranking methods
 * (`thompson`, `bayesian_mcmc`).
 *
 * These methods are inherently Monte Carlo; the Python reference seeds NumPy's
 * PCG64 generator, which cannot be reproduced bit-for-bit here. This generator
 * is deterministic given a seed so results are reproducible, but it is not
 * expected to match NumPy — the corresponding tests assert structural ranking
 * properties rather than exact values.
 */
export class SeededRng {
  private state: number;

  constructor(seed: number) {
    // SplitMix32 seeding.
    this.state = (seed >>> 0) || 0x9e3779b9;
  }

  /** Uniform in `[0, 1)`. */
  random(): number {
    // mulberry32
    this.state = (this.state + 0x6d2b79f5) | 0;
    let t = this.state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  /** Standard-normal sample scaled to `N(mean, std²)` via Box-Muller. */
  normal(mean = 0, std = 1): number {
    let u1 = this.random();
    const u2 = this.random();
    if (u1 < 1e-300) u1 = 1e-300;
    const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    return mean + std * z;
  }

  /** Gamma(shape, 1) via Marsaglia-Tsang (shape ≥ 0). */
  private gamma(shape: number): number {
    if (shape < 1) {
      // Boosting: Gamma(a) = Gamma(a+1) · U^{1/a}.
      const u = this.random();
      return this.gamma(shape + 1) * Math.pow(u < 1e-300 ? 1e-300 : u, 1 / shape);
    }
    const d = shape - 1 / 3;
    const c = 1 / Math.sqrt(9 * d);
    for (;;) {
      let x: number;
      let vCube: number;
      do {
        x = this.normal();
        vCube = 1 + c * x;
      } while (vCube <= 0);
      const v = vCube * vCube * vCube;
      const u = this.random();
      if (u < 1 - 0.0331 * x * x * x * x) return d * v;
      if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
    }
  }

  /** Beta(a, b) sample via two Gamma draws. */
  beta(a: number, b: number): number {
    const ga = this.gamma(a);
    const gb = this.gamma(b);
    const s = ga + gb;
    return s > 0 ? ga / s : 0.5;
  }
}
