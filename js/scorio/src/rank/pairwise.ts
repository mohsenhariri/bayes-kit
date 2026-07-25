/**
 * Sequential pairwise rating methods (Elo, TrueSkill, Glicko).
 * Port of `scorio/rank/pairwise.py`.
 *
 * Each `(question, trial)` event induces head-to-head outcomes for every model
 * pair; ratings are updated online, so final ratings depend on stream order.
 */

import { rankScores } from "./internal/rankScores.js";
import { normCdf, normPdf } from "./internal/special.js";
import { shape3, validateInput, type TensorInput } from "./internal/tensor.js";
import type { BaseRankOptions, RankResult } from "./internal/result.js";

/** Tie policy for `(1,1)` and `(0,0)` outcomes. */
export type PairwiseTieHandling = "skip" | "draw" | "correct_draw_only";

const PYTHON_FLOAT_PATTERN = /^[+-]?(?:(?:(?:\d(?:_?\d)*(?:\.(?:\d(?:_?\d)*)?)?|\.\d(?:_?\d)*)(?:[eE][+-]?\d(?:_?\d)*)?)|inf(?:inity)?|nan)$/i;

/** Runtime equivalent of Python's `float(value)` for ordinary JS scalars. */
function pythonFloat(value: unknown, errorMessage: string): number {
  if (typeof value === "number") return value;
  if (typeof value === "boolean") return value ? 1 : 0;
  if (typeof value !== "string") throw new TypeError(errorMessage);

  const text = value.trim();
  if (!PYTHON_FLOAT_PATTERN.test(text)) throw new Error(errorMessage);
  if (/^[+-]?nan$/i.test(text)) return Number.NaN;
  if (/^[+-]?inf(?:inity)?$/i.test(text)) return text.startsWith("-") ? -Infinity : Infinity;
  return Number(text.replace(/_/g, ""));
}

function defaultIfUndefined<T>(value: T | undefined, fallback: T): T {
  return value === undefined ? fallback : value;
}

function pythonTruthy(value: unknown): boolean {
  if (value == null) return false;
  if (typeof value === "boolean") return value;
  if (typeof value === "number") return value !== 0;
  if (typeof value === "bigint") return value !== 0n;
  if (typeof value === "string" || Array.isArray(value)) return value.length > 0;
  return true;
}

function validateTieHandling(t: unknown): PairwiseTieHandling {
  const tiePolicy = String(t);
  if (
    tiePolicy !== "skip" &&
    tiePolicy !== "draw" &&
    tiePolicy !== "correct_draw_only"
  ) {
    throw new Error('tie_handling must be one of: "skip", "draw", "correct_draw_only"');
  }
  return tiePolicy;
}

/** Options for {@link elo}. */
export interface EloOptions extends BaseRankOptions {
  K?: number;
  initialRating?: number;
  tieHandling?: PairwiseTieHandling;
}

/** Rank models with sequential Elo updates on induced pairwise matches. */
export function elo(R: TensorInput, options: EloOptions = {}): RankResult {
  const method = defaultIfUndefined(options.method, "competition");
  const tensor = validateInput(R);
  const [L, M, N] = shape3(tensor);

  const K = pythonFloat(
    defaultIfUndefined(options.K, 32),
    "K must be a positive finite scalar",
  );
  if (!Number.isFinite(K) || K <= 0) {
    throw new Error(`K must be a positive finite scalar; got ${K}`);
  }
  const initialRating = pythonFloat(
    defaultIfUndefined(options.initialRating, 1500),
    "initial_rating must be finite.",
  );
  if (!Number.isFinite(initialRating)) throw new Error("initial_rating must be finite.");
  const tiePolicy = validateTieHandling(
    defaultIfUndefined(options.tieHandling, "correct_draw_only"),
  );

  const ratings = new Array<number>(L).fill(initialRating);

  for (let t = 0; t < N; t++) {
    for (let q = 0; q < M; q++) {
      // Every match induced by one event is evaluated against the same
      // pre-event rating snapshot.  Applying the accumulated delta once keeps
      // the update invariant to model label/order, matching Python.
      const eventRatings = ratings.slice();
      const ratingDelta = new Array<number>(L).fill(0);
      for (let i = 0; i < L; i++) {
        for (let j = i + 1; j < L; j++) {
          const ri = tensor[i]![q]![t]!;
          const rj = tensor[j]![q]![t]!;
          let Si: number;
          let Sj: number;
          if (ri === rj) {
            if (tiePolicy === "skip") continue;
            if (tiePolicy === "draw") {
              Si = 0.5;
              Sj = 0.5;
            } else {
              if (ri === 1) {
                Si = 0.5;
                Sj = 0.5;
              } else continue;
            }
          } else if (ri > rj) {
            Si = 1;
            Sj = 0;
          } else {
            Si = 0;
            Sj = 1;
          }
          const Ri = eventRatings[i]!;
          const Rj = eventRatings[j]!;
          const Ei = 1 / (1 + 10 ** ((Rj - Ri) / 400));
          const Ej = 1 - Ei;
          ratingDelta[i]! += K * (Si - Ei);
          ratingDelta[j]! += K * (Sj - Ej);
        }
      }
      for (let i = 0; i < L; i++) ratings[i]! += ratingDelta[i]!;
    }
  }
  return { ranking: rankScores(ratings, method), scores: ratings };
}

/** Options for {@link trueskill}. */
export interface TrueSkillOptions extends BaseRankOptions {
  muInitial?: number;
  sigmaInitial?: number;
  beta?: number;
  tau?: number;
  tieHandling?: PairwiseTieHandling;
  drawMargin?: number;
}

/** Rank models with a two-player TrueSkill update stream. */
export function trueskill(R: TensorInput, options: TrueSkillOptions = {}): RankResult {
  const method = defaultIfUndefined(options.method, "competition");
  const tensor = validateInput(R);
  const [L, M, N] = shape3(tensor);

  // Python performs all five float conversions before checking their ranges.
  const muInitial = pythonFloat(
    defaultIfUndefined(options.muInitial, 25),
    "mu_initial must be finite.",
  );
  const sigmaInitial = pythonFloat(
    defaultIfUndefined(options.sigmaInitial, 25 / 3),
    "sigma_initial must be a positive finite scalar.",
  );
  const beta = pythonFloat(
    defaultIfUndefined(options.beta, 25 / 6),
    "beta must be a positive finite scalar.",
  );
  const tau = pythonFloat(
    defaultIfUndefined(options.tau, 25 / 300),
    "tau must be a nonnegative finite scalar.",
  );
  const drawMargin = pythonFloat(
    defaultIfUndefined(options.drawMargin, 0),
    "draw_margin must be a nonnegative finite scalar.",
  );
  if (!Number.isFinite(muInitial)) throw new Error("mu_initial must be finite.");
  if (!Number.isFinite(sigmaInitial) || sigmaInitial <= 0)
    throw new Error("sigma_initial must be a positive finite scalar.");
  if (!Number.isFinite(beta) || beta <= 0)
    throw new Error("beta must be a positive finite scalar.");
  if (!Number.isFinite(tau) || tau < 0)
    throw new Error("tau must be a nonnegative finite scalar.");
  if (!Number.isFinite(drawMargin) || drawMargin < 0)
    throw new Error("draw_margin must be a nonnegative finite scalar.");
  const tiePolicy = validateTieHandling(
    defaultIfUndefined(options.tieHandling, "skip"),
  );

  const mu = new Array<number>(L).fill(muInitial);
  let sigma = new Array<number>(L).fill(sigmaInitial);

  const winCorrections = (t: number, epsilon: number): [number, number] => {
    const x = t - epsilon;
    let v: number;
    let w: number;
    if (x < -10) {
      // Stable lower-tail inverse-Mills expansions.  Computing phi/Phi and
      // v*(v+x) directly loses all precision in this range.
      const y = -x;
      const inverseY = 1 / y;
      const inverseY2 = inverseY * inverseY;
      v =
        y +
        inverseY *
          (1 + inverseY2 * (-2 + inverseY2 * (10 + inverseY2 * -74)));
      w = 1 + inverseY2 * (-1 + inverseY2 * (6 + inverseY2 * -50));
    } else {
      const denom = normCdf(x);
      v = normPdf(x) / denom;
      w = v * (v + x);
    }
    return [v, Math.min(Math.max(w, 0), 1)];
  };

  const drawCorrections = (t: number, epsilon: number): [number, number] => {
    // Conditional-Gaussian limit for a zero-width draw interval.
    if (epsilon <= Math.sqrt(Number.EPSILON) * (1 + Math.abs(t))) return [-t, 1];

    const a = -epsilon - t;
    const b = epsilon - t;
    if (a < -10 && b > 10) return [0, 0];

    const denom = normCdf(b) - normCdf(a);
    let v = (normPdf(a) - normPdf(b)) / denom;
    let variance =
      1 + (a * normPdf(a) - b * normPdf(b)) / denom - v * v;

    if (!Number.isFinite(v) || !Number.isFinite(variance) || variance < 0) {
      // Tail fallback matching scipy.stats.truncnorm's limiting moments.
      const reflected = b < 0;
      const lower = reflected ? -b : a;
      const upper = reflected ? -a : b;
      if (lower <= 0) return [0, 0];
      const width = upper - lower;
      const scaledWidth = lower * width;
      if (scaledWidth > 50) {
        const [tailV, tailW] = winCorrections(-lower, 0);
        v = tailV;
        variance = 1 - tailW;
      } else if (scaledWidth < 1e-4) {
        const meanOffset = width * (0.5 - scaledWidth / 12);
        variance = (width * width) / 12;
        v = lower + meanOffset;
      } else {
        const denominator = Math.expm1(scaledWidth);
        const meanOffset = 1 / lower - width / denominator;
        variance =
          1 / (lower * lower) -
          (width * width * Math.exp(scaledWidth)) / (denominator * denominator);
        v = lower + meanOffset;
      }
      if (reflected) v = -v;
    }
    return [v, Math.min(Math.max(1 - variance, 0), 1)];
  };

  type PairUpdate = [number, number, number, number];
  const updateDecisive = (
    mu1: number,
    s1: number,
    mu2: number,
    s2: number,
    player1Wins: boolean,
  ): PairUpdate => {
    const c = Math.sqrt(2 * beta ** 2 + s1 ** 2 + s2 ** 2);
    const epsilon = drawMargin / c;
    let t: number;
    let v: number;
    let w: number;
    let mu1New: number;
    let mu2New: number;
    if (player1Wins) {
      t = (mu1 - mu2) / c;
      [v, w] = winCorrections(t, epsilon);
      mu1New = mu1 + (s1 ** 2 / c) * v;
      mu2New = mu2 - (s2 ** 2 / c) * v;
    } else {
      t = (mu2 - mu1) / c;
      [v, w] = winCorrections(t, epsilon);
      mu2New = mu2 + (s2 ** 2 / c) * v;
      mu1New = mu1 - (s1 ** 2 / c) * v;
    }
    const sigma1New = s1 * Math.sqrt(Math.max(1 - (s1 ** 2 / c ** 2) * w, 1e-12));
    const sigma2New = s2 * Math.sqrt(Math.max(1 - (s2 ** 2 / c ** 2) * w, 1e-12));
    return [mu1New, sigma1New, mu2New, sigma2New];
  };

  const updateDraw = (mu1: number, s1: number, mu2: number, s2: number): PairUpdate => {
    const c = Math.sqrt(2 * beta ** 2 + s1 ** 2 + s2 ** 2);
    const epsilon = drawMargin / c;
    const t = (mu1 - mu2) / c;
    const [v, w] = drawCorrections(t, epsilon);
    const mu1New = mu1 + (s1 ** 2 / c) * v;
    const mu2New = mu2 - (s2 ** 2 / c) * v;
    const sigma1New = s1 * Math.sqrt(Math.max(1 - (s1 ** 2 / c ** 2) * w, 1e-12));
    const sigma2New = s2 * Math.sqrt(Math.max(1 - (s2 ** 2 / c ** 2) * w, 1e-12));
    return [mu1New, sigma1New, mu2New, sigma2New];
  };

  for (let t = 0; t < N; t++) {
    for (let q = 0; q < M; q++) {
      const eventMu = mu.slice();
      const eventSigma = sigma.slice();
      const priorVariance = eventSigma.map((s) => s * s);
      const precisionIncrement = new Array<number>(L).fill(0);
      const naturalIncrement = new Array<number>(L).fill(0);

      for (let i = 0; i < L; i++) {
        for (let j = i + 1; j < L; j++) {
          const ri = tensor[i]![q]![t]!;
          const rj = tensor[j]![q]![t]!;
          let pair: PairUpdate;
          if (ri === rj) {
            if (tiePolicy === "skip") continue;
            if (tiePolicy === "correct_draw_only" && ri === 0) continue;
            pair = updateDraw(
              eventMu[i]!,
              eventSigma[i]!,
              eventMu[j]!,
              eventSigma[j]!,
            );
          } else {
            pair = updateDecisive(
              eventMu[i]!,
              eventSigma[i]!,
              eventMu[j]!,
              eventSigma[j]!,
              ri > rj,
            );
          }

          const players: [number, number, number][] = [
            [i, pair[0], pair[1]],
            [j, pair[2], pair[3]],
          ];
          for (const [player, pairMu, pairSigma] of players) {
            const pairVariance = pairSigma * pairSigma;
            precisionIncrement[player]! +=
              1 / pairVariance - 1 / priorVariance[player]!;
            naturalIncrement[player]! +=
              pairMu / pairVariance - eventMu[player]! / priorVariance[player]!;
          }
        }
      }

      for (let player = 0; player < L; player++) {
        const posteriorPrecision =
          1 / priorVariance[player]! + precisionIncrement[player]!;
        const posteriorNatural =
          eventMu[player]! / priorVariance[player]! + naturalIncrement[player]!;
        sigma[player] = Math.sqrt(1 / posteriorPrecision);
        mu[player] = posteriorNatural / posteriorPrecision;
      }
    }
    sigma = sigma.map((s) => Math.sqrt(s ** 2 + tau ** 2));
  }
  return { ranking: rankScores(mu, method), scores: mu };
}

/** Options for {@link glicko}. */
export interface GlickoOptions extends BaseRankOptions {
  initialRating?: number;
  initialRd?: number;
  c?: number;
  rdMax?: number;
  tieHandling?: PairwiseTieHandling;
  /** Include final rating deviations in the result. Default `false`. */
  returnDeviation?: boolean;
}

/** Glicko result; `deviation` is present when `returnDeviation` is true. */
export interface GlickoResult extends RankResult {
  deviation?: number[];
}

/** Rank models with Glicko rating and rating-deviation updates. */
export function glicko(R: TensorInput, options: GlickoOptions = {}): GlickoResult {
  const method = defaultIfUndefined(options.method, "competition");
  const tensor = validateInput(R);
  const [L, M, N] = shape3(tensor);

  // Match Python's conversion/validation order exactly.
  const initialRating = pythonFloat(
    defaultIfUndefined(options.initialRating, 1500),
    "initial_rating must be finite.",
  );
  const initialRd = pythonFloat(
    defaultIfUndefined(options.initialRd, 350),
    "initial_rd must be > 0 and finite.",
  );
  if (!Number.isFinite(initialRating)) throw new Error("initial_rating must be finite.");
  if (!Number.isFinite(initialRd) || initialRd <= 0)
    throw new Error("initial_rd must be > 0 and finite.");

  const rdMax = pythonFloat(
    defaultIfUndefined(options.rdMax, 350),
    "rd_max must be > 0 and finite.",
  );
  if (!Number.isFinite(rdMax) || rdMax <= 0)
    throw new Error("rd_max must be > 0 and finite.");

  const c = pythonFloat(
    defaultIfUndefined(options.c, 0),
    "c must be >= 0 and finite.",
  );
  if (!Number.isFinite(c) || c < 0) throw new Error("c must be >= 0 and finite.");
  const tiePolicy = validateTieHandling(
    defaultIfUndefined(options.tieHandling, "correct_draw_only"),
  );

  let rating = new Array<number>(L).fill(initialRating);
  let rd = new Array<number>(L).fill(Math.min(initialRd, rdMax));

  const q = Math.log(10) / 400;
  const g = (rdOpp: number): number =>
    1 / Math.sqrt(1 + (3 * q ** 2 * rdOpp ** 2) / Math.PI ** 2);
  const expected = (ri: number, rj: number, gj: number): number =>
    1 / (1 + 10 ** (-(gj * (ri - rj)) / 400));

  for (let t = 0; t < N; t++) {
    for (let m = 0; m < M; m++) {
      if (c > 0) rd = rd.map((v) => Math.min(Math.sqrt(v ** 2 + c ** 2), rdMax));

      const opponents: number[][] = Array.from({ length: L }, () => []);
      const results: number[][] = Array.from({ length: L }, () => []);
      for (let i = 0; i < L; i++) {
        for (let j = i + 1; j < L; j++) {
          const ri = tensor[i]![m]![t]!;
          const rj = tensor[j]![m]![t]!;
          let si: number;
          let sj: number;
          if (ri === rj) {
            if (tiePolicy === "skip") continue;
            if (tiePolicy === "draw") {
              si = 0.5;
              sj = 0.5;
            } else {
              if (ri === 1) {
                si = 0.5;
                sj = 0.5;
              } else continue;
            }
          } else if (ri > rj) {
            si = 1;
            sj = 0;
          } else {
            si = 0;
            sj = 1;
          }
          opponents[i]!.push(j);
          results[i]!.push(si);
          opponents[j]!.push(i);
          results[j]!.push(sj);
        }
      }

      const newRating = rating.slice();
      const newRd = rd.slice();
      for (let i = 0; i < L; i++) {
        if (opponents[i]!.length === 0) continue;
        let denom = 0;
        let delta = 0;
        for (let o = 0; o < opponents[i]!.length; o++) {
          const opp = opponents[i]![o]!;
          const s = results[i]![o]!;
          const gOpp = g(rd[opp]!);
          const E = expected(rating[i]!, rating[opp]!, gOpp);
          denom += gOpp ** 2 * E * (1 - E);
          delta += gOpp * (s - E);
        }
        if (denom <= 0 || !Number.isFinite(denom)) continue;
        const d2 = 1 / (q ** 2 * denom);
        const invVar = 1 / rd[i]! ** 2 + 1 / d2;
        if (invVar <= 0 || !Number.isFinite(invVar)) continue;
        let rdNew = Math.sqrt(1 / invVar);
        rdNew = Math.min(Math.max(rdNew, 1e-12), rdMax);
        newRating[i] = rating[i]! + (q / invVar) * delta;
        newRd[i] = rdNew;
      }
      rating = newRating;
      rd = newRd;
    }
  }
  const result: GlickoResult = { ranking: rankScores(rating, method), scores: rating };
  if (pythonTruthy(options.returnDeviation)) result.deviation = rd;
  return result;
}
