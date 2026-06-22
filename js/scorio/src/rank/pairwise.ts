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

function validateTieHandling(t: string): PairwiseTieHandling {
  if (t !== "skip" && t !== "draw" && t !== "correct_draw_only") {
    throw new Error('tie_handling must be one of: "skip", "draw", "correct_draw_only"');
  }
  return t;
}

/** Options for {@link elo}. */
export interface EloOptions extends BaseRankOptions {
  K?: number;
  initialRating?: number;
  tieHandling?: PairwiseTieHandling;
}

/** Rank models with sequential Elo updates on induced pairwise matches. */
export function elo(R: TensorInput, options: EloOptions = {}): RankResult {
  const method = options.method ?? "competition";
  const K = options.K ?? 32;
  const initialRating = options.initialRating ?? 1500;
  const tiePolicy = validateTieHandling(options.tieHandling ?? "correct_draw_only");
  if (!Number.isFinite(K) || K <= 0) {
    throw new Error(`K must be a positive finite scalar; got ${K}`);
  }
  if (!Number.isFinite(initialRating)) throw new Error("initial_rating must be finite.");

  const tensor = validateInput(R);
  const [L, M, N] = shape3(tensor);
  const ratings = new Array<number>(L).fill(initialRating);

  for (let t = 0; t < N; t++) {
    for (let q = 0; q < M; q++) {
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
          const Ri = ratings[i]!;
          const Rj = ratings[j]!;
          const Ei = 1 / (1 + 10 ** ((Rj - Ri) / 400));
          const Ej = 1 - Ei;
          ratings[i] = Ri + K * (Si - Ei);
          ratings[j] = Rj + K * (Sj - Ej);
        }
      }
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
  const method = options.method ?? "competition";
  const muInitial = options.muInitial ?? 25;
  const sigmaInitial = options.sigmaInitial ?? 25 / 3;
  const beta = options.beta ?? 25 / 6;
  const tau = options.tau ?? 25 / 300;
  const drawMargin = options.drawMargin ?? 0;
  if (!Number.isFinite(muInitial)) throw new Error("mu_initial must be finite.");
  if (!Number.isFinite(sigmaInitial) || sigmaInitial <= 0)
    throw new Error("sigma_initial must be a positive finite scalar.");
  if (!Number.isFinite(beta) || beta <= 0)
    throw new Error("beta must be a positive finite scalar.");
  if (!Number.isFinite(tau) || tau < 0)
    throw new Error("tau must be a nonnegative finite scalar.");
  if (!Number.isFinite(drawMargin) || drawMargin < 0)
    throw new Error("draw_margin must be a nonnegative finite scalar.");
  const tiePolicy = validateTieHandling(options.tieHandling ?? "skip");

  const tensor = validateInput(R);
  const [L, M, N] = shape3(tensor);
  const mu = new Array<number>(L).fill(muInitial);
  let sigma = new Array<number>(L).fill(sigmaInitial);

  const vWin = (t: number, epsilon: number): number => {
    const x = t - epsilon;
    const denom = normCdf(x);
    if (denom < 1e-12) return -x;
    return normPdf(x) / denom;
  };
  const wWin = (t: number, epsilon: number): number => {
    const v = vWin(t, epsilon);
    return v * (v + t - epsilon);
  };
  const vDraw = (t: number, epsilon: number): number => {
    const a = -epsilon - t;
    const b = epsilon - t;
    const denom = normCdf(b) - normCdf(a);
    if (denom < 1e-12) return 0;
    return (normPdf(a) - normPdf(b)) / denom;
  };
  const wDraw = (t: number, epsilon: number): number => {
    const a = -epsilon - t;
    const b = epsilon - t;
    const denom = normCdf(b) - normCdf(a);
    if (denom < 1e-12) return 1;
    const v = vDraw(t, epsilon);
    const term = (b * normPdf(b) - a * normPdf(a)) / denom;
    return v * v + term;
  };

  const updateDecisive = (i: number, j: number, player1Wins: boolean): void => {
    const mu1 = mu[i]!;
    const mu2 = mu[j]!;
    const s1 = sigma[i]!;
    const s2 = sigma[j]!;
    const c = Math.sqrt(2 * beta ** 2 + s1 ** 2 + s2 ** 2);
    const epsilon = drawMargin / c;
    let t: number;
    let v: number;
    let w: number;
    if (player1Wins) {
      t = (mu1 - mu2) / c;
      v = vWin(t, epsilon);
      w = wWin(t, epsilon);
      mu[i] = mu1 + (s1 ** 2 / c) * v;
      mu[j] = mu2 - (s2 ** 2 / c) * v;
    } else {
      t = (mu2 - mu1) / c;
      v = vWin(t, epsilon);
      w = wWin(t, epsilon);
      mu[j] = mu2 + (s2 ** 2 / c) * v;
      mu[i] = mu1 - (s1 ** 2 / c) * v;
    }
    sigma[i] = s1 * Math.sqrt(Math.max(1 - (s1 ** 2 / c ** 2) * w, 1e-12));
    sigma[j] = s2 * Math.sqrt(Math.max(1 - (s2 ** 2 / c ** 2) * w, 1e-12));
  };

  const updateDraw = (i: number, j: number): void => {
    const mu1 = mu[i]!;
    const mu2 = mu[j]!;
    const s1 = sigma[i]!;
    const s2 = sigma[j]!;
    const c = Math.sqrt(2 * beta ** 2 + s1 ** 2 + s2 ** 2);
    const epsilon = drawMargin / c;
    const t = (mu1 - mu2) / c;
    const v = vDraw(t, epsilon);
    const w = wDraw(t, epsilon);
    mu[i] = mu1 + (s1 ** 2 / c) * v;
    mu[j] = mu2 - (s2 ** 2 / c) * v;
    sigma[i] = s1 * Math.sqrt(Math.max(1 - (s1 ** 2 / c ** 2) * w, 1e-12));
    sigma[j] = s2 * Math.sqrt(Math.max(1 - (s2 ** 2 / c ** 2) * w, 1e-12));
  };

  for (let t = 0; t < N; t++) {
    for (let q = 0; q < M; q++) {
      for (let i = 0; i < L; i++) {
        for (let j = i + 1; j < L; j++) {
          const ri = tensor[i]![q]![t]!;
          const rj = tensor[j]![q]![t]!;
          if (ri === rj) {
            if (tiePolicy === "skip") continue;
            if (tiePolicy === "correct_draw_only" && ri === 0) continue;
            updateDraw(i, j);
            continue;
          }
          updateDecisive(i, j, ri > rj);
        }
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
}

/** Rank models with Glicko rating and rating-deviation updates. */
export function glicko(R: TensorInput, options: GlickoOptions = {}): RankResult {
  const method = options.method ?? "competition";
  const initialRating = options.initialRating ?? 1500;
  const initialRd = options.initialRd ?? 350;
  const c = options.c ?? 0;
  const rdMax = options.rdMax ?? 350;
  if (!Number.isFinite(initialRating)) throw new Error("initial_rating must be finite.");
  if (!Number.isFinite(initialRd) || initialRd <= 0)
    throw new Error("initial_rd must be > 0 and finite.");
  if (rdMax <= 0) throw new Error("rd_max must be > 0");
  if (c < 0) throw new Error("c must be >= 0");
  const tiePolicy = validateTieHandling(options.tieHandling ?? "correct_draw_only");

  const tensor = validateInput(R);
  const [L, M, N] = shape3(tensor);
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
  return { ranking: rankScores(rating, method), scores: rating };
}
