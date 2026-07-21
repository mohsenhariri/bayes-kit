"""Scorio package for Bayesian evaluation and ranking of LLMs.

Modules
------------------
- ``scorio.eval`` provides scalar metrics such as Bayes@N, average metrics,
  and Pass-family metrics with uncertainty helpers.
- ``scorio.rank`` provides ranking methods based on evaluation metrics,
  pairwise models, voting, IRT, graph methods, and more.
- ``scorio.sinf`` provides sequential inference helpers for adaptive stopping
  and allocation workflows.
- ``scorio.aggregate`` (also available as ``scorio.agg``) provides
  test-time-scaling answer aggregation: confidence signals from token
  log-probabilities (self-certainty, DeepConf, entropy), process-reward
  aggregation, offline selection/voting rules (Best-of-N, majority vote,
  weighted vote, Majority-of-the-Bests), and online early-stopping rules.
- ``scorio.utils`` provides ranking utilities shared across modules.
- ``scorio.categorical`` provides a signal-based rubric evaluation pipeline that
  loads per-completion JSONL files, computes corpus-level thresholds, applies
  rubric schemas to classify completions, and evaluates models with Bayes@N.

"""

__version__ = "0.2.2"

from . import aggregate, eval, rank, sinf, utils

agg = aggregate

__all__ = ["agg", "aggregate", "eval", "rank", "sinf", "utils", "categorical"]
