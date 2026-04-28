"""Scorio package for Bayesian evaluation and ranking of LLMs.

Modules
------------------
- ``scorio.eval`` provides scalar metrics such as Bayes@N, average metrics,
  and Pass-family metrics with uncertainty helpers.
- ``scorio.rank`` provides ranking methods based on evaluation metrics,
  pairwise models, voting, IRT, graph methods, and more.
- ``scorio.sinf`` provides sequential inference helpers for adaptive stopping
  and allocation workflows.
- ``scorio.utils`` provides ranking utilities shared across modules.

"""

__version__ = "0.2.2"

from . import eval, rank, sinf, utils

__all__ = ["eval", "rank", "sinf", "utils"]
