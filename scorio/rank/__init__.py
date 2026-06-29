"""Ranking methods for comparing multiple models.

This module implements ranking estimators for binary response tensors
used in ``scorio`` evaluation workflows.

Notation
--------

All methods use the same core notation:

- :math:`R \\in \\{0,1\\}^{L \\times M \\times N}` is the response tensor.
- :math:`L` is the number of models.
- :math:`M` is the number of questions.
- :math:`N` is the number of trials per question.
- :math:`R_{lmn}=1` means model :math:`l` solves question :math:`m` on trial
  :math:`n`.

For single-trial input, ``(L, M)`` is promoted to ``(L, M, 1)``.

Longitudinal variants of :func:`scorio.rank.dynamic_irt`
(``variant="growth"`` and ``variant="state_space"``) interpret axis
:math:`N` as ordered measurement time rather than i.i.d. trials.

Every ranking method computes raw scores :math:`s_l` and then maps those
scores to ranks:

.. math::
    s_l = f_l(R; \\psi), \\qquad
    r = \\operatorname{rank\\_scores}(s)[\\text{method}]

By default, methods return ``r`` as a one-dimensional array of shape ``(L,)``.
With ``return_scores=True``, methods return ``(r, s)``.

Available Methods
-----------------

Evaluation metric-based: `avg`, `bayes`, `pass_at_k`, `pass_hat_k`,
`g_pass_at_k_tau`, `mg_pass_at_k`.

Paired-comparison probabilistic models: `bradley_terry`, `bradley_terry_map`,
`bradley_terry_davidson`, `bradley_terry_davidson_map`, `rao_kupper`, `rao_kupper_map`.

Accuracy-based: `inverse_difficulty`.

Pairwise rating systems: `elo`, `glicko`, `trueskill`.

Bayesian methods: `thompson`, `bayesian_mcmc`.

Voting methods: `borda`, `copeland`, `win_rate`, `minimax`, `schulze`,
`ranked_pairs`, `kemeny_young`, `nanson`, `baldwin`, `majority_judgment`.

Item Response Theory: `rasch`, `rasch_map`, `rasch_2pl`, `rasch_2pl_map`,
`rasch_3pl`, `rasch_3pl_map`, `rasch_mml`, `rasch_mml_credible`, `dynamic_irt`,
`mirt` (compensatory multidimensional IRT).

Graph-based: `pagerank`, `spectral`, `rank_centrality`, `alpharank`, `nash`.

Seriation-based: `serial_rank`.

Hodge-theoretic: `hodge_rank`.

Listwise and setwise choice models (Luce family): `plackett_luce`, `plackett_luce_map`,
`davidson_luce`, `davidson_luce_map`, `bradley_terry_luce`, `bradley_terry_luce_map`.

Examples
--------

>>> import numpy as np
>>> from scorio import rank
>>> # Generate sample data: 3 models, 5 questions, 4 trials
>>> R = np.random.randint(0, 2, size=(3, 5, 4))
>>> # Get rankings using different methods
>>> ranks_bayes = rank.bayes(R, w=np.array([0.0, 1.0]))
>>> ranks_pass_at_k = rank.pass_at_k(R, k=2)
>>> # Get raw scores when needed
>>> ranking, scores = rank.avg(R, return_scores=True)
"""

# Bayesian methods
from .bayesian import bayesian_mcmc, thompson

# Paired-comparison probabilistic models
from .bradley_terry import (
    bradley_terry,
    bradley_terry_davidson,
    bradley_terry_davidson_map,
    bradley_terry_map,
    rao_kupper,
    rao_kupper_map,
)

# Evaluation metric-based ranking methods
from .eval_ranking import (
    avg,
    bayes,
    g_pass_at_k_tau,
    mg_pass_at_k,
    pass_at_k,
    pass_hat_k,
)

# Graph-based methods
from .graph import alpharank, nash, pagerank, spectral

# Hodge-theoretic
from .hodge_rank import hodge_rank

# Item Response Theory
from .irt import (
    dynamic_irt,
    mirt,
    rasch,
    rasch_2pl,
    rasch_2pl_map,
    rasch_3pl,
    rasch_3pl_map,
    rasch_map,
    rasch_mml,
    rasch_mml_credible,
)

# Listwise and setwise choice models
from .listwise import (
    bradley_terry_luce,
    bradley_terry_luce_map,
    davidson_luce,
    davidson_luce_map,
    plackett_luce,
    plackett_luce_map,
)

# Pairwise rating systems
from .pairwise import elo, glicko, trueskill

# Pointwise methods
from .pointwise import inverse_difficulty

# Prior classes for MAP estimation
from .priors import (
    CauchyPrior,
    CustomPrior,
    EmpiricalPrior,
    GaussianPrior,
    LaplacePrior,
    Prior,
    UniformPrior,
)
from .rank_centrality import rank_centrality

# Seriation-based
from .serial_rank import serial_rank

# Voting methods
from .voting import (
    baldwin,
    borda,
    copeland,
    kemeny_young,
    majority_judgment,
    minimax,
    nanson,
    ranked_pairs,
    schulze,
    win_rate,
)

__all__ = [
    # Priors
    "Prior",
    "GaussianPrior",
    "LaplacePrior",
    "CauchyPrior",
    "UniformPrior",
    "CustomPrior",
    "EmpiricalPrior",
    # Eval-based
    "avg",
    "bayes",
    "pass_at_k",
    "pass_hat_k",
    "g_pass_at_k_tau",
    "mg_pass_at_k",
    # Accuracy
    "inverse_difficulty",
    # Pairwise
    "elo",
    "glicko",
    "trueskill",
    # Bradley-Terry
    "bradley_terry",
    "bradley_terry_map",
    "bradley_terry_davidson",
    "bradley_terry_davidson_map",
    # Rao-Kupper
    "rao_kupper",
    "rao_kupper_map",
    # Bayesian
    "thompson",
    "bayesian_mcmc",
    # Voting
    "borda",
    "copeland",
    "win_rate",
    "minimax",
    "schulze",
    "ranked_pairs",
    "kemeny_young",
    "nanson",
    "baldwin",
    "majority_judgment",
    # IRT
    "rasch",
    "rasch_map",
    "rasch_2pl",
    "rasch_2pl_map",
    "rasch_3pl",
    "rasch_3pl_map",
    "rasch_mml",
    "rasch_mml_credible",
    "dynamic_irt",
    "mirt",
    # Graph
    "pagerank",
    "spectral",
    "alpharank",
    "nash",
    "rank_centrality",
    # Seriation-based
    "serial_rank",
    # Hodge-theoretic
    "hodge_rank",
    # Plackett-Luce
    "plackett_luce",
    "plackett_luce_map",
    "davidson_luce",
    "davidson_luce_map",
    "bradley_terry_luce",
    "bradley_terry_luce_map",
]
