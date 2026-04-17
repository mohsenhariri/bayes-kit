r"""Evaluation metrics and uncertainty estimators.

This module implements evaluation methods for model response tensors
 (``R``) used in ``scorio`` evaluation workflows.

Notation
--------
Let :math:`R \in \{0,\ldots,C\}^{M \times N}` be an outcome matrix.

- :math:`M` is the number of questions.
- :math:`N` is the number of trials per question.
- :math:`R_{\alpha i}` is the outcome for question :math:`\alpha` on trial
  :math:`i`.

Binary metrics use :math:`R \in \{0,1\}^{M \times N}`.

Return Pattern
---------------------
Point estimators return a scalar score. Companion ``*_ci`` functions return
``(mu, sigma, lo, hi)``, where ``mu`` is the estimated score, ``sigma`` is the
posterior standard deviation under the method assumptions, and ``lo`` and
``hi`` define a normal-approximation credible interval.

Available Families
------------------
- Bayes family: ``bayes`` and ``bayes_ci``.
- Average family: ``avg`` and ``avg_ci``.
- Pass family: ``pass_at_k`` and ``pass_hat_k`` with their ``*_ci`` variants.
- AUC@K family: ``auc_at_k`` and ``auc_at_k_ci``.
- Majority family: ``maj_at_k`` and ``maj_at_k_ci``.
- Generalized pass family: ``g_pass_at_k``, ``g_pass_at_k_tau``,
  ``mg_pass_at_k``, and their ``*_ci`` variants.
- Geometric family: ``geom_at_k``/``geom_at_k_ci``,
  ``geo_spectrum_at_k``/``geo_spectrum_at_k_ci``,
  ``threshold_spectrum_at_k``/``threshold_spectrum_at_k_ci``,
  ``geo_spectrum_star_at_k``/``geo_spectrum_star_at_k_ci``,
  spectrum-weight helpers, ``geometric_pass_favoring_at_k``,
  ``geometric_unanimous_favoring_at_k``, and ``sqrt_pu_over_p_at_k``.
- Max-reward family: ``max_at_k`` and ``max_at_k_ci``.
"""

from .auc import auc_at_k, auc_at_k_ci
from .avg import avg, avg_ci
from .bayes import bayes, bayes_ci
from .geom import (
    geo_spectrum_at_k,
    geo_spectrum_at_k_ci,
    geo_spectrum_star_at_k,
    geo_spectrum_star_at_k_ci,
    geom_at_k,
    geom_at_k_ci,
    geometric_pass_favoring_at_k,
    geometric_unanimous_favoring_at_k,
    mg_spectrum_weights,
    sqrt_pu_over_p_at_k,
    threshold_spectrum_at_k,
    threshold_spectrum_at_k_ci,
    unanimous_spectrum_weights,
)
from .gpass import (
    g_pass_at_k,
    g_pass_at_k_ci,
    g_pass_at_k_tau,
    g_pass_at_k_tau_ci,
    mg_pass_at_k,
    mg_pass_at_k_ci,
)
from .maj import maj_at_k, maj_at_k_ci
from .max_reward import max_at_k, max_at_k_ci
from .pass_at_k import pass_at_k, pass_at_k_ci, pass_hat_k, pass_hat_k_ci

unanimous_at_k = pass_hat_k
unanimous_at_k_ci = pass_hat_k_ci

__all__ = [
    # Bayes@N
    "bayes",
    "bayes_ci",
    # Avg@N with Bayesian uncertainty
    "avg",
    "avg_ci",
    # Pass-family point metrics
    "pass_at_k",
    "pass_at_k_ci",
    # G-Pass-family point metrics
    "pass_hat_k",
    "pass_hat_k_ci",
    "unanimous_at_k",
    "unanimous_at_k_ci",
    "g_pass_at_k",
    "g_pass_at_k_ci",
    "g_pass_at_k_tau",
    "g_pass_at_k_tau_ci",
    "mg_pass_at_k",
    "mg_pass_at_k_ci",
    # Majority
    "maj_at_k",
    "maj_at_k_ci",
    # AUC@K family
    "auc_at_k",
    "auc_at_k_ci",
    # Max-reward family
    "max_at_k",
    "max_at_k_ci",
    # Geometric / spectrum combinations
    "threshold_spectrum_at_k",
    "threshold_spectrum_at_k_ci",
    "unanimous_spectrum_weights",
    "mg_spectrum_weights",
    "geom_at_k",
    "geom_at_k_ci",
    "geo_spectrum_at_k",
    "geo_spectrum_at_k_ci",
    "geo_spectrum_star_at_k",
    "geo_spectrum_star_at_k_ci",
    "geometric_pass_favoring_at_k",
    "geometric_unanimous_favoring_at_k",
    "sqrt_pu_over_p_at_k",
]
