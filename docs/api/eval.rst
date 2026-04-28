scorio.eval
===========

Evaluation metrics for outcome matrices. Import the public API with
``from scorio import eval`` and call functions as ``eval.bayes(...)``,
``eval.pass_at_k(...)``, and so on.

Shared conventions
------------------

``R`` is an :math:`M \times N` matrix, where rows are questions and columns
are sampled trials. Binary metrics require entries in :math:`\{0,1\}`.
Categorical metrics use entries in :math:`\{0,\ldots,C\}` together with a
weight or reward vector ``w`` of length :math:`C+1`.

Point estimators return a scalar score. Companion ``*_ci`` functions return
``(mu, sigma, lo, hi)``, where ``mu`` is the posterior mean or point estimate,
``sigma`` is the posterior standard deviation under the metric's uncertainty
model, and ``lo`` and ``hi`` define a normal-approximation credible interval.

.. currentmodule:: scorio.eval

Bayes@N
-------

.. autofunction:: bayes

.. autofunction:: bayes_ci


avg@N
-----

.. autofunction:: avg

.. autofunction:: avg_ci


Pass@k
------

``unanimous_at_k`` and ``unanimous_at_k_ci`` are aliases for
``pass_hat_k`` and ``pass_hat_k_ci``.

.. autofunction:: pass_at_k

.. autofunction:: pass_hat_k

.. autofunction:: pass_at_k_ci

.. autofunction:: pass_hat_k_ci


AUC@K
------

.. autofunction:: auc_at_k

.. autofunction:: auc_at_k_ci


Majority
--------

.. autofunction:: maj_at_k

.. autofunction:: maj_at_k_ci


Generalized Pass@k
------------------

.. autofunction:: g_pass_at_k

.. autofunction:: g_pass_at_k_tau

.. autofunction:: mg_pass_at_k

.. autofunction:: g_pass_at_k_ci

.. autofunction:: g_pass_at_k_tau_ci

.. autofunction:: mg_pass_at_k_ci

GeoSpectrum
------------------

.. autofunction:: geom_at_k

.. autofunction:: geom_at_k_ci

.. autofunction:: geom_ds_at_k

.. autofunction:: geom_ds_at_k_ci

.. autofunction:: geo_spectrum_at_k

.. autofunction:: geo_spectrum_at_k_ci

.. autofunction:: geo_spectrum_star_at_k

.. autofunction:: geo_spectrum_star_at_k_ci

.. autofunction:: threshold_spectrum_at_k

.. autofunction:: threshold_spectrum_at_k_ci


Max-Reward
----------

.. autofunction:: max_at_k

.. autofunction:: max_at_k_ci
