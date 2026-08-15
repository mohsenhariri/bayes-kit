Changelog
=========

All notable changes to this project will be documented in this file.

Unreleased
----------

Added
~~~~~

- **Aggregation subpackage** (``scorio.aggregate``): test-time-scaling
  answer aggregation across five categories: confidence signals from token
  log-probabilities, process-reward aggregation, offline selection/voting over a
  candidate pool of ``answers`` and per-candidate ``scores`` (higher is better),
  confidence-guided selection and stopping, and other online early-stopping
  rules. Selection rules return the selected answer by default and can
  optionally return a representative candidate index and, for score-aware
  rules, its score. Measuring accuracy is left to ``scorio.eval``. Signals and
  selection compose: many literature methods are a ``(signal, rule)`` pair
  (e.g. DeepConf offline voting is ``weighted_majority_vote`` fed
  ``deepconf_confidence``).

  - Confidence signals (``scorio.aggregate.confidence``), per-trace scalar from
    the chosen-token log-probabilities / top-\ ``k`` log-probabilities:
    ``mean_logprob()`` / ``sequence_logprob()`` / ``perplexity()`` (sample-and-rank
    and likelihood-weighted voting; Adiwardana et al., 2020; Wang et al., 2023),
    ``self_certainty()`` (KL-from-uniform of the next-token law; Kang et al.,
    2025), ``token_confidence()`` / ``deepconf_confidence()`` (DeepConf
    negative-mean-top-\ ``k`` confidence with average / tail / bottom-percentile /
    lowest-group reductions; Fu et al., 2025), ``token_entropy()`` /
    ``varentropy()`` (Malinin & Gales, 2021; entropix, 2024),
    ``max_softmax_probability()`` / ``logprob_margin()`` (Hendrycks & Gimpel, 2017;
    Scheffer et al., 2001), and ``picsar()`` (reasoning + answer log-likelihood
    selector; Leang et al., 2026).
  - Reward aggregation (``scorio.aggregate.prm``): ``prm_aggregate()`` reduces
    per-step process-reward scores to a per-trace reward via
    ``last`` / ``min`` / ``mean`` / ``prod`` / ``max`` (Lightman et al., 2023;
    Wang et al., 2024).
  - Reward-based selection: ``best_of_n()`` (Cobbe et al., 2021),
    ``majority_of_the_bests()`` / ``mob()`` (Rakhsha et al., 2025), and
    ``best_of_majority()`` (highest-reward answer among the frequently produced
    ones; Di et al., 2025).
  - Vote-based aggregation: ``majority_vote()`` (Wang et al., 2023),
    ``weighted_majority_vote()`` (Li et al., 2023),
    ``softmax_weighted_vote()`` (CISC temperature-softmax-weighted vote bridging
    majority vote and Best-of-N; Taubenfeld et al., 2025),
    ``rank_weighted_vote()`` (rank-invariant Borda vote; Kang et al., 2025),
    ``logit_weighted_vote()`` (threshold-shifted log-odds vote with negative
    votes for low-quality candidates; Kuang et al., 2025), and
    ``filtered_vote()`` (keep the top-scoring candidates, then vote; DeepConf;
    Fu et al., 2025; Cobbe et al., 2021).
  - Calibrated scalar-verifier aggregation
    (``scorio.aggregate.calibration``): ``fit_kde_vote_calibration()`` fits
    correct/incorrect KDEs and a binned final-answer correctness calibrator;
    ``kde_weighted_vote()`` applies the resulting non-parametric vote (Kuang et
    al., 2025). It consumes one scalar verification probability per response;
    step-level scores must use the same fixed reduction during calibration and
    inference.
  - Online early stopping (``scorio.aggregate.online``):
    ``adaptive_consistency_stop()``, its full observed-support Dirichlet variant
    ``adaptive_consistency_dirichlet_stop()``, and its finite-horizon,
    unseen-answer CRP comparator ``adaptive_consistency_crp_stop()`` (Aggarwal
    et al., 2023); ``esc_stop()`` (Li et al., 2024); and
    ``deepconf_stop_threshold()`` / ``deepconf_online_stop()`` (Fu et al.,
    2025).
  - Confidence-guided aggregation (``scorio.aggregate.cges``):
    ``cges_vote()`` selects a final answer and ``cges_stop()`` checks an online
    stopping threshold (Aghazadeh et al., 2026).

Changed
~~~~~~~

- **Python evaluation internals** now share validated NumPy sufficient
  statistics and stable finite/posterior count-score kernels. The documented
  functional API, signatures, and valid-input aggregation semantics are
  unchanged.
- Python evaluation inputs now reject fractional, non-finite, and non-integral
  values before conversion, and bounded credible intervals cannot invert when
  their mean lies outside the requested bounds. The JavaScript and Julia ports
  retain the 0.2.2 coercion and clipping behavior pending a synchronized port;
  their old edge-case compatibility tests no longer describe Python.

Fixed
~~~~~

- Large finite banks and high latent budgets no longer overflow binomial/Beta
  coefficient products in Pass, AUC, generalized-pass, Geom, or spectrum
  metrics and credible intervals.
- Max@k now preserves rare-event probabilities, posterior uncertainty under
  reward translations, and finite uncertainty for large finite reward scales.

Version 0.2.2 (2026-04-28)
--------------------------

Added
~~~~~

- **Geometric evaluation metrics**

  - ``geom_at_k()`` and ``geom_at_k_ci()`` for questionwise geometric blends
    of Pass@k and Unanimous@k.
  - ``geom_ds_at_k()`` and ``geom_ds_at_k_ci()`` for dataset-level
    Pass/Unanimous endpoint blends.
  - ``geo_spectrum_at_k()`` and ``geo_spectrum_at_k_ci()`` for configurable
    GeoSpectrum metrics with threshold-spectrum weights.
  - ``geo_spectrum_star_at_k()`` and ``geo_spectrum_star_at_k_ci()`` for the
    default upper-half GeoSpectrum operating point.
  - ``threshold_spectrum_at_k()`` for finite-bank threshold-spectrum summaries.
  - ``threshold_spectrum_at_k_ci()`` for the corresponding latent posterior
    target, including resampling budgets larger than the observed bank.

Version 0.1.0 (2025-12-15)
--------------------------

Initial release

Added
~~~~~

- **Pass@k metrics**
  
  - Standard ``pass_at_k()`` (at least one correct)
  - ``pass_hat_k()`` / ``g_pass_at_k()`` (all correct)
  - ``g_pass_at_k_tau()`` with threshold parameter
  - ``mg_pass_at_k()``
