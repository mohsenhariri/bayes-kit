Changelog
=========

All notable changes to this project will be documented in this file.

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
  - ``threshold_spectrum_at_k()`` and ``threshold_spectrum_at_k_ci()`` for
    finite-bank threshold-spectrum summaries.

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


