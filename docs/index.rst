scorio Documentation
====================

.. image:: _static/scorio.svg
   :alt: Scorio
   :align: center
   :class: scorio-home-logo
   :width: 220px

**scorio** implements the Bayes@N framework for Bayesian performance evaluation with uncertainty quantification.

.. image:: https://img.shields.io/badge/arXiv-2510.04265-b31b1b.svg
   :target: https://arxiv.org/abs/2510.04265
   :alt: arXiv

.. image:: https://img.shields.io/pypi/v/scorio.svg
   :target: https://pypi.org/project/scorio/
   :alt: PyPI version

.. image:: https://img.shields.io/pypi/pyversions/scorio.svg
   :target: https://pypi.org/project/scorio/
   :alt: Python versions

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://github.com/mohsenhariri/scorio/blob/main/LICENSE
   :alt: License: MIT

Installation
------------

Install from PyPI:

.. code-block:: bash

   pip install scorio

Quick Start
-----------

.. code-block:: python

   import numpy as np
   from scorio import eval

   # Binary outcomes: M=2 questions, N=5 trials
   R = np.array([[0, 1, 1, 0, 1],
                 [1, 1, 0, 1, 1]])

   # Rubric weights for binary outcomes
   w = np.array([0.0, 1.0])

   # Bayesian evaluation
   mu, sigma = eval.bayes(R, w)
   print(f"μ = {mu:.4f}, σ = {sigma:.4f}")

   # Pass@k metrics
   pass_k = eval.pass_at_k(R, k=2)
   print(f"Pass@2 = {pass_k:.4f}")

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quickstart
   examples

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/eval
   api/rank
   api/utils
   api/categorical

.. toctree::
   :maxdepth: 1
   :caption: Additional Information

   citation
   contributing
   changelog

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
