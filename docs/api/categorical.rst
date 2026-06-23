scorio.categorical
==================

Signal-based rubric evaluation pipeline. Loads per-completion JSONL signal
files, computes corpus-level thresholds, applies rubric schemas to classify
completions, and evaluates models with ``scorio.eval.bayes``.

Install the optional dependency with ``pip install scorio[categorical]``.

.. code-block:: python

   from scorio.categorical import evaluate_all

   # df is a pandas DataFrame from load_records() or built synthetically
   results = evaluate_all(df)
   # {"2.1": {"model_A": (mu, sigma), ...}, ...}

.. currentmodule:: scorio.categorical

IO
--

.. autofunction:: load_records

.. autoclass:: FileResult
   :members:

Thresholds
----------

.. autoclass:: scorio.categorical.thresholds.Thresholds
   :members:

.. autodata:: scorio.categorical.thresholds.SIGNAL_TO_COLUMN

.. autodata:: scorio.categorical.thresholds.BINARY_SIGNALS

Evaluation
----------

.. autofunction:: evaluate_schema

.. autofunction:: evaluate_all
