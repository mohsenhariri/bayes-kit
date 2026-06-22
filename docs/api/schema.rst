scorio.schema
=============

Signal-based rubric evaluation pipeline. Loads per-completion JSONL signal
files, computes corpus-level thresholds, applies rubric schemas to classify
completions, and evaluates models with ``scorio.eval.bayes``.

Install the optional dependency with ``pip install scorio[schema]``.

.. code-block:: python

   from scorio.schema import evaluate_all

   # df is a pandas DataFrame from load_records() or built synthetically
   results = evaluate_all(df)
   # {"2.1": {"model_A": (mu, sigma), ...}, ...}

.. currentmodule:: scorio.schema

IO
--

.. autofunction:: load_records

.. autoclass:: FileResult
   :members:

Thresholds
----------

.. autoclass:: scorio.schema.thresholds.Thresholds
   :members:

.. autodata:: scorio.schema.thresholds.SIGNAL_TO_COLUMN

.. autodata:: scorio.schema.thresholds.BINARY_SIGNALS

Evaluation
----------

.. autofunction:: evaluate_criterion

.. autofunction:: evaluate_all
