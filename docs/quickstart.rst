Quick Start
===========

This guide will help you get started with scorio for performance evaluation.

Basic Concepts
--------------

Data Format
~~~~~~~~~~~

scorio works with outcome matrices:

- **R**: M × N integer matrix where:
  
  - M = number of questions being evaluated
  - N = number of trials/samples per question
  - Entries are categories in {0, ..., C}

- **w**: Weight vector of length C+1 mapping categories to scores

Binary Evaluation
-----------------

For binary outcomes (correct/incorrect):

.. code-block:: python

   import numpy as np
   from scorio import eval

   # 2 question, 5 trials each
   # 0 = incorrect, 1 = correct
   R = np.array([[0, 1, 1, 0, 1],
                 [1, 1, 0, 1, 1]])

   # Weight vector: 0→0.0, 1→1.0
   w = np.array([0.0, 1.0])

   # Bayesian evaluation
   mu, sigma = eval.bayes(R, w)
   print(f"Estimate: {mu:.4f} ± {sigma:.4f}")

Multi-Category Evaluation
--------------------------

For outcomes with multiple categories:

.. code-block:: python

   # 0 = incorrect, 1 = partial, 2 = correct
   R = np.array([[0, 1, 2, 2, 1],
                 [1, 1, 0, 2, 2]])

   # Weight vector for 3 categories
   w = np.array([0.0, 0.5, 1.0])

   mu, sigma = eval.bayes(R, w)
   print(f"Estimate: {mu:.4f} ± {sigma:.4f}")

Using Prior Knowledge
---------------------

Incorporate prior outcomes:

.. code-block:: python

   R = np.array([[0, 1, 2, 2, 1],
                 [1, 1, 0, 2, 2]])
   w = np.array([0.0, 0.5, 1.0])

   # Prior outcomes (2 trials per question)
   R0 = np.array([[0, 2],
                  [1, 2]])

   mu, sigma = eval.bayes(R, w, R0)
   print(f"With prior: {mu:.4f} ± {sigma:.4f}")

Pass@k Metrics
--------------

Standard Pass@k (at least one correct):

.. code-block:: python

   R = np.array([[0, 1, 1, 0, 1],
                 [1, 1, 0, 1, 1]])

   # Probability at least 1 of 2 samples is correct
   pass_2 = eval.pass_at_k(R, k=2)
   print(f"Pass@2: {pass_2:.4f}")

Pass^k (all correct):

.. code-block:: python

   # Probability all 2 samples are correct
   pass_hat_2 = eval.pass_hat_k(R, k=2)
   print(f"Pass^2: {pass_hat_2:.4f}")

Generalized Pass@k with threshold:

.. code-block:: python

   # Probability at least 50% of k samples are correct
   g_pass = eval.g_pass_at_k_tau(R, k=3, tau=0.5)
   print(f"G-Pass@3(τ=0.5): {g_pass:.4f}")

Simple Average
--------------

For basic accuracy:

.. code-block:: python

   R = np.array([[0, 1, 1, 0, 1],
                 [1, 1, 0, 1, 1]])

   avg, avg_sigma = eval.avg(R)
   print(f"Average: {avg:.4f} ± {avg_sigma:.4f}")

Next Steps
----------

- See :doc:`examples` for more detailed use cases
- Check :doc:`api/eval` for complete API documentation
- Read the paper: https://arxiv.org/abs/2510.04265
