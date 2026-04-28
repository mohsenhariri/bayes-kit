Examples
========

This page provides detailed examples of using scorio for various evaluation scenarios.

Example 1: LLM Code Generation
-------------------------------

Evaluating a language model on code generation tasks:

.. code-block:: python

   import numpy as np
   from scorio import eval

   # 3 models tested on 10 coding problems each
   # 0 = fails, 1 = passes
   R = np.array([
       [1, 1, 0, 1, 1, 0, 1, 1, 1, 0],  # Model A: 70% correct
       [1, 0, 1, 1, 0, 1, 1, 1, 0, 1],  # Model B: 70% correct
       [1, 1, 1, 0, 1, 1, 1, 1, 1, 1],  # Model C: 90% correct
   ])

   w = np.array([0.0, 1.0])

   # Bayesian evaluation with uncertainty
   for i, model in enumerate(['Model A', 'Model B', 'Model C']):
       mu, sigma = eval.bayes(R[i:i+1], w)
       print(f"{model}: {mu:.3f} ± {sigma:.3f}")

   # Pass@5: probability at least 1 of 5 attempts succeeds
   for i, model in enumerate(['Model A', 'Model B', 'Model C']):
       pass_5 = eval.pass_at_k(R[i:i+1], k=5)
       print(f"{model} Pass@5: {pass_5:.3f}")

Example 2: Graded Responses
----------------------------

Evaluating with partial credit:

.. code-block:: python

   # Essay grading: 0=poor, 1=fair, 2=good, 3=excellent
   R = np.array([
       [1, 2, 2, 3, 1, 2, 3, 2],  # Grader 1
       [2, 2, 1, 3, 2, 2, 3, 2],  # Grader 2
   ])

   # Weight mapping: scale 0-3 to 0-1
   w = np.array([0.0, 0.33, 0.67, 1.0])

   mu, sigma = eval.bayes(R, w)
   print(f"Overall quality: {mu:.3f} ± {sigma:.3f}")

Example 3: Comparing Models
----------------------------

Statistical comparison of multiple models:

.. code-block:: python

   # 5 models, 20 test cases each
   np.random.seed(42)
   R = np.array([
       np.random.binomial(1, 0.65, 20),  # Model 1: ~65% accuracy
       np.random.binomial(1, 0.70, 20),  # Model 2: ~70% accuracy
       np.random.binomial(1, 0.75, 20),  # Model 3: ~75% accuracy
       np.random.binomial(1, 0.80, 20),  # Model 4: ~80% accuracy
       np.random.binomial(1, 0.85, 20),  # Model 5: ~85% accuracy
   ])

   w = np.array([0.0, 1.0])

   results = []
   for i in range(5):
       mu, sigma = eval.bayes(R[i:i+1], w)
       results.append((f"Model {i+1}", mu, sigma))

   # Display results
   print("Model Comparison:")
   print("-" * 40)
   for name, mu, sigma in results:
       lower = mu - 1.96 * sigma  # 95% CI
       upper = mu + 1.96 * sigma
       print(f"{name}: {mu:.3f} [{lower:.3f}, {upper:.3f}]")

Example 4: Using Prior Information
-----------------------------------

Incorporating historical performance:

.. code-block:: python

   # New test results
   R = np.array([[1, 0, 1, 1, 0]])

   w = np.array([0.0, 1.0])

   # Without prior
   mu_no_prior, sigma_no_prior = eval.bayes(R, w)
   print(f"No prior: {mu_no_prior:.3f} ± {sigma_no_prior:.3f}")

   # With prior from 10 previous tests
   R0 = np.array([[1, 1, 0, 1, 1, 1, 0, 1, 1, 1]])

   mu_prior, sigma_prior = eval.bayes(R, w, R0)
   print(f"With prior: {mu_prior:.3f} ± {sigma_prior:.3f}")
   print(f"Uncertainty reduction: {(1 - sigma_prior/sigma_no_prior)*100:.1f}%")

Example 5: Pass@k Analysis
---------------------------

Analyzing pass rates with different k values:

.. code-block:: python

   R = np.array([[0, 1, 1, 0, 1, 1, 0, 1, 1, 0]])

   print("Pass@k analysis:")
   for k in [1, 2, 3, 5, 10]:
       pass_k = eval.pass_at_k(R, k)
       print(f"Pass@{k}: {pass_k:.4f}")

Example 6: mG-Pass@k
----------------------------------------

Using mG-Pass@k for consistency measurement:

.. code-block:: python

   # Model with inconsistent performance
   R_inconsistent = np.array([[1, 0, 1, 0, 1, 0, 1, 0]])

   # Model with consistent performance
   R_consistent = np.array([[1, 1, 1, 1, 0, 0, 0, 0]])

   for k in [2, 4, 6]:
       mg_incon = eval.mg_pass_at_k(R_inconsistent, k)
       mg_con = eval.mg_pass_at_k(R_consistent, k)
       print(f"k={k}: inconsistent={mg_incon:.3f}, consistent={mg_con:.3f}")

Example 7: Complete Workflow
-----------------------------

Full evaluation pipeline:

.. code-block:: python

   import numpy as np
   from scorio import eval

   # Generate test data
   np.random.seed(123)
   M, N = 3, 20
   R = np.random.randint(0, 2, size=(M, N))

   w = np.array([0.0, 1.0])

   print("=" * 50)
   print("EVALUATION REPORT")
   print("=" * 50)

   # 1. Simple metrics
   print("\n1. Simple Metrics:")
   for i in range(M):
       avg, avg_sigma = eval.avg(R[i:i+1])
       print(f"   Question {i+1} average: {avg:.3f} ± {avg_sigma:.3f}")

   # 2. Bayesian evaluation
   print("\n2. Bayesian Evaluation:")
   for i in range(M):
       mu, sigma = eval.bayes(R[i:i+1], w)
       print(f"   Question {i+1}: μ={mu:.3f}, σ={sigma:.3f}")

   # 3. Pass@k metrics
   print("\n3. Pass@k Metrics (k=5):")
   for i in range(M):
       pass_k = eval.pass_at_k(R[i:i+1], k=5)
       pass_hat = eval.pass_hat_k(R[i:i+1], k=5)
       print(f"   Question {i+1}: Pass@5={pass_k:.3f}, Pass^5={pass_hat:.3f}")

   # 4. Stability analysis
   print("\n4. Stability Analysis (mG-Pass@k, k=3):")
   for i in range(M):
       mg = eval.mg_pass_at_k(R[i:i+1], k=3)
       print(f"   Question {i+1}: {mg:.3f}")

   print("=" * 50)
