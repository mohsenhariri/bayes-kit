scorio.aggregate
================

.. currentmodule:: scorio.aggregate

Test-time-scaling aggregation. Import the public API with
``from scorio import aggregate`` or its short alias ``from scorio import agg``.
The two names refer to the same module, so ``scorio.agg.best_of_n(...)`` and
``scorio.aggregate.best_of_n(...)`` are equivalent. The methods fall into four
categories:

#. **Confidence signals** turn a trace's token log-probabilities / top-\ :math:`k`
   log-probabilities into a scalar confidence -- the ``scores`` the selection
   rules consume.
#. **Reward aggregation** reduces a process reward model's per-step scores to one
   per-trace reward.
#. **Offline selection** collapses a fixed pool of ``answers`` (+ optional
   ``scores``) into one predicted answer.
#. **Online early stopping** decides when to stop sampling traces or generating a
   trace.

Signals and selection **compose**: many literature methods are a
``(signal, selection rule)`` pair. For example, DeepConf offline voting is
``weighted_majority_vote`` fed :func:`deepconf_confidence`; self-certainty
Best-of-N is ``best_of_n`` fed :func:`self_certainty`; length-normalized
likelihood-weighted self-consistency is ``weighted_majority_vote`` fed
``exp(mean_logprob)``.

Shared conventions
------------------

Each **selection** rule collapses a pool of sampled candidates for one question
into a single predicted answer. The inputs are two aligned arrays:

- ``answers``: ``(N,)`` for one question or ``(M, N)`` for a batch of
  ``M`` questions with ``N`` candidates each. Labels are compared by equality,
  so any hashable value works (string, int, canonicalized expression). Entries
  that are ``None``, ``""`` or ``NaN`` are treated as unparsable and excluded
  from every rule.
- ``scores``: per-candidate reward-model / verifier / confidence scores of
  the same shape (higher is better). Required by every rule except
  :func:`majority_vote`, which uses only ``answers``.

By default, every offline selection rule returns the selected answer per
question: a scalar for ``(N,)`` input, or an ``(M,)`` object array for ``(M, N)``
input. Optional flags can append information about a representative candidate:

- Every selection rule supports ``return_index=True``, which also returns a
  representative candidate index
  (``-1`` when a question has no valid answer).
- Score-aware selection rules also support ``return_score=True``, which returns
  the score of that representative candidate (``NaN`` when a question has no
  valid answer). For :func:`best_of_n`, this is the maximizing (``argmax``)
  score. :func:`majority_vote` does not support ``return_score`` because it does
  not take scores.

When both flags are supported and requested, values are returned in the order
``(selected, index, score)``. Requesting only one flag returns
``(selected, index)`` or ``(selected, score)``.

These rules select answers but do not calculate accuracy. To measure the
accuracy of a rule across a benchmark, gather the correctness of the selected
candidates and pass it to :doc:`scorio.eval <eval>` (e.g. ``eval.avg`` /
``eval.bayes``) for an accuracy with a credible interval.

Confidence signals
------------------

Per-trace scalars computed from a trace's own token statistics -- the chosen-token
log-probabilities ``(T,)`` and the per-position top-\ :math:`k` log-probabilities
``(T, k)`` -- with no external verifier. Most are **confidences** (higher = more
confident). The *uncertainties* :func:`perplexity` and :func:`token_entropy` are
**lower = more confident** and must be reoriented for selectors that expect
higher scores. :func:`varentropy` is instead a dispersion diagnostic: neither
direction is generally more confident, so it should not be used alone as a
selection score without a task-specific interpretation. Choose the
transformation to match the selection rule:

- Order-based selectors, including :func:`best_of_n`,
  :func:`majority_of_the_bests`, :func:`rank_weighted_vote`, and
  :func:`filtered_vote` with ``weighted=False``, use only score ordering. Pass
  higher-oriented confidences directly; negate :func:`perplexity` or
  :func:`token_entropy`.
- Additive raw-weight voting, including :func:`weighted_majority_vote` with
  ``aggregate="sum"`` and :func:`filtered_vote` with ``weighted=True``, uses
  score magnitudes and signs. Supply non-negative weights such as
  ``exp(mean_logprob)``, ``exp(sequence_logprob)``, or ``1 / perplexity``. Do
  not use raw negative log-probabilities or ``-perplexity`` as vote weights.
- :func:`softmax_weighted_vote` applies its own softmax and accepts raw
  higher-is-better scores. :func:`logit_weighted_vote` with its default
  transform expects verifier probabilities strictly between zero and one.

.. autofunction:: mean_logprob

.. autofunction:: sequence_logprob

.. autofunction:: perplexity

.. autofunction:: self_certainty

.. autofunction:: token_confidence

.. autofunction:: deepconf_confidence

.. autofunction:: token_entropy

.. autofunction:: varentropy

.. autofunction:: max_softmax_probability

.. autofunction:: logprob_margin

.. autofunction:: picsar


Reward-model aggregation
------------------------

.. autofunction:: prm_aggregate


Reward-based selection
----------------------

.. autofunction:: best_of_n

.. autofunction:: majority_of_the_bests

``mob`` is an alias for ``majority_of_the_bests``.

.. autofunction:: best_of_majority


Vote-based aggregation
----------------------

Every voting rule groups candidates by answer equality and returns the winning
group's answer; they differ only in how each candidate's vote is weighted and
which candidates are allowed to vote. Several are strict generalizations of the
basic rules: ``softmax_weighted_vote`` and ``rank_weighted_vote`` each expose a
single knob that continuously bridges :func:`majority_vote` and
:func:`best_of_n`, and ``filtered_vote`` recovers :func:`majority_vote`,
:func:`weighted_majority_vote`, and :func:`best_of_n` at the limits of its
``keep`` argument.

.. autofunction:: majority_vote

.. autofunction:: weighted_majority_vote

.. autofunction:: softmax_weighted_vote

.. autofunction:: rank_weighted_vote

.. autofunction:: logit_weighted_vote

.. autofunction:: filtered_vote


Online early stopping
---------------------

These rules turn a fixed-budget aggregator into an adaptive one. The
sampling-level rules watch the stream of extracted answers and report when to
stop drawing traces; the trace-level DeepConf rules report the token at which a
trace's running confidence falls below a warmup-calibrated threshold.

.. autofunction:: adaptive_consistency_stop

.. autofunction:: esc_stop

.. autofunction:: deepconf_stop_threshold

.. autofunction:: deepconf_online_stop
