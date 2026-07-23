r"""Answer aggregation and selection for test-time scaling.

This subpackage covers the *sampling-and-aggregation* branch of test-time
scaling: sample :math:`N` candidate responses for a question, then decide what to
do with the pool. It sits upstream of :mod:`scorio.eval` (which scores one model)
and is the natural companion to :mod:`scorio.sinf.vote` (which decides *how many*
candidates to sample).

The methods fall into five categories:

1. **Confidence signals** (:mod:`~scorio.aggregate.confidence`) turn a trace's own
   token log-probabilities / top-:math:`k` log-probabilities into a scalar
   confidence -- the ``scores`` the selection rules consume, with no external
   verifier.
2. **Reward aggregation** (:mod:`~scorio.aggregate.prm`) reduces a process reward
   model's per-step scores to one per-trace reward.
3. **Offline selection and calibration** (:mod:`~scorio.aggregate.best_of`,
   :mod:`~scorio.aggregate.vote`, :mod:`~scorio.aggregate.calibration`) prepares
   fitted calibration state or collapses a fixed pool of ``answers`` (+ optional
   ``scores``) into one predicted answer.
4. **Confidence-guided aggregation** (:mod:`~scorio.aggregate.cges`) uses
   aligned confidence values for both answer selection and early stopping.
5. **Online early stopping** (:mod:`~scorio.aggregate.online`) decides when to
   stop sampling traces or stop generating a trace.

Signals + selection compose: many literature methods are a
(confidence/reward signal, selection rule) pair -- e.g. DeepConf offline voting
is ``weighted_majority_vote`` fed :func:`deepconf_confidence`, and self-certainty
Best-of-N is ``best_of_n`` fed :func:`self_certainty`.

Calibrated voting methods consume one scalar verification score per complete
response. An ORM score can be used directly. A PRM's step-level scores require
an explicit trace reduction such as :func:`prm_aggregate`; the same reduction
must be used for calibration and inference.

Setting
-------
Each question contributes a candidate pool of two aligned arrays:

- :math:`Z \in \mathcal{A}^{M \times N}` -- **answers**, the extracted final
  answer of each candidate. Rows are compared by equality, so any hashable
  label works (a string, an int, a canonicalized expression). Unparsable
  entries (``None``, ``""``, ``NaN``) are ignored.
- :math:`S \in \mathbb{R}^{M \times N}` -- **scores**, a per-candidate reward /
  verifier / confidence value, **higher is better**. Scores attached to a valid
  answer are assumed finite; a ``NaN`` score is not treated as "missing" (only
  the *answer* marks a candidate invalid), so pre-clean the scores if a verifier
  can emit ``NaN``.

:math:`M` is the number of questions and :math:`N` the number of candidates per
question, matching the ``(M, N)`` layout used elsewhere in ``scorio``. A single
question may be passed as a 1-D ``(N,)`` array. Rules that do not need scores
(``majority_vote``) take only ``answers``.

Return contract
---------------
Every selection rule returns **only the selection**: the chosen answer per
question (a scalar for 1-D input, else an ``(M,)`` object array), or
``(selected, index)`` with ``return_index=True``, where ``index`` points at a
representative candidate (``-1`` when a question has no valid answer). Selection
rules do not estimate their own accuracy -- to evaluate one, look up the
correctness of the selected candidates and feed it to :mod:`scorio.eval` (e.g.
``eval.avg`` / ``eval.bayes``) for a point estimate and credible interval.

Methods
-------
Confidence signals (:mod:`~scorio.aggregate.confidence`), per-trace scalar from
token log-probabilities (higher = more confident, except the flagged
*uncertainties*):

- ``mean_logprob`` / ``sequence_logprob`` / ``perplexity``: sequence-likelihood
  confidences (Adiwardana et al., 2020; Wang et al., 2023).
- ``self_certainty``: KL-from-uniform of the next-token law (Kang et al., 2025).
- ``token_confidence`` / ``deepconf_confidence``: DeepConf negative-mean-top-k
  confidence and its trace reductions (Fu et al., 2025).
- ``token_entropy`` / ``varentropy``: top-k entropy and its variance (Malinin &
  Gales, 2021; entropix, 2024) -- *uncertainties*.
- ``max_softmax_probability`` / ``logprob_margin``: top-1 probability and
  top1-top2 margin (Hendrycks & Gimpel, 2017; Scheffer et al., 2001).
- ``picsar``: reasoning + answer log-likelihood selector (Leang et al., 2026).

Reward aggregation (:mod:`~scorio.aggregate.prm`):

- ``prm_aggregate``: reduce per-step PRM scores to a per-trace reward via
  ``last`` / ``min`` / ``mean`` / ``prod`` / ``max`` (Lightman et al., 2023;
  Wang et al., 2024).

Reward-based selection (:mod:`~scorio.aggregate.best_of`):

- ``best_of_n``: answer of the highest-scoring candidate (Cobbe et al., 2021).
- ``majority_of_the_bests`` (alias ``mob``): mode of the bootstrapped Best-of-N
  distribution (Rakhsha et al., 2025).
- ``best_of_majority``: highest-reward answer among the frequently-produced ones
  (Di et al., 2025).

Vote-based aggregation (:mod:`~scorio.aggregate.vote`):

- ``majority_vote``: most frequent answer / self-consistency (Wang et al., 2023).
- ``weighted_majority_vote``: answer maximizing summed/mean score
  (verifier-weighted voting; Li et al., 2023).
- ``softmax_weighted_vote``: temperature-softmax-weighted vote bridging majority
  vote and Best-of-N (CISC; Taubenfeld et al., 2025).
- ``rank_weighted_vote``: Borda / rank-weighted vote, invariant to monotone
  rescaling of the scores (self-certainty voting; Kang et al., 2025).
- ``logit_weighted_vote``: threshold-shifted log-odds weighted vote with
  negative votes for low-quality candidates (Kuang et al., 2025).
- ``filtered_vote``: keep the top-scoring candidates, then (weighted) vote
  (DeepConf; Fu et al., 2025; Cobbe et al., 2021).

Calibrated scalar-verifier aggregation
(:mod:`~scorio.aggregate.calibration`):

- ``fit_kde_vote_calibration``: fit correct/incorrect scalar-score KDEs and a
  binned final-answer correctness calibrator (Kuang et al., 2025).
- ``kde_weighted_vote``: combine the fitted scalar-score density ratio with an
  estimated response-pool reliability term (Kuang et al., 2025).

Confidence-guided aggregation (:mod:`~scorio.aggregate.cges`):

- ``cges_vote``: select the answer with the largest CGES score.
- ``cges_stop``: stop sampling when a CGES score reaches a threshold
  (Aghazadeh et al., 2026).

Online early stopping (:mod:`~scorio.aggregate.online`):

- ``adaptive_consistency_stop``: stop sampling when the top-two answer counts
  make the majority decided (Aggarwal et al., 2023).
- ``adaptive_consistency_dirichlet_stop``: use the full observed-answer
  Dirichlet posterior (Aggarwal et al., 2023).
- ``adaptive_consistency_crp_stop``: model unseen answers with the paper's
  finite-horizon CRP comparator (Aggarwal et al., 2023).
- ``esc_stop``: stop sampling when a window of samples fully agrees (Li et al.,
  2024).
- ``deepconf_stop_threshold`` / ``deepconf_online_stop``: warmup-calibrated
  confidence threshold and per-trace early-termination token (Fu et al., 2025).
"""

from .best_of import best_of_majority, best_of_n, majority_of_the_bests, mob
from .calibration import (
    KDEVoteCalibration,
    fit_kde_vote_calibration,
    kde_weighted_vote,
)
from .cges import CGES_OTHER, cges_stop, cges_vote
from .confidence import (
    deepconf_confidence,
    logprob_margin,
    max_softmax_probability,
    mean_logprob,
    perplexity,
    picsar,
    self_certainty,
    sequence_logprob,
    token_confidence,
    token_entropy,
    varentropy,
)
from .online import (
    adaptive_consistency_crp_stop,
    adaptive_consistency_dirichlet_stop,
    adaptive_consistency_stop,
    deepconf_online_stop,
    deepconf_stop_threshold,
    esc_stop,
)
from .prm import prm_aggregate
from .vote import (
    filtered_vote,
    logit_weighted_vote,
    majority_vote,
    rank_weighted_vote,
    softmax_weighted_vote,
    weighted_majority_vote,
)

__all__ = [
    # confidence signals
    "mean_logprob",
    "sequence_logprob",
    "perplexity",
    "self_certainty",
    "token_confidence",
    "deepconf_confidence",
    "token_entropy",
    "varentropy",
    "max_softmax_probability",
    "logprob_margin",
    "picsar",
    # reward aggregation
    "prm_aggregate",
    # reward-based selection
    "best_of_n",
    "majority_of_the_bests",
    "mob",
    "best_of_majority",
    # vote-based aggregation
    "majority_vote",
    "weighted_majority_vote",
    "softmax_weighted_vote",
    "rank_weighted_vote",
    "logit_weighted_vote",
    "filtered_vote",
    # calibrated scalar-verifier aggregation
    "KDEVoteCalibration",
    "fit_kde_vote_calibration",
    "kde_weighted_vote",
    # confidence-guided aggregation
    "CGES_OTHER",
    "cges_vote",
    "cges_stop",
    # online early stopping
    "adaptive_consistency_stop",
    "adaptive_consistency_dirichlet_stop",
    "adaptive_consistency_crp_stop",
    "esc_stop",
    "deepconf_stop_threshold",
    "deepconf_online_stop",
]
