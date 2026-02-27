"""
Item Response Theory (IRT) ranking methods.

This module estimates latent model abilities and question parameters under
binary IRT families.

Notation
--------

Let :math:`R \\in \\{0,1\\}^{L \\times M \\times N}` and
:math:`k_{lm}=\\sum_{n=1}^{N} R_{lmn}`.
Model abilities are :math:`\\theta_l`; item parameters include difficulty
:math:`b_m`, discrimination :math:`a_m`, and optional pseudo-guessing
:math:`c_m`.

A general binary IRT response model is

.. math::
    P(R_{lmn}=1 \\mid \\theta_l, a_m, b_m, c_m)
    = c_m + (1-c_m)\\sigma\\left(a_m(\\theta_l-b_m)\\right).

Special cases:

- 1PL (Rasch): :math:`a_m=1`, :math:`c_m=0`.
- 2PL: :math:`c_m=0`, free :math:`a_m` and :math:`b_m`.
- 3PL: free :math:`a_m`, :math:`b_m`, and :math:`c_m`.

Rankings are induced by ability scores :math:`s_l`, typically
:math:`s_l=\\hat\\theta_l` or a posterior summary of :math:`\\theta_l`.

The module includes maximum-likelihood and joint maximum-likelihood estimators,
MAP variants with configurable priors, and MML-EAP estimators.
"""

from typing import Literal, TypeAlias

import numpy as np
from scipy.optimize import minimize

from scorio.utils import rank_scores

from ._base import sigmoid, validate_input
from ._types import RankMethod, RankResult
from .priors import GaussianPrior, Prior

DynamicIrtVariant: TypeAlias = Literal["linear", "growth", "state_space"]
DynamicScoreTargetInput: TypeAlias = Literal[
    "initial",
    "final",
    "mean",
    "gain",
    "baseline",
    "start",
    "end",
    "average",
    "delta",
    "trend",
]


def _to_binomial_counts(R: np.ndarray) -> tuple[np.ndarray, int]:
    """
    Convert (L, M, N) Bernoulli trials into per-(model,item) binomial counts.

    Returns:
        k_correct: float array of shape (L, M) with counts in [0, n_trials]
        n_trials: int number of trials per (model, item)
    """
    R = validate_input(R)
    k_correct = R.sum(axis=2, dtype=float)
    n_trials = int(R.shape[2])
    return k_correct, n_trials


def _validate_positive_int(name: str, value: int, min_value: int = 1) -> int:
    """Validate a positive integer hyperparameter."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    ivalue = int(value)
    if ivalue < min_value:
        raise ValueError(f"{name} must be >= {min_value}, got {ivalue}")
    return ivalue


def _coerce_ability_prior(prior: Prior | float) -> Prior:
    """Normalize ability prior argument to a Prior instance."""
    if isinstance(prior, (int, float)):
        prior_var = float(prior)
        if not np.isfinite(prior_var) or prior_var <= 0.0:
            raise ValueError("prior variance must be a positive finite scalar.")
        return GaussianPrior(mean=0.0, var=prior_var)
    if isinstance(prior, Prior):
        return prior
    raise TypeError(
        f"prior must be a Prior object or float, got {type(prior).__name__}"
    )


def _validate_nonnegative_float(name: str, value: float) -> float:
    """Validate a finite non-negative scalar hyperparameter."""
    fvalue = float(value)
    if not np.isfinite(fvalue) or fvalue < 0.0:
        raise ValueError(f"{name} must be a finite scalar >= 0.0, got {value!r}")
    return fvalue


def _validate_guessing_upper(guessing_upper: float) -> float:
    """Validate 3PL guessing upper bound."""
    value = float(guessing_upper)
    if not np.isfinite(value) or not (0.0 < value < 1.0):
        raise ValueError("guessing_upper must be in (0, 1) and finite.")
    return value


def _validate_fix_guessing(
    fix_guessing: float | None, guessing_upper: float
) -> float | None:
    """Validate optional fixed 3PL guessing parameter."""
    if fix_guessing is None:
        return None
    value = float(fix_guessing)
    if not np.isfinite(value) or not (0.0 <= value <= guessing_upper):
        raise ValueError(
            f"fix_guessing must be in [0, guessing_upper={guessing_upper}] and finite."
        )
    return value


def _validate_time_points(
    time_points: np.ndarray | None, n_time: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Validate and normalize longitudinal measurement times.

    Returns:
        raw_time: user-facing time points (shape ``(n_time,)``)
        time_unit: normalized times in ``[0, 1]`` used for optimization
    """
    if time_points is None:
        raw_time = np.linspace(0.0, 1.0, n_time, dtype=float)
    else:
        raw_time = np.asarray(time_points, dtype=float)
        if raw_time.ndim != 1 or raw_time.shape[0] != n_time:
            raise ValueError(
                "time_points must be a 1D array with length equal to R.shape[2]."
            )
        if not np.all(np.isfinite(raw_time)):
            raise ValueError("time_points must contain only finite values.")
        if np.any(np.diff(raw_time) <= 0.0):
            raise ValueError("time_points must be strictly increasing.")

    if n_time < 2:
        return raw_time, np.zeros(n_time, dtype=float)

    span = float(raw_time[-1] - raw_time[0])
    if not np.isfinite(span) or span <= 0.0:
        raise ValueError("time_points must span a positive interval.")

    time_unit = (raw_time - raw_time[0]) / span
    return raw_time, time_unit


def _validate_dynamic_score_target(score_target: str) -> str:
    """Validate dynamic scoring target and normalize aliases."""
    target = str(score_target).strip().lower()
    aliases = {
        "baseline": "initial",
        "start": "initial",
        "end": "final",
        "average": "mean",
        "delta": "gain",
        "trend": "gain",
    }
    target = aliases.get(target, target)
    if target not in {"initial", "final", "mean", "gain"}:
        raise ValueError(
            "score_target must be one of "
            "{'initial', 'final', 'mean', 'gain'} "
            "(aliases: baseline, start, end, average, delta, trend)."
        )
    return target


def _score_dynamic_path(theta_path: np.ndarray, score_target: str) -> np.ndarray:
    """Convert a per-model ability trajectory into ranking scores."""
    target = _validate_dynamic_score_target(score_target)
    if target == "initial":
        return theta_path[:, 0]
    if target == "final":
        return theta_path[:, -1]
    if target == "mean":
        return theta_path.mean(axis=1)
    return theta_path[:, -1] - theta_path[:, 0]


def rasch(
    R: np.ndarray,
    method: RankMethod = "competition",
    return_scores: bool = False,
    max_iter: int = 500,
    return_item_params: bool = False,
) -> (
    np.ndarray
    | tuple[np.ndarray, np.ndarray]
    | tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]
):
    """
    Rank models with Rasch (1PL) IRT via joint MLE.

    Method context:
        Each model ``l`` has latent ability ``theta_l`` and each question ``m``
        has difficulty ``b_m``. We estimate both by maximizing the binomial
        likelihood over per-question correct counts.

    Args:
        R: Binary outcome tensor with shape ``(L, M, N)`` or matrix
            ``(L, M)`` (treated as ``N=1``).
        method: Tie-handling rule passed to :func:`scorio.utils.rank_scores`.
        return_scores: If ``True``, return ``(ranking, scores)``.
        max_iter: Positive maximum number of L-BFGS iterations.
        return_item_params: If True, also returns estimated item parameters
            (difficulty). Implies returning scores.

    Returns:
        Ranking array of shape ``(L,)``.
        If ``return_scores=True``, also returns ability scores ``theta``
        (shape ``(L,)``).
        If ``return_item_params=True``, also returns
        ``{"difficulty": b}`` (shape ``(M,)``).

    Notation:
        ``k_{lm} = sum_n R_{lmn}`` is the correct-count for model ``l`` and
        question ``m``.

    Formula:
        .. math::
            k_{lm} \\sim \\mathrm{Binomial}\\left(N,\\sigma(\\theta_l-b_m)\\right)

        .. math::
            b \\leftarrow b - \\frac{1}{M}\\sum_m b_m

    References:
        Rasch, G. (1960). Probabilistic Models for Some Intelligence and
        Attainment Tests.

    Examples:
        >>> import numpy as np
        >>> from scorio import rank
        >>> R = np.array([
        ...     [[1, 1], [1, 1]],
        ...     [[0, 0], [0, 0]],
        ... ])
        >>> ranks, scores = rank.rasch(R, return_scores=True)
        >>> ranks.tolist()
        [1, 2]
    """
    max_iter = _validate_positive_int("max_iter", max_iter)
    k_correct, n_trials = _to_binomial_counts(R)

    theta, beta = _estimate_rasch_abilities(k_correct, n_trials, max_iter=max_iter)
    scores = theta

    ranking = rank_scores(scores)[method]
    if return_item_params:
        return ranking, scores, {"difficulty": beta}
    return (ranking, scores) if return_scores else ranking


def rasch_map(
    R: np.ndarray,
    prior: Prior | float = 1.0,
    method: RankMethod = "competition",
    return_scores: bool = False,
    max_iter: int = 500,
    return_item_params: bool = False,
) -> (
    np.ndarray
    | tuple[np.ndarray, np.ndarray]
    | tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]
):
    """
    Rank models with Rasch (1PL) IRT via MAP estimation.

    Method context:
        Same likelihood as :func:`rasch`, with an additional prior penalty on
        abilities ``theta`` for shrinkage and numerical stability.

    Args:
        R: Binary outcome tensor with shape ``(L, M, N)`` or matrix
            ``(L, M)`` (treated as ``N=1``).
        prior: Ability prior. A ``float`` is interpreted as Gaussian prior
            variance; otherwise must be a ``Prior`` instance.
        method: Tie-handling rule passed to :func:`scorio.utils.rank_scores`.
        return_scores: If ``True``, return ``(ranking, scores)``.
        max_iter: Positive maximum number of L-BFGS iterations.
        return_item_params: If True, also returns estimated item parameters.
            Implies returning scores.

    Returns:
        Ranking array of shape ``(L,)``.
        If ``return_scores=True``, also returns MAP ability scores ``theta``.
        If ``return_item_params=True``, also returns
        ``{"difficulty": b}``.

    Formula:
        .. math::
            \\hat\\theta,\\hat b
            = \\arg\\min_{\\theta,b}
            \\left[
            -\\sum_{l,m}\\log p(k_{lm}\\mid\\theta_l,b_m)
            + \\mathrm{penalty}(\\theta)
            \\right]

    References:
        Mislevy, R. J. (1986). Bayes modal estimation in item response models.
        Psychometrika.

    Examples:
        >>> import numpy as np
        >>> from scorio import rank
        >>> R = np.array([
        ...     [[1, 1], [1, 1]],
        ...     [[0, 0], [0, 0]],
        ... ])
        >>> rank.rasch_map(R, prior=1.0).tolist()
        [1, 2]
    """
    max_iter = _validate_positive_int("max_iter", max_iter)
    k_correct, n_trials = _to_binomial_counts(R)
    prior = _coerce_ability_prior(prior)

    theta, beta = _estimate_rasch_abilities_map(
        k_correct, n_trials, prior, max_iter=max_iter
    )
    scores = theta

    ranking = rank_scores(scores)[method]
    if return_item_params:
        return ranking, scores, {"difficulty": beta}
    return (ranking, scores) if return_scores else ranking


def rasch_2pl(
    R: np.ndarray,
    method: RankMethod = "competition",
    return_scores: bool = False,
    max_iter: int = 500,
    return_item_params: bool = False,
    reg_discrimination: float = 0.01,
) -> (
    np.ndarray
    | tuple[np.ndarray, np.ndarray]
    | tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]
):
    """
    Rank models with 2PL IRT via joint (optionally regularized) JMLE.

    Method context:
        Extends Rasch with item discrimination ``a_m > 0``, so items can differ
        in how strongly they separate abilities. By default, a small L2 penalty
        is applied on ``log(a)`` for numerical stability.

    Args:
        R: Binary outcome tensor with shape ``(L, M, N)`` or matrix
            ``(L, M)`` (treated as ``N=1``).
        method: Tie-handling rule passed to :func:`scorio.utils.rank_scores`.
        return_scores: If ``True``, return ``(ranking, scores)``.
        max_iter: Positive maximum number of L-BFGS iterations.
        return_item_params: If True, also returns estimated item parameters
            (difficulty and discrimination). Implies returning scores.
        reg_discrimination: Non-negative L2 penalty weight on ``log(a)``.
            Set to ``0.0`` for pure (unpenalized) JMLE.

    Returns:
        Ranking array of shape ``(L,)``.
        If ``return_scores=True``, also returns ability scores ``theta``.
        If ``return_item_params=True``, also returns
        ``{"difficulty": b, "discrimination": a}``.

    Formula:
        .. math::
            k_{lm} \\sim \\mathrm{Binomial}
            \\left(N,\\sigma\\left(a_m(\\theta_l-b_m)\\right)\\right)

    References:
        Birnbaum, A. (1968). Some Latent Trait Models and Their Use in
        Inferring an Examinee's Ability. In Statistical Theories of
        Mental Test Scores.

    Examples:
        >>> import numpy as np
        >>> from scorio import rank
        >>> R = np.array([
        ...     [[1, 1], [1, 1]],
        ...     [[0, 0], [0, 0]],
        ... ])
        >>> rank.rasch_2pl(R).tolist()
        [1, 2]
    """
    max_iter = _validate_positive_int("max_iter", max_iter)
    reg_discrimination = _validate_nonnegative_float(
        "reg_discrimination", reg_discrimination
    )
    k_correct, n_trials = _to_binomial_counts(R)

    theta, beta, a = _estimate_2pl_abilities(
        k_correct,
        n_trials,
        max_iter=max_iter,
        reg_discrimination=reg_discrimination,
    )
    scores = theta

    ranking = rank_scores(scores)[method]
    if return_item_params:
        return ranking, scores, {"difficulty": beta, "discrimination": a}
    return (ranking, scores) if return_scores else ranking


def rasch_2pl_map(
    R: np.ndarray,
    prior: Prior | float = 1.0,
    method: RankMethod = "competition",
    return_scores: bool = False,
    max_iter: int = 500,
    return_item_params: bool = False,
    reg_discrimination: float = 0.01,
) -> (
    np.ndarray
    | tuple[np.ndarray, np.ndarray]
    | tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]
):
    """
    Rank models with 2PL IRT via MAP estimation.

    Method context:
        Same 2PL likelihood as :func:`rasch_2pl`, with a prior penalty on model
        abilities ``theta`` and an optional L2 penalty on ``log(a)``.

    Args:
        R: Binary outcome tensor with shape ``(L, M, N)`` or matrix
            ``(L, M)`` (treated as ``N=1``).
        prior: Ability prior. A ``float`` is interpreted as Gaussian prior
            variance; otherwise must be a ``Prior`` instance.
        method: Tie-handling rule passed to :func:`scorio.utils.rank_scores`.
        return_scores: If ``True``, return ``(ranking, scores)``.
        max_iter: Positive maximum number of L-BFGS iterations.
        return_item_params: If True, also returns estimated item parameters.
            Implies returning scores.
        reg_discrimination: Non-negative L2 penalty weight on ``log(a)``.
            Set to ``0.0`` to remove item-discrimination regularization.

    Returns:
        Ranking array of shape ``(L,)``.
        If ``return_scores=True``, also returns MAP ability scores ``theta``.
        If ``return_item_params=True``, also returns
        ``{"difficulty": b, "discrimination": a}``.

    Examples:
        >>> import numpy as np
        >>> from scorio import rank
        >>> R = np.array([
        ...     [[1, 1], [1, 1]],
        ...     [[0, 0], [0, 0]],
        ... ])
        >>> rank.rasch_2pl_map(R, prior=1.0).tolist()
        [1, 2]
    """
    max_iter = _validate_positive_int("max_iter", max_iter)
    reg_discrimination = _validate_nonnegative_float(
        "reg_discrimination", reg_discrimination
    )
    k_correct, n_trials = _to_binomial_counts(R)
    prior = _coerce_ability_prior(prior)

    theta, beta, a = _estimate_2pl_abilities_map(
        k_correct,
        n_trials,
        prior,
        max_iter=max_iter,
        reg_discrimination=reg_discrimination,
    )
    scores = theta

    ranking = rank_scores(scores)[method]
    if return_item_params:
        return ranking, scores, {"difficulty": beta, "discrimination": a}
    return (ranking, scores) if return_scores else ranking


def dynamic_irt(
    R: np.ndarray,
    variant: DynamicIrtVariant = "linear",
    method: RankMethod = "competition",
    return_scores: bool = False,
    max_iter: int = 500,
    return_item_params: bool = False,
    time_points: np.ndarray | None = None,
    score_target: DynamicScoreTargetInput = "final",
    slope_reg: float = 0.01,
    state_reg: float = 1.0,
    assume_time_axis: bool = False,
) -> (
    np.ndarray
    | tuple[np.ndarray, np.ndarray]
    | tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]
):
    """
    Rank models with dynamic (longitudinal) IRT variants.

    Method context:
        ``variant="linear"`` is a static Rasch baseline over aggregated counts.
        ``variant="growth"`` fits a longitudinal logistic growth model
        with per-model baseline ``theta0_l`` and slope ``theta1_l``:

        .. math::
            \\theta_{ln}=\\theta_{0,l}+\\theta_{1,l}t_n

        ``variant="state_space"`` fits a dynamic Rasch trajectory
        :math:`\\theta_{ln}` with random-walk smoothness regularization:

        .. math::
            P(R_{lmn}=1)=\\sigma\\left(\\theta_{ln}-b_m\\right)

        .. math::
            \\mathrm{penalty}=\\lambda\\sum_{l,n>0}
            \\frac{\\left(\\theta_{ln}-\\theta_{l,n-1}\\right)^2}{t_n-t_{n-1}}

    Args:
        R: Binary outcome tensor with shape ``(L, M, N)`` or matrix
            ``(L, M)`` (treated as ``N=1``).
        variant: ``"linear"``, ``"growth"``, or ``"state_space"``.
        method: Tie-handling rule passed to :func:`scorio.utils.rank_scores`.
        return_scores: If ``True``, return ``(ranking, scores)``.
        max_iter: Positive maximum number of L-BFGS iterations.
        return_item_params: If True, also returns estimated item parameters.
            Implies returning scores.
        time_points: Optional ordered measurement times of length ``N``.
            If ``None``, uses equally spaced times in ``[0, 1]``.
            Used only for longitudinal variants.
        score_target: Longitudinal score extracted from ability paths for
            ranking in growth and state-space variants. One of
            ``{"initial", "final", "mean", "gain"}``.
        slope_reg: Non-negative L2 regularization weight on growth slopes.
            Used only for ``variant="growth"``.
        state_reg: Non-negative random-walk smoothness penalty in
            ``variant="state_space"``.
        assume_time_axis: Safety switch for longitudinal variants.
            Set ``True`` to acknowledge that axis-2 of ``R`` is ordered time,
            not i.i.d. sampling trials.

    Returns:
        Ranking array of shape ``(L,)``.
        If ``return_scores=True``, also returns scores:
        ``theta`` for ``linear`` and dynamic target scores for
        ``growth``/``state_space``.
        If ``return_item_params=True``, also returns
        ``{"difficulty": b}`` (linear),
        or for longitudinal variants:
        ``{"difficulty": b, "ability_path": theta_path, ...}``.

    Formula:
        .. math::
            P(R_{lmn}=1)
            = \\sigma\\left(\\theta_{0,l} + \\theta_{1,l} t_n - b_m\\right)

    References:
        Verhelst, N. D., & Glas, C. A. (1993). A dynamic generalization
        of the Rasch model. Psychometrika.

        Wang, C., & Nydick, S. W. (2020). On Longitudinal Item Response
        Theory Models: A Didactic. Journal of Educational and Behavioral
        Statistics.

    Examples:
        >>> import numpy as np
        >>> from scorio import rank
        >>> R = np.array([
        ...     [[1, 1, 1], [1, 1, 1]],
        ...     [[0, 0, 0], [0, 0, 0]],
        ... ])
        >>> rank.dynamic_irt(R, variant="linear").tolist()
        [1, 2]
    """
    max_iter = _validate_positive_int("max_iter", max_iter)
    variant_name = str(variant).strip().lower()
    R = validate_input(R)
    k_correct = R.sum(axis=2, dtype=float)
    n_trials = int(R.shape[2])
    score_target_name = _validate_dynamic_score_target(score_target)
    slope_reg = _validate_nonnegative_float("slope_reg", slope_reg)
    state_reg = _validate_nonnegative_float("state_reg", state_reg)

    if variant_name == "linear":
        if score_target_name != "final":
            raise ValueError(
                "score_target is only used for longitudinal variants "
                "('growth' and 'state_space')."
            )
        theta, beta = _estimate_rasch_abilities(k_correct, n_trials, max_iter=max_iter)
        scores = theta

    elif variant_name == "growth":
        if not assume_time_axis:
            raise ValueError(
                "variant='growth' interprets axis-2 as ordered longitudinal time. "
                "Set assume_time_axis=True to proceed."
            )
        raw_time, time_unit = _validate_time_points(time_points, n_trials)
        theta0, theta1, beta = _estimate_growth_model_abilities(
            R,
            time_unit,
            max_iter=max_iter,
            slope_reg=slope_reg,
        )
        theta_path = theta0[:, None] + theta1[:, None] * time_unit[None, :]
        scores = _score_dynamic_path(theta_path, score_target_name)
    elif variant_name == "state_space":
        if not assume_time_axis:
            raise ValueError(
                "variant='state_space' interprets axis-2 as ordered longitudinal "
                "time. Set assume_time_axis=True to proceed."
            )
        raw_time, time_unit = _validate_time_points(time_points, n_trials)
        theta_path, beta = _estimate_state_space_abilities(
            R,
            time_unit,
            max_iter=max_iter,
            state_reg=state_reg,
        )
        scores = _score_dynamic_path(theta_path, score_target_name)
    else:
        raise ValueError(
            f"Unknown variant: {variant_name}. "
            "Use 'linear', 'growth', or 'state_space'."
        )

    ranking = rank_scores(scores)[method]
    if return_item_params:
        if variant_name == "linear":
            return ranking, scores, {"difficulty": beta}
        if variant_name == "growth":
            return (
                ranking,
                scores,
                {
                    "difficulty": beta,
                    "baseline": theta0,
                    "slope": theta1,
                    "ability_path": theta_path,
                    "time_points": raw_time,
                },
            )
        return (
            ranking,
            scores,
            {
                "difficulty": beta,
                "ability_path": theta_path,
                "time_points": raw_time,
                "gain": theta_path[:, -1] - theta_path[:, 0],
            },
        )
    return (ranking, scores) if return_scores else ranking


def _estimate_rasch_abilities(
    k_correct: np.ndarray, n_trials: int, max_iter: int = 500
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate Rasch abilities via JMLE.

    Args:
        k_correct: Shape (L, M) with counts in [0, n_trials].
        n_trials: Number of trials per (model, item).
    """
    L, M = k_correct.shape

    def negative_log_likelihood(params):
        theta = params[:L]
        beta = params[L:]
        beta = beta - beta.mean()  # Identifiability constraint

        # P(correct) = sigmoid(theta - beta)
        diff = theta[:, None] - beta[None, :]  # (L, M)
        prob = sigmoid(diff)
        prob = np.clip(prob, 1e-10, 1 - 1e-10)

        nll = -np.sum(
            k_correct * np.log(prob) + (n_trials - k_correct) * np.log(1 - prob)
        )
        return nll

    # Initialize from observed proportions
    p_lm = np.clip((k_correct + 0.5) / (n_trials + 1.0), 1e-6, 1 - 1e-6)
    model_scores = p_lm.mean(axis=1)
    question_difficulty = p_lm.mean(axis=0)

    theta_init = np.log(model_scores / (1 - model_scores))
    beta_init = -np.log(question_difficulty / (1 - question_difficulty))
    params_init = np.concatenate([theta_init, beta_init])

    result = minimize(
        negative_log_likelihood,
        params_init,
        method="L-BFGS-B",
        options={"maxiter": max_iter},
    )

    theta = result.x[:L]
    beta = result.x[L:]
    beta = beta - beta.mean()
    return theta, beta


def _estimate_rasch_abilities_map(
    k_correct: np.ndarray, n_trials: int, prior: Prior, max_iter: int = 500
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate Rasch abilities via MAP with configurable prior on abilities.

    Args:
        k_correct: Shape (L, M) with counts in [0, n_trials].
        n_trials: Number of trials per (model, item).
        prior: Prior distribution on ability parameters.
        max_iter: Maximum optimization iterations.
    """
    L, M = k_correct.shape

    def negative_log_posterior(params):
        theta = params[:L]
        beta = params[L:]
        beta = beta - beta.mean()  # Identifiability constraint

        # P(correct) = sigmoid(theta - beta)
        diff = theta[:, None] - beta[None, :]  # (L, M)
        prob = sigmoid(diff)
        prob = np.clip(prob, 1e-10, 1 - 1e-10)

        # Negative log-likelihood
        nll = -np.sum(
            k_correct * np.log(prob) + (n_trials - k_correct) * np.log(1 - prob)
        )

        # Prior penalty on abilities (negative log-prior)
        prior_penalty = prior.penalty(theta)

        return nll + prior_penalty

    # Initialize from observed proportions
    p_lm = np.clip((k_correct + 0.5) / (n_trials + 1.0), 1e-6, 1 - 1e-6)
    model_scores = p_lm.mean(axis=1)
    question_difficulty = p_lm.mean(axis=0)

    theta_init = np.log(model_scores / (1 - model_scores))
    beta_init = -np.log(question_difficulty / (1 - question_difficulty))
    params_init = np.concatenate([theta_init, beta_init])

    result = minimize(
        negative_log_posterior,
        params_init,
        method="L-BFGS-B",
        options={"maxiter": max_iter},
    )

    theta = result.x[:L]
    beta = result.x[L:]
    beta = beta - beta.mean()
    return theta, beta


def _estimate_2pl_abilities(
    k_correct: np.ndarray,
    n_trials: int,
    max_iter: int = 500,
    reg_discrimination: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate 2PL abilities via JMLE.
    """
    L, M = k_correct.shape

    def negative_log_likelihood(params):
        theta = params[:L]
        beta = params[L : L + M]
        log_a = params[L + M :]

        beta = beta - beta.mean()
        a = np.exp(np.clip(log_a, -3, 3))  # Discrimination in reasonable range

        # P(correct) = sigmoid(a * (theta - beta))
        diff = theta[:, None] - beta[None, :]
        logit = a[None, :] * diff
        prob = sigmoid(logit)
        prob = np.clip(prob, 1e-10, 1 - 1e-10)

        nll = -np.sum(
            k_correct * np.log(prob) + (n_trials - k_correct) * np.log(1 - prob)
        )

        # Optional regularization on discrimination.
        nll += reg_discrimination * np.sum(log_a**2)

        return nll

    # Initialize
    p_lm = np.clip((k_correct + 0.5) / (n_trials + 1.0), 1e-6, 1 - 1e-6)
    model_scores = p_lm.mean(axis=1)
    question_difficulty = p_lm.mean(axis=0)

    theta_init = np.log(model_scores / (1 - model_scores))
    beta_init = -np.log(question_difficulty / (1 - question_difficulty))
    log_a_init = np.zeros(M)  # Start with discrimination = 1
    params_init = np.concatenate([theta_init, beta_init, log_a_init])

    result = minimize(
        negative_log_likelihood,
        params_init,
        method="L-BFGS-B",
        options={"maxiter": max_iter},
    )

    theta = result.x[:L]
    beta = result.x[L : L + M]
    beta = beta - beta.mean()
    a = np.exp(np.clip(result.x[L + M :], -3, 3))
    return theta, beta, a


def _estimate_2pl_abilities_map(
    k_correct: np.ndarray,
    n_trials: int,
    prior: Prior,
    max_iter: int = 500,
    reg_discrimination: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate 2PL abilities via MAP with configurable prior on abilities.

    Args:
        k_correct: Shape (L, M) with counts in [0, n_trials].
        n_trials: Number of trials per (model, item).
        prior: Prior distribution on ability parameters.
        max_iter: Maximum optimization iterations.
    """
    L, M = k_correct.shape

    def negative_log_posterior(params):
        theta = params[:L]
        beta = params[L : L + M]
        log_a = params[L + M :]

        beta = beta - beta.mean()
        a = np.exp(np.clip(log_a, -3, 3))  # Discrimination in reasonable range

        # P(correct) = sigmoid(a * (theta - beta))
        diff = theta[:, None] - beta[None, :]
        logit = a[None, :] * diff
        prob = sigmoid(logit)
        prob = np.clip(prob, 1e-10, 1 - 1e-10)

        # Negative log-likelihood
        nll = -np.sum(
            k_correct * np.log(prob) + (n_trials - k_correct) * np.log(1 - prob)
        )

        # Optional regularization on discrimination.
        nll += reg_discrimination * np.sum(log_a**2)

        # Prior penalty on abilities (negative log-prior)
        prior_penalty = prior.penalty(theta)

        return nll + prior_penalty

    # Initialize
    p_lm = np.clip((k_correct + 0.5) / (n_trials + 1.0), 1e-6, 1 - 1e-6)
    model_scores = p_lm.mean(axis=1)
    question_difficulty = p_lm.mean(axis=0)

    theta_init = np.log(model_scores / (1 - model_scores))
    beta_init = -np.log(question_difficulty / (1 - question_difficulty))
    log_a_init = np.zeros(M)  # Start with discrimination = 1
    params_init = np.concatenate([theta_init, beta_init, log_a_init])

    result = minimize(
        negative_log_posterior,
        params_init,
        method="L-BFGS-B",
        options={"maxiter": max_iter},
    )

    theta = result.x[:L]
    beta = result.x[L : L + M]
    beta = beta - beta.mean()
    a = np.exp(np.clip(result.x[L + M :], -3, 3))
    return theta, beta, a


def _estimate_growth_model_abilities(
    R: np.ndarray,
    time_unit: np.ndarray,
    max_iter: int = 500,
    slope_reg: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate a longitudinal Rasch (1PL) model with per-model growth.

    We fit the logistic growth model:

        P(R[l,m,n]=1) = σ(θ0_l + θ1_l * t_n - b_m)

    where:
        - θ0_l is the baseline ability (trial n=0),
        - θ1_l is a per-model trend over trials,
        - b_m is item difficulty.

    This is a more faithful longitudinal IRT formulation than regressing mean
    accuracy over trials, because it:
        - respects the Bernoulli likelihood,
        - retains item difficulties,
        - keeps probabilities in (0, 1) via the logistic link.

    Args:
        R: Binary tensor of shape (L, M, N).
        time_unit: Normalized time points in ``[0, 1]`` with shape ``(N,)``.
        max_iter: Maximum iterations for optimization.
        slope_reg: Non-negative L2 penalty on growth slopes.

    Returns:
        Tuple of:
            - theta0: (L,) baseline abilities
            - theta1: (L,) per-model slopes over trials
            - beta: (M,) item difficulties (mean-centered)
    """
    R = validate_input(R)
    L, M, N = R.shape
    time_unit = np.asarray(time_unit, dtype=float)
    if time_unit.shape != (N,):
        raise ValueError("time_unit must have shape (N,) where N = R.shape[2].")

    if N < 2:
        k_correct = R.sum(axis=2, dtype=float)
        theta0, beta = _estimate_rasch_abilities(
            k_correct, n_trials=int(N), max_iter=max_iter
        )
        theta1 = np.zeros(L, dtype=float)
        return theta0, theta1, beta

    # Init: baseline from trial 0, difficulty from global solve rates.
    p0 = np.clip(R[:, :, 0].mean(axis=1), 1e-6, 1 - 1e-6)
    theta0_init = np.log(p0 / (1 - p0))
    theta1_init = np.zeros(L, dtype=float)

    p_m = np.clip(R.mean(axis=(0, 2)), 1e-6, 1 - 1e-6)
    beta_init = -np.log(p_m / (1 - p_m))

    params_init = np.concatenate([theta0_init, theta1_init, beta_init])
    R_float = R.astype(float, copy=False)

    def negative_log_likelihood(params: np.ndarray) -> float:
        theta0 = params[:L]
        theta1 = params[L : 2 * L]
        beta = params[2 * L :]
        beta = beta - beta.mean()  # Identifiability constraint

        diff = (
            theta0[:, None, None]
            + theta1[:, None, None] * time_unit[None, None, :]
            - beta[None, :, None]
        )
        prob = sigmoid(diff)
        prob = np.clip(prob, 1e-10, 1 - 1e-10)

        nll = -np.sum(R_float * np.log(prob) + (1 - R_float) * np.log(1 - prob))

        # Weak Gaussian prior on slopes for stable longitudinal estimation.
        nll += slope_reg * np.sum(theta1**2)
        return float(nll)

    result = minimize(
        negative_log_likelihood,
        params_init,
        method="L-BFGS-B",
        options={"maxiter": max_iter},
    )

    theta0 = result.x[:L]
    theta1 = result.x[L : 2 * L]
    beta = result.x[2 * L :]
    beta = beta - beta.mean()

    return theta0, theta1, beta


def _estimate_state_space_abilities(
    R: np.ndarray,
    time_unit: np.ndarray,
    max_iter: int = 500,
    state_reg: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate dynamic Rasch abilities with per-model random-walk trajectories.

    We fit:
        P(R[l,m,n]=1) = σ(θ[l,n] - b[m])
    with a quadratic smoothness penalty on first differences of θ over time.
    On an irregular time grid, we scale by the time step so the penalty is
    comparable across different spacings.

    Args:
        R: Binary tensor of shape (L, M, N).
        time_unit: Normalized time points in ``[0, 1]`` with shape ``(N,)``.
        max_iter: Maximum iterations for optimization.
        state_reg: Non-negative smoothness penalty on temporal differences.
    """
    R = validate_input(R)
    L, M, N = R.shape
    time_unit = np.asarray(time_unit, dtype=float)
    if time_unit.shape != (N,):
        raise ValueError("time_unit must have shape (N,) where N = R.shape[2].")

    if N < 2:
        k_correct = R.sum(axis=2, dtype=float)
        theta, beta = _estimate_rasch_abilities(
            k_correct, n_trials=int(N), max_iter=max_iter
        )
        return theta[:, None], beta

    # Initialize theta path from per-time observed solve rates.
    p_ln = np.clip(R.mean(axis=1), 1e-6, 1 - 1e-6)  # (L, N)
    theta_init = np.log(p_ln / (1 - p_ln))

    p_m = np.clip(R.mean(axis=(0, 2)), 1e-6, 1 - 1e-6)
    beta_init = -np.log(p_m / (1 - p_m))

    params_init = np.concatenate([theta_init.ravel(), beta_init])
    R_float = R.astype(float, copy=False)
    dt = np.diff(time_unit)

    def negative_log_posterior(params: np.ndarray) -> float:
        theta = params[: L * N].reshape(L, N)
        beta = params[L * N :]
        beta = beta - beta.mean()

        diff = theta[:, None, :] - beta[None, :, None]
        prob = sigmoid(diff)
        prob = np.clip(prob, 1e-10, 1 - 1e-10)

        nll = -np.sum(R_float * np.log(prob) + (1 - R_float) * np.log(1 - prob))

        # Random-walk (Brownian-motion) smoothness over irregular or regular grids:
        # penalize squared increments scaled by the time step.
        step = (theta[:, 1:] - theta[:, :-1]) / np.sqrt(dt)[None, :]
        nll += state_reg * np.sum(step**2)

        # Weak anchoring for identifiability and numerical stability.
        nll += 1e-3 * np.sum(theta[:, 0] ** 2)
        return float(nll)

    result = minimize(
        negative_log_posterior,
        params_init,
        method="L-BFGS-B",
        options={"maxiter": max_iter},
    )

    theta_path = result.x[: L * N].reshape(L, N)
    beta = result.x[L * N :]
    beta = beta - beta.mean()
    return theta_path, beta


def rasch_3pl(
    R: np.ndarray,
    method: RankMethod = "competition",
    return_scores: bool = False,
    max_iter: int = 500,
    fix_guessing: float | None = None,
    return_item_params: bool = False,
    reg_discrimination: float = 0.01,
    reg_guessing: float = 0.1,
    guessing_upper: float = 0.5,
) -> (
    np.ndarray
    | tuple[np.ndarray, np.ndarray]
    | tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]
):
    """
    Rank models with 3PL IRT via joint (optionally regularized) JMLE.

    Method context:
        Extends 2PL with item-specific pseudo-guessing ``c_m``. Estimated
        guessing is constrained to ``[0, guessing_upper]``; optionally a fixed
        value can be used. By default, small L2 penalties are applied on
        ``log(a)`` and guessing logits for numerical stability.

    Args:
        R: Binary outcome tensor with shape ``(L, M, N)`` or matrix
            ``(L, M)`` (treated as ``N=1``).
        method: Tie-handling rule passed to :func:`scorio.utils.rank_scores`.
        return_scores: If ``True``, return ``(ranking, scores)``.
        max_iter: Positive maximum number of L-BFGS iterations.
        fix_guessing: If provided, fixes the guessing parameter to this value
            for all questions; must lie in ``[0, guessing_upper]``.
        return_item_params: If True, also returns estimated item parameters.
            Implies returning scores.
        reg_discrimination: Non-negative L2 penalty weight on ``log(a)``.
            Set to ``0.0`` for pure (unpenalized) JMLE.
        reg_guessing: Non-negative L2 penalty weight on guessing logits.
            Set to ``0.0`` for pure (unpenalized) JMLE.
        guessing_upper: Upper bound for item guessing ``c_m``. Must be in
            ``(0, 1)``. Default ``0.5`` is suitable for binary outcomes.

    Returns:
        Ranking array of shape ``(L,)``.
        If ``return_scores=True``, also returns ability scores ``theta``.
        If ``return_item_params=True``, also returns
        ``{"difficulty": b, "discrimination": a, "guessing": c}``.

    Formula:
        .. math::
            p_{lm} = c_m + (1-c_m)\\sigma\\left(a_m(\\theta_l-b_m)\\right)

    References:
        Lord, F. M. (1980). Applications of Item Response Theory to
        Practical Testing Problems. Routledge.

        Birnbaum, A. (1968). Some Latent Trait Models and Their Use in
        Inferring an Examinee's Ability. In Statistical Theories of
        Mental Test Scores.

    Examples:
        >>> import numpy as np
        >>> from scorio import rank
        >>> R = np.array([
        ...     [[1, 1], [1, 1]],
        ...     [[0, 0], [0, 0]],
        ... ])
        >>> rank.rasch_3pl(R, fix_guessing=0.25).tolist()
        [1, 2]
    """
    max_iter = _validate_positive_int("max_iter", max_iter)
    reg_discrimination = _validate_nonnegative_float(
        "reg_discrimination", reg_discrimination
    )
    reg_guessing = _validate_nonnegative_float("reg_guessing", reg_guessing)
    guessing_upper = _validate_guessing_upper(guessing_upper)
    fix_guessing = _validate_fix_guessing(fix_guessing, guessing_upper)
    k_correct, n_trials = _to_binomial_counts(R)

    theta, beta, a, c = _estimate_3pl_abilities(
        k_correct,
        n_trials,
        max_iter=max_iter,
        fix_guessing=fix_guessing,
        reg_discrimination=reg_discrimination,
        reg_guessing=reg_guessing,
        guessing_upper=guessing_upper,
    )
    scores = theta

    ranking = rank_scores(scores)[method]
    if return_item_params:
        return (
            ranking,
            scores,
            {"difficulty": beta, "discrimination": a, "guessing": c},
        )
    return (ranking, scores) if return_scores else ranking


def rasch_3pl_map(
    R: np.ndarray,
    prior: Prior | float = 1.0,
    method: RankMethod = "competition",
    return_scores: bool = False,
    max_iter: int = 500,
    fix_guessing: float | None = None,
    return_item_params: bool = False,
    reg_discrimination: float = 0.01,
    reg_guessing: float = 0.1,
    guessing_upper: float = 0.5,
) -> (
    np.ndarray
    | tuple[np.ndarray, np.ndarray]
    | tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]
):
    """
    Rank models with 3PL IRT via MAP estimation.

    Method context:
        Same 3PL likelihood as :func:`rasch_3pl`, with prior regularization on
        model abilities ``theta`` and optional L2 regularization on item
        parameters.

    Args:
        R: Binary outcome tensor with shape ``(L, M, N)`` or matrix
            ``(L, M)`` (treated as ``N=1``).
        prior: Ability prior. A ``float`` is interpreted as Gaussian prior
            variance; otherwise must be a ``Prior`` instance.
        method: Tie-handling rule passed to :func:`scorio.utils.rank_scores`.
        return_scores: If ``True``, return ``(ranking, scores)``.
        max_iter: Positive maximum number of L-BFGS iterations.
        fix_guessing: Optional fixed guessing parameter in
            ``[0, guessing_upper]``.
        return_item_params: If ``True``, also return item parameters.
        reg_discrimination: Non-negative L2 penalty weight on ``log(a)``.
        reg_guessing: Non-negative L2 penalty weight on guessing logits.
        guessing_upper: Upper bound for item guessing ``c_m`` in ``(0, 1)``.
            Default is ``0.5`` for binary outcomes.

    Returns:
        Ranking array of shape ``(L,)``.
        If ``return_scores=True``, also returns MAP ability scores ``theta``.
        If ``return_item_params=True``, also returns
        ``{"difficulty": b, "discrimination": a, "guessing": c}``.
    """
    max_iter = _validate_positive_int("max_iter", max_iter)
    reg_discrimination = _validate_nonnegative_float(
        "reg_discrimination", reg_discrimination
    )
    reg_guessing = _validate_nonnegative_float("reg_guessing", reg_guessing)
    guessing_upper = _validate_guessing_upper(guessing_upper)
    fix_guessing = _validate_fix_guessing(fix_guessing, guessing_upper)
    k_correct, n_trials = _to_binomial_counts(R)
    prior = _coerce_ability_prior(prior)

    theta, beta, a, c = _estimate_3pl_abilities_map(
        k_correct,
        n_trials,
        prior,
        max_iter=max_iter,
        fix_guessing=fix_guessing,
        reg_discrimination=reg_discrimination,
        reg_guessing=reg_guessing,
        guessing_upper=guessing_upper,
    )
    scores = theta

    ranking = rank_scores(scores)[method]
    if return_item_params:
        return (
            ranking,
            scores,
            {"difficulty": beta, "discrimination": a, "guessing": c},
        )
    return (ranking, scores) if return_scores else ranking


def _estimate_3pl_abilities(
    k_correct: np.ndarray,
    n_trials: int,
    max_iter: int = 500,
    fix_guessing: float | None = None,
    reg_discrimination: float = 0.01,
    reg_guessing: float = 0.1,
    guessing_upper: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate 3PL abilities via JMLE.

    Args:
        k_correct: Shape (L, M) with counts in [0, n_trials].
        n_trials: Number of trials per (model, item).
        max_iter: Maximum iterations for optimization.
        fix_guessing: If provided, use fixed guessing parameter for all items.
        reg_discrimination: L2 penalty weight on discrimination logits.
        reg_guessing: L2 penalty weight on guessing logits.
        guessing_upper: Upper bound for estimated guessing parameters.
    """
    L, M = k_correct.shape

    def negative_log_likelihood(params):
        theta = params[:L]
        beta = params[L : L + M]
        log_a = params[L + M : L + 2 * M]

        if fix_guessing is None:
            # Estimate guessing parameters using a bounded logit transform.
            logit_c = params[L + 2 * M :]
            c = guessing_upper * sigmoid(logit_c)
        else:
            c = np.full(M, fix_guessing)

        beta = beta - beta.mean()  # Identifiability
        a = np.exp(np.clip(log_a, -3, 3))

        # P(correct) = c + (1-c) * sigmoid(a * (theta - beta))
        diff = theta[:, None] - beta[None, :]  # (L, M)
        logit = a[None, :] * diff
        base_prob = sigmoid(logit)
        prob = c[None, :] + (1 - c[None, :]) * base_prob
        prob = np.clip(prob, 1e-10, 1 - 1e-10)

        nll = -np.sum(
            k_correct * np.log(prob) + (n_trials - k_correct) * np.log(1 - prob)
        )

        # Optional regularization
        nll += reg_discrimination * np.sum(log_a**2)
        if fix_guessing is None:
            nll += reg_guessing * np.sum(logit_c**2)

        return nll

    # Initialize
    p_lm = np.clip((k_correct + 0.5) / (n_trials + 1.0), 1e-6, 1 - 1e-6)
    model_scores = p_lm.mean(axis=1)
    question_difficulty = p_lm.mean(axis=0)

    theta_init = np.log(model_scores / (1 - model_scores))
    beta_init = -np.log(question_difficulty / (1 - question_difficulty))
    log_a_init = np.zeros(M)

    if fix_guessing is None:
        # Initialize guessing at midpoint of [0, guessing_upper].
        logit_c_init = np.zeros(M)  # sigmoid(0) * guessing_upper
        params_init = np.concatenate([theta_init, beta_init, log_a_init, logit_c_init])
    else:
        params_init = np.concatenate([theta_init, beta_init, log_a_init])

    result = minimize(
        negative_log_likelihood,
        params_init,
        method="L-BFGS-B",
        options={"maxiter": max_iter},
    )

    theta = result.x[:L]
    beta = result.x[L : L + M]
    beta = beta - beta.mean()
    log_a = result.x[L + M : L + 2 * M]
    a = np.exp(np.clip(log_a, -3, 3))

    if fix_guessing is None:
        logit_c = result.x[L + 2 * M :]
        c = guessing_upper * sigmoid(logit_c)
    else:
        c = np.full(M, float(fix_guessing))

    return theta, beta, a, c


def _estimate_3pl_abilities_map(
    k_correct: np.ndarray,
    n_trials: int,
    prior: Prior,
    max_iter: int = 500,
    fix_guessing: float | None = None,
    reg_discrimination: float = 0.01,
    reg_guessing: float = 0.1,
    guessing_upper: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate 3PL abilities via MAP with configurable prior on abilities.

    Notes:
    - The 3PL model is often weakly identified without priors; we regularize:
        (i) θ via `prior.penalty(theta)`
        (ii) log a and (optionally) logit c via small quadratic penalties
             (interpretable as weak Gaussian priors)
    """
    L, M = k_correct.shape

    def negative_log_posterior(params):
        theta = params[:L]
        beta = params[L : L + M]
        log_a = params[L + M : L + 2 * M]

        if fix_guessing is None:
            logit_c = params[L + 2 * M :]
            c = guessing_upper * sigmoid(logit_c)
        else:
            c = np.full(M, float(fix_guessing))

        beta = beta - beta.mean()
        a = np.exp(np.clip(log_a, -3, 3))

        diff = theta[:, None] - beta[None, :]
        logit = a[None, :] * diff
        base_prob = sigmoid(logit)
        prob = c[None, :] + (1 - c[None, :]) * base_prob
        prob = np.clip(prob, 1e-10, 1 - 1e-10)

        nll = -np.sum(
            k_correct * np.log(prob) + (n_trials - k_correct) * np.log(1 - prob)
        )

        # Priors and regularization.
        nll += prior.penalty(theta)
        nll += reg_discrimination * np.sum(log_a**2)
        if fix_guessing is None:
            nll += reg_guessing * np.sum(logit_c**2)

        return float(nll)

    # Initialize
    p_lm = np.clip((k_correct + 0.5) / (n_trials + 1.0), 1e-6, 1 - 1e-6)
    model_scores = p_lm.mean(axis=1)
    question_difficulty = p_lm.mean(axis=0)

    theta_init = np.log(model_scores / (1 - model_scores))
    beta_init = -np.log(question_difficulty / (1 - question_difficulty))
    log_a_init = np.zeros(M)

    if fix_guessing is None:
        logit_c_init = np.zeros(M)  # => c ≈ guessing_upper / 2
        params_init = np.concatenate([theta_init, beta_init, log_a_init, logit_c_init])
    else:
        params_init = np.concatenate([theta_init, beta_init, log_a_init])

    result = minimize(
        negative_log_posterior,
        params_init,
        method="L-BFGS-B",
        options={"maxiter": max_iter},
    )

    theta = result.x[:L]
    beta = result.x[L : L + M]
    beta = beta - beta.mean()

    log_a = result.x[L + M : L + 2 * M]
    a = np.exp(np.clip(log_a, -3, 3))

    if fix_guessing is None:
        logit_c = result.x[L + 2 * M :]
        c = guessing_upper * sigmoid(logit_c)
    else:
        c = np.full(M, float(fix_guessing))

    return theta, beta, a, c


def rasch_mml(
    R: np.ndarray,
    method: RankMethod = "competition",
    return_scores: bool = False,
    max_iter: int = 100,
    em_iter: int = 20,
    n_quadrature: int = 21,
    return_item_params: bool = False,
) -> (
    np.ndarray
    | tuple[np.ndarray, np.ndarray]
    | tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]
):
    """
    Rank models with Rasch MML (EM + quadrature) and EAP scoring.

    Method context:
        Integrates out abilities under a population prior (standard normal),
        estimates item difficulties by EM, then computes expected-a-posteriori
        (EAP) model abilities.

    Args:
        R: Binary outcome tensor with shape ``(L, M, N)`` or matrix
            ``(L, M)`` (treated as ``N=1``).
        method: Tie-handling rule passed to :func:`scorio.utils.rank_scores`.
        return_scores: If ``True``, return ``(ranking, scores)``.
        max_iter: Positive max optimizer iterations in each M-step item update.
        em_iter: Positive number of EM iterations.
        n_quadrature: Number of Gauss-Hermite nodes (integer ``>=2``).
        return_item_params: If True, also returns estimated item parameters.
            Implies returning scores.

    Returns:
        Ranking array of shape ``(L,)``.
        If ``return_scores=True``, also returns EAP ability scores.
        If ``return_item_params=True``, also returns
        ``{"difficulty": b, "ability_sd": sd(theta|R)}``.

    Formula:
        .. math::
            \\hat\\theta_l^{\\mathrm{EAP}}
            = \\sum_q w_{lq}\\,\\theta_q,
            \\quad
            w_{lq} \\propto p(k_l\\mid\\theta_q,b)\\,w_q

    References:
        Bock, R. D., & Aitkin, M. (1981). Marginal maximum likelihood
        estimation of item parameters: Application of an EM algorithm.
        Psychometrika, 46(4), 443-459.

        Mislevy, R. J. (1986). Bayes modal estimation in item response
        models. Psychometrika, 51(2), 177-195.

    Examples:
        >>> import numpy as np
        >>> from scorio import rank
        >>> R = np.array([
        ...     [[1, 1], [1, 1]],
        ...     [[0, 0], [0, 0]],
        ... ])
        >>> rank.rasch_mml(R).tolist()
        [1, 2]
    """
    max_iter = _validate_positive_int("max_iter", max_iter)
    em_iter = _validate_positive_int("em_iter", em_iter)
    n_quadrature = _validate_positive_int("n_quadrature", n_quadrature, min_value=2)
    k_correct, n_trials = _to_binomial_counts(R)

    theta, beta, posterior, theta_q = _estimate_rasch_mml(
        k_correct,
        n_trials,
        max_iter=max_iter,
        em_iter=em_iter,
        n_quadrature=n_quadrature,
    )
    scores = theta

    ranking = rank_scores(scores)[method]
    if return_item_params:
        theta_sd = _posterior_sd(posterior, theta_q)
        return ranking, scores, {"difficulty": beta, "ability_sd": theta_sd}
    return (ranking, scores) if return_scores else ranking


def rasch_mml_credible(
    R: np.ndarray,
    quantile: float = 0.05,
    method: RankMethod = "competition",
    return_scores: bool = False,
    max_iter: int = 100,
    em_iter: int = 20,
    n_quadrature: int = 21,
) -> RankResult:
    """
    Rank models by a posterior quantile under Rasch MML.

    Method context:
        Uses the discrete posterior from :func:`rasch_mml` and ranks by
        posterior quantile ``Q_q(theta_l | R)``. Lower quantiles provide
        conservative, uncertainty-aware ordering.

    Args:
        R: Binary outcome tensor with shape ``(L, M, N)`` or matrix
            ``(L, M)`` (treated as ``N=1``).
        quantile: Posterior quantile ``q`` in ``(0, 1)``.
        method: Tie-handling rule passed to :func:`scorio.utils.rank_scores`.
        return_scores: If ``True``, return ``(ranking, scores)``.
        max_iter: Positive max optimizer iterations in each M-step item update.
        em_iter: Positive number of EM iterations.
        n_quadrature: Number of Gauss-Hermite nodes (integer ``>=2``).

    Returns:
        Ranking array of shape ``(L,)``.
        If ``return_scores=True``, also returns posterior-quantile scores.

    Formula:
        .. math::
            s_l = Q_q(\\theta_l\\mid R)
    """
    if not (0.0 < quantile < 1.0):
        raise ValueError("quantile must be in (0, 1)")
    max_iter = _validate_positive_int("max_iter", max_iter)
    em_iter = _validate_positive_int("em_iter", em_iter)
    n_quadrature = _validate_positive_int("n_quadrature", n_quadrature, min_value=2)

    k_correct, n_trials = _to_binomial_counts(R)
    _, beta, posterior, theta_q = _estimate_rasch_mml(
        k_correct,
        n_trials,
        max_iter=max_iter,
        em_iter=em_iter,
        n_quadrature=n_quadrature,
    )

    scores = _posterior_quantile(posterior, theta_q, quantile)
    ranking = rank_scores(scores)[method]
    return (ranking, scores) if return_scores else ranking


def _posterior_sd(posterior: np.ndarray, theta_q: np.ndarray) -> np.ndarray:
    """
    Posterior SD for each row of a discrete posterior over theta_q.
    """
    posterior = np.asarray(posterior, dtype=float)
    theta_q = np.asarray(theta_q, dtype=float)
    mean = posterior @ theta_q
    second = posterior @ (theta_q**2)
    var = np.maximum(second - mean**2, 0.0)
    return np.sqrt(var)


def _posterior_quantile(
    posterior: np.ndarray, theta_q: np.ndarray, q: float
) -> np.ndarray:
    """
    Posterior quantile for each row of a discrete posterior over theta_q.
    """
    if not (0.0 < q < 1.0):
        raise ValueError("q must be in (0, 1)")

    posterior = np.asarray(posterior, dtype=float)
    theta_q = np.asarray(theta_q, dtype=float)

    order = np.argsort(theta_q)
    theta_sorted = theta_q[order]
    post_sorted = posterior[:, order]
    cdf = np.cumsum(post_sorted, axis=1)

    out = np.empty(posterior.shape[0], dtype=float)
    for i in range(out.size):
        j = int(np.searchsorted(cdf[i], q, side="left"))
        if j >= theta_sorted.size:
            j = theta_sorted.size - 1
        out[i] = theta_sorted[j]
    return out


def _estimate_rasch_mml(
    k_correct: np.ndarray,
    n_trials: int,
    max_iter: int = 100,
    em_iter: int = 20,
    n_quadrature: int = 21,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate Rasch model via Marginal Maximum Likelihood with EM.

    Args:
        k_correct: Shape (L, M) with counts in [0, n_trials].
        n_trials: Number of trials per (model, item).
        max_iter: Max Newton-Raphson iterations per M-step.
        em_iter: Number of EM iterations.
        n_quadrature: Number of quadrature points for integration.

    Returns:
        EAP ability estimates for each model.
    """
    L, M = k_correct.shape

    # Gauss-Hermite quadrature points and weights
    # Transform to standard normal: θ = √2 * x
    x_gh, w_gh = np.polynomial.hermite.hermgauss(n_quadrature)
    theta_q = np.sqrt(2) * x_gh  # Quadrature points
    w_q = w_gh / np.sqrt(np.pi)  # Normalized weights

    # Initialize difficulties from observed proportions
    p_lm = np.clip((k_correct + 0.5) / (n_trials + 1.0), 1e-6, 1 - 1e-6)
    question_difficulty = p_lm.mean(axis=0)
    beta = -np.log((question_difficulty + 0.01) / (1 - question_difficulty + 0.01))
    beta = beta - beta.mean()

    def _make_item_nll(k_m, posterior):
        def item_nll(b):
            nll = 0.0
            for q in range(n_quadrature):
                prob = sigmoid(theta_q[q] - b)
                prob = np.clip(prob, 1e-10, 1 - 1e-10)
                log_p = k_m * np.log(prob) + (n_trials - k_m) * np.log(1 - prob)
                nll -= np.sum(posterior[:, q] * log_p)
            return nll

        return item_nll

    # EM algorithm
    for _ in range(em_iter):
        # E-step: Compute posterior weights for each model at each quadrature point
        # P(θ_q | data) ∝ P(data | θ_q) * P(θ_q)
        log_lik = np.zeros((L, n_quadrature))
        for q in range(n_quadrature):
            diff = theta_q[q] - beta  # (M,)
            prob = sigmoid(diff)
            prob = np.clip(prob, 1e-10, 1 - 1e-10)
            # Log likelihood for each model at this quadrature point
            log_lik[:, q] = np.sum(
                k_correct * np.log(prob) + (n_trials - k_correct) * np.log(1 - prob),
                axis=1,
            )

        # Posterior weights (softmax over quadrature points)
        log_lik_max = log_lik.max(axis=1, keepdims=True)
        lik = np.exp(log_lik - log_lik_max) * w_q[None, :]
        posterior = lik / lik.sum(axis=1, keepdims=True)  # (L, n_quadrature)

        # M-step: Update item difficulties
        for m in range(M):
            k_m = k_correct[:, m]
            item_nll = _make_item_nll(k_m, posterior)

            result = minimize(
                item_nll,
                beta[m],
                method="L-BFGS-B",
                options={"maxiter": max_iter},
            )
            beta[m] = result.x[0]

        # Re-center difficulties
        beta = beta - beta.mean()

    # Final E-step: Compute EAP ability estimates
    log_lik = np.zeros((L, n_quadrature))
    for q in range(n_quadrature):
        diff = theta_q[q] - beta
        prob = sigmoid(diff)
        prob = np.clip(prob, 1e-10, 1 - 1e-10)
        log_lik[:, q] = np.sum(
            k_correct * np.log(prob) + (n_trials - k_correct) * np.log(1 - prob),
            axis=1,
        )

    log_lik_max = log_lik.max(axis=1, keepdims=True)
    lik = np.exp(log_lik - log_lik_max) * w_q[None, :]
    posterior = lik / lik.sum(axis=1, keepdims=True)

    # EAP = E[θ | data] = Σ θ_q * P(θ_q | data)
    abilities = np.sum(posterior * theta_q[None, :], axis=1)

    return abilities, beta, posterior, theta_q


__all__ = [
    "rasch",
    "rasch_map",
    "rasch_mml",
    "rasch_mml_credible",
    "rasch_2pl",
    "rasch_2pl_map",
    "rasch_3pl",
    "rasch_3pl_map",
    "dynamic_irt",
]
