"""
Prior penalties for MAP ranking methods.

Method context:
    Several ranking methods in :mod:`scorio.rank` estimate latent log-strengths
    ``theta`` via a MAP objective of the form

    .. math::
        \\mathcal{L}_{\\text{MAP}}(\\theta)
        =
        \\mathcal{L}_{\\text{NLL}}(\\theta)
        +
        P(\\theta),

    where ``P(theta)`` is the prior penalty (negative log-prior up to constants).
    This module defines reusable prior classes implementing ``P(theta)`` through
    a common ``Prior`` interface.

"""

from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np


class Prior(ABC):
    """
    Abstract interface for prior penalties on log-strength parameters.

    Method context:
        Ranking MAP solvers optimize a negative objective and add a prior penalty
        term. Subclasses define that term via :meth:`penalty`.

    Notes:
        ``theta`` should be interpreted as centered log-strengths in most Scorio
        MAP estimators because strengths are identifiable only up to an additive
        constant in log-space.
    """

    @abstractmethod
    def penalty(self, theta: np.ndarray) -> float:
        """
        Evaluate the prior penalty ``P(theta)``.

        Args:
            theta: Log-strength vector of shape ``(L,)``.

        Returns:
            Scalar penalty to be added to a negative log-likelihood objective.

        Notation:
            ``theta_i`` denotes the latent log-strength for model ``i``.

        Formula:
            Subclasses implement concrete forms of

            .. math::
                P(\\theta) = -\\log p(\\theta) + \\text{constant}.
        """
        pass


class EmpiricalPrior(Prior):
    """
    Empirical Gaussian prior from prior outcome tensor ``R0``.

    Method context:
        Builds a model-specific prior mean from empirical accuracies in ``R0``
        and applies a shared Gaussian variance around those means.
        This is useful when a deterministic baseline (for example greedy decoding)
        should inform stochastic ranking.

    References:
        Hariri, M., Samandar, A., Hinczewski, M., & Chaudhary, V. (2026).
        Don't Pass@k: A Bayesian Framework for Large Language Model Evaluation.
        https://arxiv.org/abs/2510.04265

    Args:
        R0: Prior outcomes with shape ``(L, M, D)`` or ``(L, M)``.
            ``(L, M)`` is treated as ``D=1``.
        var: Positive Gaussian variance around empirical logit means.
        eps: Clipping constant for stable logit transform.

    Examples:
        >>> import numpy as np
        >>> R0 = np.array([
        ...     [1, 1, 1, 0, 1],
        ...     [0, 1, 0, 0, 1],
        ... ])
        >>> prior = EmpiricalPrior(R0, var=1.0)
        >>> prior.prior_mean.shape
        (2,)

    Notation:
        ``acc_i``: mean empirical accuracy for model ``i`` in ``R0``.
        ``mu_i``: empirical logit mean.

    Formula:
        .. math::
            \\mu_i = \\operatorname{logit}(\\operatorname{clip}(acc_i, \\epsilon, 1-\\epsilon))

        .. math::
            P(\\theta) = \\frac{1}{2\\,\\mathrm{var}}
            \\sum_i (\\theta_i - \\mu_i)^2

    Notes:
        Prior means are centered to match BT-style identifiability constraints.
    """

    def __init__(self, R0: np.ndarray, var: float = 1.0, eps: float = 1e-6):
        """
        Initialize an empirical Gaussian prior from prior observations.

        Args:
            R0: Prior outcomes, shape ``(L, M, D)`` or ``(L, M)``.
            var: Positive variance around empirical means.
            eps: Logit clipping constant in ``(0, 0.5)`` recommended.

        Raises:
            ValueError: If ``var <= 0`` or ``R0`` has invalid dimensions.
        """
        if var <= 0:
            raise ValueError("Variance must be positive")

        R0 = np.asarray(R0)

        # Handle (L, M) shape by adding D=1 dimension
        if R0.ndim == 2:
            R0 = R0[:, :, np.newaxis]
        elif R0.ndim != 3:
            raise ValueError(
                f"R0 must be 2D (L, M) or 3D (L, M, D), got ndim={R0.ndim}"
            )

        self.R0 = R0
        self.var = var
        self.eps = eps

        # Compute prior mean for each model from R0 accuracy
        # acc_l = mean accuracy of model l across all questions and trials
        L = R0.shape[0]
        acc = np.array([R0[l].mean() for l in range(L)])

        # Clip to avoid log(0) or log(inf)
        acc = np.clip(acc, eps, 1 - eps)

        # Logit transform: log-odds as prior mean for log-strength
        self.prior_mean = np.log(acc / (1 - acc))

        # Center the prior means (BT model is identified up to a constant)
        self.prior_mean = self.prior_mean - self.prior_mean.mean()

    def penalty(self, theta: np.ndarray) -> float:
        """
        Evaluate empirical-Gaussian penalty around learned prior means.

        Args:
            theta: Log-strength vector of shape ``(L,)``.

        Returns:
            Scalar penalty value.

        Raises:
            ValueError: If ``theta`` length differs from prior model count.
        """
        if len(theta) != len(self.prior_mean):
            raise ValueError(
                f"theta length ({len(theta)}) must match number of models "
                f"({len(self.prior_mean)})"
            )
        return ((theta - self.prior_mean) ** 2).sum() / (2 * self.var)


class GaussianPrior(Prior):
    """
    Isotropic Gaussian prior on log-strengths.

    Method context:
        Standard L2-style regularization used in many MAP ranking objectives.

    Args:
        mean: Prior location.
        var: Positive prior variance.

    Returns:
        ``penalty(theta)`` returns a scalar quadratic penalty.

    Formula:
        .. math::
            P(\\theta) = \\frac{1}{2\\,\\mathrm{var}}\\sum_i (\\theta_i-\\mathrm{mean})^2

    Examples:
        >>> prior = GaussianPrior(mean=0.0, var=1.0)
        >>> prior.penalty(np.array([0.5, -0.5]))
        0.25
    """

    def __init__(self, mean: float = 0.0, var: float = 1.0):
        """
        Initialize Gaussian prior parameters.

        Args:
            mean: Prior location.
            var: Positive prior variance.

        Raises:
            ValueError: If ``var <= 0``.
        """
        if var <= 0:
            raise ValueError("Variance must be positive")
        self.mean = mean
        self.var = var

    def penalty(self, theta: np.ndarray) -> float:
        """
        Evaluate quadratic Gaussian penalty.

        Args:
            theta: Log-strength vector.

        Returns:
            Scalar penalty value.
        """
        return ((theta - self.mean) ** 2).sum() / (2 * self.var)


class LaplacePrior(Prior):
    """
    Laplace prior on log-strengths.

    Method context:
        L1-style shrinkage prior that can be more robust than Gaussian around
        outliers and may encourage sparse deviations from ``loc``.

    Args:
        loc: Prior location (mode).
        scale: Positive scale.

    Examples:
        >>> prior = LaplacePrior(loc=0.0, scale=1.0)
        >>> prior.penalty(np.array([0.5, -0.5]))
        1.0

    Formula:
        .. math::
            P(\\theta) = \\frac{1}{\\mathrm{scale}}
            \\sum_i |\\theta_i - \\mathrm{loc}|
    """

    def __init__(self, loc: float = 0.0, scale: float = 1.0):
        """
        Initialize Laplace prior parameters.

        Args:
            loc: Prior location.
            scale: Positive scale.

        Raises:
            ValueError: If ``scale <= 0``.
        """
        if scale <= 0:
            raise ValueError("Scale must be positive")
        self.loc = loc
        self.scale = scale

    def penalty(self, theta: np.ndarray) -> float:
        """
        Evaluate absolute-deviation Laplace penalty.

        Args:
            theta: Log-strength vector.

        Returns:
            Scalar penalty value.
        """
        return np.abs(theta - self.loc).sum() / self.scale


class CauchyPrior(Prior):
    """
    Cauchy prior on log-strengths.

    Method context:
        Heavy-tailed prior that penalizes extreme values less aggressively than
        Gaussian/Laplace priors.

    Args:
        loc: Prior location.
        scale: Positive scale.

    Examples:
        >>> prior = CauchyPrior(loc=0.0, scale=1.0)
        >>> prior.penalty(np.array([2.0, -2.0]))
        3.218...

    Formula:
        .. math::
            P(\\theta) = \\sum_i \\log\\left(1 +
            \\left(\\frac{\\theta_i-\\mathrm{loc}}{\\mathrm{scale}}\\right)^2\\right)
    """

    def __init__(self, loc: float = 0.0, scale: float = 1.0):
        """
        Initialize Cauchy prior parameters.

        Args:
            loc: Prior location.
            scale: Positive scale.

        Raises:
            ValueError: If ``scale <= 0``.
        """
        if scale <= 0:
            raise ValueError("Scale must be positive")
        self.loc = loc
        self.scale = scale

    def penalty(self, theta: np.ndarray) -> float:
        """
        Evaluate heavy-tailed Cauchy penalty.

        Args:
            theta: Log-strength vector.

        Returns:
            Scalar penalty value.
        """
        z = (theta - self.loc) / self.scale
        return np.log1p(z**2).sum()


class UniformPrior(Prior):
    """
    Improper uniform prior on log-strengths.

    Method context:
        Disables regularization in MAP routines (equivalent to ML objective).

    Examples:
        >>> prior = UniformPrior()
        >>> prior.penalty(np.array([100.0, -100.0]))
        0.0

    Formula:
        .. math::
            P(\\theta) = 0
    """

    def penalty(self, theta: np.ndarray) -> float:
        """
        Return zero penalty.

        Args:
            theta: Log-strength vector (ignored).

        Returns:
            Always ``0.0``.
        """
        return 0.0


class CustomPrior(Prior):
    """
    User-defined prior penalty wrapper.

    Method context:
        Allows custom regularization while preserving the ``Prior`` interface
        expected by MAP estimators.

    Args:
        penalty_fn: Callable mapping ``theta`` to a scalar penalty.

    Examples:
        >>> def horseshoe_penalty(theta):
        ...     return np.log1p(theta**2).sum()
        >>> prior = CustomPrior(horseshoe_penalty)
        >>> float(prior.penalty(np.array([0.0, 1.0]))) > 0.0
        True
    """

    def __init__(self, penalty_fn: Callable[[np.ndarray], float]):
        """
        Initialize custom prior with a user-specified penalty callable.

        Args:
            penalty_fn: Callable accepting ``np.ndarray`` and returning scalar.

        Raises:
            ValueError: If ``penalty_fn`` is not callable.
        """
        if not callable(penalty_fn):
            raise ValueError("penalty_fn must be callable")
        self._penalty_fn = penalty_fn

    def penalty(self, theta: np.ndarray) -> float:
        """
        Evaluate user-provided penalty function.

        Args:
            theta: Log-strength vector.

        Returns:
            Scalar penalty from ``penalty_fn(theta)``.
        """
        return self._penalty_fn(theta)


__all__ = [
    "Prior",
    "EmpiricalPrior",
    "GaussianPrior",
    "LaplacePrior",
    "CauchyPrior",
    "UniformPrior",
    "CustomPrior",
]
