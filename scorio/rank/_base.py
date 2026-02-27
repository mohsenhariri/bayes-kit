"""
Common utilities for ranking methods.

Notation used across ``scorio.rank``:

- :math:`R \\in \\{0,1\\}^{L \\times M \\times N}` is the response tensor.
- :math:`R_{lmn}=1` indicates a correct response by model :math:`l` on
  question :math:`m` and trial :math:`n`.
- :math:`W_{ij}` denotes decisive pairwise wins of model :math:`i` over
  model :math:`j`.
- :math:`T_{ij}` denotes pairwise ties between models :math:`i` and
  :math:`j`.

The helper routines in this module build validated tensor representations and
the sufficient pairwise count matrices used by many ranking estimators.
"""

import numpy as np
import numpy.typing as npt


def validate_input(  # noqa: C901, PLR0912
    R: npt.ArrayLike,  # noqa: N803
    binary_only: bool = True,  # noqa: FBT001, FBT002
) -> npt.NDArray[np.int_]:
    """
    Validate and convert input to proper 3D array.

    Args:
        R: Input array of shape (L, M, N) or (L, M) where:
           - L = number of models
           - M = number of questions
           - N = number of trials (optional, defaults to 1 if shape is (L, M))
        binary_only: If True (default), enforce binary outcomes {0, 1}
                     (bool and numeric 0/1, including 0.0/1.0, are accepted).
                     If False, allow any integer-valued outcomes for integer
                     dtype inputs. Float dtype inputs are still restricted to
                     binary 0.0/1.0.

    Returns:
        Validated numpy array of shape (L, M, N) with integer dtype.

    Raises:
        ValueError: If input has invalid dimensions or non-binary values (when binary_only=True).
    """
    R = np.asarray(R)  # noqa: N806

    # Handle 2D input (L, M) by adding trial dimension
    if R.ndim == 2:
        R = R[:, :, np.newaxis]  # Shape becomes (L, M, 1)  # noqa: N806
    elif R.ndim != 3:
        raise ValueError(
            f"Input R must be a 2D array of shape (L, M) or 3D array of shape (L, M, N), got shape {R.shape}"
        )

    # Booleans are valid binary inputs; cast directly to {0,1}.
    if np.issubdtype(R.dtype, np.bool_):
        R = R.astype(int, copy=False)
    else:
        # Validate real numeric inputs before casting.
        if not np.issubdtype(R.dtype, np.number):
            raise ValueError(f"Input R must be numeric, got dtype {R.dtype}")

        if np.issubdtype(R.dtype, np.complexfloating):
            raise ValueError("Input R must contain real-valued outcomes")

        if not np.isfinite(R).all():
            raise ValueError("Input R must not contain NaN or Inf values")

        if np.issubdtype(R.dtype, np.floating):
            # Float inputs are accepted only for binary data.
            if not np.all((R == 0) | (R == 1)):
                raise ValueError(
                    "Float inputs must be binary values (0.0 or 1.0). "
                    "Use integer dtype for multiclass outcomes."
                )
        elif binary_only and not np.all((R == 0) | (R == 1)):
            raise ValueError("Input R must contain only binary values (0 or 1)")

        R = R.astype(int, copy=False)

    L, M, N = R.shape
    if L < 2:
        raise ValueError(f"Need at least 2 models to rank, got L={L}")
    if M < 1:
        raise ValueError(f"Need at least 1 question, got M={M}")
    if N < 1:
        raise ValueError(f"Need at least 1 trial, got N={N}")

    return R


def build_pairwise_wins(R: np.ndarray) -> np.ndarray:
    """
    Build pairwise win count matrix from binary response tensor.

    For each pair (i, j), counts the number of (question, trial) instances
    where model i answered correctly and model j answered incorrectly.

    Args:
        R: Binary tensor of shape (L, M, N).

    Returns:
        Win matrix of shape (L, L) where wins[i, j] = number of times
        model i beats model j.
    """
    L, _, _ = R.shape
    wins = np.zeros((L, L), dtype=float)

    for i in range(L):
        for j in range(i + 1, L):
            # Model i wins when R[i] == 1 and R[j] == 0
            i_wins = ((R[i] == 1) & (R[j] == 0)).sum()
            j_wins = ((R[j] == 1) & (R[i] == 0)).sum()

            wins[i, j] = i_wins
            wins[j, i] = j_wins

    return wins


def build_pairwise_counts(
    R: npt.NDArray[np.int_],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Build pairwise win and tie count matrices from binary response tensor.

    Args:
        R: Binary tensor of shape (L, M, N).

    Returns:
        Tuple of (wins, ties) matrices, each of shape (L, L).
        - wins[i, j] = number of times model i beats model j
        - ties[i, j] = number of times both models have same outcome
    """
    L, _, _ = R.shape
    wins = np.zeros((L, L), dtype=float)
    ties = np.zeros((L, L), dtype=float)

    for i in range(L):
        for j in range(i + 1, L):
            i_wins = ((R[i] == 1) & (R[j] == 0)).sum()
            j_wins = ((R[j] == 1) & (R[i] == 0)).sum()
            both_same = (R[i] == R[j]).sum()

            wins[i, j] = i_wins
            wins[j, i] = j_wins
            ties[i, j] = both_same
            ties[j, i] = both_same

    return wins, ties


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid function."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


__all__ = [
    "validate_input",
    "build_pairwise_wins",
    "build_pairwise_counts",
    "sigmoid",
]
