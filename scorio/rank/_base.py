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


def is_strongly_connected(adjacency: npt.ArrayLike) -> bool:
    """Return whether every vertex is reachable in both graph directions."""
    graph = np.asarray(adjacency, dtype=bool)
    if graph.ndim != 2 or graph.shape[0] != graph.shape[1]:
        raise ValueError("adjacency must be a square matrix")
    n_vertices = graph.shape[0]
    if n_vertices <= 1:
        return True

    def reachable(edges: np.ndarray) -> npt.NDArray[np.bool_]:
        seen = np.zeros(n_vertices, dtype=bool)
        stack = [0]
        seen[0] = True
        while stack:
            vertex = stack.pop()
            for neighbour in np.flatnonzero(edges[vertex] & ~seen):
                seen[neighbour] = True
                stack.append(int(neighbour))
        return seen

    return bool(reachable(graph).all() and reachable(graph.T).all())


def average_equivalent_scores(
    scores: npt.ArrayLike,
    sufficient_statistics: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """
    Average scores for observationally indistinguishable model rows.

    Optimization and Monte Carlo error must not turn exact likelihood
    symmetries into arbitrary strict ranks. ``sufficient_statistics`` may be
    the original response tensor or a lower-dimensional sufficient statistic;
    its first axis must index the same models as ``scores``.
    """
    values = np.asarray(scores, dtype=float)
    statistics = np.asarray(sufficient_statistics)
    if values.ndim != 1:
        raise ValueError("scores must be a one-dimensional array")
    if statistics.ndim < 1 or statistics.shape[0] != values.size:
        raise ValueError("sufficient_statistics must have one row for every score")

    rows = statistics.reshape(values.size, -1)
    _, groups = np.unique(rows, axis=0, return_inverse=True)
    result = values.copy()
    for group in range(int(groups.max()) + 1):
        members = groups == group
        if np.count_nonzero(members) > 1:
            result[members] = float(np.mean(values[members]))
    return result


def average_event_exchangeable_scores(
    scores: npt.ArrayLike,
    observations: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Average exact model orbits under permutations of observation columns.

    Two models belong to the same orbit when some simultaneous permutation of
    model rows and observation columns preserves the complete data matrix and
    maps one model to the other.  Such labels are not distinguished by any
    exchangeable model fitted to these observations, even when the symmetry
    requires moving several model rows at once.
    """
    values = np.asarray(scores, dtype=float)
    data = np.asarray(observations)
    if data.ndim < 2 or data.shape[0] != values.size:
        raise ValueError("observations must have one row for every score")
    data = data.reshape(values.size, -1)

    def projection_matches(source_rows: list[int], target_rows: list[int]) -> bool:
        source = data[source_rows]
        target = data[target_rows]
        source_order = np.lexsort(source[::-1, :])
        target_order = np.lexsort(target[::-1, :])
        return bool(np.array_equal(source[:, source_order], target[:, target_order]))

    # A row's multiset of values is an inexpensive, exact automorphism
    # invariant.  The backtracking projection test below supplies the full
    # proof; this signature only reduces its candidate set.
    row_signatures = [tuple(np.sort(row).tolist()) for row in data]
    parent = np.arange(values.size)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    def find_automorphism(source: int, target: int) -> np.ndarray | None:
        """Find a row bijection that extends source -> target, if one exists."""
        if row_signatures[source] != row_signatures[target]:
            return None

        source_rows = [source]
        target_rows = [target]
        used_source = np.zeros(values.size, dtype=bool)
        used_target = np.zeros(values.size, dtype=bool)
        used_source[source] = True
        used_target[target] = True
        mapping = np.full(values.size, -1, dtype=int)
        mapping[source] = target
        signature_sizes = {
            signature: row_signatures.count(signature)
            for signature in set(row_signatures)
        }
        source_order = sorted(
            (index for index in range(values.size) if index != source),
            key=lambda index: signature_sizes[row_signatures[index]],
        )

        if not projection_matches(source_rows, target_rows):
            return None

        def search() -> bool:
            if len(source_rows) == values.size:
                return True

            next_source = next(
                index for index in source_order if not used_source[index]
            )
            compatible: list[int] = []
            candidate_source_rows = [*source_rows, next_source]
            for raw_candidate_target in np.flatnonzero(~used_target):
                candidate_target = int(raw_candidate_target)
                if row_signatures[next_source] != row_signatures[candidate_target]:
                    continue
                if projection_matches(
                    candidate_source_rows,
                    [*target_rows, candidate_target],
                ):
                    compatible.append(candidate_target)
            if not compatible:
                return False

            used_source[next_source] = True
            source_rows.append(next_source)
            for candidate_target in compatible:
                used_target[candidate_target] = True
                target_rows.append(candidate_target)
                mapping[next_source] = candidate_target
                if search():
                    return True
                mapping[next_source] = -1
                target_rows.pop()
                used_target[candidate_target] = False
            source_rows.pop()
            used_source[next_source] = False
            return False

        return mapping if search() else None

    for first in range(values.size):
        for second in range(first + 1, values.size):
            if find(first) == find(second):
                continue
            automorphism = find_automorphism(first, second)
            if automorphism is None:
                continue
            for source, target in enumerate(automorphism):
                union(source, int(target))

    groups = np.array([find(index) for index in range(values.size)])
    return average_equivalent_scores(values, groups[:, None])


__all__ = [
    "validate_input",
    "build_pairwise_wins",
    "build_pairwise_counts",
    "sigmoid",
    "is_strongly_connected",
    "average_equivalent_scores",
    "average_event_exchangeable_scores",
]
