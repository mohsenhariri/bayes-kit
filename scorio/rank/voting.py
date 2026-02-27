"""
Voting-based ranking methods.

This module adapts social-choice rules to model ranking from response tensors.

Notation
--------

Let :math:`R \\in \\{0,1\\}^{L \\times M \\times N}` and define question-level
grades

.. math::
    k_{lm} = \\sum_{n=1}^{N} R_{lmn} \\in \\{0,1,\\ldots,N\\}.

Each question :math:`m` is treated as one voter over models based on
:math:`k_{:m}`. Voting rules produce scores :math:`s_l` from these per-question
preferences, then ranks are derived from :math:`s`.

When :math:`N=1`, each question induces a two-level ordering (correct versus
incorrect), so many voting rules reduce to simple majority-style behavior.
"""

from functools import cmp_to_key

import numpy as np
from scipy import optimize
from scipy.optimize import Bounds, LinearConstraint
from scipy.sparse import coo_matrix
from scipy.stats import rankdata

from scorio.utils import rank_scores

from ._base import validate_input
from ._types import RankMethod, RankResult


def _per_question_correct_counts(R: np.ndarray) -> np.ndarray:
    """Return per-question correct counts k_{lm} with shape (L, M)."""
    return np.asarray(R, dtype=int).sum(axis=2)


def _pairwise_preference_counts(
    k: np.ndarray,
    tie_policy: str = "half",
) -> np.ndarray:
    """
    Build pairwise preference counts from per-question scores.

    Args:
        k: Array of shape (L, M) with per-question scores (larger is better).
        tie_policy: How to treat per-question ties for a pair (i,j):
            - "ignore": ties contribute 0 to both directions.
            - "half": ties contribute 0.5 to both directions (default).

    Returns:
        P: Array of shape (L, L) where P[i, j] is the number of questions that
           prefer i over j (possibly fractional if tie_policy="half").
    """
    k = np.asarray(k, dtype=float)
    if k.ndim != 2:
        raise ValueError(f"k must have shape (L, M), got {k.shape}")
    L, M = k.shape

    if tie_policy not in {"ignore", "half"}:
        raise ValueError("tie_policy must be one of {'ignore','half'}")

    P = np.zeros((L, L), dtype=float)
    for i in range(L):
        for j in range(i + 1, L):
            i_over_j = float(np.sum(k[i] > k[j]))
            j_over_i = float(np.sum(k[j] > k[i]))
            if tie_policy == "half":
                ties = float(M - i_over_j - j_over_i)
                i_over_j += 0.5 * ties
                j_over_i += 0.5 * ties
            P[i, j] = i_over_j
            P[j, i] = j_over_i
    return P


def _topological_level_scores(adj: np.ndarray) -> np.ndarray:
    """
    Convert a directed acyclic relation (adjacency) into tie-aware scores.

    Args:
        adj: Bool array (L, L) where adj[i, j]=True means i should rank above j.

    Returns:
        scores: Float array (L,), higher is better; tied nodes get equal score.
    """
    adj = np.asarray(adj, dtype=bool)
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError("adj must be square (L, L)")
    L = adj.shape[0]

    remaining = np.ones(L, dtype=bool)
    indeg = adj.sum(axis=0).astype(int)

    scores = np.zeros(L, dtype=float)
    current_score = float(L)

    while remaining.any():
        zero_indeg = remaining & (indeg == 0)
        if not zero_indeg.any():
            # Should not happen for a DAG; fall back to tying remaining items.
            scores[remaining] = current_score
            break

        nodes = np.flatnonzero(zero_indeg)
        scores[nodes] = current_score
        current_score -= 1.0

        # Remove nodes and update indegrees
        remaining[nodes] = False
        for u in nodes:
            for v in np.flatnonzero(adj[u]):
                indeg[v] -= 1

    return scores


def borda(
    R: np.ndarray,
    method: RankMethod = "competition",
    return_scores: bool = False,
) -> RankResult:
    """
    Rank models with Borda count on per-question rankings.

    Method context:
        Each question acts as a voter that ranks models by per-question
        correct count ``k[l, m]``. Borda assigns positional points per
        question and sums them across questions; ties use averaged ranks.

    Args:
        R: Binary outcome tensor with shape ``(L, M, N)`` or matrix
            ``(L, M)`` (treated as ``N=1``).
        method: Tie-handling rule passed to :func:`scorio.utils.rank_scores`.
        return_scores: If ``True``, return ``(ranking, scores)``.

    Returns:
        Ranking array of shape ``(L,)``.
        If ``return_scores=True``, also returns Borda scores
        (shape ``(L,)``).

    Notation:
        ``k_{lm} = sum_{n=1}^N R_{lmn}`` and ``r_{lm}`` is model ``l``'s
        tie-averaged descending rank among ``k_{:,m}`` (rank 1 is best).

    Formula:
        .. math::
            s_l^{\\mathrm{Borda}}
            = \\sum_{m=1}^{M} (L - r_{lm})

    References:
        de Borda, J.-C. (1781/1784). Mémoire sur les élections au scrutin.
        In Histoire de l'Académie Royale des Sciences.

        Brandt, F., Conitzer, V., Endriss, U., Lang, J., & Procaccia, A. D.
        (2016). Handbook of Computational Social Choice. Cambridge University
        Press.

    Examples:
        >>> import numpy as np
        >>> from scorio import rank
        >>> R = np.array([
        ...     [[1, 1, 1], [1, 0, 0]],  # k: [3, 1]
        ...     [[1, 1, 0], [0, 1, 0]],  # k: [2, 1]
        ...     [[0, 0, 0], [1, 1, 1]],  # k: [0, 3]
        ... ])
        >>> ranks, scores = rank.borda(R, return_scores=True)
        >>> ranks.tolist()
        [1, 3, 2]
        >>> scores.round(2).tolist()
        [2.5, 1.5, 2.0]

    Notes:
        Adding a constant to every model's score does not change the ranking.
    """
    R = validate_input(R)
    k = _per_question_correct_counts(R)
    L, M = k.shape

    scores = np.zeros(L, dtype=float)
    for m in range(M):
        # rank 1 = best (largest k), ties get average rank
        r = rankdata(-k[:, m], method="average")
        scores += L - r  # positional Borda points (L-1..0 up to a constant)

    ranking = rank_scores(scores)[method]

    return (ranking, scores) if return_scores else ranking


def copeland(
    R: np.ndarray,
    method: RankMethod = "competition",
    return_scores: bool = False,
) -> RankResult:
    """
    Rank models with Copeland pairwise-majority scores.

    Method context:
        For each model pair ``(i, j)``, count how many questions prefer ``i``
        over ``j`` by comparing per-question correct counts. A strict majority
        contributes ``+1`` to the winner and ``-1`` to the loser; tied
        pairwise contests contribute ``0``.

    Args:
        R: Binary outcome tensor with shape ``(L, M, N)`` or matrix
            ``(L, M)`` (treated as ``N=1``).
        method: Tie-handling rule passed to :func:`scorio.utils.rank_scores`.
        return_scores: If ``True``, return ``(ranking, scores)``.

    Returns:
        Ranking array of shape ``(L,)``.
        If ``return_scores=True``, also returns Copeland scores
        (shape ``(L,)``).

    Notation:
        ``k_{im} = sum_{n=1}^N R_{imn}`` and
        ``W^{(q)}_{ij} = sum_{m=1}^{M} 1[k_{im} > k_{jm}]``.

    Formula:
        .. math::
            s_i^{\\mathrm{Copeland}}
            = \\sum_{j\\neq i}
            \\operatorname{sign}\\!\\left(
            W^{(q)}_{ij} - W^{(q)}_{ji}
            \\right)

    References:
        Copeland, A. H. (1951). A Reasonable Social Welfare Function.
        Seminar on Applications of Mathematics to the Social Sciences,
        University of Michigan.

        Brandt, F., Conitzer, V., Endriss, U., Lang, J., & Procaccia, A. D.
        (2016). Handbook of Computational Social Choice. Cambridge University
        Press.

    Examples:
        >>> import numpy as np
        >>> from scorio import rank
        >>> R = np.array([
        ...     [[1, 1], [0, 0], [1, 0]],  # k: [2, 0, 1]
        ...     [[1, 0], [1, 1], [0, 0]],  # k: [1, 2, 0]
        ...     [[0, 0], [1, 0], [1, 1]],  # k: [0, 1, 2]
        ... ])
        >>> ranks, scores = rank.copeland(R, return_scores=True)
        >>> ranks.tolist()
        [1, 1, 1]
        >>> scores.tolist()
        [0.0, 0.0, 0.0]

    Notes:
        This implementation compares pairwise question-level majorities and
        does not use magnitude of per-question margins beyond the sign.
    """
    R = validate_input(R)
    k = _per_question_correct_counts(R)
    L, _ = k.shape

    scores = np.zeros(L, dtype=float)
    for i in range(L):
        for j in range(i + 1, L):
            i_over_j = float(np.sum(k[i] > k[j]))
            j_over_i = float(np.sum(k[j] > k[i]))
            if i_over_j > j_over_i:
                scores[i] += 1.0
                scores[j] -= 1.0
            elif j_over_i > i_over_j:
                scores[i] -= 1.0
                scores[j] += 1.0

    ranking = rank_scores(scores)[method]

    return (ranking, scores) if return_scores else ranking


def win_rate(
    R: np.ndarray,
    method: RankMethod = "competition",
    return_scores: bool = False,
) -> RankResult:
    """
    Rank models by pairwise question-level win rate.

    Method context:
        For each model pair, count on how many questions model ``i`` has a
        higher per-question correct count than model ``j``. A model's score
        is the fraction of decisive pairwise outcomes it wins.

    Args:
        R: Binary outcome tensor with shape ``(L, M, N)`` or matrix
            ``(L, M)`` (treated as ``N=1``).
        method: Tie-handling rule passed to :func:`scorio.utils.rank_scores`.
        return_scores: If ``True``, return ``(ranking, scores)``.

    Returns:
        Ranking array of shape ``(L,)``.
        If ``return_scores=True``, also returns win-rate scores in ``[0, 1]``
        (shape ``(L,)``).

    Notation:
        ``k_{im} = sum_{n=1}^N R_{imn}`` and
        ``W^{(q)}_{ij} = sum_{m=1}^{M} 1[k_{im} > k_{jm}]``.

    Formula:
        .. math::
            s_i^{\\mathrm{winrate}}
            = \\frac{\\sum_{j\\neq i} W^{(q)}_{ij}}
                   {\\sum_{j\\neq i} \\left(W^{(q)}_{ij} + W^{(q)}_{ji}\\right)}

    Examples:
        >>> import numpy as np
        >>> from scorio import rank
        >>> R = np.array([
        ...     [[1, 1], [1, 1]],
        ...     [[0, 0], [0, 0]],
        ... ])
        >>> ranks, scores = rank.win_rate(R, return_scores=True)
        >>> ranks.tolist()
        [1, 2]
        >>> scores.round(2).tolist()
        [1.0, 0.0]

    Notes:
        If a model has no decisive pairwise outcomes against any opponent, its
        score is set to ``0.5``.
    """
    R = validate_input(R)
    k = _per_question_correct_counts(R)
    L, _ = k.shape

    wins = np.zeros((L, L), dtype=float)
    for i in range(L):
        for j in range(i + 1, L):
            wins[i, j] = float(np.sum(k[i] > k[j]))
            wins[j, i] = float(np.sum(k[j] > k[i]))

    total_wins = wins.sum(axis=1)
    total_comparisons = wins.sum(axis=1) + wins.sum(axis=0)

    scores = np.full(L, 0.5, dtype=float)
    mask = total_comparisons > 0
    scores[mask] = total_wins[mask] / total_comparisons[mask]
    ranking = rank_scores(scores)[method]

    return (ranking, scores) if return_scores else ranking


def minimax(
    R: np.ndarray,
    variant: str = "margin",
    tie_policy: str = "half",
    method: RankMethod = "competition",
    return_scores: bool = False,
) -> RankResult:
    """
    Rank models with the Minimax (Simpson-Kramer) Condorcet rule.

    Method context:
        Build pairwise question-level preference counts ``P[i, j]`` from
        per-question correct counts. Each model is scored by its worst
        pairwise defeat; smaller worst defeat is better.

    Args:
        R: Binary outcome tensor with shape ``(L, M, N)`` or matrix
            ``(L, M)`` (treated as ``N=1``).
        variant: Defeat-strength definition:
            - ``"margin"``: use margin of defeat
            - ``"winning_votes"``: use opponent's winning-vote count
        tie_policy: How per-question ties contribute to pairwise counts:
            ``"ignore"`` or ``"half"``.
        method: Tie-handling rule passed to :func:`scorio.utils.rank_scores`.
        return_scores: If ``True``, return ``(ranking, scores)``.

    Returns:
        Ranking array of shape ``(L,)``.
        If ``return_scores=True``, also returns minimax scores
        (shape ``(L,)``), where values are non-positive and larger is better.

    Notation:
        ``P_{ij}`` is pairwise preference count and
        ``Delta_{ij} = P_{ij} - P_{ji}``.

    Formula:
        .. math::
            s_i^{\\mathrm{minimax}}
            = -\\max_{j\\neq i} \\max(0, \\Delta_{ji})

        .. math::
            s_i^{\\mathrm{wv}}
            = -\\max_{j : P_{ji} > P_{ij}} P_{ji}

    References:
        Brandt, F., Conitzer, V., Endriss, U., Lang, J., & Procaccia, A. D.
        (2016). Handbook of Computational Social Choice. Cambridge University
        Press.

    Examples:
        >>> import numpy as np
        >>> from scorio import rank
        >>> R = np.array([
        ...     [[1, 1], [1, 1], [1, 1]],
        ...     [[1, 0], [1, 0], [1, 0]],
        ...     [[0, 0], [0, 0], [0, 0]],
        ... ])
        >>> ranks, scores = rank.minimax(R, return_scores=True)
        >>> ranks.tolist()
        [1, 2, 2]
        >>> scores.tolist()
        [-0.0, -3.0, -3.0]
    """
    R = validate_input(R)
    k = _per_question_correct_counts(R)
    P = _pairwise_preference_counts(k, tie_policy=tie_policy)
    margin = P - P.T

    variant = str(variant)
    if variant not in {"margin", "winning_votes"}:
        raise ValueError("variant must be one of {'margin','winning_votes'}")

    L = P.shape[0]
    scores = np.zeros(L, dtype=float)
    for i in range(L):
        defeats = []
        for j in range(L):
            if i == j:
                continue
            if margin[j, i] > 0:
                if variant == "margin":
                    defeats.append(float(margin[j, i]))
                else:
                    defeats.append(float(P[j, i]))
        scores[i] = -(max(defeats) if defeats else 0.0)

    ranking = rank_scores(scores)[method]
    return (ranking, scores) if return_scores else ranking


def schulze(
    R: np.ndarray,
    tie_policy: str = "half",
    method: RankMethod = "competition",
    return_scores: bool = False,
) -> RankResult:
    """
    Rank models with the Schulze beatpath Condorcet method.

    Method context:
        Build pairwise preference counts ``P`` from per-question outcomes.
        Initialize direct victories, then run a strongest-path closure:
        path strength is the maximum bottleneck strength across all directed
        paths. Model ``i`` is preferred to ``j`` when ``p[i, j] > p[j, i]``.

    Args:
        R: Binary outcome tensor with shape ``(L, M, N)`` or matrix
            ``(L, M)`` (treated as ``N=1``).
        tie_policy: How per-question ties contribute to pairwise counts:
            ``"ignore"`` or ``"half"``.
        method: Tie-handling rule passed to :func:`scorio.utils.rank_scores`.
        return_scores: If ``True``, return ``(ranking, scores)``.

    Returns:
        Ranking array of shape ``(L,)``.
        If ``return_scores=True``, also returns level scores derived from the
        beatpath dominance relation (shape ``(L,)``).

    Notation:
        ``P_{ij}`` is pairwise preference count and ``p_{ij}`` is strongest
        path strength from ``i`` to ``j``.

    Formula:
        .. math::
            p_{ij} =
            \\begin{cases}
            P_{ij}, & P_{ij} > P_{ji} \\\\
            0, & \\text{otherwise}
            \\end{cases}

        .. math::
            p_{jk} \\leftarrow
            \\max\\bigl(p_{jk}, \\min(p_{ji}, p_{ik})\\bigr)

    References:
        Schulze, M. (2010). A new monotonic, clone-independent, reversal
        symmetric, and Condorcet-consistent single-winner election method.
        Social Choice and Welfare.

        Brandt, F., Conitzer, V., Endriss, U., Lang, J., & Procaccia, A. D.
        (2016). Handbook of Computational Social Choice. Cambridge University
        Press.
    """
    R = validate_input(R)
    k = _per_question_correct_counts(R)
    P = _pairwise_preference_counts(k, tie_policy=tie_policy)
    L = P.shape[0]

    # p[i,j] = strength of the strongest path from i to j
    p = np.zeros((L, L), dtype=float)
    for i in range(L):
        for j in range(L):
            if i == j:
                continue
            if P[i, j] > P[j, i]:
                p[i, j] = P[i, j]

    for i in range(L):
        for j in range(L):
            if i == j:
                continue
            for k_ in range(L):
                if i == k_ or j == k_:
                    continue
                p[j, k_] = max(p[j, k_], min(p[j, i], p[i, k_]))

    beats = p > p.T
    scores = _topological_level_scores(beats)

    ranking = rank_scores(scores)[method]
    return (ranking, scores) if return_scores else ranking


def ranked_pairs(
    R: np.ndarray,
    strength: str = "margin",
    tie_policy: str = "half",
    method: RankMethod = "competition",
    return_scores: bool = False,
) -> RankResult:
    """
    Rank models with the Ranked Pairs (Tideman) Condorcet method.

    Method context:
        Compute directed pairwise victories and sort them by strength. Lock
        victories in descending order, skipping any edge that would create a
        directed cycle. The locked acyclic graph induces the final ranking.

    Args:
        R: Binary outcome tensor with shape ``(L, M, N)`` or matrix
            ``(L, M)`` (treated as ``N=1``).
        strength: Primary edge-strength key:
            - ``"margin"``: absolute pairwise margin ``|P_ij - P_ji|``
            - ``"winning_votes"``: winning-vote count ``P_ij``
        tie_policy: How per-question ties contribute to pairwise counts:
            ``"ignore"`` or ``"half"``.
        method: Tie-handling rule passed to :func:`scorio.utils.rank_scores`.
        return_scores: If ``True``, return ``(ranking, scores)``.

    Returns:
        Ranking array of shape ``(L,)``.
        If ``return_scores=True``, also returns topological level scores from
        the locked graph (shape ``(L,)``).

    Notation:
        ``P_{ij}`` is pairwise preference count and
        ``Delta_{ij} = P_{ij} - P_{ji}``.

    Formula:
        Lock directed wins ``i -> j`` in nonincreasing order of selected
        strength, accepting a lock only if it keeps the locked graph acyclic.

    References:
        Tideman, T. N. (1987). Independence of clones as a criterion for voting
        rules. Social Choice and Welfare.

        Brandt, F., Conitzer, V., Endriss, U., Lang, J., & Procaccia, A. D.
        (2016). Handbook of Computational Social Choice. Cambridge University
        Press.
    """
    R = validate_input(R)
    k = _per_question_correct_counts(R)
    P = _pairwise_preference_counts(k, tie_policy=tie_policy)
    L = P.shape[0]

    margin = P - P.T
    strength = str(strength)
    if strength not in {"margin", "winning_votes"}:
        raise ValueError("strength must be one of {'margin','winning_votes'}")

    victories: list[tuple[float, float, int, int]] = []
    for i in range(L):
        for j in range(i + 1, L):
            if margin[i, j] == 0:
                continue
            if margin[i, j] > 0:
                winner, loser = i, j
            else:
                winner, loser = j, i

            m = float(abs(margin[i, j]))
            wv = float(P[winner, loser])
            primary = m if strength == "margin" else wv
            victories.append((primary, wv, winner, loser))

    # Sort by primary strength, then winning votes, then deterministic IDs.
    victories.sort(key=lambda t: (-t[0], -t[1], t[2], t[3]))

    locked = np.zeros((L, L), dtype=bool)

    def has_path(src: int, dst: int) -> bool:
        stack = [src]
        seen = {src}
        while stack:
            u = stack.pop()
            if u == dst:
                return True
            for v in np.flatnonzero(locked[u]):
                if v not in seen:
                    seen.add(int(v))
                    stack.append(int(v))
        return False

    for _, _, winner, loser in victories:
        # Adding winner->loser creates a cycle iff loser can reach winner.
        if has_path(loser, winner):
            continue
        locked[winner, loser] = True

    scores = _topological_level_scores(locked)
    ranking = rank_scores(scores)[method]
    return (ranking, scores) if return_scores else ranking


def kemeny_young(
    R: np.ndarray,
    tie_policy: str = "half",
    method: RankMethod = "competition",
    return_scores: bool = False,
    time_limit: float | None = None,
    tie_aware: bool = True,
) -> RankResult:
    """
    Rank models with Kemeny-Young rank aggregation via exact MILP.

    Method context:
        Treat each question as a voter and aggregate pairwise preferences into
        ``P``. Kemeny-Young selects a total order maximizing agreement with
        pairwise counts.

    Args:
        R: Binary outcome tensor with shape ``(L, M, N)`` or matrix
            ``(L, M)`` (treated as ``N=1``).
        tie_policy: How per-question ties contribute to pairwise counts:
            ``"ignore"`` or ``"half"``.
        method: Tie-handling rule passed to :func:`scorio.utils.rank_scores`.
        return_scores: If ``True``, return ``(ranking, scores)``.
        time_limit: Optional positive MILP time limit in seconds.
        tie_aware: If ``True`` (default), derive a tie-aware preorder by
            checking which pairwise orders are forced across optimal Kemeny
            solutions. If ``False``, return the single optimal order selected
            by the MILP solver.

    Returns:
        Ranking array of shape ``(L,)``.
        If ``return_scores=True``, also returns Kemeny scores
        (shape ``(L,)``). For ``tie_aware=True``, these are level scores from
        the forced-order DAG over all optimal Kemeny solutions. For
        ``tie_aware=False``, scores are the number of opponents ranked below
        each model in the selected optimal order.

    Notation:
        ``P_{ij}`` is the aggregated pairwise preference count.
        ``y_{ij} in {0,1}`` indicates whether model ``i`` is ranked above
        model ``j``.

    Formula:
        .. math::
            \\max_{y}
            \\sum_{i \\ne j} P_{ij} y_{ij}

        .. math::
            y_{ij} + y_{ji} = 1, \\quad
            y_{ij} + y_{jk} + y_{ki} \\le 2
            \\;\\; (\\forall i,j,k \\text{ distinct})

    References:
        Kemeny, J. G. (1959). Mathematics without Numbers. Daedalus.

        Young, H. P. (1977). Extending Condorcet's rule.
        Journal of Economic Theory.

        Brandt, F., Conitzer, V., Endriss, U., Lang, J., & Procaccia, A. D.
        (2016). Handbook of Computational Social Choice. Cambridge University
        Press.

    Notes:
        Exact Kemeny optimization is NP-hard in general; this implementation
        uses ``scipy.optimize.milp`` on the induced linear-ordering ILP.
        Tie-aware mode solves additional MILPs (up to one per model pair).
    """
    R = validate_input(R)
    if time_limit is not None:
        time_limit = float(time_limit)
        if not np.isfinite(time_limit) or time_limit <= 0.0:
            raise ValueError("time_limit must be a positive finite scalar.")

    k = _per_question_correct_counts(R)
    P = _pairwise_preference_counts(k, tie_policy=tie_policy)
    L = P.shape[0]

    # Variables y_{ij} for all ordered pairs i != j: y_{ij}=1 if i ranked above j.
    # Number of variables: L*(L-1)
    def var_index(i: int, j: int) -> int:
        if i == j:
            raise ValueError("No variable for i==j")
        return i * (L - 1) + (j - 1 if j > i else j)

    n_vars = L * (L - 1)
    c = np.zeros(n_vars, dtype=float)
    integrality = np.ones(n_vars, dtype=int)

    for i in range(L):
        for j in range(L):
            if i == j:
                continue
            c[var_index(i, j)] = -float(P[i, j])  # maximize => minimize negative

    bounds = Bounds(lb=np.zeros(n_vars), ub=np.ones(n_vars))

    # Equality constraints: y_{ij} + y_{ji} = 1 for all i<j
    eq_rows = []
    eq_cols = []
    eq_data = []
    eq_lb = []
    eq_ub = []
    row = 0
    for i in range(L):
        for j in range(i + 1, L):
            eq_rows.extend([row, row])
            eq_cols.extend([var_index(i, j), var_index(j, i)])
            eq_data.extend([1.0, 1.0])
            eq_lb.append(1.0)
            eq_ub.append(1.0)
            row += 1

    A_eq = coo_matrix((eq_data, (eq_rows, eq_cols)), shape=(row, n_vars)).tocsr()
    constraints: list[LinearConstraint] = [
        LinearConstraint(A_eq, np.asarray(eq_lb), np.asarray(eq_ub))
    ]

    # Triangle (3-cycle) elimination constraints:
    # y_{ij} + y_{jk} + y_{ki} <= 2 and y_{ik} + y_{kj} + y_{ji} <= 2 for i<j<k
    tri_rows = []
    tri_cols = []
    tri_data = []
    tri_lb = []
    tri_ub = []
    row = 0
    for i in range(L):
        for j in range(i + 1, L):
            for k_ in range(j + 1, L):
                # i -> j -> k -> i
                tri_rows.extend([row, row, row])
                tri_cols.extend([var_index(i, j), var_index(j, k_), var_index(k_, i)])
                tri_data.extend([1.0, 1.0, 1.0])
                tri_lb.append(-np.inf)
                tri_ub.append(2.0)
                row += 1

                # i -> k -> j -> i
                tri_rows.extend([row, row, row])
                tri_cols.extend([var_index(i, k_), var_index(k_, j), var_index(j, i)])
                tri_data.extend([1.0, 1.0, 1.0])
                tri_lb.append(-np.inf)
                tri_ub.append(2.0)
                row += 1

    A_tri = coo_matrix((tri_data, (tri_rows, tri_cols)), shape=(row, n_vars)).tocsr()
    constraints.append(
        LinearConstraint(A_tri, np.asarray(tri_lb, dtype=float), np.asarray(tri_ub))
    )

    options = None if time_limit is None else {"time_limit": float(time_limit)}
    res = optimize.milp(
        c,
        integrality=integrality,
        bounds=bounds,
        constraints=constraints,
        options=options,
    )
    if res.x is None:
        raise RuntimeError("MILP solver failed to return a solution")

    y = np.zeros((L, L), dtype=float)
    for i in range(L):
        for j in range(L):
            if i == j:
                continue
            y[i, j] = res.x[var_index(i, j)]

    if not tie_aware or not res.success:
        # In a selected total order, score_i = number of candidates below i.
        scores = y.sum(axis=1)
        ranking = rank_scores(scores)[method]
        return (ranking, scores) if return_scores else ranking

    # Tie-aware Kemeny: keep only pairwise orders that are forced in all
    # optimal solutions. For each pair, test whether reversing the selected
    # orientation can still achieve the same optimal objective value.
    opt_value = float(res.fun)
    opt_tol = 1e-9 * max(1.0, abs(opt_value))
    base_lb = np.zeros(n_vars, dtype=float)
    base_ub = np.ones(n_vars, dtype=float)

    def can_be_optimal_with(i_above_j: int, j_below_i: int) -> bool:
        idx = var_index(i_above_j, j_below_i)
        lb = base_lb.copy()
        ub = base_ub.copy()
        lb[idx] = 1.0
        ub[idx] = 1.0
        fixed_res = optimize.milp(
            c,
            integrality=integrality,
            bounds=Bounds(lb=lb, ub=ub),
            constraints=constraints,
            options=options,
        )
        if fixed_res.x is None:
            # Solver could not certify the constrained subproblem;
            # conservatively treat the order as potentially optimal.
            return True
        if not np.isfinite(fixed_res.fun):
            return True
        if float(fixed_res.fun) <= opt_value + opt_tol:
            return True
        # If the constrained solve did not terminate optimally, do not force
        # the opposite order to be impossible.
        return bool(not fixed_res.success)

    forced = np.zeros((L, L), dtype=bool)
    for i in range(L):
        for j in range(i + 1, L):
            if y[i, j] >= 0.5:
                winner, loser = i, j
                reverse_optimal = can_be_optimal_with(j, i)
            else:
                winner, loser = j, i
                reverse_optimal = can_be_optimal_with(i, j)

            if not reverse_optimal:
                forced[winner, loser] = True

    scores = _topological_level_scores(forced)
    ranking = rank_scores(scores)[method]
    return (ranking, scores) if return_scores else ranking


def nanson(
    R: np.ndarray,
    rank_ties: str = "average",
    method: RankMethod = "competition",
    return_scores: bool = False,
) -> RankResult:
    """
    Rank models with Nanson's Borda-elimination rule.

    Method context:
        Recompute Borda scores among currently active models each round, then
        eliminate all models scoring strictly below the round mean.

    Args:
        R: Binary outcome tensor with shape ``(L, M, N)`` or matrix
            ``(L, M)`` (treated as ``N=1``).
        rank_ties: Tie rule passed to :func:`scipy.stats.rankdata` when
            computing per-question Borda ranks among active models.
        method: Tie-handling rule passed to :func:`scorio.utils.rank_scores`.
        return_scores: If ``True``, return ``(ranking, scores)``.

    Returns:
        Ranking array of shape ``(L,)``.
        If ``return_scores=True``, also returns survival-round scores
        (shape ``(L,)``), where larger means eliminated later.

    Notation:
        At round ``t``, active set is ``A_t`` and Borda score in that round is
        ``s_i^{(t)}`` for ``i in A_t``.

    Formula:
        .. math::
            E_t = \\{ i \\in A_t : s_i^{(t)} < \\overline{s}^{(t)} \\}
            ,\\quad
            A_{t+1} = A_t \\setminus E_t

    References:
        Brandt, F., Conitzer, V., Endriss, U., Lang, J., & Procaccia, A. D.
        (2016). Handbook of Computational Social Choice. Cambridge University
        Press.

    Notes:
        If no active model is strictly below the mean, elimination stops and
        all remaining models tie.
    """
    R = validate_input(R)
    k = _per_question_correct_counts(R)
    L, M = k.shape

    alive = np.ones(L, dtype=bool)
    survival = np.zeros(L, dtype=float)
    round_idx = 0.0

    while alive.sum() > 1:
        idx = np.flatnonzero(alive)
        k_sub = k[idx]  # (L_alive, M)

        borda_sub = np.zeros(idx.size, dtype=float)
        for m in range(M):
            r = rankdata(-k_sub[:, m], method=rank_ties)
            borda_sub += idx.size - r

        mean_score = float(borda_sub.mean())
        to_eliminate = borda_sub < mean_score
        if not np.any(to_eliminate):
            # No one below mean: stop with a tie among remaining.
            break

        eliminated = idx[to_eliminate]
        survival[eliminated] = round_idx
        alive[eliminated] = False
        round_idx += 1.0

    # Remaining candidates survive the longest
    survival[alive] = round_idx
    ranking = rank_scores(survival)[method]
    return (ranking, survival) if return_scores else ranking


def baldwin(
    R: np.ndarray,
    rank_ties: str = "average",
    method: RankMethod = "competition",
    return_scores: bool = False,
) -> RankResult:
    """
    Rank models with Baldwin's iterative Borda-elimination rule.

    Method context:
        Recompute Borda scores among active models each round and eliminate the
        model(s) with the minimum score. This implementation removes all models
        tied at the minimum in a round.

    Args:
        R: Binary outcome tensor with shape ``(L, M, N)`` or matrix
            ``(L, M)`` (treated as ``N=1``).
        rank_ties: Tie rule passed to :func:`scipy.stats.rankdata` when
            computing per-question Borda ranks among active models.
        method: Tie-handling rule passed to :func:`scorio.utils.rank_scores`.
        return_scores: If ``True``, return ``(ranking, scores)``.

    Returns:
        Ranking array of shape ``(L,)``.
        If ``return_scores=True``, also returns survival-round scores
        (shape ``(L,)``), where larger means eliminated later.

    Notation:
        At round ``t``, active set is ``A_t`` and Borda score is
        ``s_i^{(t)}`` for ``i in A_t``.

    Formula:
        .. math::
            E_t = \\arg\\min_{i \\in A_t} s_i^{(t)}
            ,\\quad
            A_{t+1} = A_t \\setminus E_t

    References:
        Brandt, F., Conitzer, V., Endriss, U., Lang, J., & Procaccia, A. D.
        (2016). Handbook of Computational Social Choice. Cambridge University
        Press.

    Notes:
        If all active models tie at the minimum in a round, elimination stops
        and all remaining models tie.
    """
    R = validate_input(R)
    k = _per_question_correct_counts(R)
    L, M = k.shape

    alive = np.ones(L, dtype=bool)
    survival = np.zeros(L, dtype=float)
    round_idx = 0.0

    while alive.sum() > 1:
        idx = np.flatnonzero(alive)
        k_sub = k[idx]

        borda_sub = np.zeros(idx.size, dtype=float)
        for m in range(M):
            r = rankdata(-k_sub[:, m], method=rank_ties)
            borda_sub += idx.size - r

        min_score = float(np.min(borda_sub))
        to_eliminate = borda_sub == min_score
        if np.all(to_eliminate):
            # All tied: stop with a tie among remaining.
            break

        eliminated = idx[to_eliminate]
        survival[eliminated] = round_idx
        alive[eliminated] = False
        round_idx += 1.0

    survival[alive] = round_idx
    ranking = rank_scores(survival)[method]
    return (ranking, survival) if return_scores else ranking


def majority_judgment(
    R: np.ndarray,
    method: RankMethod = "competition",
    return_scores: bool = False,
) -> RankResult:
    """
    Rank models with Majority Judgment on per-question trial counts.

    Method context:
        Each model receives per-question integer grades
        ``k_{lm} in {0, ..., N}``. Models are first compared by median grade.
        Ties are broken with majority-gauge style iterative median removal.

    Args:
        R: Binary outcome tensor with shape ``(L, M, N)`` or matrix
            ``(L, M)`` (treated as ``N=1``).
        method: Tie-handling rule passed to :func:`scorio.utils.rank_scores`.
        return_scores: If ``True``, return ``(ranking, scores)``.

    Returns:
        Ranking array of shape ``(L,)``.
        If ``return_scores=True``, also returns level scores induced by the
        MJ comparison preorder (shape ``(L,)``).

    Notation:
        ``k_{lm} = sum_{n=1}^N R_{lmn}`` is the grade assigned by question
        ``m`` to model ``l``.

    Formula:
        Rank by median grade, then for tied models iteratively remove one
        occurrence of the current median grade from each tied model and repeat
        median comparison.

    References:
        Balinski, M., & Laraki, R. (2011).
        Majority Judgment: Measuring, Ranking, and Electing. MIT Press.

    Notes:
        The implementation uses the lower median for even sample sizes.
    """
    R = validate_input(R)
    k = _per_question_correct_counts(R)
    L, M = k.shape
    N = int(R.shape[2])

    counts = np.zeros((L, N + 1), dtype=int)
    for i in range(L):
        counts[i] = np.bincount(k[i], minlength=N + 1)

    def lower_median_grade(hist: np.ndarray, total: int) -> int:
        # Lower median index in sorted ascending list is (total-1)//2
        target = (total - 1) // 2
        cum = 0
        for g, c_ in enumerate(hist):
            cum += int(c_)
            if cum > target:
                return int(g)
        return int(len(hist) - 1)

    def compare(i: int, j: int) -> int:
        hi = counts[i].copy()
        hj = counts[j].copy()
        ti = M
        tj = M
        while ti > 0 and tj > 0:
            gi = lower_median_grade(hi, ti)
            gj = lower_median_grade(hj, tj)
            if gi != gj:
                return -1 if gi > gj else 1
            # Remove one ballot at the current median grade from both
            hi[gi] -= 1
            hj[gj] -= 1
            ti -= 1
            tj -= 1

        # No separating median found after full iterative stripping:
        # candidates are tied under this MJ comparison.
        return 0

    # Deterministic sort using majority-judgment comparison
    order = list(range(L))
    order.sort(key=cmp_to_key(compare))

    scores = np.zeros(L, dtype=float)
    current_score = float(L)
    start = 0
    while start < L:
        end = start + 1
        while end < L and compare(order[start], order[end]) == 0:
            end += 1
        scores[np.asarray(order[start:end], dtype=int)] = current_score
        current_score -= 1.0
        start = end

    ranking = rank_scores(scores)[method]
    return (ranking, scores) if return_scores else ranking


__all__ = [
    "borda",
    "copeland",
    "win_rate",
    "minimax",
    "schulze",
    "ranked_pairs",
    "kemeny_young",
    "nanson",
    "baldwin",
    "majority_judgment",
]
