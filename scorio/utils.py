import math
from collections.abc import Sequence
from typing import Literal

import numpy as np
from scipy.stats import kendalltau, norm, rankdata, spearmanr, weightedtau


def rank_scores(
    scores_in_id_order: Sequence[float] | np.ndarray,
    tol: float = 1e-12,
    sigmas_in_id_order: Sequence[float] | np.ndarray | None = None,
    confidence: float = 0.95,
    ci_tie_method: Literal[
        "zscore_adjacent", "ci_overlap_adjacent"
    ] = "zscore_adjacent",
) -> dict[str, np.ndarray]:
    """
    Convert scores to ranks using multiple ranking methods (with optional confidence-aware ties).

    This is a utility that many evaluation scripts already use: it converts a list of
    scores into four standard rank conventions (competition/min, competition/max,
    dense, average/fractional). Higher scores get better (lower) ranks.

    Extension: if `sigmas_in_id_order` is provided, this function ALSO returns a
    second set of ranks that treats two adjacent scores as tied when they are not
    separable at the requested confidence level.

    Args:
        scores_in_id_order: Scores aligned by ID order.
        tol: Tolerance threshold for treating scores as equal (default: 1e-12).
        sigmas_in_id_order: Optional per-score uncertainty aligned by ID order.
            If provided, uncertainty-aware ranks are added under keys suffixed with "_ci".
        confidence: Confidence used for the uncertainty-aware tie rule.
        ci_tie_method: How to decide ties when sigmas are provided:
            - "zscore_adjacent" (paper Sec. 2.9): tie if z < Φ^{-1}(confidence)
            - "ci_overlap_adjacent": tie if confidence-level CIs overlap

    Returns:
        dict with four base ranking methods:
            - "competition": Min-rank competition (1, 2, 2, 4)
            - "competition_max": Max-rank competition (1, 3, 3, 4)
            - "dense": Dense ranking (1, 2, 2, 3)
            - "avg": Fractional/average ranking (1, 2.5, 2.5, 4)

        If `sigmas_in_id_order` is provided, four extra keys are included:
            - "competition_ci"
            - "competition_max_ci"
            - "dense_ci"
            - "avg_ci"

    Examples:
        >>> import numpy as np
        >>> scores = [95.0, 87.5, 87.5, 80.0, 75.0]
        >>> ranks = rank_scores(scores)
        >>> ranks["competition"].tolist() == [1, 2, 2, 4, 5]
        True
        >>> ranks["dense"].tolist() == [1, 2, 2, 3, 4]
        True

        With uncertainty: the top two are close and get tied at 95%:

        >>> sigmas = [1.0, 1.0, 0.1, 0.1, 0.1]
        >>> ranks2 = rank_scores(scores, sigmas_in_id_order=sigmas, confidence=0.95)
        >>> "dense_ci" in ranks2
        True
    """
    scores = np.asarray(scores_in_id_order, dtype=float)
    if scores.ndim != 1:
        raise ValueError("scores_in_id_order must be a 1D sequence.")
    order = np.argsort(-scores)  # descending order
    sorted_scores = scores[order]

    # Group near-equal scores (within tolerance)
    grouped_scores = sorted_scores.copy()
    for i in range(1, len(grouped_scores)):
        if abs(grouped_scores[i] - grouped_scores[i - 1]) <= tol:
            grouped_scores[i] = grouped_scores[i - 1]

    def ranker(method: str, arr_sorted: np.ndarray) -> np.ndarray:
        ranks_sorted = rankdata(-arr_sorted, method=method)
        ranks = np.empty_like(ranks_sorted)
        ranks[order] = ranks_sorted
        return ranks

    out: dict[str, np.ndarray] = {
        "competition": ranker("min", grouped_scores),  # 1,2,2,4,5
        "competition_max": ranker("max", grouped_scores),  # 1,3,3,4,5
        "dense": ranker("dense", grouped_scores),  # 1,2,2,3,4
        "avg": ranker("average", grouped_scores),  # 1.0,2.5,2.5,4.0,5.0
    }

    if sigmas_in_id_order is not None:
        sigmas = np.asarray(sigmas_in_id_order, dtype=float)
        if sigmas.shape != scores.shape:
            raise ValueError("sigmas_in_id_order must have the same length as scores.")
        mus_s = scores[order]
        sig_s = sigmas[order]
        ci_grouped = grouped_scores.copy()

        if ci_tie_method == "zscore_adjacent":
            z_thresh = float(norm.ppf(confidence))  # one-sided
            for i in range(1, len(ci_grouped)):
                # If already equal by tol, keep tied
                if abs(ci_grouped[i] - ci_grouped[i - 1]) <= tol:
                    ci_grouped[i] = ci_grouped[i - 1]
                    continue
                denom = math.sqrt(sig_s[i - 1] ** 2 + sig_s[i] ** 2)
                if denom == 0.0:
                    continue
                z = abs(mus_s[i - 1] - mus_s[i]) / denom
                if z < z_thresh:
                    ci_grouped[i] = ci_grouped[i - 1]
        elif ci_tie_method == "ci_overlap_adjacent":
            z = float(norm.ppf(0.5 + confidence / 2.0))
            for i in range(1, len(ci_grouped)):
                if abs(ci_grouped[i] - ci_grouped[i - 1]) <= tol:
                    ci_grouped[i] = ci_grouped[i - 1]
                    continue
                lo_prev = mus_s[i - 1] - z * sig_s[i - 1]
                hi_cur = mus_s[i] + z * sig_s[i]
                if lo_prev <= hi_cur:
                    ci_grouped[i] = ci_grouped[i - 1]
        else:
            raise ValueError("Unknown ci_tie_method.")

        out.update(
            {
                "competition_ci": ranker("min", ci_grouped),
                "competition_max_ci": ranker("max", ci_grouped),
                "dense_ci": ranker("dense", ci_grouped),
                "avg_ci": ranker("average", ci_grouped),
            }
        )

    return out


def compare_rankings(
    ranked_list_a,
    ranked_list_b,
    method="all",
):
    """
    Compare two rankings using multiple correlation metrics.

    Computes Kendall's tau, Spearman's rho, and weighted Kendall's tau
    to measure agreement between two rankings.

    Args:
        ranked_list_a: First ranking (numeric array or list).
        ranked_list_b: Second ranking (numeric array or list).
        method: Which metric to return: "kendall", "spearman",
            "weighted_kendall", or "all" (default).

    Returns:
        If method is not "all", returns a ``(statistic, pvalue)`` tuple.
        If method is "all", returns a dictionary with:
        ``kendalltau``, ``spearmanr``, ``weighted_kendalltau``,
        ``fraction_mismatched``, and ``max_disp``.

    Raises:
        ValueError: If lists have different lengths or contain NaN/inf.
        TypeError: If lists are not numeric.

    Notes:
        ``scipy.stats.weightedtau`` does not compute p-values (pvalue is NaN).
        Rankings are compared element-wise at matching indices.

    Examples:
        >>> import numpy as np
        >>> rank_a = [1, 2, 3, 4, 5]
        >>> rank_b = [1, 3, 2, 4, 5]
        >>> tau, pval = compare_rankings(rank_a, rank_b, method="kendall")
        >>> round(tau, 2)
        0.8
        >>> results = compare_rankings(rank_a, rank_b, method="all")
        >>> round(results["fraction_mismatched"], 2)
        0.4
    """
    allowed_methods = {"kendall", "spearman", "weighted_kendall", "all"}
    if method not in allowed_methods:
        raise ValueError(
            f"method must be one of {sorted(allowed_methods)}; got {method!r}"
        )

    n = len(ranked_list_a) if len(ranked_list_a) == len(ranked_list_b) else 0
    if n == 0:
        raise ValueError("Ranked lists must have the same non-zero length.")

    g = np.asarray(ranked_list_a)
    t = np.asarray(ranked_list_b)

    if not (np.issubdtype(g.dtype, np.number) and np.issubdtype(t.dtype, np.number)):
        raise TypeError("ranked lists must be numeric.")

    if not (np.isfinite(g).all() and np.isfinite(t).all()):
        raise ValueError("ranked lists must not contain NaN or inf.")

    diffs = (t - g).astype(float)
    fraction_mismatched = float(np.sum(diffs != 0) / n)

    max_disp = float(np.max(np.abs(diffs))) / (n - 1) if n > 1 else 0.0

    tau_res = kendalltau(g, t)
    rho_res = spearmanr(g, t)
    wtau_res = weightedtau(g, t)

    results = {
        "kendall": (float(tau_res.statistic), float(tau_res.pvalue)),
        "spearman": (float(rho_res.statistic), float(rho_res.pvalue)),
        # Note: scipy.stats.weightedtau does not compute p-values, so pvalue is always NaN
        "weighted_kendall": (float(wtau_res.statistic), float(wtau_res.pvalue)),
        "all": {
            "kendalltau": (float(tau_res.statistic), float(tau_res.pvalue)),
            "spearmanr": (float(rho_res.statistic), float(rho_res.pvalue)),
            "weighted_kendalltau": (float(wtau_res.statistic), float(wtau_res.pvalue)),
            "fraction_mismatched": fraction_mismatched,
            "max_disp": max_disp,
        },
    }

    return results[method]


# Lehmer code ranking hash functions (for permutations without ties)


def lehmer_hash(ranked_list):
    """
    Convert a permutation to its Lehmer code (factorial number system).

    The Lehmer code provides a bijection between permutations and integers
    in the range [0, n!-1], useful for hashing, indexing, or enumerating
    all possible permutations.

    Args:
        ranked_list: Permutation of integers ``0..n-1`` as a list or array.
            Must contain no ties (all values unique).

    Returns:
        int: Unique hash value in range [0, n!-1] where n = len(ranked_list).

    Notes:
        - Time complexity: O(n²) due to inversion counting
        - Space complexity: O(1)
        - This function does NOT handle ties. All elements must be distinct.
        - Pre-computes factorials for efficiency

    Algorithm:
        For each position i, counts inversions (elements to the right that
        are smaller) and encodes them in the factorial number system.

    Examples:
        >>> lehmer_hash([0, 1, 2])
        0
        >>> lehmer_hash([2, 1, 0])
        5
        >>> lehmer_hash([0, 2, 1])
        1
        >>> lehmer_hash([1, 2, 0])
        3

    References:
        Lehmer, D. H. (1960). Teaching combinatorial tricks to a computer.
        In *Combinatorial Analysis* (Proceedings of Symposia in Applied Mathematics,
        Vol. 10, pp. 179–193). American Mathematical Society. MR 0113289.
    """
    perm = list(ranked_list)
    n = len(perm)

    if any(not isinstance(x, (int, np.integer)) for x in perm):
        raise TypeError("ranked_list must be a permutation of integers 0..n-1.")
    if set(perm) != set(range(n)):
        raise ValueError("ranked_list must be a permutation of 0..n-1 with no ties.")

    # Pre-compute factorials to avoid repeated computation
    factorials = [1] * n
    for i in range(1, n):
        factorials[i] = factorials[i - 1] * i

    hash_value = 0
    for i in range(n):
        # Count inversions: elements to the right that are smaller
        inversions = sum(1 for j in range(i + 1, n) if perm[j] < perm[i])
        hash_value += inversions * factorials[n - 1 - i]

    return hash_value


def lehmer_unhash(hash_value, n):
    """
    Convert a Lehmer code (hash) back to its permutation.

    Inverse operation of lehmer_hash. Reconstructs the original permutation
    from its integer representation in the factorial number system.

    Args:
        hash_value: Integer in range [0, n!-1] representing a permutation.
        n: Length of the permutation to generate.

    Returns:
        list: Permutation of integers [0, 1, ..., n-1] corresponding to hash_value.

    Raises:
        ValueError: If hash_value >= n! (invalid hash for given n).

    Notes:
        - Time complexity: O(n²) due to element removal tracking
        - Space complexity: O(n)
        - Pre-computes factorials for efficiency

    Examples:
        >>> lehmer_unhash(0, 3)
        [0, 1, 2]
        >>> lehmer_unhash(5, 3)
        [2, 1, 0]
        >>> lehmer_unhash(1, 3)
        [0, 2, 1]
        >>> lehmer_unhash(3, 3)
        [1, 2, 0]

    References:
        Lehmer, D. H. (1960). Teaching combinatorial tricks to a computer.
        In *Combinatorial Analysis* (Proceedings of Symposia in Applied Mathematics,
        Vol. 10, pp. 179–193). American Mathematical Society. MR 0113289.
    """
    # Validate input
    max_hash = math.factorial(n)
    if hash_value < 0 or hash_value >= max_hash:
        raise ValueError(
            f"hash_value must be in range 0..{n}!-1 = {max_hash - 1}; got {hash_value}"
        )

    # Pre-compute factorials to avoid repeated computation
    factorials = [1] * n
    for i in range(1, n):
        factorials[i] = factorials[i - 1] * i

    # Track which elements have been used
    available = list(range(n))
    result = []

    for i in range(n):
        f = factorials[n - 1 - i]
        idx = hash_value // f
        hash_value %= f
        result.append(available.pop(idx))

    return result


def ordered_bell(n: int):
    """
    Compute Fubini numbers (ordered Bell numbers) F[0..n].

    F[n] counts the number of weak orderings (rankings with ties) on n elements.
    """
    F = [0] * (n + 1)
    F[0] = 1
    for m in range(1, n + 1):
        s = 0
        for k in range(1, m + 1):
            s += math.comb(m, k) * F[m - k]
        F[m] = s
    return F


def comb_rank_lex(indices, n, k):
    """
    Rank a k-combination given by sorted indices in lexicographic order.

    Args:
        indices: Sorted list of k integers from 0..n-1
        n: Size of ground set
        k: Size of combination

    Returns:
        Integer rank in [0, C(n,k)-1]
    """
    r = 0
    prev = -1
    for pos in range(k):
        start = prev + 1
        end = indices[pos]
        remaining = k - pos - 1
        for x in range(start, end):
            r += math.comb(n - 1 - x, remaining)
        prev = indices[pos]
    return r


def comb_unrank_lex(r: int, n: int, k: int):
    """
    Unrank the r-th k-combination of {0..n-1} in lexicographic order.

    Args:
        r: Combination rank
        n: Size of ground set
        k: Size of combination

    Returns:
        Sorted list of k integers

    Raises:
        ValueError: If rank is out of range
    """
    if k == 0:
        return []
    if r < 0 or r >= math.comb(n, k):
        raise ValueError("Combination rank out of range.")
    combo = []
    x = 0
    for pos in range(k):
        rem = k - pos - 1
        while True:
            cnt = math.comb(n - 1 - x, rem) if (n - 1 - x) >= rem else 0
            if r < cnt:
                combo.append(x)
                x += 1
                break
            r -= cnt
            x += 1
    return combo


def blocks_from_rank_list(rank_list, tol=1e-12):
    """
    Convert a ranking to canonical ordered tie blocks.

    Args:
        rank_list: List where rank_list[i] is the rank of item i
        tol: Tolerance for comparing float ranks

    Returns:
        List of tie blocks (each block is a sorted list of item IDs)
        ordered from best to worst rank
    """
    r = np.asarray(rank_list, dtype=float)
    n = len(r)
    if n == 0:
        return []
    ids = np.arange(n)

    # Sort by rank ascending (best first), then by ID for determinism
    order = np.lexsort((ids, r))
    r_sorted = r[order]
    ids_sorted = ids[order]

    blocks = []
    cur = [int(ids_sorted[0])]
    for i in range(1, n):
        if abs(r_sorted[i] - r_sorted[i - 1]) <= tol:
            cur.append(int(ids_sorted[i]))
        else:
            blocks.append(sorted(cur))
            cur = [int(ids_sorted[i])]
    blocks.append(sorted(cur))
    return blocks


def ranking_hash(rank_list, tol=1e-12):
    """
    Perfect collision-free hash for rankings with ties.

    Encodes any ranking (with or without ties) into a unique integer using
    ordered Bell numbers (Fubini numbers). This is the theoretically optimal
    approach that fully preserves all ranking information.

    Args:
        rank_list: List where rank_list[i] is the rank of item i.
                  Lower values = better ranks. Ties are represented by
                  equal values. Example: [0, 1, 1, 3] means item 0 is first,
                  items 1 and 2 are tied for second, item 3 is fourth.
        tol: Tolerance for treating float ranks as equal (default: 1e-12).

    Returns:
        int: Unique hash value in range [0, F(n)-1] where F(n) is the
             n-th ordered Bell number. F(n) = exact count of all possible
             rankings with ties for n items.

    Notes:
        - Time complexity: O(n²) for encoding
        - Space complexity: O(n) for storing ordered Bell numbers
        - **Collision-free**: Different rankings always get different hashes
        - **Complete encoding**: Hash fully captures both order and tie structure
        - **Optimal**: Uses exactly log₂(F(n)) bits for F(n) possible rankings

    Algorithm:
        Uses ordered Bell (Fubini) numbers to enumerate all weak orderings.
        For each tie group, encodes both the subset selection (which items)
        and the lexicographic ordering within the subset.

    Examples:
        >>> ranking_hash([0, 1, 2])  # No ties
        0
        >>> ranking_hash([0, 1, 1])  # Items 1,2 tied at second place
        2
        >>> ranking_hash([0, 0, 1])  # Items 0,1 tied at first place
        9
        >>> ranking_hash([0, 0, 0])  # All tied
        12
        >>> ranking_hash([1, 0, 1])  # Items 0,2 tied at rank 1, item 1 at rank 0
        5

    Comparison to factorial-based hashing:
        For example, ``ranking_hash([0, 1, 1]) == 2``,
        ``ranking_hash([0, 0, 1]) == 9``, and ``ranking_hash([1, 0, 1]) == 5``.
        These hashes differ, unlike simpler approaches that lose tie information.

    References:
        Ordered Bell numbers (a.k.a. Fubini numbers), OEIS A000670:
        https://en.wikipedia.org/wiki/Ordered_Bell_number
        https://oeis.org/A000670

        Weak orderings / total preorders (rankings with ties):
        https://en.wikipedia.org/wiki/Total_preorder

        Combinatorial number system (ranking/unranking k-subsets):
        https://en.wikipedia.org/wiki/Combinatorial_number_system

    See Also:
        :func:`unhash_ranking`: Inverse operation to reconstruct ranking from hash.
    """
    blocks = blocks_from_rank_list(rank_list, tol=tol)
    n = len(rank_list)
    F = ordered_bell(n)

    remaining = list(range(n))  # Ground set by ID order
    remaining_set = set(remaining)

    h = 0
    for block in blocks:
        m = len(remaining)
        k = len(block)

        # Add contributions from all smaller tie-group sizes
        for s in range(1, k):
            h += math.comb(m, s) * F[m - s]

        # Add contribution from lexicographic position within this size
        pos = {v: i for i, v in enumerate(remaining)}
        idx = [pos[v] for v in block]  # Already sorted
        subset_rank = comb_rank_lex(idx, m, k)
        h += subset_rank * F[m - k]

        # Remove this tie group from remaining items
        remaining_set.difference_update(block)
        remaining = [x for x in remaining if x in remaining_set]

    return h


def unhash_ranking(h: int, n: int):
    """
    Reconstruct ranking with ties from its hash value.

    Inverse operation of ranking_hash. Decodes a hash back to the original
    ranking, returning competition ranks (min-rank with gaps).

    Args:
        h: Hash value in range [0, F(n)-1]
        n: Number of items to rank

    Returns:
        list: Ranking in competition format (min-rank with gaps).
              Example: [1, 2, 2, 4] means item 0 is rank 1,
              items 1 and 2 are tied at rank 2, item 3 is rank 4.

    Raises:
        ValueError: If h is out of valid range [0, F(n)-1]

    Notes:
        - Time complexity: O(n²) for decoding
        - Space complexity: O(n) for ordered Bell numbers
        - Returns competition/min-rank format with gaps after ties
        - Fully reconstructs original ranking including tie structure

    Examples:
        >>> unhash_ranking(0, 3)  # First possible ranking
        [1, 2, 3]
        >>> unhash_ranking(2, 3)  # Items 1,2 tied
        [1, 2, 2]
        >>> unhash_ranking(9, 3)  # Items 0,1 tied
        [1, 1, 3]
        >>> unhash_ranking(12, 3)  # All tied
        [1, 1, 1]
        >>> unhash_ranking(5, 3)  # Items 0,2 tied at rank 2
        [2, 1, 2]

    See Also:
        :func:`ranking_hash`: Inverse operation to hash a ranking.
    """
    F = ordered_bell(n)
    if h < 0 or h >= F[n]:
        raise ValueError(f"h out of range for n={n}. Must be 0..{F[n] - 1}.")

    remaining = list(range(n))  # IDs 0..n-1
    rank_list = [0] * n
    cur_rank = 1

    while remaining:
        m = len(remaining)

        # 1) Determine tie-group size k
        offset = 0
        for k in range(1, m + 1):
            cnt = math.comb(m, k) * F[m - k]
            if h < offset + cnt:
                h -= offset
                break
            offset += cnt
        else:
            raise RuntimeError("Unhashing failed.")

        # 2) Determine which subset and suffix hash
        suffix = F[m - k]
        subset_rank = h // suffix
        h = h % suffix

        # 3) Unrank subset, assign competition rank
        idx = comb_unrank_lex(subset_rank, m, k)
        group_ids = [remaining[i] for i in idx]
        for item in group_ids:
            rank_list[item] = cur_rank

        # 4) Remove group, advance rank (with gaps)
        chosen = set(group_ids)
        remaining = [x for x in remaining if x not in chosen]
        cur_rank += k

    return rank_list


__all__ = [
    "rank_scores",
    "compare_rankings",
    "lehmer_hash",
    "lehmer_unhash",
    "ranking_hash",
    "unhash_ranking",
]
