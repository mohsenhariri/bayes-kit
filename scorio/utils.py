import numpy as np


def competition_ranks_from_scores(scores_in_id_order, tol=1e-12):
    """
    L models with ids 1..L and their scores.
    scores_in_id_order: list/np.array of scores aligned to ids 1..
    Returns competition ranks (1,2,3,3,5,...).
    """
    scores = np.asarray(scores_in_id_order, dtype=float)
    order = np.argsort(-scores)
    ranks = np.zeros_like(scores, dtype=int)
    rank = 1
    i = 0
    while i < len(scores):
        # find tie block
        j = i
        while (
            j + 1 < len(scores) and abs(scores[order[j + 1]] - scores[order[i]]) <= tol
        ):
            j += 1
        # assign same rank to the whole block
        for t in range(i, j + 1):
            ranks[order[t]] = rank
        # next rank skips by block size
        rank += j - i + 1
        i = j + 1
    return ranks.tolist()
