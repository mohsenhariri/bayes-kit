"""Schema registry for scorio signal-based evaluation.

Each schema maps a named set of signals to a classification function that
returns a ``(category_name, description, score)`` tuple.  Schemas are
registered via :func:`_register` and consumed by
:mod:`scorio.categorical.evaluate`.
"""

from __future__ import annotations

from collections.abc import Callable

from scorio.categorical.thresholds import Thresholds

_SCHEMA_REGISTRY: dict[str, dict] = {}

__all__ = ["_SCHEMA_REGISTRY"]

ClassifyFn = Callable[
    [dict[str, str], dict[str, float | None], Thresholds],
    tuple[str, str, float],
]


def _register(name: str, signals: list[str], classify_fn: ClassifyFn) -> None:
    """Add a schema entry to the registry."""
    _SCHEMA_REGISTRY[name] = {
        "cid": name,
        "name": name,
        "signals": signals,
        "classify": classify_fn,
    }


# Confident & Correct  -> C1 x R1
def _cls_2_1(
    lvl: dict[str, str], val: dict[str, float | None], th: Thresholds
) -> tuple[str, str, float]:
    """Classify calibration quality by confidence level vs. correctness."""
    c1, r1 = lvl.get("C1", "low"), lvl.get("R1", "0")
    if c1 == "high" and r1 == "1":
        return ("Confident & Correct",
                "Model knew what it was doing - ideal case.", 1.0)
    if c1 == "high" and r1 == "0":
        return ("Confidently Wrong",
                "Most dangerous failure mode - fluent but incorrect.", 0.1)
    if c1 == "low" and r1 == "1":
        return ("Lucky or Cautious",
                "Model was uncertain but arrived at the right answer.", 0.6)
    return ("Uncertain & Wrong",
            "Model was unsure and got it wrong - expected, less concerning.", 0.2)

_register("Confident & Correct", ["C1", "R1"], _cls_2_1)


# Token Surprise vs. Correctness -> T_lp_min x R1
def _cls_2_12(
    lvl: dict[str, str], val: dict[str, float | None], th: Thresholds
) -> tuple[str, str, float]:
    """Classify whether peak token surprise correlates with correctness."""
    tlm, r1 = lvl.get("T_lp_min", "low"), lvl.get("R1", "0")
    if tlm == "low" and r1 == "0":
        return ("Critical Token Derailed Answer",
                "A single uncertain token decision may have derailed the answer.", 0.1)
    if tlm == "low" and r1 == "1":
        return ("Risky Token Bet Paid Off",
                "Model took a risky token bet and it paid off.", 0.6)
    if tlm == "high" and r1 == "1":
        return ("Smooth Correct",
                "No extreme token uncertainty, correct answer - clean generation.", 0.9)
    return ("Smooth but Wrong",
            "No extreme token surprise yet wrong - the error isn't from a single uncertain moment.", 0.3)

_register("Token Surprise vs. Correctness", ["T_lp_min", "R1"], _cls_2_12)


# Format Compliance & Correctness -> R2 x R1
def _cls_2_5(
    lvl: dict[str, str], val: dict[str, float | None], th: Thresholds
) -> tuple[str, str, float]:
    """Classify format compliance against answer correctness."""
    r2, r1 = lvl.get("R2", "0"), lvl.get("R1", "0")
    if r2 == "1" and r1 == "1":
        return ("Full Compliance", "Followed format and correct.", 1.0)
    if r2 == "1" and r1 == "0":
        return ("Obedient but Wrong", "Followed format but wrong content.", 0.3)
    if r2 == "0" and r1 == "1":
        return ("Correct but Unformatted", "Right answer buried in unstructured output - extraction risk.", 0.6)
    return ("Fully Non-Compliant", "No format compliance and incorrect.", 0.0)

_register("Format Compliance & Correctness", ["R2", "R1"], _cls_2_5)


# Difficulty-Adjusted Correctness -> P2 x R1
def _cls_2_2(
    lvl: dict[str, str], val: dict[str, float | None], th: Thresholds
) -> tuple[str, str, float]:
    """Classify correctness relative to problem difficulty."""
    p2, r1 = lvl.get("P2", "low"), lvl.get("R1", "0")
    if p2 == "high" and r1 == "1":
        return ("Hard Problem Solved", "Solved a hard problem - impressive.", 1.0)
    if p2 == "high" and r1 == "0":
        return ("Hard Problem Failed", "Failed a hard problem - expected.", 0.4)
    if p2 == "low" and r1 == "1":
        return ("Easy Problem Solved", "Solved an easy problem - routine.", 0.7)
    return ("Easy Problem Failed", "Failed an easy problem - concerning.", 0.1)

_register("Difficulty-Adjusted Correctness", ["P2", "R1"], _cls_2_2)


# IO Ratio Profile -> P3 x C3 x R1
#
# P3 = prompt_sum_logprob  (Σ log P over prompt tokens; always negative)
# C3 = completion_sum_logprob  (Σ log P over completion tokens; always negative)
#
# "high" = above corpus median (less negative) → brief and/or familiar sequence
# "low"  = below corpus median (more negative) → extended and/or surprising sequence
def _cls_3_18(
    lvl: dict[str, str], val: dict[str, float | None], th: Thresholds
) -> tuple[str, str, float]:
    """Classify prompt and completion log-probability totals against correctness."""
    p3, c3, r1 = lvl.get("P3", "low"), lvl.get("C3", "low"), lvl.get("R1", "0")
    if p3 == "low" and c3 == "high" and r1 == "1":
        return ("Insightful Compression", "High-surprisal prompt, brief confident completion, correct.", 1.0)
    if p3 == "high" and c3 == "low" and r1 == "1":
        return ("Appropriate Expansion", "Familiar prompt, extended or uncertain completion, correct.", 0.8)
    if p3 == "low" and c3 == "low" and r1 == "0":
        return ("Complex Problem, Wasted Effort", "High-surprisal prompt, extended uncertain completion, wrong.", 0.1)
    if p3 == "high" and c3 == "high" and r1 == "1":
        return ("Proportional IO", "Familiar prompt, brief confident completion, correct.", 0.8)
    if p3 == "high" and c3 == "high" and r1 == "0":
        return ("Simple Problem, Brief Failure", "Familiar prompt, brief confident completion, wrong.", 0.2)
    return ("Mixed IO Profile", "P3={}, C3={}, R1={}.".format(p3, c3, r1),
            0.5 if r1 == "1" else 0.2)

_register("IO Ratio Profile", ["P3", "C3", "R1"], _cls_3_18)
