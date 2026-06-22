# Schemas are designed and stored here

# These schemas will score the dataframe after thresholds.py is run on io.py and score the individual completions in the df
# From there, they will be put in R matrix form and run scorio bayes@k on them in evaluate.py


_CRITERION_REGISTRY: dict[str, dict] = {}

__all__ = ["_CRITERION_REGISTRY"]


def _register(cid: str, name: str, signals: list[str], classify_fn):
    _CRITERION_REGISTRY[cid] = {
        "id": cid,
        "name": name,
        "signals": signals,
        "classify": classify_fn,
    }

# 2.1 - Confident & Correct (Calibration) -> C1 x R1
def _cls_2_1(lvl, val, th):
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

_register("2.1", "Confident & Correct (Calibration)", ["C1", "R1"], _cls_2_1)

# 2.12 - Token Surprise vs. Correctness -> T_lp_min x R1
def _cls_2_12(lvl, val, th):
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

_register("2.12", "Token Surprise vs. Correctness", ["T_lp_min", "R1"], _cls_2_12)

# 2.5 - Format Compliance & Correctness -> R2 x R1
def _cls_2_5(lvl, val, th):
    r2, r1 = lvl.get("R2", "0"), lvl.get("R1", "0")
    if r2 == "1" and r1 == "1":
        return ("Full Compliance", "Followed format and correct.", 1.0)
    if r2 == "1" and r1 == "0":
        return ("Obedient but Wrong", "Followed format but wrong content.", 0.3)
    if r2 == "0" and r1 == "1":
        return ("Correct but Unformatted", "Right answer buried in unstructured output - extraction risk.", 0.6)
    return ("Fully Non-Compliant", "No format compliance and incorrect.", 0.0)

_register("2.5", "Format Compliance & Correctness", ["R2", "R1"], _cls_2_5)

# 2.2 - Difficulty-Adjusted Correctness -> P2 x R1
def _cls_2_2(lvl, val, th):
    p2, r1 = lvl.get("P2", "low"), lvl.get("R1", "0")
    if p2 == "high" and r1 == "1":
        return ("Hard Problem Solved", "Solved a hard problem - impressive.", 1.0)
    if p2 == "high" and r1 == "0":
        return ("Hard Problem Failed", "Failed a hard problem - expected.", 0.4)
    if p2 == "low" and r1 == "1":
        return ("Easy Problem Solved", "Solved an easy problem - routine.", 0.7)
    return ("Easy Problem Failed", "Failed an easy problem - concerning.", 0.1)

_register("2.2", "Difficulty-Adjusted Correctness", ["P2", "R1"], _cls_2_2)

# 3.18 - IO Ratio Profile -> P3 x C3 x R1
def _cls_3_18(lvl, val, th):
    p3, c3, r1 = lvl.get("P3", "low"), lvl.get("C3", "low"), lvl.get("R1", "0")
    if p3 == "low" and c3 == "high" and r1 == "1":
        return ("Insightful Compression", "Complex prompt, short completion, correct.", 1.0)
    if p3 == "high" and c3 == "low" and r1 == "1":
        return ("Appropriate Expansion", "Short prompt, long completion, correct.", 0.8)
    if p3 == "low" and c3 == "low" and r1 == "0":
        return ("Complex Problem, Wasted Effort", "Complex, long, wrong.", 0.1)
    if p3 == "high" and c3 == "high" and r1 == "1":
        return ("Proportional IO", "Simple prompt, concise answer, correct.", 0.8)
    if p3 == "high" and c3 == "high" and r1 == "0":
        return ("Simple Problem, Brief Failure", "Simple prompt, brief wrong answer.", 0.2)
    return ("Mixed IO Profile", "P3={}, C3={}, R1={}.".format(p3, c3, r1),
            0.5 if r1 == "1" else 0.2)

_register("3.18", "IO Ratio Profile", ["P3", "C3", "R1"], _cls_3_18)