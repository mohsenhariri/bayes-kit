# Categorical Evaluation — `scorio.categorical`
---

## Overview

This categorical framework replaces that binary evaluation with a **rubric-defined outcome category**. Each attempt is mapped to one of a small number of interpretable categories by a **schema**, a function of observable runtime signals (log-probabilities, format indicators, correctness). A **Dirichlet posterior** over category probabilities then yields an uncertainty-aware rubric score at each compute budget:

$$
\mathbb{E}[w^\top \pi_\alpha \mid n_\alpha] = \sum_{c=0}^{C} w_c \frac{\alpha_c + n_{\alpha,c}}{\sum_{j=0}^{C} \alpha_j + N}
$$

where $w_c$ is the category weight, $n_{\alpha,c}$ is the category count, and a uniform Dirichlet prior $\text{Dir}(\alpha_0 = \ldots = \alpha_C = 1)$ is used throughout.

All five schemas in this module are **base-signal schemas**: they require only runtime observables (completion log-probabilities, format indicators) and a binary correctness label — no external verifier or reward model.

---

## Signals

Continuous signals are discretized into `"high"` / `"low"` levels before being passed to a schema. The relevant signals used by the schemas in this module are:

| Signal | Description  |
|--------|-------------|
| Completion avg. log-probability | Mean token log-probability of the completion. Higher = model was more confident during generation. |
| Completion total log-probability | Sum of completion token log-probabilities. Used in IO-ratio comparisons. |
| Prompt avg. log-probability | Mean prompt token log-probability under the same model. Higher = prompt is more predictable / lower difficulty. |
| Prompt total log-probability | Sum of prompt token log-probabilities. Used in IO-ratio comparisons. |
| Min token log-probability | The single least-confident token choice in the completion. A very low minimum flags a locally risky decision point. |
| `is_correct` | Binary: extracted answer matches ground truth. |
| `has_box` | Binary: model produced a `\boxed{}` final answer. |
---

### 1. Confident & Correct (Calibration)

**Signals:** Completion Confidence, Correctness

**Purpose:** Separates calibrated success from overconfident failure and uncertain success. A well-calibrated model should be confident when correct and uncertain when wrong. This schema penalises the failure mode of having high confidence paired with an incorrect answer.

| Category | Condition | Justification |
|----------|-----------|---------------|
| Confident & Correct | high C1, correct | Ideal: the model committed and was right. Full credit. |
| Lucky or Cautious | low C1, correct | Correct but uncertain: could reflect over-hedging or genuine difficulty. Partial credit — the answer is right but the model did not clearly know it. |
| Uncertain & Wrong | low C1, incorrect | Expected failure mode: the model signalled uncertainty and was wrong. Less informative about a systematic problem. |
| Confidently Wrong | high C1, incorrect | Most dangerous: the model was fluent and committed but wrong. Near-zero score to heavily penalise miscalibration. |

**Weight vector:** `[1.0, 0.5, 0.5, 0.0]` (Cat 1 through Cat 4, ordered: Confident & Correct, Lucky or Cautious, Uncertain & Wrong, Confidently Wrong). The symmetric 0.5 for the two uncertain outcomes reflects that neither uncertain-correct nor uncertain-wrong is strongly penalised relative to each other, while confidently-wrong receives near-zero.

---

### 2. Token Surprise vs. Correctness

**Signals:** Minimum Token Log-Probability, Correctness

**Purpose:** Flags attempts where a single, locally low-probability token decision may have been the pivotal event. When the minimum token log-prob is very low, the model took at least one high-risk token bet during generation.

| Category | Condition | Justification |
|----------|-----------|---------------|
| Smooth Correct | high T_lp_min, correct | No extreme uncertainty anywhere, and correct. Clean, reliable generation. |
| Risky Token Bet Paid Off | low T_lp_min, correct | A risky bet was taken and happened to succeed. Correct, but not robust |
| Smooth but Wrong | high T_lp_min, incorrect | Confidently wrong without a token-level signal. The error is systematic, not trace-level uncertainty.|
| Critical Token Derailed Answer | low T_lp_min, incorrect | A single uncertain token decision coincided with an incorrect answer. The local surprise may have been causally responsible for the failure. |

**Weight vector:** `[1.0, 0.5, 0.5, 0.0]`. The top category here is Smooth Correct. The symmetric treatment of the two middle categories reflects that their risk profiles are qualitatively different but similarly concerning.

---

### 3. Format Compliance & Correctness

**Signals:** has_boxed, Correctness

**Purpose:** Isolates format compliance from mathematical failures. A model can produce a correct derivation but fail to place the answer in the required `\boxed{}` format, causing an extractor to mark it wrong. 

| Category | Condition | Justification |
|----------|-----------|---------------|
| Full Compliance | has box, correct | Followed the format contract and got answer correct. |
| Correct but Unformatted | no box, correct | The answer is mathematically right but will likely fail extraction. A format fix would convert this to full credit. |
| Obedient but Wrong | has box, incorrect | Format was followed but content is wrong. The error is mathematical, not contractual. |
| Fully Non-Compliant | no box, incorrect | Neither format nor content correct. |

**Weight vector:** `[1.0, 0.5, 0.5, 0.0]`. The 0.5 weighting for both middle categories encodes that format and correctness are treated as orthogonal contributions of equal importance, so each partial success earns half credit.

---

### 4. Difficulty-Adjusted Correctness

**Signals:** Prompt Avg. Log-probability, Correctness

**Purpose:** Adjusts credit for correctness by prompt difficulty. A prompt with high average log-probability under the model is predictable as the model has seen similar patterns and the question has been trained in the model's representations.

| Category | Condition | Justification |
|----------|-----------|---------------|
| Hard Problem Solved | low P2, correct | The prompt was challenging relative to the model's prior, yet the model succeeded. |
| Easy Problem Solved | high P2, correct | Correct on a predictable prompt. |
| Hard Problem Failed | low P2, incorrect | Failed a hard prompt. |
| Easy Problem Failed | high P2, incorrect | Failed a prompt that was well within the model's distribution. |

**Weight vector:** `[1.0, 0.75, 0.5, 0.0]`. This is the only base-signal schema with a non-uniform spacing between partial-credit categories. The 0.75 for easy-correct and 0.5 for hard-incorrect reflect the asymmetric signal value: an easy correct answer is more expected than a hard failure.

---

### 5. IO Ratio Profile

**Signals:** Prompt Total Log-Probability, Completion Total Log-Probability, Correctness

**Purpose:** Compares how much the model expanded (or compressed) from prompt to completion relative to correctness. A short prompt with a long correct completion suggests appropriate elaboration; a long, complex prompt with a short correct completion suggests efficient reasoning. Mismatched ratios paired with incorrect answers indicate wasted or insufficient effort.

| Category | Condition | Justification |
|----------|-----------|---------------|
| Insightful Compression | low P3, high C3, correct | Complex prompt handled with a long, thorough completion and correct. High inference efficiency. |
| Appropriate Expansion | high P3, low C3, correct | Simple prompt answered concisely and correctly. Good proportionality. |
| Proportional IO | high P3, high C3, correct | Brief prompt with a brief correct answer. |
| Mixed IO Profile | correct | Correct but with an atypical IO ratio. Partial credit for the correct answer. |
| Simple Problem, Brief Failure | high P3, high C3, incorrect | Simple prompt produced a brief wrong answer. |
| Complex Problem, Wasted Effort | low P3, low C3, incorrect | Hard prompt, long completion, still wrong. |
| Mixed IO Profile  | incorrect | Incorrect with an atypical IO ratio. |

**Weight vector:** `[1.0, 0.5, 0.25, 0.0, 0.0, 0.5/0.2]`. The IO Ratio schema has five categories plus two weighted fallbacks for mixed-ratio cases. The 0.5 fallback for correct mixed-ratio attempts and 0.2 for incorrect mixed-ratio attempts encode the asymmetry between correctness and IO-ratio mismatch.

---

## Usage

```python
from scorio.categorical.schemas import _SCHEMA_REGISTRY
from scorio.categorical.thresholds import Thresholds

# Load thresholds from your dataset
th = Thresholds.from_dataframe(df)

# Inspect registered schemas
for name, entry in _SCHEMA_REGISTRY.items():
    print(name, "->", entry["signals"])

# Classify a single attempt
schema = _SCHEMA_REGISTRY["Confident & Correct (Calibration)"]
levels = {"C1": "high", "R1": "1"}
values = {"C1": -0.05, "R1": 1.0}
category, description, score = schema["classify"](levels, values, th)
```

See `scorio.categorical.evaluate` for the full Dirichlet–Bayes evaluation pipeline.
