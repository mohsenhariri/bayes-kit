# Scorio Datasets

Each release pairs repeated model attempts with correctness labels, verifier scores, and token-level probabilities so the same data can support test-time-scaling evaluation, model ranking, and answer selection.

The releases serve different purposes. Scorio Trace is a 20-model math collection for comparing reasoning models. Scorio Lite combines competition-math and superGPQA samples from four model configurations in a standard Hugging Face dataset. Scorio Math and Scorio GPQA provide the corresponding domains as Hugging Face Storage Buckets with richer top-20 token probability distributions.

## Choose a dataset

| Dataset | Sampled attempts | Coverage | Token information | Storage |
| --- | ---: | --- | --- | --- |
| [Scorio Trace](#scorio-trace) | 192,000 | 20 model configurations on four math benchmarks | Realized-token log probabilities and ranks | Hugging Face dataset |
| [Scorio Lite](#scorio-lite) | 1,211,520 | Four model configurations across five competition-math splits and superGPQA | Realized-token log probabilities and ranks | Hugging Face dataset |
| [Scorio Math](#scorio-math) | 59,520 | Four model configurations across five competition-math benchmarks | Top-20 token probability distributions | Hugging Face Storage Bucket |
| [Scorio GPQA](#scorio-gpqa) | 1,152,000 | Four model configurations on 3,600 superGPQA questions across 72 fields | Top-20 token probability distributions | Hugging Face Storage Bucket |

Use Scorio Lite when you want one convenient dataset spanning both domains and do not need complete candidate distributions. Use Scorio Math or Scorio GPQA when token entropy, self-certainty, probability margins, or other top-k confidence signals are part of the experiment. Use Scorio Trace when the wider 20-model comparison is the priority.

## Data organization

The core unit is a candidate pool: the sampled attempts for one model on one question. Every dataset here has 80 attempts per pool. The notebooks order each pool by seed, so taking the first `n` attempts gives a reproducible sample-budget sweep.

The dataset exploration notebooks describe the schemas and loading options in detail. They show how to work with:

- model, task, question, and seed identifiers;
- prompts, generated reasoning, and extracted final answers;
- ground truth and binary correctness labels;
- verifier labels, scores, and contextual probabilities;
- aggregate or per-token log probabilities and ranks;
- top-20 prompt and completion distributions in Scorio Math and Scorio GPQA.

## What each directory contains

Every dataset directory follows the same four-notebook workflow:

1. The dataset notebook loads and explores records and fields.
2. `eval.ipynb` scores model performance and uncertainty.
3. `rank.ipynb` compares and ranks models.
4. `aggregate.ipynb` selects one answer from multiple sampled candidates.

### Scorio Trace

[Scorio Trace](https://huggingface.co/datasets/harimo/scorio-trace) contains 192,000 reasoning traces: 20 model configurations, four math benchmarks, 30 questions per benchmark, and 80 runs per question. Its default `meta` configuration contains all models and aggregate token statistics; per-model configurations add full token, log-probability, and vocabulary-rank lists. The dataset stores the realized token at each position rather than a top-k candidate distribution.

- [`trace/trace.ipynb`](trace/trace.ipynb): load and explore the dataset.
- [`trace/eval.ipynb`](trace/eval.ipynb): score model performance and uncertainty.
- [`trace/rank.ipynb`](trace/rank.ipynb): compare and rank models.
- [`trace/aggregate.ipynb`](trace/aggregate.ipynb): choose one answer from several samples.
- [`trace/README.md`](trace/README.md): dataset-specific overview.

### Scorio Lite

[Scorio Lite](https://huggingface.co/datasets/harimo/scorio-lite) contains 1,211,520 attempts from four model configurations, with 80 seeds per question. It covers five competition-math splits and a fixed sample of 3,600 superGPQA questions. The `meta-*` configurations omit the six token-level lists for lighter analysis, while per-model configurations add prompt and completion tokens, log probabilities, and vocabulary ranks. Scorio Lite does not contain top-20 candidate distributions.

- [`lite/lite.ipynb`](lite/lite.ipynb): load and explore the dataset.
- [`lite/eval.ipynb`](lite/eval.ipynb): score model performance and uncertainty.
- [`lite/rank.ipynb`](lite/rank.ipynb): compare and rank models.
- [`lite/aggregate.ipynb`](lite/aggregate.ipynb): choose one answer from several samples.
- [`lite/README.md`](lite/README.md): dataset-specific overview.

### Scorio Math

[Scorio Math](https://huggingface.co/buckets/harimo/scorio-math) contains 59,520 attempts from four model configurations across five competition-math benchmarks, with 80 attempts per model and question. Each Parquet file is one complete candidate pool. The Bucket stores the top-20 candidate distribution at every prompt and completion token position, making it the math release for detailed token-level confidence and uncertainty analysis.

- [`math/math.ipynb`](math/math.ipynb): load and explore the dataset.
- [`math/eval.ipynb`](math/eval.ipynb): score model performance and uncertainty.
- [`math/rank.ipynb`](math/rank.ipynb): compare and rank models.
- [`math/aggregate.ipynb`](math/aggregate.ipynb): choose one answer from several samples.
- [`math/README.md`](math/README.md): dataset-specific overview.

### Scorio GPQA

[Scorio GPQA](https://huggingface.co/buckets/harimo/scorio-gpqa) contains 1,152,000 attempts from four model configurations on a fixed sample of 3,600 superGPQA questions. The sample spans 72 fields with 50 questions per field, and every model has 80 attempts per question. Each Parquet file is one candidate pool, including the top-20 token distributions needed for detailed confidence and uncertainty analysis.

- [`gpqa/gpqa.ipynb`](gpqa/gpqa.ipynb): load and explore the dataset.
- [`gpqa/eval.ipynb`](gpqa/eval.ipynb): score model performance and uncertainty.
- [`gpqa/rank.ipynb`](gpqa/rank.ipynb): compare and rank models.
- [`gpqa/aggregate.ipynb`](gpqa/aggregate.ipynb): choose one answer from several samples.
- [`gpqa/README.md`](gpqa/README.md): dataset-specific overview.

## Installation and Storage Bucket support

Hugging Face Storage Buckets are recent. Hugging Face [introduced Buckets on March 10,
2026](https://huggingface.co/blog/storage-buckets), and direct Bucket loading was added to
the `datasets` library in [version 5.0.0](https://github.com/huggingface/datasets/releases/tag/5.0.0).
The Scorio Math and Scorio GPQA notebooks therefore require `datasets>=5.0.0` and a recent
`huggingface_hub` client. Scorio Trace and Scorio Lite use standard dataset repositories,
but the same environment can run all four releases.

Install the notebook dependencies in the Python environment that runs Jupyter:

```bash
python -m pip install --upgrade \
  scorio \
  "datasets>=5.0.0" \
  "huggingface_hub>=1.5.0" \
  pandas \
  pyarrow \
  ipykernel
```

Restart the Jupyter kernel after upgrading. Confirm that the active kernel sees the new
versions before running a Bucket notebook:

```python
import datasets
import huggingface_hub

print("datasets:", datasets.__version__)
print("huggingface_hub:", huggingface_hub.__version__)
```

An older client may interpret `buckets/<owner>/<name>` as a local directory. The resulting
`FileNotFoundError` contains a path such as `/current/working/directory/buckets/...`, even
though the remote Bucket exists. Upgrade `datasets` and `huggingface_hub` in the active
kernel environment, restart the kernel, and run the notebook from the first cell.
