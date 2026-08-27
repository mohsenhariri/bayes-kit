# Scorio GPQA Dataset

The [Scorio GPQA dataset](https://huggingface.co/buckets/harimo/scorio-gpqa) is a collection of sampled reasoning answers from four model configurations on 3,600 superGPQA questions across 72 fields, with correctness labels, verifier scores, and top-20 token probability distributions.

- [`gpqa.ipynb`](gpqa.ipynb): load and explore the dataset.
- [`eval.ipynb`](eval.ipynb): score model performance and uncertainty.
- [`rank.ipynb`](rank.ipynb): compare and rank models.
- [`aggregate.ipynb`](aggregate.ipynb): choose one answer from several samples.

Install the dependencies with:

```bash
pip install "datasets>=5.0.0" "huggingface_hub>=1.5.0" pandas pyarrow ipykernel scorio
```

Restart the Jupyter kernel after upgrading `datasets` or `huggingface_hub`.
