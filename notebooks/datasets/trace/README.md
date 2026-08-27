# Scorio Trace Dataset

The [Scorio Trace dataset](https://huggingface.co/datasets/harimo/scorio-trace) is a collection of sampled math-reasoning answers with correctness labels, verifier scores, and token log probabilities.

- [`trace.ipynb`](trace.ipynb): load and explore the dataset.
- [`eval.ipynb`](eval.ipynb): score model performance and uncertainty.
- [`rank.ipynb`](rank.ipynb): compare and rank models.
- [`aggregate.ipynb`](aggregate.ipynb): choose one answer from several samples.

Install the dependencies with:

```bash
pip install scorio datasets
```
