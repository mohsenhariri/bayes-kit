# Scorio Math Dataset

The [Scorio Math dataset](https://huggingface.co/buckets/harimo/scorio-math) is a collection of sampled math-reasoning answers from four model configurations across five competition-math benchmarks, with correctness labels, verifier scores, and top-20 token probability distributions.

- [`math.ipynb`](math.ipynb): load and explore the dataset.
- [`eval.ipynb`](eval.ipynb): score model performance and uncertainty.
- [`rank.ipynb`](rank.ipynb): compare and rank models.
- [`aggregate.ipynb`](aggregate.ipynb): choose one answer from several samples.

Install the dependencies with:

```bash
pip install scorio datasets
```
