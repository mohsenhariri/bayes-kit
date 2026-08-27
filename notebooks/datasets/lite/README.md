# Scorio Lite Dataset

The [Scorio Lite dataset](https://huggingface.co/datasets/harimo/scorio-lite) is a collection of sampled reasoning answers from four model configurations across competition-math and superGPQA splits, with correctness labels, verifier scores, and token log probabilities.

- [`lite.ipynb`](lite.ipynb): load and explore the dataset.
- [`eval.ipynb`](eval.ipynb): score model performance and uncertainty.
- [`rank.ipynb`](rank.ipynb): compare and rank models.
- [`aggregate.ipynb`](aggregate.ipynb): choose one answer from several samples.

Install the dependencies with:

```bash
pip install scorio datasets
```
