# `scorio.eval` Method References

Every public function in `scorio.eval` takes a question-by-trial matrix `R` as
its first argument. `R` has shape `M × N`, where each row is one question and
each column is one sampled trial. A one-dimensional input is treated as a
single question. The geometric and spectrum APIs are intentionally omitted
from this reference.

We introduced `Bayes@N` and the `scorio.eval` interface in
[Don't Pass@k: A Bayesian Framework for Large Language Model Evaluation](https://arxiv.org/abs/2510.04265) (ICLR 2026).

https://github.com/user-attachments/assets/7cb72b44-7e24-40f5-b198-4c102fe2d184

If you use the evaluation APIs, please cite this work:

```bibtex
@inproceedings{hariri2026dont,
  title={Don't Pass@k: A Bayesian Framework for Large Language Model Evaluation},
  author={Hariri, Mohsen and Samandar, Amirhossein and Hinczewski, Michael and Chaudhary, Vipin},
  booktitle={International Conference on Learning Representations},
  year={2026},
  url={https://proceedings.iclr.cc/paper_files/paper/2026/file/f04edfc65463d020629673a4bc4c58e7-Paper-Conference.pdf},
  eprint={2510.04265},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  note={Latest version available at \url{https://arxiv.org/abs/2510.04265}}
}
```

Notes:

- Binary methods expect entries in `{0, 1}`. `bayes`, `avg`, and `max_at_k`
  also accept categorical outcomes in `{0, ..., C}` when given a length-`C+1`
  weight or reward vector `w`. Their interval companions support the same
  categorical form.
- `R0` holds prior outcomes for the same questions. It is available in
  `bayes`, `bayes_ci`, and `max_at_k_ci`; its row count must match `R`.
- `bayes` and `avg` return `(score, sigma)`. The finite-bank `*at_k` point
  estimators return one scalar. Functions ending in `_ci` return
  `(mu, sigma, lo, hi)`, using a normal approximation for the credible
  interval.
- Point estimators that select `k` trials require `1 <= k <= N`. The pass,
  majority, and AUC interval functions keep the same restriction. The
  Max@k interval function describes a posterior resampling target and allows
  any integer `k >= 1`, including `k > N`.
- `pass_hat_k`, `unanimous_at_k`, and `g_pass_at_k` are the same all-success
  metric under three common names. Their `_ci` functions are equivalent too.
- `g_pass_at_k_tau` uses `tau=0` for Pass@k and `tau=1` for the all-success
  metric. `mg_pass_at_k` follows the published discrete threshold sum; for odd
  `k`, it is an approximation to the corresponding continuous area.
- By default, binary posterior uncertainty uses independent `Beta(1, 1)` base
  priors; the pass, threshold, majority, and AUC interval families expose
  `alpha0` and `beta0` overrides. Categorical uncertainty in `bayes`/`bayes_ci`,
  `avg`/`avg_ci`, and `max_at_k_ci` uses a `Dirichlet(1, ..., 1)` base prior;
  where supported, `R0` augments its counts. Bounds only clip the reported
  interval; they do not change `mu` or `sigma`.

## Bayesian and Average Metrics

| `scorio.eval.[method_name]` | Returns | Method | Reference |
| --- | --- | --- | --- |
| `bayes` | `(mu, sigma)` | Bayes@N posterior score | [Paper](https://openreview.net/forum?id=PTXi3Ef4sT) · [API](./bayes.py) · [BibTeX](#bibtex-hariri2026dont) |
| `bayes_ci` | `(mu, sigma, lo, hi)` | Bayes@N with a credible interval | [Paper](https://openreview.net/forum?id=PTXi3Ef4sT) · [API](./bayes.py) · [BibTeX](#bibtex-hariri2026dont) |
| `avg` | `(average, sigma)` | Avg@N with uncertainty on the average scale | [Paper](https://openreview.net/forum?id=PTXi3Ef4sT) · [API](./avg.py) · [BibTeX](#bibtex-hariri2026dont) |
| `avg_ci` | `(average, sigma, lo, hi)` | Avg@N with a credible interval | [Paper](https://openreview.net/forum?id=PTXi3Ef4sT) · [API](./avg.py) · [BibTeX](#bibtex-hariri2026dont) |

## Pass and Threshold Metrics

| `scorio.eval.[method_name]` | Returns | Method | Reference |
| --- | --- | --- | --- |
| `pass_at_k` | `score` | At least one of `k` selected trials succeeds | [Paper](https://arxiv.org/abs/2107.03374) · [API](./pass_at_k.py) · [BibTeX](#bibtex-chen2021evaluating) |
| `pass_at_k_ci` | `(mu, sigma, lo, hi)` | Posterior Pass@k summary | [API](./pass_at_k.py) · [Bayes@N](https://openreview.net/forum?id=PTXi3Ef4sT) |
| `pass_hat_k` | `score` | Every selected trial succeeds; also written Pass^k | [Paper](https://arxiv.org/abs/2406.12045) · [API](./pass_at_k.py) · [BibTeX](#bibtex-yao2024taubench) |
| `pass_hat_k_ci` | `(mu, sigma, lo, hi)` | Posterior all-success summary | [API](./pass_at_k.py) · [Bayes@N](https://openreview.net/forum?id=PTXi3Ef4sT) |
| `unanimous_at_k` | `score` | Alias for `pass_hat_k` | [API](./__init__.py) |
| `unanimous_at_k_ci` | `(mu, sigma, lo, hi)` | Alias for `pass_hat_k_ci` | [API](./__init__.py) |
| `g_pass_at_k` | `score` | G-Pass@k at the all-success threshold | [Paper](https://aclanthology.org/2025.findings-acl.905/) · [API](./gpass.py) · [BibTeX](#bibtex-liu2025stable-reasoning) |
| `g_pass_at_k_ci` | `(mu, sigma, lo, hi)` | Posterior G-Pass@k summary | [API](./gpass.py) · [Bayes@N](https://openreview.net/forum?id=PTXi3Ef4sT) |
| `g_pass_at_k_tau` | `score` | At least `max(1, ceil(tau * k))` selected trials succeed | [Paper](https://aclanthology.org/2025.findings-acl.905/) · [API](./gpass.py) · [BibTeX](#bibtex-liu2025stable-reasoning) |
| `g_pass_at_k_tau_ci` | `(mu, sigma, lo, hi)` | Posterior thresholded G-Pass@k summary | [API](./gpass.py) · [Bayes@N](https://openreview.net/forum?id=PTXi3Ef4sT) |
| `mg_pass_at_k` | `score` | Published discrete mean G-Pass@k summary | [Paper](https://aclanthology.org/2025.findings-acl.905/) · [API](./gpass.py) · [BibTeX](#bibtex-liu2025stable-reasoning) |
| `mg_pass_at_k_ci` | `(mu, sigma, lo, hi)` | Posterior mean G-Pass@k summary | [API](./gpass.py) · [Bayes@N](https://openreview.net/forum?id=PTXi3Ef4sT) |
| `maj_at_k` | `score` | Strict-majority specialization of thresholded G-Pass@k | [API](./maj.py) |
| `maj_at_k_ci` | `(mu, sigma, lo, hi)` | Posterior strict-majority summary | [API](./maj.py) · [Bayes@N](https://openreview.net/forum?id=PTXi3Ef4sT) |

## AUC@K Metrics

| `scorio.eval.[method_name]` | Returns | Method | Reference |
| --- | --- | --- | --- |
| `auc_at_k` | `score` | Normalized trapezoidal area over Pass@1 through Pass@k | [Paper](https://arxiv.org/abs/2601.08763) · [API](./auc.py) · [BibTeX](#bibtex-hu2026rewarding) |
| `auc_at_k_ci` | `(mu, sigma, lo, hi)` | Posterior AUC@K summary | [API](./auc.py) · [Bayes@N](https://openreview.net/forum?id=PTXi3Ef4sT) |

## Max-Reward Metrics

| `scorio.eval.[method_name]` | Returns | Method | Reference |
| --- | --- | --- | --- |
| `max_at_k` | `score` | Expected best reward among `k` selected trials | [Paper](https://arxiv.org/abs/2510.23393) · [API](./max_reward.py) · [BibTeX](#bibtex-bagirov2025best) |
| `max_at_k_ci` | `(mu, sigma, lo, hi)` | Scorio's categorical posterior extension of Max@k | [API](./max_reward.py) · [Bayes@N](https://openreview.net/forum?id=PTXi3Ef4sT) |

## References

<a id="bibtex-hariri2026dont"></a>
### `hariri2026dont`

```bibtex
@inproceedings{hariri2026dont,
  title={Don't Pass@k: A Bayesian Framework for Large Language Model Evaluation},
  author={Hariri, Mohsen and Samandar, Amirhossein and Hinczewski, Michael and Chaudhary, Vipin},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://openreview.net/forum?id=PTXi3Ef4sT},
  doi={10.48550/arXiv.2510.04265}
}
```

<a id="bibtex-chen2021evaluating"></a>
### `chen2021evaluating`

```bibtex
@article{chen2021evaluating,
  title   = {Evaluating Large Language Models Trained on Code},
  author  = {Chen, Mark and Tworek, Jerry and Jun, Heewoo and Yuan, Qiming and Pinto, Henrique Ponde de Oliveira and Kaplan, Jared and Edwards, Harri and Burda, Yuri and Joseph, Nicholas and Brockman, Greg and others},
  journal = {arXiv preprint arXiv:2107.03374},
  year    = {2021},
  doi     = {10.48550/arXiv.2107.03374},
  url     = {https://arxiv.org/abs/2107.03374}
}
```

<a id="bibtex-yao2024taubench"></a>
### `yao2024taubench`

```bibtex
@misc{yao2024taubench,
  title         = {{$\tau$}-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains},
  author        = {Yao, Shunyu and Shinn, Noah and Razavi, Pedram and Narasimhan, Karthik},
  year          = {2024},
  eprint        = {2406.12045},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  doi           = {10.48550/arXiv.2406.12045},
  url           = {https://arxiv.org/abs/2406.12045}
}
```

<a id="bibtex-liu2025stable-reasoning"></a>
### `liu2025stable_reasoning`

```bibtex
@inproceedings{liu2025stable_reasoning,
  title     = {Are Your {LLM}s Capable of Stable Reasoning?},
  author    = {Liu, Junnan and Liu, Hongwei and Xiao, Linchen and Wang, Ziyi and Liu, Kuikun and Gao, Songyang and Zhang, Wenwei and Zhang, Songyang and Chen, Kai},
  booktitle = {Findings of the Association for Computational Linguistics: ACL 2025},
  pages     = {17594--17632},
  year      = {2025},
  publisher = {Association for Computational Linguistics},
  doi       = {10.18653/v1/2025.findings-acl.905},
  url       = {https://aclanthology.org/2025.findings-acl.905/}
}
```

<a id="bibtex-hu2026rewarding"></a>
### `hu2026rewarding`

```bibtex
@article{hu2026rewarding,
  title   = {Rewarding the Rare: Uniqueness-Aware {RL} for Creative Problem Solving in {LLM}s},
  author  = {Hu, Zhiyuan and Wang, Yucheng and He, Yufei and Wu, Jiaying and Zhao, Yilun and Ng, See-Kiong and Breazeal, Cynthia and Luu, Anh Tuan and Park, Hae Won and Hooi, Bryan},
  journal = {arXiv preprint arXiv:2601.08763},
  year    = {2026},
  doi     = {10.48550/arXiv.2601.08763},
  url     = {https://arxiv.org/abs/2601.08763}
}
```

<a id="bibtex-bagirov2025best"></a>
### `bagirov2025best`

```bibtex
@article{bagirov2025best,
  title   = {The Best of {N} Worlds: Aligning Reinforcement Learning with Best-of-{N} Sampling via max@k Optimisation},
  author  = {Bagirov, Farid and Arkhipov, Mikhail and Sycheva, Ksenia and Glukhov, Evgeniy and Bogomolov, Egor},
  journal = {arXiv preprint arXiv:2510.23393},
  year    = {2025},
  doi     = {10.48550/arXiv.2510.23393},
  url     = {https://arxiv.org/abs/2510.23393}
}
```
