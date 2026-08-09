# `scorio.aggregate` Method References

`scorio.aggregate` scores sampled responses, selects final answers, and decides
when to stop sampling or generation. Offline selectors accept `(N,)` for one
question or `(M, N)` for a batch, with questions in rows and sampled responses
in columns.

We introduce the `scorio.aggregate` APIs in
[Test-Time Scaling in Reasoning LLMs: Inference Regimes, Evaluation, and Reproducibility](https://arxiv.org/abs/2608.04001).

If you use these aggregation APIs, please cite this work:

```bibtex
@misc{hariri2026testtime,
  title         = {Test-Time Scaling in Reasoning {LLM}s: Inference Regimes, Evaluation, and Reproducibility},
  author        = {Hariri, Mohsen and Chen, Weicong and Shahini, Nahal and Singh, Vikash and Ye, Kai and Samandar, Amirhossein and Ganguly, Debargha and Sankar, Sreehari and Zhang, Yanyan and Wang, Shouren and Peng, Jerry and Zhang, Biyao and Hinczewski, Michael and Chaudhary, Vipin},
  year          = {2026},
  eprint        = {2608.04001},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  doi           = {10.48550/arXiv.2608.04001},
  url           = {https://arxiv.org/abs/2608.04001}
}
```

Notes:

- Import the API as `from scorio import aggregate` or `from scorio import agg`.
- Answers are grouped by equality and must be hashable. Put tuple labels in an
  explicitly shaped `dtype=object` array; otherwise NumPy may treat tuple
  elements as extra axes. `None`, `""`, and `NaN` answers are ignored. A row
  with no valid answers returns `None`, with index `-1` and score `NaN` when
  requested.
- `scores` must have the same shape as `answers`; higher is better. Scores for
  valid answers must be finite. `NaN` is not treated as a missing score.
- By default, selectors return a scalar for `(N,)` input or an `(M,)` object
  array for `(M, N)` input. `return_index=True` adds the
  representative candidate index. Score-based selectors also accept
  `return_score=True`; with both flags, the result is
  `(selected, index, score)`. Except for `best_of_n`, the returned score belongs
  to the representative candidate, not the group's aggregate weight.
- Confidence functions score one trace at a time. Chosen-token log-probabilities
  have shape `(T,)`; top-k log-probabilities have shape `(T, k)` or use a ragged
  length-`T` list. Entropy and self-certainty renormalize the observed top-k
  support, not the full vocabulary.
- `perplexity` and `token_entropy` are lower-is-better. Negate them for
  order-based selectors. For additive weighted voting, use non-negative weights
  such as reciprocal perplexity and exponentiate raw log-probabilities.
  `varentropy` has no fixed confidence direction.
- `prm_aggregate` reduces PRM step scores to one number per trace. Use the same
  reduction when fitting and applying calibrated methods.
- Call `fit_kde_vote_calibration` with held-out labeled responses from the same
  generator, verifier, and target distribution used at inference. KDE voting
  and CGES accept only finite probabilities in `(0, 1)`.
- Answer and score ties favor first appearance.
  `adaptive_consistency_crp_stop` uses a finite-horizon Monte Carlo estimate
  controlled by `seed`, `n_alpha`, and `n_simulations`.
- `filtered_vote(keep=1)` keeps one candidate; `keep=1.0` keeps all candidates.
- `best_of_majority` adapts the paper's Pass@k method to select one answer. The
  default `alpha=0` skips frequency filtering; use a positive threshold to
  enable it. When several traces share an answer, `aggregate` pools their
  scores. `aggregate="max"` is closest to the paper's highest-reward surviving
  response.
- Selection rules do not estimate accuracy. Evaluate correctness with
  `scorio.eval`, such as `eval.avg`, `eval.bayes`, or their interval variants.

## Confidence signals

| `scorio.aggregate.[method_name]` | Returns | Method | Reference |
| --- | --- | --- | --- |
| `mean_logprob` | `score` | Length-normalized chosen-token log-likelihood | [Meena](https://arxiv.org/abs/2001.09977) · [Self-Consistency](https://arxiv.org/abs/2203.11171) · [API](./confidence.py) · [BibTeX](#bibtex-adiwardana2020towards) · [BibTeX](#bibtex-wang2023selfconsistency) |
| `sequence_logprob` | `score` | Unnormalized sequence log-likelihood | [Paper](https://arxiv.org/abs/2203.11171) · [API](./confidence.py) · [BibTeX](#bibtex-wang2023selfconsistency) |
| `perplexity` | `score` | Exponentiated mean negative log-likelihood; lower is better | [Paper](https://arxiv.org/abs/2001.09977) · [API](./confidence.py) · [BibTeX](#bibtex-adiwardana2020towards) |
| `self_certainty` | `score` | Top-k KL-from-uniform trace confidence | [Paper](https://arxiv.org/abs/2502.18581) · [API](./confidence.py) · [BibTeX](#bibtex-kang2025scalable) |
| `token_confidence` | `(T,) array` | DeepConf negative-mean-top-k token confidence | [Paper](https://arxiv.org/abs/2508.15260) · [API](./confidence.py) · [BibTeX](#bibtex-fu2025deep) |
| `deepconf_confidence` | `score` | Mean, tail, bottom-group, or lowest-group DeepConf trace confidence | [Paper](https://arxiv.org/abs/2508.15260) · [API](./confidence.py) · [BibTeX](#bibtex-fu2025deep) |
| `token_entropy` | `score` | Top-k Shannon entropy; lower is better | [Paper](https://arxiv.org/abs/2002.07650) · [API](./confidence.py) · [BibTeX](#bibtex-malinin2021uncertainty) |
| `varentropy` | `score` | Top-k surprisal variance diagnostic | [Paper](https://doi.org/10.1109/TIT.2013.2291007) · [entropix](https://github.com/xjdr-alt/entropix) · [API](./confidence.py) · [BibTeX](#bibtex-kontoyiannis2014optimal) |
| `max_softmax_probability` | `score` | Reduced top-1 token probability | [Paper](https://arxiv.org/abs/1610.02136) · [API](./confidence.py) · [BibTeX](#bibtex-hendrycks2017baseline) |
| `logprob_margin` | `score` | Reduced top1-top2 log-probability or probability margin | [Paper](https://doi.org/10.1007/3-540-44816-0_31) · [API](./confidence.py) · [BibTeX](#bibtex-scheffer2001active) |
| `picsar` | `score` | Reasoning-plus-answer log-likelihood selector | [Paper](https://aclanthology.org/2026.findings-acl.1577/) · [API](./confidence.py) · [BibTeX](#bibtex-leang2026picsar) |

## Reward aggregation

| `scorio.aggregate.[method_name]` | Returns | Method | Reference |
| --- | --- | --- | --- |
| `prm_aggregate` | `score` | Reduce per-step PRM scores by `last`, `min`, `mean`, `prod`, or `max` | [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) · [Math-Shepherd](https://aclanthology.org/2024.acl-long.510/) · [API](./prm.py) · [BibTeX](#bibtex-lightman2023verify) · [BibTeX](#bibtex-wang2024mathshepherd) |

## Reward-based selection

| `scorio.aggregate.[method_name]` | Returns | Method | Reference |
| --- | --- | --- | --- |
| `best_of_n` | `selection` | Answer of the highest-scoring candidate | [Paper](https://arxiv.org/abs/2110.14168) · [API](./best_of.py) · [BibTeX](#bibtex-cobbe2021training) |
| `majority_of_the_bests` | `selection` | Exact mode of the bootstrapped Best-of-N answer distribution | [Paper](https://arxiv.org/abs/2511.18630) · [API](./best_of.py) · [BibTeX](#bibtex-rakhsha2025majority) |
| `mob` | `selection` | Alias for `majority_of_the_bests` | [API](./best_of.py) |
| `best_of_majority` | `selection` | Highest pooled reward among frequency-gated answers | [Paper](https://arxiv.org/abs/2510.03199) · [API](./best_of.py) · [BibTeX](#bibtex-di2025best) |

## Vote-based aggregation

| `scorio.aggregate.[method_name]` | Returns | Method | Reference |
| --- | --- | --- | --- |
| `majority_vote` | `selection` | Most frequent valid answer (self-consistency) | [Paper](https://arxiv.org/abs/2203.11171) · [API](./vote.py) · [BibTeX](#bibtex-wang2023selfconsistency) |
| `weighted_majority_vote` | `selection` | Answer maximizing summed or mean candidate score | [Paper](https://aclanthology.org/2023.acl-long.291/) · [API](./vote.py) · [BibTeX](#bibtex-li2023making) |
| `softmax_weighted_vote` | `selection` | Temperature-softmax Confidence-Informed Self-Consistency | [Paper](https://aclanthology.org/2025.findings-acl.1030/) · [API](./vote.py) · [BibTeX](#bibtex-taubenfeld2025confidence) |
| `rank_weighted_vote` | `selection` | Rank-weighted Borda vote, invariant to monotone score rescaling | [Paper](https://arxiv.org/abs/2502.18581) · [API](./vote.py) · [BibTeX](#bibtex-kang2025scalable) |
| `logit_weighted_vote` | `selection` | Threshold-shifted log-odds or linear vote with negative weights | [Paper](https://openreview.net/forum?id=x85kiYqL4y) · [API](./vote.py) · [BibTeX](#bibtex-kuang2026optimal) |
| `filtered_vote` | `selection` | Vote among the top-scoring retained candidates | [DeepConf](https://arxiv.org/abs/2508.15260) · [Verifier voting](https://arxiv.org/abs/2110.14168) · [API](./vote.py) · [BibTeX](#bibtex-fu2025deep) · [BibTeX](#bibtex-cobbe2021training) |

## Calibrated scalar-verifier aggregation

| `scorio.aggregate.[method_name]` | Returns | Method | Reference |
| --- | --- | --- | --- |
| `KDEVoteCalibration` | fitted state | Read-only KDE and binned-correctness calibrator | [Paper](https://openreview.net/forum?id=x85kiYqL4y) · [API](./calibration.py) · [BibTeX](#bibtex-kuang2026optimal) |
| `fit_kde_vote_calibration` | `KDEVoteCalibration` | Fit correct/incorrect score KDEs and a quantile-binned calibrator | [Paper](https://openreview.net/forum?id=x85kiYqL4y) · [API](./calibration.py) · [BibTeX](#bibtex-kuang2026optimal) |
| `kde_weighted_vote` | `selection` | Non-parametric density-ratio vote with a response-pool reliability term | [Paper](https://openreview.net/forum?id=x85kiYqL4y) · [API](./calibration.py) · [BibTeX](#bibtex-kuang2026optimal) |

## Confidence-guided aggregation

| `scorio.aggregate.[method_name]` | Returns | Method | Reference |
| --- | --- | --- | --- |
| `CGES_OTHER` | sentinel | Sentinel for a correct answer not yet observed | [Paper](https://arxiv.org/abs/2511.02603) · [API](./cges.py) · [BibTeX](#bibtex-aghazadeh2026cges) |
| `cges_vote` | `selection` | Answer with the largest Confidence-Guided Early Stopping score | [Paper](https://arxiv.org/abs/2511.02603) · [API](./cges.py) · [BibTeX](#bibtex-aghazadeh2026cges) |
| `cges_stop` | `stop` or `(stop, probability)` | Stop when a CGES score reaches a threshold | [Paper](https://arxiv.org/abs/2511.02603) · [API](./cges.py) · [BibTeX](#bibtex-aghazadeh2026cges) |

`cges_vote` and `cges_stop` ignore `CGES_OTHER` by default. Use
`allow_other=True` with `cges_vote` or `include_other=True` with `cges_stop` to
include it.

## Online early stopping

| `scorio.aggregate.[method_name]` | Returns | Method | Reference |
| --- | --- | --- | --- |
| `adaptive_consistency_stop` | `stop` or `(stop, probability)` | Top-two Beta approximation to Adaptive-Consistency | [Paper](https://arxiv.org/abs/2305.11860) · [API](./online.py) · [BibTeX](#bibtex-aggarwal2023sample) |
| `adaptive_consistency_dirichlet_stop` | `stop` or `(stop, probability)` | Full observed-support Dirichlet leader probability | [Paper](https://arxiv.org/abs/2305.11860) · [API](./online.py) · [BibTeX](#bibtex-aggarwal2023sample) |
| `adaptive_consistency_crp_stop` | `stop` or `(stop, probability)` | Finite-horizon CRP comparator that models unseen answers | [Paper](https://arxiv.org/abs/2305.11860) · [API](./online.py) · [BibTeX](#bibtex-aggarwal2023sample) |
| `esc_stop` | `stop` | Stop when every answer in the current sampling window agrees | [Paper](https://openreview.net/forum?id=ndR8Ytrzhh) · [API](./online.py) · [BibTeX](#bibtex-li2024escape) |
| `deepconf_stop_threshold` | `threshold` | Warmup quantile for retaining a target fraction of traces | [Paper](https://arxiv.org/abs/2508.15260) · [API](./online.py) · [BibTeX](#bibtex-fu2025deep) |
| `deepconf_online_stop` | `index` or `None` | Index of the first sliding-window endpoint below threshold | [Paper](https://arxiv.org/abs/2508.15260) · [API](./online.py) · [BibTeX](#bibtex-fu2025deep) |

The Adaptive-Consistency functions use the sampled answer prefix. `esc_stop`
uses the current fixed window. `deepconf_online_stop` returns the zero-based
token index at which generation should stop, or `None` to finish the trace.

## References

<a id="bibtex-adiwardana2020towards"></a>
### `adiwardana2020towards`

```bibtex
@article{adiwardana2020towards,
  title   = {Towards a Human-like Open-Domain Chatbot},
  author  = {Adiwardana, Daniel and Luong, Minh-Thang and So, David R. and Hall, Jamie and Fiedel, Noah and Thoppilan, Romal and Yang, Zi and Kulshreshtha, Apoorv and Nemade, Gaurav and Lu, Yifeng and Le, Quoc V.},
  journal = {arXiv preprint arXiv:2001.09977},
  year    = {2020},
  doi     = {10.48550/arXiv.2001.09977},
  url     = {https://arxiv.org/abs/2001.09977}
}
```

<a id="bibtex-wang2023selfconsistency"></a>
### `wang2023selfconsistency`

```bibtex
@inproceedings{wang2023selfconsistency,
  title     = {Self-Consistency Improves Chain of Thought Reasoning in Language Models},
  author    = {Wang, Xuezhi and Wei, Jason and Schuurmans, Dale and Le, Quoc V. and Chi, Ed H. and Narang, Sharan and Chowdhery, Aakanksha and Zhou, Denny},
  booktitle = {International Conference on Learning Representations},
  year      = {2023},
  doi       = {10.48550/arXiv.2203.11171},
  url       = {https://arxiv.org/abs/2203.11171}
}
```

<a id="bibtex-kang2025scalable"></a>
### `kang2025scalable`

```bibtex
@inproceedings{kang2025scalable,
  title     = {Scalable Best-of-{N} Selection for Large Language Models via Self-Certainty},
  author    = {Kang, Zhewei and Zhao, Xuandong and Song, Dawn},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2025},
  doi       = {10.48550/arXiv.2502.18581},
  url       = {https://arxiv.org/abs/2502.18581}
}
```

<a id="bibtex-fu2025deep"></a>
### `fu2025deep`

```bibtex
@inproceedings{fu2025deep,
  title={Deep Think with Confidence},
  author={Yichao Fu and Xuewei Wang and Hao Zhang and Yuandong Tian and Jiawei Zhao},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://openreview.net/forum?id=8LqHs0KIM7}
}
```

<a id="bibtex-malinin2021uncertainty"></a>
### `malinin2021uncertainty`

```bibtex
@inproceedings{malinin2021uncertainty,
  title     = {Uncertainty Estimation in Autoregressive Structured Prediction},
  author    = {Malinin, Andrey and Gales, Mark},
  booktitle = {International Conference on Learning Representations},
  year      = {2021},
  doi       = {10.48550/arXiv.2002.07650},
  url       = {https://arxiv.org/abs/2002.07650}
}
```

<a id="bibtex-kontoyiannis2014optimal"></a>
### `kontoyiannis2014optimal`

```bibtex
@article{kontoyiannis2014optimal,
  title   = {Optimal Lossless Data Compression: Non-Asymptotics and Asymptotics},
  author  = {Kontoyiannis, Ioannis and Verd{\'u}, Sergio},
  journal = {IEEE Transactions on Information Theory},
  volume  = {60},
  number  = {2},
  pages   = {777--795},
  year    = {2014},
  doi     = {10.1109/TIT.2013.2291007}
}
```

<a id="bibtex-hendrycks2017baseline"></a>
### `hendrycks2017baseline`

```bibtex
@inproceedings{hendrycks2017baseline,
  title     = {A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks},
  author    = {Hendrycks, Dan and Gimpel, Kevin},
  booktitle = {International Conference on Learning Representations},
  year      = {2017},
  doi       = {10.48550/arXiv.1610.02136},
  url       = {https://arxiv.org/abs/1610.02136}
}
```

<a id="bibtex-scheffer2001active"></a>
### `scheffer2001active`

```bibtex
@inproceedings{scheffer2001active,
  title     = {Active Hidden Markov Models for Information Extraction},
  author    = {Scheffer, Tobias and Decomain, Christian and Wrobel, Stefan},
  booktitle = {Advances in Intelligent Data Analysis},
  series    = {Lecture Notes in Computer Science},
  volume    = {2189},
  pages     = {309--318},
  year      = {2001},
  publisher = {Springer},
  doi       = {10.1007/3-540-44816-0_31}
}
```

<a id="bibtex-leang2026picsar"></a>
### `leang2026picsar`

```bibtex
@inproceedings{leang2026picsar,
  title     = {{PiCSAR}: Probabilistic Confidence Selection And Ranking for Reasoning Chains},
  author    = {Leang, Joshua Ong Jun and Zhao, Zheng and Gema, Aryo Pradipta and Yang, Sohee and Kwan, Wai-Chung and He, Xuanli and Li, Wenda and Minervini, Pasquale and Giunchiglia, Eleonora and Cohen, Shay B.},
  booktitle = {Findings of the Association for Computational Linguistics: ACL 2026},
  year      = {2026},
  doi       = {10.18653/v1/2026.findings-acl.1577},
  url       = {https://aclanthology.org/2026.findings-acl.1577/}
}
```

<a id="bibtex-lightman2023verify"></a>
### `lightman2023verify`

```bibtex
@article{lightman2023verify,
  title   = {Let's Verify Step by Step},
  author  = {Lightman, Hunter and Kosaraju, Vineet and Burda, Yura and Edwards, Harri and Baker, Bowen and Lee, Teddy and Leike, Jan and Schulman, John and Sutskever, Ilya and Cobbe, Karl},
  journal = {arXiv preprint arXiv:2305.20050},
  year    = {2023},
  doi     = {10.48550/arXiv.2305.20050},
  url     = {https://arxiv.org/abs/2305.20050}
}
```

<a id="bibtex-wang2024mathshepherd"></a>
### `wang2024mathshepherd`

```bibtex
@inproceedings{wang2024mathshepherd,
  title     = {Math-Shepherd: Verify and Reinforce {LLM}s Step-by-step without Human Annotations},
  author    = {Wang, Peiyi and Li, Lei and Shao, Zhihong and Xu, Runxin and Dai, Damai and Li, Yifei and Chen, Deli and Wu, Yu and Sui, Zhifang},
  booktitle = {Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages     = {9426--9439},
  year      = {2024},
  publisher = {Association for Computational Linguistics},
  doi       = {10.18653/v1/2024.acl-long.510},
  url       = {https://aclanthology.org/2024.acl-long.510/}
}
```

<a id="bibtex-cobbe2021training"></a>
### `cobbe2021training`

```bibtex
@article{cobbe2021training,
  title   = {Training Verifiers to Solve Math Word Problems},
  author  = {Cobbe, Karl and Kosaraju, Vineet and Bavarian, Mohammad and Chen, Mark and Jun, Heewoo and Kaiser, Lukasz and Plappert, Matthias and Tworek, Jerry and Hilton, Jacob and Nakano, Reiichiro and Hesse, Christopher and Schulman, John},
  journal = {arXiv preprint arXiv:2110.14168},
  year    = {2021},
  doi     = {10.48550/arXiv.2110.14168},
  url     = {https://arxiv.org/abs/2110.14168}
}
```

<a id="bibtex-rakhsha2025majority"></a>
### `rakhsha2025majority`

```bibtex
@inproceedings{rakhsha2025majority,
 author = {Rakhsha, Amin and Madan, Kanika and Zhang, Tianyu and Farahmand, Amir-massoud and Khasahmadi, Amir},
 booktitle = {Advances in Neural Information Processing Systems},
 doi = {10.52202/085713-1268},
 title = {Majority of the Bests: Improving Best-of-N via Bootstrapping},
 url = {https://proceedings.neurips.cc/paper_files/paper/2025/file/36556567e8437f137da23047309155dd-Paper-Conference.pdf},
 year = {2025}
}
```

<a id="bibtex-di2025best"></a>
### `di2025best`

```bibtex
@inproceedings{di2026bestofmajority,
  title={Best-of-Majority: Minimax-Optimal Strategy for Pass@k Inference Scaling},
  author={Qiwei Di and Kaixuan Ji and Xuheng Li and Heyang Zhao and Quanquan Gu},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://openreview.net/forum?id=a6CVQpjbXq}
}
```

<a id="bibtex-li2023making"></a>
### `li2023making`

```bibtex
@inproceedings{li2023making,
  title     = {Making Language Models Better Reasoners with Step-Aware Verifier},
  author    = {Li, Yifei and Lin, Zeqi and Zhang, Shizhuo and Fu, Qiang and Chen, Bei and Lou, Jian-Guang and Chen, Weizhu},
  booktitle = {Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages     = {5315--5333},
  year      = {2023},
  publisher = {Association for Computational Linguistics},
  doi       = {10.18653/v1/2023.acl-long.291},
  url       = {https://aclanthology.org/2023.acl-long.291/}
}
```

<a id="bibtex-taubenfeld2025confidence"></a>
### `taubenfeld2025confidence`

```bibtex
@inproceedings{taubenfeld2025confidence,
  title     = {Confidence Improves Self-Consistency in {LLM}s},
  author    = {Taubenfeld, Amir and Sheffer, Tom and Ofek, Eran and Feder, Amir and Goldstein, Ariel and Gekhman, Zorik and Yona, Gal},
  booktitle = {Findings of the Association for Computational Linguistics: ACL 2025},
  pages     = {20090--20111},
  year      = {2025},
  publisher = {Association for Computational Linguistics},
  doi       = {10.18653/v1/2025.findings-acl.1030},
  url       = {https://aclanthology.org/2025.findings-acl.1030/}
}
```

<a id="bibtex-kuang2026optimal"></a>
### `kuang2026optimal`

```bibtex
@inproceedings{kuang2026optimal,
  title     = {Optimal Aggregation of {LLM} and {PRM} Signals for Efficient Test-Time Scaling},
  author    = {Kuang, Peng and Wang, Yanli and Han, Xiaoyu and Liu, Yaowenqi and Xu, Kaidi and Wang, Haohan},
  booktitle = {International Conference on Learning Representations},
  year      = {2026},
  doi       = {10.48550/arXiv.2510.13918},
  url       = {https://openreview.net/forum?id=x85kiYqL4y}
}
```

<a id="bibtex-aghazadeh2026cges"></a>
### `aghazadeh2026cges`

```bibtex
@misc{aghazadeh2026cges,
      title={CGES: Confidence-Guided Early Stopping for Efficient and Accurate Self-Consistency}, 
      author={Ehsan Aghazadeh and Ahmad Ghasemi and Hedyeh Beyhaghi and Hossein Pishro-Nik},
      year={2026},
      eprint={2511.02603},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2511.02603}, 
}
```

<a id="bibtex-aggarwal2023sample"></a>
### `aggarwal2023sample`

```bibtex
@inproceedings{aggarwal2023sample,
    title = "Let{'}s Sample Step by Step: Adaptive-Consistency for Efficient Reasoning and Coding with {LLM}s",
    author = "Aggarwal, Pranjal  and
      Madaan, Aman  and
      Yang, Yiming  and
      Mausam",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2023",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.emnlp-main.761/",
    doi = "10.18653/v1/2023.emnlp-main.761"
}
```

<a id="bibtex-li2024escape"></a>
### `li2024escape`

```bibtex
@inproceedings{li2024escape,
  title     = {Escape Sky-high Cost: Early-stopping Self-Consistency for Multi-step Reasoning},
  author    = {Li, Yiwei and Yuan, Peiwen and Feng, Shaoxiong and Pan, Boyuan and Wang, Xinglin and Sun, Bin and Wang, Heda and Li, Kan},
  booktitle = {International Conference on Learning Representations},
  year      = {2024},
  url       = {https://openreview.net/forum?id=ndR8Ytrzhh}
}
```
