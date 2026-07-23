# `scorio.rank` Method References

Methods generally accept a binary model-by-question response matrix
or model-by-question-by-trial tensor; `bayes` additionally supports integer
categorical outcomes. Several methods adapt their source model by constructing
pairwise or setwise outcomes from each question-trial event.

We introduce the `scorio.rank` APIs in
[Ranking Reasoning LLMs under Test-Time Scaling](https://aclanthology.org/2026.acl-long.1544/)
(ACL 2026).

If you use these ranking APIs, please cite this work:

```bibtex
@inproceedings{hariri-etal-2026-ranking,
  title     = {Ranking Reasoning {LLM}s under Test-Time Scaling},
  author    = {Hariri, Mohsen and Hinczewski, Michael and Ma, Jing and Chaudhary, Vipin},
  booktitle = {Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  month     = jul,
  year      = {2026},
  address   = {San Diego, California, United States},
  publisher = {Association for Computational Linguistics},
  pages     = {33437--33478},
  doi       = {10.18653/v1/2026.acl-long.1544},
  url       = {https://aclanthology.org/2026.acl-long.1544/}
}
```

Notes:

- Unregularized Bradley-Terry and decisive/composite Luce maximum-likelihood
  fits require a strongly connected directed win graph. Their tie-model
  extensions require the analogous directed win/tie or winner/co-winner support
  graph. These APIs raise when no finite MLE exists; use the corresponding MAP
  estimator for separated data.
- `plackett_luce` is the decisive pairwise Bradley-Terry restriction of Luce
  and drops tied outcomes; `bradley_terry_luce` is a rank-broken composite
  likelihood. Only `davidson_luce` directly models a tied winner set as one
  normalized event. The corresponding MAP APIs use the same constructions.
- Elo and TrueSkill remain sequential across question-trial events, but all
  matches within one event use a common pre-event state so results do not depend
  on arbitrary model index order. `trueskill` is an induced two-player
  approximation, not the original paper's full multiplayer factor graph.
- `thompson` is an offline posterior expected-rank estimator using Beta posterior
  samples. It is inspired by Thompson sampling.
- `inverse_difficulty` is a question-difficulty reweighting heuristic.
- `mg_pass_at_k` follows the paper's discrete threshold-grid approximation to
  the G-Pass AUC. For odd `k`, that approximation does not attain exactly one
  even for a perfect model.
- `dynamic_irt` defaults to a static Rasch fit. Its `growth` and `state_space`
  variants are longitudinal, reinterpret the trial axis as ordered time, and
  require `assume_time_axis=True`, at least two time points, and positive
  slope/random-walk regularization. Time is normalized to `[0, 1]`, so a growth
  slope is change over the observed time span; state-space paths also use a weak
  Gaussian anchor on their initial level.
- Joint Rasch fits reject complete or quasi-separated response patterns rather
  than returning finite boundary proxies. Joint 2PL/3PL fits require positive
  discrimination regularization to identify scale and run an additional
  label-equivariant sensitivity check for weak/nonconvex fits; ambiguous or
  saturated solutions raise. Rasch MML instead reports extended item estimates
  (`-inf` for all-correct, `+inf` for all-wrong) and retains finite EAP scores.
- For this binary complete-event construction, `nash` has payoff
  `A[i,j] = accuracy[i] - accuracy[j]`; its label-invariant equilibrium is
  uniform over all maximum-accuracy models.
- `ranked_pairs` resolves otherwise equal-strength victories by winning-vote
  count and then model indices. `kemeny_young` returns only a proven MILP
  optimum and raises if the configured solver limit is reached first.

## Evaluation Metric-Based Ranking Methods

| `scorio.rank.[method_name]` | Paper | Reference |
| --- | --- | --- |
| `avg` | - | [API](./eval_ranking.py) |
| `bayes` | [Don't Pass@k: A Bayesian Framework for Large Language Model Evaluation](https://openreview.net/forum?id=PTXi3Ef4sT) | [API](./eval_ranking.py) · [BibTeX](#bibtex-hariri2026don) |
| `pass_at_k` | [Evaluating Large Language Models Trained on Code](https://arxiv.org/abs/2107.03374) | [API](./eval_ranking.py) · [BibTeX](#bibtex-chen2021evaluating) |
| `pass_hat_k` | tau-bench [`pass^k`](https://arxiv.org/abs/2406.12045): all `k` draws succeed | [API](./eval_ranking.py) · [BibTeX](#bibtex-yao2024taubench) |
| `g_pass_at_k_tau` | [Are Your LLMs Capable of Stable Reasoning?](https://aclanthology.org/2025.findings-acl.905/) | [API](./eval_ranking.py) · [BibTeX](#bibtex-liu2025stable-reasoning) |
| `mg_pass_at_k` | [Are Your LLMs Capable of Stable Reasoning?](https://aclanthology.org/2025.findings-acl.905/) | [API](./eval_ranking.py) · [BibTeX](#bibtex-liu2025stable-reasoning) |

## Paired-Comparison Probabilistic Models

| `scorio.rank.[method_name]` | Paper | Reference |
| --- | --- | --- |
| `bradley_terry` | [Rank Analysis of Incomplete Block Designs: The Method of Paired Comparisons](https://doi.org/10.1093/biomet/39.3-4.324) | [API](./bradley_terry.py) · [BibTeX](#bibtex-bradleyterry1952) |
| `bradley_terry_map` | [Efficient Bayesian Inference for Generalized Bradley--Terry Models](https://doi.org/10.1080/10618600.2012.638220) | [API](./bradley_terry.py) · [BibTeX](#bibtex-carondoucet2012) |
| `bradley_terry_davidson` | [On Extending the Bradley--Terry Model to Accommodate Ties in Paired Comparison Experiments](https://doi.org/10.1080/01621459.1970.10481082) | [API](./bradley_terry.py) · [BibTeX](#bibtex-davidson1970bties) |
| `bradley_terry_davidson_map` | [On Extending the Bradley--Terry Model to Accommodate Ties in Paired Comparison Experiments](https://doi.org/10.1080/01621459.1970.10481082) · [Efficient Bayesian Inference for Generalized Bradley--Terry Models](https://doi.org/10.1080/10618600.2012.638220) | [API](./bradley_terry.py) · [BibTeX](#bibtex-davidson1970bties) · [BibTeX](#bibtex-carondoucet2012) |
| `rao_kupper` | [Ties in Paired-Comparison Experiments: A Generalization of the Bradley--Terry Model](https://doi.org/10.1080/01621459.1967.10482901) | [API](./bradley_terry.py) · [BibTeX](#bibtex-rao1967ties) |
| `rao_kupper_map` | [Ties in Paired-Comparison Experiments: A Generalization of the Bradley--Terry Model](https://doi.org/10.1080/01621459.1967.10482901) · [Efficient Bayesian Inference for Generalized Bradley--Terry Models](https://doi.org/10.1080/10618600.2012.638220) | [API](./bradley_terry.py) · [BibTeX](#bibtex-rao1967ties) · [BibTeX](#bibtex-carondoucet2012) |


## Pointwise Methods (Accuracy-Based)

| `scorio.rank.[method_name]` | Paper | Reference |
| --- | --- | --- |
| `inverse_difficulty` | Question-difficulty reweighting heuristic | [API](./pointwise.py) |

## Pairwise Rating Systems

| `scorio.rank.[method_name]` | Paper | Reference |
| --- | --- | --- |
| `elo` | [The Rating of Chessplayers, Past and Present](https://archive.org/details/ratingofchesspla0000eloa) | [API](./pairwise.py) · [BibTeX](#bibtex-elo1978) |
| `trueskill` | [TrueSkill(TM): A Bayesian Skill Rating System](https://proceedings.neurips.cc/paper_files/paper/2006/file/f44ee263952e65b3610b8ba51229d1f9-Paper.pdf) | [API](./pairwise.py) · [BibTeX](#bibtex-herbrich2006trueskill) |
| `glicko` | [Parameter Estimation in Large Dynamic Paired Comparison Experiments](https://doi.org/10.1111/1467-9876.00159) | [API](./pairwise.py) · [BibTeX](#bibtex-glickman1999) |

## Bayesian Methods

| `scorio.rank.[method_name]` | Paper | Reference |
| --- | --- | --- |
| `thompson` | Posterior expected rank via Beta sampling, inspired by [Thompson sampling](https://doi.org/10.1093/biomet/25.3-4.285) | [API](./bayesian.py) · [BibTeX](#bibtex-thompson1933) · [BibTeX](#bibtex-russo2018ts) |
| `bayesian_mcmc` | [Rank Analysis of Incomplete Block Designs: The Method of Paired Comparisons](https://doi.org/10.1093/biomet/39.3-4.324) · [Equation of State Calculations by Fast Computing Machines](https://doi.org/10.1063/1.1699114) · [Monte Carlo Sampling Methods Using Markov Chains and Their Applications](https://doi.org/10.1093/biomet/57.1.97) | [API](./bayesian.py) · [BibTeX](#bibtex-bradleyterry1952) · [BibTeX](#bibtex-metropolis1953) · [BibTeX](#bibtex-hastings1970) |

## Voting Methods

| `scorio.rank.[method_name]` | Paper | Reference |
| --- | --- | --- |
| `borda` | [Mémoire sur les élections au scrutin](https://webusers.imj-prg.fr/~alexandre.guilbaud/LX2U1/Borda_Memoire_sur_les_elections_au_scrutin_MARS_1781_extrait.pdf) | [API](./voting.py) · [BibTeX](#bibtex-borda1781) |
| `copeland` | [A Reasonable Social Welfare Function](https://bibbase.org/network/publication/copeland-areasonablesocialwelfarefunction-1951) | [API](./voting.py) · [BibTeX](#bibtex-copeland1951) |
| `win_rate` | Micro-averaged strict question-level pairwise win fraction over decisive outcomes | [API](./voting.py) |
| `minimax` | Simpson-Kramer minimax rule (computational treatment in [Handbook of Computational Social Choice](https://doi.org/10.1017/CBO9781107446984)) | [API](./voting.py) · [BibTeX](#bibtex-brandt2016compsocchoice) |
| `schulze` | [A new monotonic, clone-independent, reversal symmetric, and Condorcet-consistent single-winner election method](https://doi.org/10.1007/s00355-010-0475-4) | [API](./voting.py) · [BibTeX](#bibtex-schulze2010) |
| `ranked_pairs` | [Independence of clones as a criterion for voting rules](https://doi.org/10.1007/BF00433944) | [API](./voting.py) · [BibTeX](#bibtex-tideman1987) |
| `kemeny_young` | [Mathematics without Numbers](https://www.jstor.org/stable/20026581) · [Extending Condorcet's rule](https://doi.org/10.1016/0022-0531(77)90012-6) | [API](./voting.py) · [BibTeX](#bibtex-kemeny1959) · [BibTeX](#bibtex-young1977) |
| `nanson` | [Methods of Election](https://www.biodiversitylibrary.org/itemdetails/106382) | [API](./voting.py) · [BibTeX](#bibtex-nanson1883) · [BibTeX](#bibtex-brandt2016compsocchoice) |
| `baldwin` | [The technique of the Nanson preferential majority system of election](https://www.biodiversitylibrary.org/part/302140) | [API](./voting.py) · [BibTeX](#bibtex-baldwin1926) · [BibTeX](#bibtex-brandt2016compsocchoice) |
| `majority_judgment` | [Majority Judgment: Measuring, Ranking, and Electing](https://doi.org/10.7551/mitpress/9780262015134.001.0001) | [API](./voting.py) · [BibTeX](#bibtex-balinskilaraki2011) |



## Item Response Theory Methods

| `scorio.rank.[method_name]` | Paper | Reference |
| --- | --- | --- |
| `rasch` | [Probabilistic Models for Some Intelligence and Attainment Tests](https://archive.org/details/probabilisticmod0000rasc) | [API](./irt.py) · [BibTeX](#bibtex-rasch1960rasch) |
| `rasch_map` | [Bayes Modal Estimation in Item Response Models](https://doi.org/10.1007/BF02293979) | [API](./irt.py) · [BibTeX](#bibtex-mislevy1986) |
| `rasch_mml` | [Marginal Maximum Likelihood Estimation of Item Parameters: Application of an EM Algorithm](https://doi.org/10.1007/BF02293801) | [API](./irt.py) · [BibTeX](#bibtex-bockaitkin1981) |
| `rasch_mml_credible` | [Marginal Maximum Likelihood Estimation of Item Parameters: Application of an EM Algorithm](https://doi.org/10.1007/BF02293801) · [Bayes Modal Estimation in Item Response Models](https://doi.org/10.1007/BF02293979) | [API](./irt.py) · [BibTeX](#bibtex-bockaitkin1981) · [BibTeX](#bibtex-mislevy1986) |
| `rasch_2pl` | [Some Latent Trait Models and Their Use in Inferring an Examinee's Ability](https://faculty.ucmerced.edu/jvevea/classes/290_21/readings/week%209/Birnbaum.pdf) | [API](./irt.py) · [BibTeX](#bibtex-birnbaum1968latent) |
| `rasch_2pl_map` | [Some Latent Trait Models and Their Use in Inferring an Examinee's Ability](https://faculty.ucmerced.edu/jvevea/classes/290_21/readings/week%209/Birnbaum.pdf) · [Bayes Modal Estimation in Item Response Models](https://doi.org/10.1007/BF02293979) | [API](./irt.py) · [BibTeX](#bibtex-birnbaum1968latent) · [BibTeX](#bibtex-mislevy1986) |
| `rasch_3pl` | [Some Latent Trait Models and Their Use in Inferring an Examinee's Ability](https://faculty.ucmerced.edu/jvevea/classes/290_21/readings/week%209/Birnbaum.pdf) | [API](./irt.py) · [BibTeX](#bibtex-birnbaum1968latent) |
| `rasch_3pl_map` | [Some Latent Trait Models and Their Use in Inferring an Examinee's Ability](https://faculty.ucmerced.edu/jvevea/classes/290_21/readings/week%209/Birnbaum.pdf) · [Bayes Modal Estimation in Item Response Models](https://doi.org/10.1007/BF02293979) | [API](./irt.py) · [BibTeX](#bibtex-birnbaum1968latent) · [BibTeX](#bibtex-mislevy1986) |
| `dynamic_irt` | [A Dynamic Generalization of the Rasch Model](https://doi.org/10.1007/BF02294648) · [On Longitudinal Item Response Theory Models: A Didactic](https://doi.org/10.3102/1076998619882026) | [API](./irt.py) · [BibTeX](#bibtex-verhelst1993dynamicrasch) · [BibTeX](#bibtex-wang2019longitudinalirt) |
| `mirt` | [mirt: A Multidimensional Item Response Theory Package for the R Environment](https://doi.org/10.18637/jss.v048.i06) · [Multidimensional Item Response Theory](https://doi.org/10.1007/978-0-387-89976-3) · [Bock--Aitkin MML-EM](https://doi.org/10.1007/BF02293801) | [API](./irt.py) · [BibTeX](#bibtex-chalmers2012mirt) · [BibTeX](#bibtex-reckase2009mirt) · [BibTeX](#bibtex-bockaitkin1981) |

## Graph-Based Methods

| `scorio.rank.[method_name]` | Paper | Reference |
| --- | --- | --- |
| `pagerank` | [The PageRank Citation Ranking: Bringing Order to the Web](https://ilpubs.stanford.edu/422/) | [API](./graph.py) · [BibTeX](#bibtex-page1999pagerank) |
| `spectral` | [The Perron–Frobenius theorem and the ranking of football teams](https://doi.org/10.1137/1035004) | [API](./graph.py) · [BibTeX](#bibtex-keener1993perron) |
| `alpharank` | [α-Rank: Multi-Agent Evaluation by Evolution](https://doi.org/10.1038/s41598-019-45619-9) | [API](./graph.py) · [BibTeX](#bibtex-omidshafiei2019alpharank) |
| `nash` | [Open-ended Learning in Symmetric Zero-sum Games](https://proceedings.mlr.press/v97/balduzzi19a.html) · [Re-evaluating Evaluation](https://proceedings.neurips.cc/paper_files/paper/2018/file/cdf1035c34ec380218a8cc9a43d438f9-Paper.pdf) | [API](./graph.py) · [BibTeX](#bibtex-balduzzi2019openended) · [BibTeX](#bibtex-balduzzi2018reevaluating) |
| `rank_centrality` | [Rank Centrality: Ranking from Pairwise Comparisons](https://doi.org/10.1287/opre.2016.1534) | [API](./rank_centrality.py) · [BibTeX](#bibtex-negahban2017rankcentrality) |

## Seriation-Based Methods

| `scorio.rank.[method_name]` | Paper | Reference |
| --- | --- | --- |
| `serial_rank` | [Spectral Ranking Using Seriation](https://jmlr.org/papers/v17/16-035.html) | [API](./serial_rank.py) · [BibTeX](#bibtex-fogel2016serialrank) |

## Hodge-Theoretic Methods

| `scorio.rank.[method_name]` | Paper | Reference |
| --- | --- | --- |
| `hodge_rank` | [Statistical Ranking and Combinatorial Hodge Theory](https://arxiv.org/abs/0811.1067) | [API](./hodge_rank.py) · [BibTeX](#bibtex-jiang2009hodgerank) |

## Listwise and Setwise Choice Models

| `scorio.rank.[method_name]` | Paper | Reference |
| --- | --- | --- |
| `plackett_luce` | [The Analysis of Permutations](https://doi.org/10.2307/2346567) · [MM Algorithms for Generalized Bradley--Terry Models](https://doi.org/10.1214/aos/1079120141) | [API](./listwise.py) · [BibTeX](#bibtex-plackett1975permutations) · [BibTeX](#bibtex-hunter2004mmbradleyterry) |
| `plackett_luce_map` | [Individual Choice Behavior: A Theoretical Analysis](https://archive.org/details/individualchoice0000luce) · [Efficient Bayesian Inference for Generalized Bradley--Terry Models](https://doi.org/10.1080/10618600.2012.638220) | [API](./listwise.py) · [BibTeX](#bibtex-luce1959choice) · [BibTeX](#bibtex-carondoucet2012) |
| `davidson_luce` | [Davidson--Luce Model for Multi-item Choice with Ties](https://arxiv.org/abs/1909.07123) | [API](./listwise.py) · [BibTeX](#bibtex-firth2019davidsonluce) |
| `davidson_luce_map` | [Davidson--Luce Model for Multi-item Choice with Ties](https://arxiv.org/abs/1909.07123) · [Efficient Bayesian Inference for Generalized Bradley--Terry Models](https://doi.org/10.1080/10618600.2012.638220) | [API](./listwise.py) · [BibTeX](#bibtex-firth2019davidsonluce) · [BibTeX](#bibtex-carondoucet2012) |
| `bradley_terry_luce` | [Individual Choice Behavior: A Theoretical Analysis](https://archive.org/details/individualchoice0000luce) · [Generalized Linear Models](https://doi.org/10.1007/978-1-4899-3242-6) | [API](./listwise.py) · [BibTeX](#bibtex-luce1959choice) · [BibTeX](#bibtex-mccullaghnelder1989glm) |
| `bradley_terry_luce_map` | [Individual Choice Behavior: A Theoretical Analysis](https://archive.org/details/individualchoice0000luce) · [Generalized Linear Models](https://doi.org/10.1007/978-1-4899-3242-6) · [Efficient Bayesian Inference for Generalized Bradley--Terry Models](https://doi.org/10.1080/10618600.2012.638220) | [API](./listwise.py) · [BibTeX](#bibtex-luce1959choice) · [BibTeX](#bibtex-mccullaghnelder1989glm) · [BibTeX](#bibtex-carondoucet2012) |



## References

### Evaluation Metrics

<a id="bibtex-chen2021evaluating"></a>
### `chen2021evaluating`

```bibtex
@article{chen2021evaluating,
  title={Evaluating Large Language Models Trained on Code},
  author={Chen, Mark and Tworek, Jerry and Jun, Heewoo and Yuan, Qiming and Pinto, Henrique Ponde de Oliveira and Kaplan, Jared and Edwards, Harri and Burda, Yuri and Joseph, Nicholas and Brockman, Greg and others},
  journal={arXiv preprint arXiv:2107.03374},
  year={2021},
  doi={10.48550/arXiv.2107.03374},
  url={https://arxiv.org/abs/2107.03374}
}
```

<a id="bibtex-yao2024taubench"></a>
### `yao2024taubench`

```bibtex
@misc{yao2024taubench,
      title={{$\tau$}-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains},
      author={Shunyu Yao and Noah Shinn and Pedram Razavi and Karthik Narasimhan},
      year={2024},
      eprint={2406.12045},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      doi={10.48550/arXiv.2406.12045},
      url={https://arxiv.org/abs/2406.12045},
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

<a id="bibtex-hariri2026don"></a>
### `hariri2026don`

```bibtex
@inproceedings{hariri2026don,
  title     = {Don't Pass@k: A Bayesian Framework for Large Language Model Evaluation},
  author    = {Hariri, Mohsen and Samandar, Amirhossein and Hinczewski, Michael and Chaudhary, Vipin},
  booktitle = {The Fourteenth International Conference on Learning Representations},
  year      = {2026},
  url       = {https://openreview.net/forum?id=PTXi3Ef4sT},
  doi       = {10.48550/arXiv.2510.04265}
}
```

### Bayesian and Pairwise Probabilistic Methods

<a id="bibtex-thompson1933"></a>
### `Thompson1933`

```bibtex
@article{Thompson1933,
  title   = {On the Likelihood that One Unknown Probability Exceeds Another in View of the Evidence of Two Samples},
  author  = {Thompson, William R.},
  journal = {Biometrika},
  volume  = {25},
  number  = {3-4},
  pages   = {285--294},
  year    = {1933},
  doi     = {10.1093/biomet/25.3-4.285}
}
```

<a id="bibtex-russo2018ts"></a>
### `Russo2018TS`

```bibtex
@article{Russo2018TS,
  title   = {A Tutorial on Thompson Sampling},
  author  = {Russo, Daniel J. and Van Roy, Benjamin and Kazerouni, Abbas and Osband, Ian and Wen, Zheng},
  journal = {Foundations and Trends in Machine Learning},
  volume  = {11},
  number  = {1},
  pages   = {1--96},
  year    = {2018},
  doi     = {10.1561/2200000070}
}
```

<a id="bibtex-gelman2013bda"></a>
### `Gelman2013BDA`

```bibtex
@book{Gelman2013BDA,
  title     = {Bayesian Data Analysis},
  author    = {Gelman, Andrew and Carlin, John B. and Stern, Hal S. and Dunson, David B. and Vehtari, Aki and Rubin, Donald B.},
  edition   = {3},
  publisher = {CRC Press},
  year      = {2013},
  doi       = {10.1201/b16018}
}
```

<a id="bibtex-bradleyterry1952"></a>
### `BradleyTerry1952`

```bibtex
@article{BradleyTerry1952,
  author  = {Bradley, Ralph Allan and Terry, Milton E.},
  title   = {Rank Analysis of Incomplete Block Designs: The Method of Paired Comparisons},
  journal = {Biometrika},
  volume  = {39},
  number  = {3-4},
  pages   = {324--345},
  year    = {1952},
  doi     = {10.1093/biomet/39.3-4.324}
}
```

<a id="bibtex-metropolis1953"></a>
### `Metropolis1953`

```bibtex
@article{Metropolis1953,
  title   = {Equation of State Calculations by Fast Computing Machines},
  author  = {Metropolis, Nicholas and Rosenbluth, Arianna W. and Rosenbluth, Marshall N. and Teller, Augusta H. and Teller, Edward},
  journal = {The Journal of Chemical Physics},
  volume  = {21},
  number  = {6},
  pages   = {1087--1092},
  year    = {1953},
  doi     = {10.1063/1.1699114}
}
```

<a id="bibtex-hastings1970"></a>
### `Hastings1970`

```bibtex
@article{Hastings1970,
  title   = {Monte Carlo Sampling Methods Using Markov Chains and Their Applications},
  author  = {Hastings, W. K.},
  journal = {Biometrika},
  volume  = {57},
  number  = {1},
  pages   = {97--109},
  year    = {1970},
  doi     = {10.1093/biomet/57.1.97}
}
```

<a id="bibtex-carondoucet2012"></a>
### `CaronDoucet2012`

```bibtex
@article{CaronDoucet2012,
  title   = {Efficient Bayesian Inference for Generalized Bradley--Terry Models},
  author  = {Caron, Fran{\c{c}}ois and Doucet, Arnaud},
  journal = {Journal of Computational and Graphical Statistics},
  volume  = {21},
  number  = {1},
  pages   = {174--196},
  year    = {2012},
  doi     = {10.1080/10618600.2012.638220}
}
```

### Voting and Social Choice Methods

<a id="bibtex-borda1781"></a>
### `Borda1781`

```bibtex
@misc{Borda1781,
  author       = {de Borda, Jean-Charles},
  title        = {M{\'e}moire sur les {\'e}lections au scrutin},
  howpublished = {Histoire de l'Acad{\'e}mie Royale des Sciences, Paris},
  pages        = {657--665},
  year         = {1781},
  note         = {Often cited as appearing in the 1781 volume (issued in 1784) of the Histoire/M{\'e}moires of the Acad{\'e}mie.},
  url       = {https://webusers.imj-prg.fr/~alexandre.guilbaud/LX2U1/Borda_Memoire_sur_les_elections_au_scrutin_MARS_1781_extrait.pdf}
}
```

<a id="bibtex-brandt2016compsocchoice"></a>
### `brandt2016compsocchoice`

```bibtex
@book{brandt2016compsocchoice,
  title     = {Handbook of Computational Social Choice},
  editor    = {Brandt, Felix and Conitzer, Vincent and Endriss, Ulle and Lang, J{\'{e}}r{\^{o}}me and Procaccia, Ariel D.},
  publisher = {Cambridge University Press},
  year      = {2016},
  doi       = {10.1017/CBO9781107446984},
  url       = {https://doi.org/10.1017/CBO9781107446984},
  isbn      = {9781107446984}
}
```

<a id="bibtex-copeland1951"></a>
### `Copeland1951`

```bibtex
@misc{Copeland1951,
  author       = {Copeland, Arthur H.},
  title        = {A Reasonable Social Welfare Function},
  howpublished = {Seminar on Mathematics in Social Sciences},
  year         = {1951},
  note         = {University of Michigan.},
  url       = {https://bibbase.org/network/publication/copeland-areasonablesocialwelfarefunction-1951}
}
```

<a id="bibtex-schulze2010"></a>
### `Schulze2010`

```bibtex
@article{Schulze2010,
  author  = {Schulze, Markus},
  title   = {A new monotonic, clone-independent, reversal symmetric, and {C}ondorcet-consistent single-winner election method},
  journal = {Social Choice and Welfare},
  volume  = {36},
  number  = {2},
  pages   = {267--303},
  year    = {2011},
  doi     = {10.1007/s00355-010-0475-4}
}
```

<a id="bibtex-tideman1987"></a>
### `Tideman1987`

```bibtex
@article{Tideman1987,
  author  = {Tideman, T. N.},
  title   = {Independence of clones as a criterion for voting rules},
  journal = {Social Choice and Welfare},
  volume  = {4},
  number  = {3},
  pages   = {185--206},
  year    = {1987},
  doi     = {10.1007/BF00433944}
}
```

<a id="bibtex-young1977"></a>
### `Young1977`

```bibtex
@article{Young1977,
  author  = {Young, H. P.},
  title   = {Extending {C}ondorcet's rule},
  journal = {Journal of Economic Theory},
  volume  = {16},
  number  = {2},
  pages   = {335--353},
  year    = {1977},
  doi     = {10.1016/0022-0531(77)90012-6}
}
```

<a id="bibtex-kemeny1959"></a>
### `Kemeny1959`

```bibtex
@article{Kemeny1959,
  author  = {Kemeny, John G.},
  title   = {Mathematics without Numbers},
  journal = {Daedalus},
  volume  = {88},
  number  = {4},
  pages   = {577--591},
  year    = {1959},
  url     = {https://www.jstor.org/stable/20026581}
}
```

<a id="bibtex-nanson1883"></a>
### `Nanson1883`

```bibtex
@article{Nanson1883,
  author  = {Nanson, E. J.},
  title   = {Methods of Election},
  journal = {Transactions and Proceedings of the Royal Society of Victoria},
  volume  = {19},
  pages   = {197--240},
  year    = {1883},
  url     = {https://www.biodiversitylibrary.org/itemdetails/106382},
  note    = {Often cited as 1882 in secondary sources.}
}
```

<a id="bibtex-baldwin1926"></a>
### `Baldwin1926`

```bibtex
@article{Baldwin1926,
  author  = {Baldwin, J. M.},
  title   = {The technique of the Nanson preferential majority system of election},
  journal = {Proceedings of the Royal Society of Victoria, New Series},
  volume  = {39},
  number  = {1},
  pages   = {42--52},
  year    = {1926},
  url     = {https://www.biodiversitylibrary.org/part/302140}
}
```

<a id="bibtex-balinskilaraki2011"></a>
### `BalinskiLaraki2011`

```bibtex
@book{BalinskiLaraki2011,
  author    = {Balinski, Michel and Laraki, Rida},
  title     = {Majority Judgment: Measuring, Ranking, and Electing},
  publisher = {The MIT Press},
  year      = {2011},
  doi       = {10.7551/mitpress/9780262015134.001.0001},
  isbn      = {9780262295604}
}
```

### Pairwise Rating and Choice Models

<a id="bibtex-davidson1970bties"></a>
### `davidson1970bties`

```bibtex
@article{davidson1970bties,
  title   = {On Extending the Bradley--Terry Model to Accommodate Ties in Paired Comparison Experiments},
  author  = {Davidson, Roger R.},
  journal = {Journal of the American Statistical Association},
  volume  = {65},
  number  = {329},
  pages   = {317--328},
  year    = {1970},
  doi     = {10.1080/01621459.1970.10481082},
  url     = {https://doi.org/10.1080/01621459.1970.10481082}
}
```

<a id="bibtex-rao1967ties"></a>
### `rao1967ties`

```bibtex
@article{rao1967ties,
  title   = {Ties in Paired-Comparison Experiments: A Generalization of the Bradley--Terry Model},
  author  = {Rao, P. V. and Kupper, L. L.},
  journal = {Journal of the American Statistical Association},
  volume  = {62},
  number  = {317},
  pages   = {194--204},
  year    = {1967},
  doi     = {10.1080/01621459.1967.10482901},
  url     = {https://doi.org/10.1080/01621459.1967.10482901}
}
```

<a id="bibtex-elo1978"></a>
### `Elo1978`

```bibtex
@book{Elo1978,
  author    = {Elo, Arpad E.},
  title     = {The Rating of Chessplayers, Past and Present},
  publisher = {Arco Publishing},
  year      = {1978},
  isbn      = {0668047216},
  url       = {https://archive.org/details/ratingofchesspla0000eloa}
}
```

<a id="bibtex-glickman1999"></a>
### `Glickman1999`

```bibtex
@article{Glickman1999,
  author  = {Glickman, Mark E.},
  title   = {Parameter Estimation in Large Dynamic Paired Comparison Experiments},
  journal = {Journal of the Royal Statistical Society: Series C (Applied Statistics)},
  volume  = {48},
  number  = {3},
  pages   = {377--394},
  year    = {1999},
  doi     = {10.1111/1467-9876.00159}
}
```

<a id="bibtex-herbrich2006trueskill"></a>
### `herbrich2006trueskill`

```bibtex
@inproceedings{herbrich2006trueskill,
  author    = {Herbrich, Ralf and Minka, Tom and Graepel, Thore},
  title     = {TrueSkill\({}^{\mbox{TM}}\): {A} {B}ayesian Skill Rating System},
  booktitle = {Advances in Neural Information Processing Systems 19 (NeurIPS 2006)},
  pages     = {569--576},
  year      = {2006},
  publisher = {MIT Press},
  url       = {https://proceedings.neurips.cc/paper/2006/hash/f44ee263952e65b3610b8ba51229d1f9-Abstract.html}
}
```

<a id="bibtex-plackett1975permutations"></a>
### `plackett1975permutations`

```bibtex
@article{plackett1975permutations,
  title   = {The Analysis of Permutations},
  author  = {Plackett, R. L.},
  journal = {Applied Statistics},
  volume  = {24},
  number  = {2},
  pages   = {193--202},
  year    = {1975},
  doi     = {10.2307/2346567},
  url     = {https://doi.org/10.2307/2346567}
}
```

<a id="bibtex-luce1959choice"></a>
### `luce1959choice`

```bibtex
@book{luce1959choice,
  title     = {Individual Choice Behavior: A Theoretical Analysis},
  author    = {Luce, R. Duncan},
  year      = {1959},
  publisher = {John Wiley \& Sons},
  url       = {https://archive.org/details/individualchoice0000luce}
}
```

<a id="bibtex-hunter2004mmbradleyterry"></a>
### `hunter2004mmbradleyterry`

```bibtex
@article{hunter2004mmbradleyterry,
  title   = {{MM} Algorithms for Generalized Bradley--Terry Models},
  author  = {Hunter, David R.},
  journal = {The Annals of Statistics},
  volume  = {32},
  number  = {1},
  pages   = {384--406},
  year    = {2004},
  doi     = {10.1214/aos/1079120141},
  url     = {https://doi.org/10.1214/aos/1079120141}
}
```

<a id="bibtex-firth2019davidsonluce"></a>
### `firth2019davidsonluce`

```bibtex
@misc{firth2019davidsonluce,
  title         = {Davidson--Luce Model for Multi-item Choice with Ties},
  author        = {David Firth and Ioannis Kosmidis and Heather Turner},
  year          = {2019},
  eprint        = {1909.07123},
  archivePrefix = {arXiv},
  primaryClass  = {stat.ME},
  doi           = {10.48550/arXiv.1909.07123},
  url           = {https://arxiv.org/abs/1909.07123}
}
```

<a id="bibtex-mccullaghnelder1989glm"></a>
### `mccullaghnelder1989glm`

```bibtex
@book{mccullaghnelder1989glm,
  title     = {Generalized Linear Models},
  author    = {McCullagh, P. and Nelder, J. A.},
  publisher = {Springer},
  year      = {1989},
  doi       = {10.1007/978-1-4899-3242-6},
  url       = {https://doi.org/10.1007/978-1-4899-3242-6},
  isbn      = {9781489932426}
}
```

<a id="bibtex-deboeckwilson2004explanatory"></a>
### `deboeckwilson2004explanatory`

```bibtex
@book{deboeckwilson2004explanatory,
  title     = {Explanatory Item Response Models},
  editor    = {De Boeck, Paul and Wilson, Mark},
  publisher = {Springer},
  year      = {2004},
  doi       = {10.1007/978-1-4757-3990-9},
  url       = {https://doi.org/10.1007/978-1-4757-3990-9},
  isbn      = {9781475739909}
}
```

### IRT and Latent-Trait Methods

<a id="bibtex-rasch1960rasch"></a>
### `rasch1960rasch`

```bibtex
@book{rasch1960rasch,
  title     = {Probabilistic Models for Some Intelligence and Attainment Tests},
  author    = {Rasch, Georg},
  year      = {1960},
  publisher = {Danish Institute for Educational Research},
  address   = {Copenhagen},
  url       = {https://archive.org/details/probabilisticmod0000rasc}
}
```

<a id="bibtex-birnbaum1968latent"></a>
### `birnbaum1968latent`

```bibtex
@incollection{birnbaum1968latent,
  title     = {Some Latent Trait Models and Their Use in Inferring an Examinee's Ability},
  author    = {Birnbaum, Allan},
  booktitle = {Statistical Theories of Mental Test Scores},
  editor    = {Lord, Frederic M. and Novick, Melvin R.},
  year      = {1968},
  publisher = {Addison-Wesley},
  address   = {Reading, MA},
  url       = {https://faculty.ucmerced.edu/jvevea/classes/290_21/readings/week%209/Birnbaum.pdf}
}
```

<a id="bibtex-mislevy1986"></a>
### `Mislevy1986`

```bibtex
@article{Mislevy1986,
  title   = {Bayes Modal Estimation in Item Response Models},
  author  = {Mislevy, Robert J.},
  journal = {Psychometrika},
  volume  = {51},
  number  = {2},
  pages   = {177--195},
  year    = {1986},
  doi     = {10.1007/bf02293979}
}
```

<a id="bibtex-bockaitkin1981"></a>
### `BockAitkin1981`

```bibtex
@article{BockAitkin1981,
  title   = {Marginal Maximum Likelihood Estimation of Item Parameters: Application of an {EM} Algorithm},
  author  = {Bock, R. Darrell and Aitkin, Murray},
  journal = {Psychometrika},
  volume  = {46},
  number  = {4},
  pages   = {443--459},
  year    = {1981},
  doi     = {10.1007/bf02293801},
  url     = {https://doi.org/10.1007/bf02293801}
}
```

<a id="bibtex-chenhoudodd1998"></a>
### `ChenHouDodd1998`

```bibtex
@article{ChenHouDodd1998,
  title   = {A Comparison of Maximum Likelihood Estimation and Expected a Posteriori Estimation in {CAT} Using the Partial Credit Model},
  author  = {Chen, Ssu-Kuang and Hou, Liling and Dodd, Barbara G.},
  journal = {Educational and Psychological Measurement},
  volume  = {58},
  number  = {4},
  pages   = {569--595},
  year    = {1998},
  doi     = {10.1177/0013164498058004002}
}
```

<a id="bibtex-verhelst1993dynamicrasch"></a>
### `verhelst1993dynamicrasch`

```bibtex
@article{verhelst1993dynamicrasch,
  title   = {A Dynamic Generalization of the Rasch Model},
  author  = {Verhelst, Norman D. and Glas, Cees A. W.},
  journal = {Psychometrika},
  volume  = {58},
  number  = {3},
  pages   = {395--415},
  year    = {1993},
  doi     = {10.1007/BF02294648},
  url     = {https://doi.org/10.1007/BF02294648}
}
```

<a id="bibtex-wang2019longitudinalirt"></a>
### `wang2019longitudinalirt`

```bibtex
@article{wang2019longitudinalirt,
  title   = {On Longitudinal Item Response Theory Models: A Didactic},
  author  = {Wang, Chun and Nydick, Steven W.},
  journal = {Journal of Educational and Behavioral Statistics},
  volume  = {45},
  number  = {3},
  pages   = {339--368},
  year    = {2020},
  doi     = {10.3102/1076998619882026},
  url     = {https://doi.org/10.3102/1076998619882026}
}
```

<a id="bibtex-chalmers2012mirt"></a>
### `chalmers2012mirt`

```bibtex
@article{chalmers2012mirt,
  title   = {mirt: A Multidimensional Item Response Theory Package for the {R} Environment},
  author  = {Chalmers, R. Philip},
  journal = {Journal of Statistical Software},
  volume  = {48},
  number  = {6},
  pages   = {1--29},
  year    = {2012},
  doi     = {10.18637/jss.v048.i06},
  url     = {https://doi.org/10.18637/jss.v048.i06}
}
```

<a id="bibtex-reckase2009mirt"></a>
### `reckase2009mirt`

```bibtex
@book{reckase2009mirt,
  title     = {Multidimensional Item Response Theory},
  author    = {Reckase, Mark D.},
  publisher = {Springer},
  year      = {2009},
  doi       = {10.1007/978-0-387-89976-3},
  url       = {https://doi.org/10.1007/978-0-387-89976-3}
}
```

### Graph and Spectral Methods

<a id="bibtex-page1999pagerank"></a>
### `page1999pagerank`

```bibtex
@techreport{page1999pagerank,
  title       = {The PageRank Citation Ranking: Bringing Order to the Web},
  author      = {Page, Lawrence and Brin, Sergey and Motwani, Rajeev and Winograd, Terry},
  institution = {Stanford InfoLab},
  year        = {1999},
  number      = {1999-66},
  month       = nov,
  url         = {https://ilpubs.stanford.edu/422/},
  note        = {Previous number: {SIDL-WP-1999-0120}}
}
```

<a id="bibtex-vigna2016spectral"></a>
### `vigna2016spectral`

```bibtex
@article{vigna2016spectral,
  title   = {Spectral ranking},
  author  = {Vigna, Sebastiano},
  journal = {Network Science},
  volume  = {4},
  number  = {4},
  pages   = {433--445},
  year    = {2016},
  doi     = {10.1017/nws.2016.21},
  url     = {https://doi.org/10.1017/nws.2016.21}
}
```

<a id="bibtex-keener1993perron"></a>
### `keener1993perron`

```bibtex
@article{keener1993perron,
  title   = {The Perron-Frobenius theorem and the ranking of football teams},
  author  = {Keener, James P.},
  journal = {SIAM Review},
  volume  = {35},
  number  = {1},
  pages   = {80--93},
  year    = {1993},
  doi     = {10.1137/1035004},
  url     = {https://doi.org/10.1137/1035004}
}
```

<a id="bibtex-negahban2017rankcentrality"></a>
### `negahban2017rankcentrality`

```bibtex
@article{negahban2017rankcentrality,
  title   = {Rank Centrality: Ranking from Pairwise Comparisons},
  author  = {Negahban, Sahand and Oh, Sewoong and Shah, Devavrat},
  journal = {Operations Research},
  volume  = {65},
  number  = {1},
  pages   = {266--287},
  year    = {2017},
  doi     = {10.1287/opre.2016.1534},
  url     = {https://doi.org/10.1287/opre.2016.1534}
}
```

<a id="bibtex-omidshafiei2019alpharank"></a>
### `omidshafiei2019alpharank`

```bibtex
@article{omidshafiei2019alpharank,
  title   = {{$\alpha$}-Rank: Multi-Agent Evaluation by Evolution},
  author  = {Omidshafiei, Shayegan and Papadimitriou, Christos and Piliouras, Georgios and Tuyls, Karl and Rowland, Mark and Lespiau, Jean-Baptiste and Czarnecki, Wojciech M. and Lanctot, Marc and P{\'{e}}rolat, Julien and Munos, R{\'{e}}mi},
  journal = {Scientific Reports},
  volume  = {9},
  number  = {1},
  year    = {2019},
  doi     = {10.1038/s41598-019-45619-9},
  url     = {https://doi.org/10.1038/s41598-019-45619-9}
}
```

<a id="bibtex-balduzzi2019openended"></a>
### `balduzzi2019openended`

```bibtex
@inproceedings{balduzzi2019openended,
  author    = {Balduzzi, David and Garnelo, Marta and Bachrach, Yoram and Czarnecki, Wojciech and P{\'{e}}rolat, Julien and Jaderberg, Max and Graepel, Thore},
  title     = {Open-ended Learning in Symmetric Zero-sum Games},
  booktitle = {Proceedings of the 36th International Conference on Machine Learning (ICML 2019)},
  series    = {Proceedings of Machine Learning Research},
  volume    = {97},
  pages     = {434--443},
  year      = {2019},
  publisher = {PMLR},
  url       = {http://proceedings.mlr.press/v97/balduzzi19a.html}
}
```

<a id="bibtex-balduzzi2018reevaluating"></a>
### `balduzzi2018reevaluating`

```bibtex
@inproceedings{balduzzi2018reevaluating,
  author    = {Balduzzi, David and Tuyls, Karl and P{\'{e}}rolat, Julien and Graepel, Thore},
  title     = {Re-evaluating Evaluation},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {31},
  year      = {2018},
  publisher = {Curran Associates, Inc.},
  url       = {https://proceedings.neurips.cc/paper_files/paper/2018/file/cdf1035c34ec380218a8cc9a43d438f9-Paper.pdf}
}
```

<a id="bibtex-fogel2016serialrank"></a>
### `fogel2016serialrank`

```bibtex
@article{fogel2016serialrank,
  title   = {Spectral Ranking using Seriation},
  author  = {Fogel, Fajwel and d'Aspremont, Alexandre and Vojnovic, Milan},
  journal = {Journal of Machine Learning Research},
  volume  = {17},
  pages   = {88:1--88:45},
  year    = {2016},
  url     = {https://jmlr.org/papers/v17/16-035.html}
}
```

<a id="bibtex-jiang2009hodgerank"></a>
### `jiang2009hodgerank`

```bibtex
@misc{jiang2009hodgerank,
  title         = {Statistical Ranking and Combinatorial Hodge Theory},
  author        = {Xiaoye Jiang and Lek-Heng Lim and Yuan Yao and Yinyu Ye},
  year          = {2009},
  eprint        = {0811.1067},
  archivePrefix = {arXiv},
  primaryClass  = {stat.ML},
  doi           = {10.48550/arXiv.0811.1067},
  url           = {https://arxiv.org/abs/0811.1067v2},
  note          = {arXiv:0811.1067v2}
}
```
