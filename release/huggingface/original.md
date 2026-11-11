## Original nine-component model

This repository releases the frozen **PHASE_ONLY_EVENT_S7_TRAIN40_SCIENTIFIC_OOF** system. Its nine components are four modality-specific motion backbones, two pairwise temporal readouts, two temporal refiners and one 1,955-parameter phase fusion head. The head's size is not the size of the complete model. Up to 650 ms of causal information is admitted.

The original model achieved **Rank 1** in the archived **[GarlTTC Test leaderboard](https://www.codabench.org/competitions/17289/#/results-tab), recorded on 28 October 2026**: **35.80145 Overall MiD**, compared with **45.03** for Garl-TTC, a **20.49% reduction** over 6,762 scored tokens with **0% scorer-defined failures**. The tokens include 5,805 physical queries and aliases. These are archived official results, not a claim about today's leaderboard or a new evaluation of this release.

The separately distributed [PHASE-TTC-Adapt](https://huggingface.co/AlvaroSchwiedop/PHASE-TTC-Adapt) adds an EvTTC-trained geometric quality router. It is not the model associated with this official score.

[View the official leaderboard on Codabench →](https://www.codabench.org/competitions/17289/#/results-tab)
