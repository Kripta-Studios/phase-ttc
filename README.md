# PHASE-TTC

**A Causal Event–RGB System for Time-to-Contact Estimation**

[Paper](https://schwiedop.com/papers/phase-ttc/) ·
[Project website](https://schwiedop.com/PHASE-TTC/) ·
[Leaderboard](https://www.codabench.org/competitions/17289/#/results-tab) ·
[Reproduction guide](docs/reproducibility.md)

<a href="https://huggingface.co/AlvaroSchwiedop/PHASE-TTC"><img src="docs/media/huggingface-logo.png" width="22" height="22" alt="Hugging Face"></a> **Hugging Face:** [PHASE-TTC](https://huggingface.co/AlvaroSchwiedop/PHASE-TTC) · [PHASE-TTC-Adapt](https://huggingface.co/AlvaroSchwiedop/PHASE-TTC-Adapt) · [Recorded demo](https://huggingface.co/spaces/AlvaroSchwiedop/PHASE-TTC)

<a href="https://github.com/Kripta-Studios/phase-ttc"><img src="docs/media/github-logo.png" width="22" height="22" alt="GitHub"></a> [GitHub](https://github.com/Kripta-Studios/phase-ttc) · <a href="https://www.linkedin.com/in/%C3%A1lvaro-schwiedop-souto-92718b339/"><img src="docs/media/linkedin-logo.png" width="22" height="22" alt="LinkedIn"></a> [LinkedIn](https://www.linkedin.com/in/%C3%A1lvaro-schwiedop-souto-92718b339/)

PHASE-TTC estimates signed time-to-contact (TTC) from event streams and RGB
frames. Independent motion estimators process each modality; temporal refiners
summarize their causal histories, and a compact residual head corrects an RGB
anchor using phase, timing and observation-support descriptors.

The complete system achieves **35.8015 Overall Motion-in-Depth (MiD) error** on
GarlTTC Test, a **20.49% reduction** relative to the reported Garl-TTC reference
of 45.03, with **0% scorer-defined failures** across **6,762 scored tokens**.
In the archived leaderboard of 28 October 2026 this is the **first of three
entries**; the other two are the organizers' Garl-TTC reference and an unnamed
submission with the same score. PHASE-TTC uses up to 650 ms of causal history and
the benchmark's object regions, whereas Garl-TTC uses two observations 0.1 s
apart. This result measures the complete system: the matched controls in the
paper do not show a benefit from event access or from the phase parameterization,
so the gain is not attributed to event-RGB fusion.

A second, pre-registered submission on 18 November 2026 tested that reading. A
head that receives the RGB-branch estimates and the causal expansion of the
supplied object regions, with a band-weighted loss and **no event inputs**,
scores **23.47 Overall MiD** on GarlTTC Test (submission 1058935): 34% below the
original system and 48% below Garl-TTC. Most of the accuracy available on this
benchmark comes from the history of the supplied regions. Protocol, results and
the submission ledger are in [docs/sota_20261118](docs/sota_20261118).

## Paper

**[PHASE-TTC: From Motion to Time](paper/from_motion_to_time_20261102/PHASE_TTC_From_Motion_to_Time_Expanded.pdf)**
describes the architecture, official benchmark result, matched readout controls,
geometric adaptation and execution measurements.

- [Read online](https://schwiedop.com/papers/phase-ttc/)
- [Download the PDF](paper/from_motion_to_time_20261102/PHASE_TTC_From_Motion_to_Time_Expanded.pdf)
- [LaTeX source](paper/from_motion_to_time_20261102/main.tex)
- [Short-form LaTeX source](paper/from_motion_to_time_20261102/main_short.tex)
- [Download the source bundle](paper/PHASE_TTC_From_Motion_to_Time_Expanded_sources.zip)
- [Matched-comparison data and protocols](paper/from_motion_to_time_20261102/data/consolidation)
- [Expanded persistent-service evidence](paper/from_motion_to_time_20261102/data/extended_service_20261112)
- [Six-scene, 72-pass inference comparison](paper/from_motion_to_time_20261102/revision/extended_inference_report.md)
- [Initial three-pass comparison](paper/from_motion_to_time_20261102/revision/persistent_comparison_report.md)

The expanded PDF contains the main article and a technical supplement with implementation
details, additional experiments and reproduction records. The 13 November update
adds a complete inference chapter: 72 fresh-process PHASE/Garl passes across six
EvTTC scenes, with continuous intervals of 3.1–9.3 seconds and two queue policies.
It separates delivery coverage, native numerical agreement and response latency;
all losses and intermediate-output discrepancies remain visible. The initial
three-pass comparison remains historical evidence. Query-API timings are excluded
from the article's deployment-response comparison.

## Architecture

The released base model executes one nine-component inference chain. Training
uses grouped out-of-fold features; inference does not execute an ensemble of
the training folds.

| Components | Role |
|---|---|
| Four motion backbones | Separate transport-trained and coarse-to-fine estimators for RGB and events. |
| Two pairwise readouts | Convert modality-specific pair features and timing into phase estimates. |
| Two temporal refiners | Summarize up to eight observations per modality, within a maximum 650 ms causal information span. |
| One residual fusion head | Combine 20 phase, timing and support descriptors to correct the RGB estimate. |

The complete original system has **2,347,313 parameters**. Its
**1,955-parameter fusion head** contains two hidden layers of width 32 with
LayerNorm and SiLU. This count describes the head, not the full system. The phase
coordinate links apparent scale change to inverse TTC; the output policy
preserves approaching and receding motion and bounds emitted TTC magnitudes.
The 650 ms history describes past information, not inference latency.

Inputs require synchronized event/RGB observations and supplied object regions.
The released estimator does not include an autonomous object detector or
calibrated collision probabilities. The paper specifies the sensor preparation,
timing conventions and signed output policy.

**PHASE-TTC-Adapt** adds causal box-expansion estimates and three quality routers
of 688 parameters each. It keeps the original nine components frozen and
averages the three adapted TTC predictions. This is a separate model with a
local evaluation. It is not PHASE-TTC-Geo, the separately fitted no-event head
used for the second official submission.

## Results

### Official benchmark

| System | Inputs and temporal support | Overall MiD ↓ | Scorer-defined failure rate ↓ |
|---|---|---:|---:|
| Garl-TTC reference | Event + RGB; two observations 0.1 s apart; supplied regions | 45.03 | 0% |
| PHASE-TTC, 28 October | Event + RGB; up to 650 ms; supplied regions | 35.8015 | 0% |
| PHASE-TTC-Geo, 18 November | RGB + 600 ms supplied-box history; no event input | **23.4724** | **0%** |

Each evaluation uses private test labels and 6,762 submission tokens, including
aliases of physical queries. The [scoring records](paper/data/official_benchmark.json)
and [second-submission record](docs/sota_20261118/leaderboard/submission_1058935/RECORD.json)
document the TTC bands and failure criterion. The public leaderboard retains one
row per participant and now displays PHASE-TTC-Geo; the 28 October result remains
in its dated archive. These rows compare complete systems with different temporal
support and do not rank all published TTC methods.

### Transfer and matched readout controls

The transfer study evaluates 460 eligible queries across 32 recordings.
Released fusion has a mean absolute error of **0.813 s**, compared with
**0.851 s** for the RGB branch. Fusion improves the mean and upper-tail point
estimates, while RGB has lower median and mean relative error. The paired
recording-bootstrap intervals for the mean, median and p95 differences include
zero.

Matched experiments keep upstream models and histories fixed and fit
equal-capacity heads using source-domain features. Each learned variant uses
three seeds.

| Readout | Transfer MAE, mean ± standard deviation ↓ |
|---|---:|
| Phase correction with real events | 0.847 ± 0.038 s |
| Phase correction with zero events | 0.838 ± 0.021 s |
| Inverse-TTC correction with real events | 0.847 ± 0.041 s |
| Source-calibrated scalar phase mixture | 0.996 s |

The learned corrections outperform the tested scalar mixture. These controls
show no consistent incremental advantage from event access or from phase over
inverse-TTC parameterization on this cohort. The recordings had influenced
earlier development, so these comparisons provide component analysis rather
than independent confirmation. The experimental heads do not replace the
released weights.

### Geometric adaptation

The adaptation study separates box geometry from learned routing. All rows
below use the same geometry-valid queries and signed predictions.

| Estimator | MAE: 424 common queries ↓ | MAE: 357 road queries ↓ |
|---|---:|---:|
| Original fusion | 0.793 s | 0.282 s |
| Geometry alone | 1.158 s | 1.343 s |
| Fixed inverse-TTC mixture | **0.289 s** | 0.221 s |
| Learned router | 0.304 s | **0.159 s** |

The router improves road MAE over the fixed mixture by 0.062 s, with a 95%
recording-bootstrap interval of [0.014, 0.120] s. Its pooled advantage is
unresolved: the fixed-minus-router difference is −0.0146 s, with interval
[−0.1704, 0.0943] s. Across all 434 queries, including geometry fallbacks,
original/adapted signed MAE is **0.824/0.347 s**. These development-exposed local
results do not establish a new official adapted score.

### Execution and reproducibility

A five-second, single-object replay paced at 50 Hz completes all **250 updates**
with numerical parity and no drops. Scheduled-arrival response reaches
**29.14 ms at p95**; 201 responses meet or exceed 20 ms. The replay therefore
does not satisfy the 20 ms response deadline. RAW data are preloaded and disk
I/O is excluded. Starting the clock after input preparation still gives a p95
of 25.25 ms, so preparation alone does not explain the missed deadline.

The released head agrees with saved transfer predictions within 2.87 × 10⁻⁶ s
when replayed on recovered features. Full historical inference has a separate
unresolved discrepancy: 504 of 5,805 native-replay queries exceed the original
tolerance, with a maximum TTC difference of 0.4968 s. The archived submission
and official score remain distinct from local reproduction.

## Download and run

The Hugging Face repositories contain frozen weights, normalizers, custom
runtime code, example inputs, evaluation records and training recipes.

| Repository or record | Weights | Verified score and protocol |
|---|---|---|
| [PHASE-TTC](https://huggingface.co/AlvaroSchwiedop/PHASE-TTC) | Original nine-component weights | 35.8015 Overall MiD; submission 1016512; TRAIN40 and 650 ms causal-history protocol. |
| [PHASE-TTC-Adapt](https://huggingface.co/AlvaroSchwiedop/PHASE-TTC-Adapt) | Original system plus three local quality routers | No official server score for these router weights; 434-query development evaluation only. |
| [PHASE-TTC-Geo record](docs/sota_20261118/SUBMISSION_LEDGER.md) | Scored package is archived in the submission evidence, not either Hub repository | 23.4724 Overall MiD; submission 1058935; RGB plus supplied-box history, band loss, no event input. |
| [Demo Space](https://huggingface.co/spaces/AlvaroSchwiedop/PHASE-TTC) | Recorded examples of the original system | Instructions for local inference; not the PHASE-TTC-Geo package. |

Use a fresh **Python 3.12** environment:

```bash
pip install huggingface_hub
hf download AlvaroSchwiedop/PHASE-TTC --local-dir PHASE-TTC
cd PHASE-TTC
pip install -r requirements.txt
python predict.py --input examples/synthetic_looming.npz --verify examples/synthetic_expected.json
python evaluate_saved.py
```

CPU execution is the default; CUDA requires a compatible PyTorch installation.
To run the adapted model, replace both the Hub repository and local directory
with `PHASE-TTC-Adapt`. Pin `hf download --revision <commit>` for repeatable
downloads. The inference scripts verify file checksums before loading weights.

The synthetic example executes the selected inference graph.
`evaluate_saved.py` recomputes published local summaries from saved predictions;
it does not rerun private-label benchmark scoring. For real sensor inputs,
obtain the dataset separately and follow the bundle's `INPUT_FORMAT.md` and
the [reproduction guide](docs/reproducibility.md).

## Research code and data

Install the repository and development tools with `uv`:

```bash
git clone https://github.com/Kripta-Studios/phase-ttc.git
cd phase-ttc
uv sync --locked --all-groups --no-editable
uv run --no-sync phase-ttc --help
```

The research package supports Python 3.11–3.13. The standalone Hub inference
bundles have a tested Python 3.12 environment and their own requirements.

| Directory | Contents |
|---|---|
| `src/e_jepa_ttc/` | Data handling, representations, models, training and evaluation. |
| `operational/final_refit/` | Grouped fitting, final refit and inference procedures. |
| `operational/evttc_competition/` | Sensor preparation and EvTTC inference runtime. |
| `configs/` | Model, data and experiment configurations. |
| `tests/` | Unit, integration and regression checks. |
| `paper/from_motion_to_time_20261102/` | Current article, figures, result tables and build scripts. |

eAP/GarlTTC and EvTTC data must be obtained under their respective terms.
Raw datasets and intermediate training caches are not included in Git.
Full original retraining requires the external data, path configuration and
reconstruction of intermediate caches; a complete retraining has not been
performed as part of the public packaging validation. The adapted bundle
includes derived training features and its fixed three-seed router recipe.
See the [dataset card](docs/dataset_card.md) and
[reproduction guide](docs/reproducibility.md) for requirements.

The final model uses supervised motion estimation and phase fusion. Predictive
JEPA experiments form part of the supplementary research; they did not establish
a pretraining benefit in the tested settings. Existing checkpoints and scripts
retain the `e_jepa_ttc` import namespace and compatible `e-jepa-ttc` command.

Run the repository checks with:

```bash
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync pyright
uv run --no-sync pytest -q
```

## Build the paper

> **Current edition.** The files directly under `paper/` are the sealed
> 28 October 2026 historical snapshot and are retained byte for byte. They
> contain superseded claims (the 468-query transfer cohort and an unqualified
> "Rank 1"). For citation and publication, use the
> [corrected expanded edition](paper/from_motion_to_time_20261102/README.md).

From the repository root:

```bash
python paper/from_motion_to_time_20261102/scripts/build_expanded.py
python paper/from_motion_to_time_20261102/scripts/verify_publication.py
python paper/from_motion_to_time_20261102/scripts/package_expanded.py
```

The short-form source reuses the same tables, figures and bibliography:

```bash
cd paper/from_motion_to_time_20261102
pdflatex -interaction=nonstopmode -halt-on-error main_short.tex
bibtex main_short
pdflatex -interaction=nonstopmode -halt-on-error main_short.tex
pdflatex -interaction=nonstopmode -halt-on-error main_short.tex
```

Its final page count must be checked against the selected venue template before
submission; `main_short.tex` does not claim compliance with an unspecified venue.

The build requires NumPy, Matplotlib, Pillow, PyMuPDF, pdfLaTeX and BibTeX,
including the XCharter, newtxmath and TikZ packages. It regenerates result tables
from the accompanying evidence and checks that the compiled PDF matches its
source inputs. The packager writes the PDF, consolidated LaTeX and source ZIP
to `output/pdf/`.

## Architecture video

[![PHASE-TTC architecture explainer: real RGB frames and event streams](docs/media/phase-ttc-architecture-explainer.png)](https://github.com/Kripta-Studios/phase-ttc/releases/download/phase-ttc-architecture-video-20261111-rank1/PHASE_TTC_Architecture_260s_1080p.mp4)

**[Watch or download: 4:20, 1080p, 60 fps](https://github.com/Kripta-Studios/phase-ttc/releases/download/phase-ttc-architecture-video-20261111-rank1/PHASE_TTC_Architecture_260s_1080p.mp4)** ·
[English subtitles](https://github.com/Kripta-Studios/phase-ttc/releases/download/phase-ttc-architecture-video-20261111-rank1/PHASE_TTC_Detailed_EN.srt)

The narrated walkthrough follows real EvTTC observations through preprocessing,
motion estimation, temporal refinement and signed TTC prediction. Connected
diagrams distinguish expert values, routing weights and the RGB residual anchor.
The closing sections present the current matched readout controls, signed
geometric-adaptation results and measured execution limits.

## Scope and limitations

PHASE-TTC is a research estimator for supplied object regions. The official
complete-system result does not isolate the benefit of the phase interface,
event inputs or temporal history. Secondary studies reuse recordings inspected
during development, and additional head seeds share upstream models.

Independent transfer validation, reconciliation of the Slider-750 evaluator
discrepancy and historical inference parity remain open. Sustained physical-camera
operation, autonomous localization, multiobject latency and calibrated warning
probabilities are outside the demonstrated result. The system is not validated
for safety-critical control.

## Citation and license

**Álvaro Schwiedop Souto** · University of Santiago de Compostela

Please cite the [paper](https://schwiedop.com/papers/phase-ttc/) and use
[CITATION.cff](CITATION.cff) for software citation metadata. Pin the repository
and model revisions used in experiments. Project-authored code and PHASE-TTC
weights use the [MIT license](LICENSE). Third-party models and datasets retain
their providers' terms; the MIT license does not relicense them.

```bibtex
@misc{schwiedop2026phasettc,
  title = {PHASE-TTC: From Motion to Time -- A Causal Event--RGB System for Signed Time-to-Contact Estimation},
  author = {Schwiedop Souto, {\'A}lvaro},
  year = {2026},
  howpublished = {Research preprint},
  url = {https://schwiedop.com/papers/phase-ttc/}
}
```

[Email](mailto:alvaro@schwiedop.com) ·
[Website](https://schwiedop.com/PHASE-TTC/) ·
[LinkedIn](https://www.linkedin.com/in/%C3%A1lvaro-schwiedop-souto-92718b339/) ·
[Model card](docs/model_card.md)
