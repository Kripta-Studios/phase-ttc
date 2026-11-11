# PHASE-TTC Hugging Face release implementation plan

**Goal:** Publish the current original model, separate adapted model and a usable demo under AlvaroSchwiedop, with current documentation and independently downloadable reproducibility artifacts.

**Architecture:** Preserve all sealed weights and historical receipts. Add a small prepared-observation inference entry point around the existing sensor runtime, ship its exact source dependencies and examples, and distinguish real inference from recorded demonstrations. Stage privately and expose publicly only after downloaded-artifact checks.

**Scope approved in conversation:** User requested execution of the proposed model/adaptation/demo publication and all adjustments needed to remove obsolete model descriptions. No new training or paid compute is authorized or needed.

## Constraints

- Original model and EvTTC adaptation have distinct identities and evaluation scopes.
- Preserve byte-identical weights, normalizers and historic manifests.
- Source line endings bound by historical digests must be reconstructed in the release copy, not by changing historical manifests.
- Read-only raw datasets remain local; distribute a synthetic example and a reproducible extraction command for real inputs. A licensed excerpt may be included only with its attribution.
- No secrets, environment folders or unrelated experiments in release folders.
- Existing dirty manuscript work remains untouched.

## Tasks

- [x] Archive old model/data/reproducibility cards and write current PHASE-TTC cards; update active README.
- [x] Test and implement release input validation, manifest verification and CPU/CUDA inference CLI.
- [x] Package original and adapted models with current source snapshot, examples, corrected evaluation evidence and retraining recipe/availability boundaries.
- [x] Run examples from isolated copies, reject changed artifacts, check CPU/CUDA numerical agreement and reproduce router training if included inputs permit it.
- [x] Publish verified model repositories, verify fresh downloads and public model cards.
- [x] Publish a Space with an accurately labeled demo and working download/run instructions; verify browser behavior.
- [x] Record revisions, hashes, validations and any actual availability limits in docs/progress.md.

## Review focus

Prepared inputs must not contain targets; clocks and geometry must remain causal. Both model identities must be explicitly bound. No automatic loading of arbitrary remote code. A smoke test proves execution, not benchmark accuracy. Official private-label scores cannot be recomputed from public targets. Model card metadata must not describe an obsolete JEPA candidate. Source export must not depend on this workspace's absolute paths.

## Execution notes

Using an isolated staging directory under output/huggingface; source changes are additive release tooling and current documentation. OAuth write permission was verified by creating the private PHASE-TTC repository. The exposed user-supplied token was not used.

## Completed validation and hosting scope

Both model repositories are public and were verified from fresh downloads (synthetic and real-query inference plus saved metrics). Router refit reproduces all three archived state dictionaries exactly. Current targeted suite: 38 passed, 1 skipped; Ruff passes. Review findings addressed: per-variant receipts, fresh staging, rejection of unexpected code/model files, verification before extraction imports, and accurate evaluation documentation.

Hugging Face rejected Gradio cpu-basic creation with HTTP 402 because this account lacks PRO. Published a free static Space with clearly labeled recorded outputs, revision-pinned commands and the working Gradio app ZIP. No subscription or paid resource was created. The live public Space and model manifests were verified without authentication. Publication revisions and hashes: docs/releases/huggingface_20261111.json.
