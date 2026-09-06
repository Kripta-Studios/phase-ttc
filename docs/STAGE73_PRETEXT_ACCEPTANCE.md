# Correspondence technical acceptance

A passed its technical prerequisites before any scientific fit. This establishes
the available correspondence implementation; it does not establish TTC utility or
choose a scientific candidate by score. D and the two canonical candidate names
remain unresolved until D's independent prerequisites are complete.

The fixed generated acceptance used32 seeded band-limited eight-channel textures,
128×128 inputs, log-scale U[-0.1,0.1], normalized translation U[-0.125,0.125], no
rotation and no photometric noise. `FIXTURE_LOCK.json` recorded the fixture,
encoder seed7,100 updates, effective batch16, microbatch2, warmup5 and20% reduction
threshold before fitting. Initial and final measurements use identical query draws.
No TTC targets enter this test.

At update100, correspondence cross entropy fell from4.493998758494854 to
1.1406143344938755, a74.6191666756% reduction. Known-coordinate matching error fell
from21.9518441259861 to1.5252760564908385 input pixels. The constant-descriptor
control is6.9314718055994495. The final shuffled-correspondence CE is
12.764859765768051. These describe the fixed learnability fixture, not a real-data
accuracy estimate. No repeated fixture or substitute seed was selected.

The production `PretextTextures` adapter uses only the authorized TRAIN count view
and its normalization. A fixed hash split reserves floor10% of TRAIN tokens,
including all four textures of each held token, for diagnostic correspondence
evaluation. The encoder-fit and diagnostic tokens are disjoint. Normalization is
the full permissible TRAIN view; the diagnostic split is not independent
confirmation. The fold0 smoke had4915 encoder-fit tokens and546 diagnostic tokens.

The sampler uses the prescribed independent image perturbations and affine ranges
from document07. An invalid transform receives at most three redraws after its
initial draw. Exhausted draws are logged and replenished without consuming a fake
optimizer update. A1024-draw microbatch ceiling stops a wholly unusable pool with
an explicit data prerequisite failure. Labels and task heads are absent.

The real TRAIN pretext smoke ran10 updates continuously and5+save/reconstruct/load+5
with the same fixed encoder-only objective. Both runs matched model, optimizer,
all captured RNG states, losses, gradient norms, learning rates and the entire
texture/augmentation/diagnostic trace exactly. This included actual CUDA gradients
and the CPU augmentation generator. Three additional unit tests cover fixed held
membership, redraw/replenishment accounting and foreign-normalization rejection.

Campaign accounting is now180 confirmed technical optimizer updates in nine
technical fits, zero uncertain attempts, zero scientific optimizer updates. The
technical encoder checkpoints are not scientific initializers. Scientific A/AD
must train their own registered3000-update encoders under each producer's ancestor
exclusions and transfer only those encoder weights to newly initialized task heads.

Evidence under `artifacts/stage70_76_architecture`:

- `technical_correspondence_learning/FIXTURE_LOCK.json`
- `technical_correspondence_learning/LEARNING_ACCEPTANCE.json`
- `technical_pretext_resume/PRETEXT_RESUME_EQUIVALENCE.json`
- `preflight/PRETEXT_LIFECYCLE_UNIT.xml`
- `preflight/A_TECHNICAL_PREREQUISITES.json`
- `UPDATE_LEDGER.json`
