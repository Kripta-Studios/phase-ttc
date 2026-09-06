# TRAIN reader and correspondence integration

This implementation checkpoint is not a training lock and does not certify A
availability. No optimizer updates were performed by these unit tests.

`event_context128_reader.py` verifies the completed cache manifest, exact component
inventory, byte hashes, tensor layouts and unique row identities. Each requested
count row is copied out of a memory map that is explicitly closed before returning;
observation ZIP handles are also closed. File size/mtime changes after verification
stop subsequent reads. Metadata and labels are not neural input fields.

`TrainingView` binds ordered tokens, the cache identity, the complete fixed role
map and excluded sequences. Acquisition aliases of an excluded sequence are
rejected before a normalization read. Population moments include zero activity
inside sensor support and use only the selected TRAIN rows. A normalization from
another view is rejected. Outer inference will require a separate endpoint reader;
this TRAIN-only interface does not authorize an outer or confirmation read.

`stage73_correspondence.py` implements the document07 shared-encoder symmetric
temperature-0.07 four-neighbor soft cross entropy. It uses the prescribed common
scale, relative scale, rotation and translation ranges. The remaining photometric
choice is fixed prospectively to independent scalar gain U[0.9,1.1] and Gaussian
noise sigma0.01 in TRAIN-normalized texture units; optional cutout is disabled.
Transforms use align_corners=False and known sensor/transform validity, with no
TTC label, teacher or foreground annotation. Each direction samples at most128
valid feature centers without replacement and rejects fewer than32. The training
owner must implement the prescribed maximum-three-redraw/replenishment lifecycle.

Diagnostics include matching error in input pixels, constant-descriptor and
shuffled-correspondence controls, descriptor variance, masked fraction and query
counts. Explicit CPU generator state reproduces the augmentation draws.

Validation at this implementation checkpoint:

- Five production-reader tests pass, including excluded acquisition aliases,
  post-verification modification, malicious component paths and Windows map closure.
- Four correspondence tests pass, including exact subpixel mass, RNG replay,
  identity matching, constant descriptors, invalid inputs and encoder gradients.
- Both 64-row probe cache versions have identical counts, observation arrays and
  metadata. This does not replace an independent raw histogram reference check.
- Ruff and scoped Pyright pass. Full baseline/new failure-ID comparison, real TRAIN
  gradient/update smoke, exact training resume and the prescribed pretext learning
  acceptance test are still pending. Neither A availability nor scientific execution
  readiness has been declared.

Evidence resides under `artifacts/stage70_76_architecture/preflight/`:
`E128_READER_UNIT.xml`, `CORRESPONDENCE_UNIT.xml`, `PROBE_V1_V2_PARITY.json`.
