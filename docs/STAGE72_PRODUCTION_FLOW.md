# Stage72 production data integration

The initial real TRAIN smoke verified the cache adapter, model gradients and exact
resume without the supervised augmentations. Its historical receipt remains valid
for that narrower scope. It is not the final production-flow acceptance.

The production callback now implements document06's count augmentation before the
TRAIN-fitted normalization: one shared horizontal flip, integer x/y translation
uniformly from -4 through4 pixels, independent binomial thinning of the cached
event counts using a per-example retention probability uniformly in[.8,1], and
two polarity-wide gains uniformly in[.9,1.1], shared across all times. Binomial
sampling uses float64 integer counts; rate, gain, log and normalization use float32.
Counts are never interpolated spatially. Zero padding moves the physical support
mask with the common transform. There is no temporal rescaling or target change.
The support is sensor coverage, not an object foreground label.

The augmentation generator hashes the registered seed, logical update, position
within effective batch16 and token. It never consumes the row sampler generator.
Thus physical microbatch1 and2 produce the same sampled rows and augmentations,
and all architecture arms receive matched examples. TIMEPERM hashes token and
seed into the nine fixed-point-free permutations; chronology and targets stay in
their original slots. Its generator is independent of augmentation and sampling.

The supervised factory reseeds and rebuilds the complete model, initializes the
TRAIN-prior readout, then optionally copies only an exactly matching finite encoder
state. Every task-head tensor is identical to scratch initialization at the same
seed/prior. The executor must still enforce the pretext endpoint's full source
lineage before calling this factory; a raw state dictionary is not provenance proof.

The composite reader joins verified original and expansion count caches without
copying payloads. It requires identical representation semantics and producer code,
unique tokens, and permitted development roles. Normalization remains a single
streaming fit over each exact TRAIN view, with outer/inner acquisition exclusions.
This implementation does not yet establish D availability: the complete expansion
cache and resolved manifest still require validation.

Validation: nine count-reader/composition tests and five supervised-flow tests pass,
including actual small cache files, target mutation, microbatch-invariant sampling,
exact augmentation RNG replay, count thinning, shared spatial transforms, clock
preservation and head reset. Ruff and scoped Pyright pass. No optimizer updates
are performed by these tests.

The extended `qa_stage71_real_train_gpu.py --production-flow` runs four arms on the
same original fold0-TRAIN pool, with10 continuous versus5+resume5 updates each:
80 technical updates, never scientific candidates. Its separate receipt and output
root prevent replacing the earlier40-update smoke. All production-flow numerical
acceptance has now completed: all four arms passed10 versus5+resume5 with exact
model, optimizer, RNG, update, loss/gradient/LR history and complete
sampler/augmentation/TIMEPERM trace equality. The foreground process exited0.
Its recipe hash is
`72abfdc14921f7fef17f83006707798e1e4f8fa8e07499437f56494992227963`.
The campaign ledger records260 confirmed technical updates,17 completed optimizer
fits,0 uncertain attempts and0 scientific fits after this acceptance. Peak
allocated VRAM was below709MB for every tested arm; these short fixed-pool
measurements do not substitute for whole-campaign resource accounting.
