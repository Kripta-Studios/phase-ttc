# Visual assets

## Real EvTTC sensor observations

`figures/evttc_ccrm_poster.jpg`, `figures/evttc_query-{00,07,15}-rgb.jpg`,
`figures/evttc_query-{00,07,15}-events.png`, and
`figures/evttc_query-07-channels.png` are byte-identical copies of the real-data
illustrations already used at https://schwiedop.com/PHASE-TTC/.
They derive from EvTTC recording `CCRm-low-0-overlap-0`, source HDF5
`2024-08-08-10-49-53.hdf5`, using Blackfly left RGB frames and Prophesee left
events. These are recorded data and runtime preprocessing outputs.

EvTTC authors: Kaizhen Sun, Jinghang Li, Kuan Dai, Bangyan Liao, Wei Xiong,
and Yi Zhou. Dataset project: https://nail-hnu.github.io/EvTTC/.
The supplied attribution, copyright, conditions, and full Clear BSD license
notice are retained in `data/real_sensor_excerpt/ATTRIBUTION.txt`.
The project license does not replace these third-party terms.

`data/real_sensor_excerpt/source_provenance.json` preserves the original
website extraction inventory and hashes, including assets that are not needed
in this manuscript. `replay-data.json` retains exact timestamps, event windows,
query identities, and unclipped per-channel statistics. The byte-identical
`extract-real-replay.py` records how the website generated its media using the
research runtime and the original data; it requires access to those inputs.
`scripts/verify_sensor_excerpt.py` checks the distributed subset, extraction
script, timestamp claims, and its connection to the archived transfer CSV.

The full-scene RGB display swaps stored channels 0 and 2 to match the publisher
visualization; model crops preserve runtime channel order. Event polarity
images use log counts and occupied-pixel 95th-percentile scaling; the twelve
input planes have separate clipping for display. Images are display encodings,
not lossless tensors or learned activations. Sensor coordinates remain
separate. These selected, development-exposed examples do not add evaluation
evidence. Raw HDF5 and annotations are not included.

## FCWD event/RGB illustration

`figures/fcwd_example.png` was supplied with the project handoff. It is included
as a byte-identical copy; its SHA-256 checksum and transformation are recorded
in `data/visual_source_manifest.json`. The handoff did not identify an original
creator or a separate redistribution license. Original dataset and material
rights remain with their respective owners; the project's license does not
relicense this third-party image.

## Archived cover asset

The author-supplied cover image retained in this source inventory is no longer
used by the manuscript. Rights to the image remain with its owner.
