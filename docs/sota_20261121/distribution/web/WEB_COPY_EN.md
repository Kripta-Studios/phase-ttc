# PHASE-TTC result and reproduction status

PHASE-TTC-Geo scored **23.47238 Overall MiD** on GarlTTC Test on 18 November
2026, with **0% scorer-defined failures** across 6,762 tokens. Its band scores
were 24.37 / 23.95 / 22.89 / 18.13. The submitted system used RGB-branch
estimates and up to 600 ms of causal history from the supplied object regions;
it did not use event inputs. The original nine-component PHASE-TTC system scored
35.80145, and the Garl-TTC reference scored 45.03 under their recorded protocols.

The uploaded PHASE-TTC-Geo package recovered from CodaBench is byte-identical to
the pre-registered ZIP (SHA-256 `8590f115…`). Replay from authenticated frozen
PHASE20 endpoints reproduces all 5,805 physical predictions, 957 aliases and the
ZIP with zero TTC drift. Refitting the head from the same endpoint inputs
reproduces every seed-7 physical prediction. Seeds 13 and 21 have unstable tails,
so the seed-7 result does not establish seed-insensitive behavior.

The bounded media replay covers 12/5,805 physical queries, one per sequence. It
regenerates RGB from retained media and reuses the frozen public event tensor. It
passes the `1e-6` endpoint gate on 0/12 and the `1e-4 s` geometry-TTC gate on
5/12 (median/max drift `1.893e-4/8.430e-4 s`). No private targets were read, and
no new private Overall score was inferred. This is not end-to-end parity.

The result measures a complete benchmark system with supplied regions. It does
not establish an advantage from event access, sensor-to-output parity, detector
robustness, generalization to an independent cohort or a 20 ms deployment bound.

The account history records a third completed submission on 21 November 2026:
submission `1063261`, a box-only head using 600 ms of supplied-region history and
no image or event model, scored **27.97563 Overall MiD** with zero scorer-defined
failures. Its crucial / small / large / negative bands were 26.38540 / 28.00007 /
32.93790 / 30.89121. The RGB + geometry submission remains better at 23.47238,
a **4.50325-point gap**, and is better in all four bands (24.36932 / 23.95223 /
22.89423 / 18.12626). This comparison isolates the contribution of RGB-branch
estimates beyond the shared supplied-box history; it is not a comparison between
companies or matched end-to-end sensor systems. The public leaderboard display
after the third submission has not been archived, so no current rank is claimed.
