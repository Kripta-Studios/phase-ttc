# Full test evidence and prospective A/D availability

The operational disk minimum is now **40,000,000,000 bytes after reservation**,
under user amendment 002. RAM remains 4 GiB available after incremental reservation.
Amendment 001 and the interruptions it caused remain historical evidence.

The original 8,192-row cache and expanded 27,307-row cache completed. All component
hashes were rechecked through the production readers, including one expansion shard
losslessly recompressed with LZX. Both caches compose under the same representation
contract. Original outer TRAIN normalization/prior/source views are complete for
5,461, 5,461 and 5,462 rows; expanded views use the fixed 32,768 rows per fold.

`preflight/CANONICAL_CANDIDATES.json` prospectively resolves:

- `C_DIRECT = S73-DIRECT-D`
- `C_PPS = S73-PPS-AD`

A and D both passed their technical/data prerequisites. These names were recorded
with zero scientific optimizer updates, before any new outer prediction. The full
initial matrix remains 12 S72 and 15 S73 architecture jobs, including all eligible
pretext work and the 11,000-update EXTRA control. Failed fits cannot change the names.

The full test comparison is
`baseline_qa/BASELINE_VERSUS_CURRENT_4A8B545.json`. Its verified coverage is:

| Source | Passed | Failed | Skipped |
|---|---:|---:|---:|
| Mandatory source pin | 1743 | 37 | 23 |
| Current implementation tested | 1956 | 37 | 29 |

There are no additional failing IDs and 219 additional test IDs. Six of the new
tests are CUDA-skipped in this explicitly CPU suite; the separate measured real
TRAIN GPU acceptance remains required evidence and is already recorded for all four
arms. This comparison applies to the exact source revisions named in its receipt;
remaining orchestration/diagnostic implementation needs validation before freeze.

The first baseline attempt was resource-interrupted after 394 completed tests.
Its continuation verified the entire original inventory and executed the exact
1,409 pending tests in collection order. The initial current suite exposed one
new regression: a reference test set PyTorch threads globally during collection,
changing an older synthetic learning test. A restoring module fixture fixed it;
the subsequent complete suite verifies the older learning test passes.

Nine initial failures came from subprocess environments: `uv` selected an empty
checkout environment, and unqualified `python` on Windows selected the base Python
executable before PATH. The harness now binds offline uv to the installed environment,
checks the exact child Python version, and supplies the selected checkout's source
plus the shared installed dependencies explicitly. Both source versions passed the
four direct-Python selector tests after this correction. No dependencies were
downloaded, no source tests were waived, and every earlier failure observation and
environment recheck is retained in the comparison's source receipts.

The remaining shared failures concern legacy schema conventions and unavailable or
byte-mismatched historical fixture bindings. For example, the grouped V5 protocol
has the same physical hash in both new checkouts, while the historical original
contains the exact bytes expected by its frozen hash. The Garl lineage audit used
and verified those original bytes. Each remaining failure ID is preserved; it must
not be silently relabeled as a new campaign pass.

Actual confirmed optimizer work is now 1,680 technical updates, 447 observed
optimizer instances and 17 prescribed technical model endpoints, with no uncertain
attempts. Scientific optimizer updates and scientific ridge fits remain zero.
Statistics preparation is distinct from neural training. There is no scientific
training lock yet; this report does not claim execution readiness or campaign completion.
