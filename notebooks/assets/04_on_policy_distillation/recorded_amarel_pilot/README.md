# Recorded Amarel pilot, November 8, 2026

This directory preserves the user-downloaded audit and the locally saved notebook
before EOS-alignment implementation. The audit covers 176 trajectories and 22
updates from `on_policy_distillation_20261108T184539826093Z`. At update 22, 24 of
the latest 32 responses were capped. The notebook records the health pause and
follow-up validation progress through question 48/64 before allocation expiry.
The completed SFT control scored 40/64 strict validation matches; teacher
validation scored 52/64. These are validation measurements from the saved output.

`notebook_snapshot.json` preserves code and outputs together as historical evidence.
It is a local saved snapshot. The remote experiment specification, full trajectory
log, optimizer snapshots, and exact source fingerprints await evidence collection.
The snapshot is not imported by the new notebook. Neither weights nor an optimizer
state are included here. The original 03 checkpoint remains a separate artifact.
`SHA256SUMS.json` fingerprints these local evidence files. Source-runtime checkpoint
inventories require the separate collection step below.

The audit's validation reference NLL is 0.6758457335 at update 8 and 0.8193381555 at
update 16. Its fixed-prefix KL values are 0.9153289869 and 0.7472800836 respectively.
These measures do not establish downstream accuracy improvement. The latest text
contains repetition, multiple final-answer markers, and invented continuations.

Use `scripts/collect_opd_evidence.py` on Amarel to verify the source inventories
and collect the complete small-file evidence. Current local Downloads files named
`training_trajectories.jsonl` and `on_policy_history.json` belong to the earlier
raw-context run and must not be substituted for this pilot's records.
