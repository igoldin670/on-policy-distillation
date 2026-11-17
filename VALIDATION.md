# Release validation

Reviewed November 16, 2026. The active experiment uses the pinned `qwen38` profile.
Historical GPU measurements remain linked from the notebook evidence directories.

## What is implemented

The same profile-aware SFT and KD paths support notebook execution and CPU tests.
The new workflow includes a reference-prefix KD control, two fixed learning-rate
pilots, quality pauses, atomic recovery, deferred held-out testing, and recursive
source provenance. The mathematical correspondence is recorded in
[PAPER_ALIGNMENT.md](PAPER_ALIGNMENT.md).

Concrete fixes include the removed Transformers 5 generation argument, an obsolete
mandatory ZIP-upload cell, split mutation during teacher-prefix preparation,
final-update recovery, reference-log truncation, and pilot/full source identity.
Editing mode, run paths, stage selection, or Markdown preserves resume identity.
Changes to training code or resolved scientific inputs create a new identity.

## Local checks

The final new-environment suite ran **169 tests, all passing**. The legacy suite
reported **167 tests, OK (skipped=2)**, with its two new-architecture classes
explicitly skipped. The local release sequence completed nine commands with zero
exit codes. Formatting also passed and is included in the portable command.
Python 3.11 parsed all active helpers, scripts, tests, and 02–05 code cells.


The final portable command records each actual exit code in
`results/cpu_release_qa/qwen38_checks.json`. It also records commit identity and
whether the checkout has unpublished edits. The full source remains unchanged by QA.

- The full suite runs under Transformers 5.3.0 with tiny randomly initialized CPU
  models. It includes the actual SFT → reload → two pilots → full on-policy and
  reference-prefix controls → held-out evaluation → export → reopen sequence.
- The legacy Transformers 4.57.6 suite remains a separate compatibility check.
  New-architecture classes are explicitly skipped in that environment.
- Clean kernels execute 02 replay, both-profile 03/04 inspection, and 05 inspection.
- Static checks cover all five notebook schemas, code syntax, guidance comments,
  equations rendered with KaTeX, relative links, image integrity, and HTML export.
- Regression tests cover exact loss/gradients, causal offsets, response masks,
  genuine EOS targets, teacher freezing, partial groups, RNG isolation, snapshot
  retention, paused recovery, serialization, metric replay, and export integrity.
- Source and notebook outputs from the original 03/04 revisions are preserved in
  [historical notebooks](notebooks/assets/historical_notebooks/README.md). The later
  local missing-upload error has its own untouched snapshot.

Local QA uses PyTorch 2.10.0+cpu with Python 3.14.7. New-model checks use an isolated
Transformers 5.3.0 package set. The portable QA requirements support Python 3.11,
matching the intended Amarel interpreter family. Tiny-model measurements establish
code behavior and supply no pretrained Qwen task-performance evidence.

## Independent Cloud check

A separate clean Cloud task will check the published candidate by exact commit.
It will install user-owned Python 3.11 environments, run the full new-model release
command, and run the legacy compatibility checks. Its actual result and tested
commit will be reported with delivery. A local shell is recorded as local QA.

## Required real-GPU qualification

Notebook 05 first preserves one real student response and teacher scores. Its
additional synthetic probes exercise response-length-256 backward propagation,
one disposable optimizer update, cached student generation to 1,024 positions,
a teacher forward to 1,056 positions, and checkpoint reload. Temporary updated
weights are discarded. The report records memory, timing, and a fingerprint of
models, software, qualification source, and GPU class.

The `qwen38` KD path requires matching completed qualification evidence. A new
hardware/software combination needs its own qualification. Existing cached data
and old-model reports cannot establish these checks.

Real pretrained GPU qualification, new SFT/KD learning behavior, runtime over the
full question set, revised guards during actual optimization, and any improvement
remain pending Amarel execution. The six-hour allocation has a ten-minute reserve,
with verified continuation across allocations. Abrupt loss can discard work after
the last complete snapshot. [Run checklist](FINAL_RUN.md)
