# Release validation

Reviewed November 17, 2026. The active experiment uses the pinned `qwen38` profile.
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

The final review also corrected the new SFT entry point to use a pooled response-token
mean. Unequal-length gradient tests and the shared-runner integration cover that wiring.
SFT reports and ZIPs publish atomically and can recover interrupted packaging. Source
files are captured before training, checked against the signed implementation, and
used for exports even if the live checkout changes. Full continuation rechecks the
complete measured pilot records and locks the qualifying recipe. Editing a selection
file cannot start an untested rate or restart a completed run in pilot mode.

## Local checks

Fresh, separate Python 3.11.16 environments ran both release sequences. The new-model
suite ran **175 tests, all passing**. The legacy suite reported **173 tests,
OK (skipped=2)**, with its two new-architecture classes explicitly skipped. All
**10 new-profile commands** and **7 legacy commands** returned exit code zero.
Both sequences include Ruff formatting, equation rendering, and clean notebook kernels.
Fresh installation exposed missing kernel/widget dependencies and display-math
delimiter compatibility; these were corrected before the successful complete runs.
Notebook 02 replay redirects newly generated artwork to the ignored QA directory;
the versioned notebooks, original figures, and recorded results remain unchanged.

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

The final clean environments use PyTorch 2.10.0+cpu, Python 3.11.16, Transformers
5.3.0 / 4.57.6, and the pinned portable QA requirements. An earlier Python 3.14.7
suite provided additional local coverage. The final notebook inspections used pinned
cached metadata, tokenizers, and GSM8K data, acquired and checked separately online.
All 65 equations and 50 PNG/SVG assets passed static rendering/integrity checks.
Desktop and narrow-screen Chrome previews were inspected; focused 03/04 HTML
previews also typeset the equations with MathJax. These previews are local HTML,
not a claim of live GitHub or Amarel execution. Tiny-model measurements establish
code behavior and supply no pretrained Qwen task-performance evidence.

## Independent Cloud check

Independent Cloud validation has not passed. The first task-creation attempt was
blocked by automatic approval review's account usage limit. A retry after the stated
reset has not returned a task ID or execution evidence. The successful Python 3.11
checks above were run locally and are recorded as local QA. Cloud results must name
the exact tested commit before they can be added as independent evidence.

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
