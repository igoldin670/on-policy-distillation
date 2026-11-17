> Earlier revision record. The current profile-aware workflow and release checks
> are documented in the [project validation record](../../../VALIDATION.md).
> Original GPU executions remain preserved in the historical evidence.

# Notebook 04 validation record

Reviewed November 16–17, 2026. Local QA and the author's Amarel GPU runs have
separate provenance throughout this record.

## Recorded GPU evidence

| Record | Observed result |
| :--- | :--- |
| Earlier raw-context run | 416 downloaded responses across 52 updates, including 232 consecutive capped responses at updates 24–52 |
| Context probe | Non-thinking chat increased teacher stopping mass on one saved boundary |
| November 8 Amarel pilot | 176 responses across 22 updates, with 24/32 final sampled responses capped |
| November 16 EOS comparison | 400 responses across 18 raw-token and 32 grouped-stop updates. Both validations completed |
| First measured quality decline | At update 8, strict matches fell from 33/64 to 17/64 raw-token and 29/64 grouped-stop |
| Final raw-token result | 1/64 strict, 53/64 greedy validation caps, 24/32 recent sampled caps, NLL 0.863818 |
| Final grouped-stop result | 23/64 strict, 4/64 greedy validation caps, 19/32 recent sampled caps, NLL 0.963980 |

The [downloaded comparison](recorded_eos_comparison/README.md) passed all 69
collected file checksums, trajectory consistency checks, and frozen-checker replay.
Historical source hashes match the actual experiment. Original NLL/KL values were
read from the GPU records. Recomputing those forwards needs model weights.
The source-side collector verified retained snapshot inventories. These downloaded
evidence files contain metadata, while model and optimizer tensors remain on Amarel.

The earlier November 8 checkpoint forwards and interrupted validation recovery are
separate unfinished investigations. The completed November 16 comparison provides
newer direct evidence. The two records keep their original model identities.

## Revised behavior

New runs sign `TrainingSafetyConfig` into their experiment identity and resume
metadata. Defaults pause above eight capped responses in the latest 32, above
110% of starting reference NLL, or after a strict-accuracy decline exceeding five
percentage points. On the saved histories, the new guard requests a pause at
update 8 for both arms. This is a replay of guard decisions on historical records.

Checksummed optimizer boundaries precede validation. Atomic metric sidecars allow
interrupted validation to finish before another update. Accepted boundaries stay
protected during pruning. Quality pauses permit evaluation recovery. SLURM checks
reserve ten minutes and run before individual trajectories. A mid-group deadline
restores the clean update boundary's RNG. Reports package automatically at pauses.
A packaging error is printed after the checkpoint and pause manifest are saved.
Abrupt process or hardware loss can discard work after the latest saved boundary.

Frozen probes now record distribution entropy and separate ordinary-token, raw-EOS,
and grouped-EOS KL contributions. The historical probes predate these extra fields.

## Completed local checks

| Check | Result |
| :--- | :--- |
| Regression suite | 157 tests run, 156 passed and one new-architecture class skipped in Transformers 4.57.6 |
| New architecture suite | Nine tests passed under Transformers 5.3.0, including tiny random CPU models |
| Guard and recovery | Threshold boundaries, final-boundary quality pause, interrupted evaluation, accepted-checkpoint retention, finite-parameter rejection, and mid-group RNG recovery passed |
| Historical guard replay | First requested pause reproduced at update 8 for both recorded arms |
| Notebook 04 CPU inspection | All 18 code cells executed sequentially in a clean kernel, zero errors |
| Inspection stderr | Four reviewed offline dataset-cache notices, zero unexpected stderr outputs |
| Notebook 05 CPU inspection | All ten code cells executed sequentially, zero errors |
| Static notebook series | Schema, syntax, comments, relative links, and HTML exports passed for 01–05 |
| Equations and figures | 78 expressions rendered with KaTeX 0.16.22. All 50 PNG/SVG files passed integrity or XML checks |
| Preservation | 01–03 bytes and all pre-edit 04 saved output objects preserved |

CPU inspection uses pinned tokenizer files and cached GSM8K data. It rebuilds
1,024 practice, 64 validation, 256 primary-test, and 32 historical-test questions.
It loads no pretrained Qwen weights. Tiny-model optimization tests verify mechanics.
Executed inspection notebooks and HTML live in ignored `results/` QA directories.
The source retains saved historical outputs, with explicit warnings that the GPU
execution predates the revised safeguards. The archived source supports reproduction.
New measured figures use the audited histories. Illustrative diagrams are labeled.
Direct PNG review and offline HTML checks were completed. Live GitHub rendering
and the revised guard's CUDA behavior await separate inspection and execution.

## Reproduce checks

Use the original CPU QA environment for the historical notebooks and a separate
Transformers 5.3.0 environment for the new architecture.

```bash
python -m unittest discover -s tests
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  python scripts/validate_notebook04.py --allow-offline-cache-warnings
python scripts/validate_notebook_series.py --katex-js .cache/katex.min.js
python -m ruff check src scripts tests notebooks/04_on_policy_distillation.ipynb notebooks/05_qwen38_restart.ipynb
python -m ruff format --check src scripts tests notebooks/04_on_policy_distillation.ipynb notebooks/05_qwen38_restart.ipynb
git diff --check
```

The offline inspection requires the pinned dataset/tokenizers in cache. The warning
flag accepts only the reviewed cached-dataset notices and records their text.
Equation checks require QuickJS and the official KaTeX bundle.

## Next GPU milestone

Use [Notebook 05](../../05_qwen38_restart.ipynb) and its [run guide](../05_qwen38_restart/RUN.md)
to qualify the selected Qwen3.5-0.8B-Base student and Qwen3.8-27B teacher. Real new-model
generation/scoring, peak GPU memory, maximum-length behavior, a fresh baseline and
SFT checkpoint, revised guarded optimization, and quality gains remain pending.
The existing 03 weights belong to the historical Qwen3 student.
