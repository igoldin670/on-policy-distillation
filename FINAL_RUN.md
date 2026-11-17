# Final Amarel run checklist

## Before requesting GPUs

- Sync the final `dev` checkout, including helpers, configs, requirements, assets,
  and notebooks. Keep existing results, archives, and caches.
- Run the portable CPU check in the matching environment. Review `VALIDATION.md`
  and the independent Cloud result for the exact candidate revision.
- Use the `on-policy-qwen38` environment. Python 3.11 and the pinned Transformers
  version match the intended Amarel path.

## Qualify the allocated kernel

Run inside the allocated node, preserving SLURM's visible GPU mapping. Check the
hostname, visible devices, GPU names, and memory. Use one Jupyter kernel for this
experiment so a second teacher does not occupy its memory budget.

Run 05 in `inspect`, then `gpu_check`. Its qualification records real scoring,
synthetic worst-shape probes, one disposable update, reload verification, timing,
and per-device memory. The disposable weights are discarded. Keep the report.

## Build the supervised start

Open 03 and select `MODEL_PROFILE = "qwen38"`, `MODE = "train"`.
Leave `RESUME_RUN_DIRECTORY = None` for a new experiment. Save the printed folder.
The shared loop checkpoints every eight updates. If interrupted, put that same
folder into `RESUME_RUN_DIRECTORY` and rerun from the top with the same software.

Inspect validation results and actual responses. Preserve the new checkpoint and
its report together. The historical checkpoint remains separate.

## Run the controlled pilots

Open 04 with the same profile and set `SFT_RUN_DIRECTORY` to the completed 03 folder.
Select `MODE = "train"` and `RUN_STAGE = "pilot"`. It runs learning rates `1e-6` and
`5e-6` sequentially from that same checkpoint, with a shared frozen teacher.

The 32-update pilots retain the full 128-update scheduler horizon. Selection uses
strict validation accuracy, then NLL, then the lower rate among qualifying pilots.
Safety pauses protect quality and stopping behavior. If both fail, preserve the
negative result and finish the predeclared validation study.

## Continue and evaluate

For a qualified pilot, set `RUN_STAGE = "full"` and `RESUME_RUN_DIRECTORY` to its
parent run folder. The winning optimizer continues. The reference-prefix KD control
starts from the same SFT weights. Held-out evaluation begins after both full arms
complete. A quality-paused run can finish evaluation without restarting its optimizer.

Keep raw responses, histories, selection, runtime metadata, and checkpoints. The
small report supports review and contains no weights. Measured PNG/SVG figures are
created from actual histories. Final metrics include paired question-level intervals.

A six-hour allocation can end before all stages finish. The runner reserves ten
minutes, saves an optimizer boundary, and resumes unfinished validation by question.
A matching software environment and CUDA device count are required for optimizer
recovery. Abrupt loss can discard work after the last published checkpoint.
