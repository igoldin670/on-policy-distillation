# Next Colab session

## Starting files

Keep the verified Notebook 03 `checkpoint.zip` and its matching `report.zip`.
The recovered checkpoint is 2,164,329,683 bytes with SHA-256
`4ecf79a79f32e89944a606c41b79ce52d65841c151d25f5ead016648cd730eb6`.
The matching report SHA-256 is
`33a38d64b7b3831712d127d870b5359220d84add820c7749e17b0b08df36a316`.

Use a fresh A100 80GB kernel in VS Code. Sync the complete `dev` checkout on
Colab before executing the local notebook. Runtime files and local files are
separate. The old raw-prompt training run is archived negative evidence.

## Run the pilot

1. Set `MODE = "train"` in the first setup cell. Restart once if setup changes dependencies.
2. Set `RUN_STAGE = "pilot"`, `RESUME_RUN_DIRECTORY = None`, and the server-side
   `SFT_REPORT_PATH` / `SFT_CHECKPOINT_PATH`. Folders and ZIPs are supported.
3. Run sequentially. Confirm checkpoint hashes, kernel PID, free GPU memory,
   shared tokenizer mapping, and the teacher-template fingerprint.
4. Inspect starting-student and teacher validation tables. Teacher-generated
   validation answers are diagnostic; they are not distillation targets.
5. Confirm the aligned response-logit shapes agree while prompt lengths differ.
6. Review the 32-update pilot, its generated responses, validation NLL, fixed-prefix
   KL, and EOS/cap counts. The full test remains gated.

The pilot uses the full 128-update scheduler from the beginning. Its 32 updates
are the first part of that experiment. Validation and snapshots occur every eight
pilot updates. An engineering guard pauses early when at least 24 of the latest
32 responses hit their cap. A 25% reference-NLL increase produces a warning.
An alarm is evidence to investigate; it is not a published statistical criterion.

## Continue after review

Retain the pilot's **entire run folder**, including `resume/on_policy`, the
verified SFT-control checkpoint, JSON records, and sampled trajectories. On a
replacement runtime, upload that folder and the original 03 artifacts. A report
archive alone cannot resume training because it excludes weights and optimizer state.

Restart the notebook kernel to release the teacher and student. In the same code
revision, set `RUN_STAGE = "full"` and `RESUME_RUN_DIRECTORY` to the pilot folder,
then execute from the top. Cached within-run evaluations are reused. The saved
student, optimizer, scheduler, and RNG resume at an exact optimizer boundary.
A changed context/template/code signature requires a new experiment.

Full final-test generation, paired comparisons, and final checkpoint export run
only after 128 successful updates. Judge numeric agreement, formatting, stopping,
changed answers, and uncertainty together. A lower training KL is one diagnostic.

## Preserve and disconnect

Download the small report separately from model/recovery files. Packaging verifies
archive entries; file transfer requires your runtime's file browser. For unreliable
large transfers, split the ZIP into 200 MiB pieces on Colab, download every numbered
piece, join them locally in order, and compare the resulting SHA-256 with the server.
Keep the original server files available until verification passes.

A notebook-kernel restart releases that kernel's variables. Other notebook kernels
may still occupy the GPU. Compare the printed PID with `nvidia-smi` before loading
another teacher. Process termination should target an identified obsolete kernel.
