# Historical Qwen3 investigation guide

## Current project direction

The November 16 EOS comparison has completed and its downloaded records passed
local auditing. Both pilots failed their quality gates. The first measured strict
accuracy decline occurs at update 8. See the [audited evidence](recorded_eos_comparison/README.md).

For the selected Qwen3.5/Qwen3.8 upgrade, use [Notebook 05's run guide](../05_qwen38_restart/RUN.md).
The immediate GPU milestone is one student response and frozen-teacher scoring.
A fresh new-model baseline and SFT checkpoint follow after that qualification.

The remaining instructions retain the historical Qwen3 reproduction workflow.
A new guarded run has a new signature. Old snapshots retain their original policy.


For a fresh Colab run using the 03 archives, follow [COLAB_RUN.md](COLAB_RUN.md).
This Amarel guide describes the optional historical-checkpoint investigation.

## Current evidence and next experiment

The November 8 pilot reached 22 updates and the cap-rate health guard. Its
follow-up validation was interrupted by SLURM's two-hour limit. The original
snapshot and completed SFT control are historical evidence. The next experiment
compares raw-token and semantic-STOP forward KL from the same verified 03 weights.

The 03 checkpoint SHA-256 is
`4ecf79a79f32e89944a606c41b79ce52d65841c151d25f5ead016648cd730eb6`.
Its matching report SHA-256 is
`33a38d64b7b3831712d127d870b5359220d84add820c7749e17b0b08df36a316`.
Keep both at `artifacts/sft03/` in the remote project checkout.

## 1. Sync and collect the historical evidence

Sync the complete updated project from your laptop, including `src/`, `scripts/`,
assets, and the notebook. A remote Jupyter connection runs local notebook cells
against the remote filesystem; it does not synchronize helper files.
Preserve remote `results/`, model caches, and input archives during transfer.

From your laptop terminal, copy the updated project and its helper modules.
These exclusions keep runtime artifacts and local environments separate.

```bash
rsync -avP \
    --exclude='.git/' --exclude='.cache/' --exclude='results/' \
    --exclude='artifacts/' --exclude='.venv/' --exclude='venv/' \
    --exclude='__pycache__/' \
    /home/isaac/Code/on-policy/on-policy-distillation/ \
    igg11@amarel.rutgers.edu:~/on-policy-distillation/
```

On Amarel's login node, the following read-only source inspection verifies the
16/22 snapshot inventories and completed SFT-control checkpoint, then packages
JSON records and inventories. It creates a new evidence folder and ZIP. It never
loads models or modifies the recorded source run.

```bash
cd "$HOME/on-policy-distillation"
source "$HOME/venvs/on-policy/bin/activate"
python scripts/collect_opd_evidence.py \
    results/on_policy_distillation_20261108T184539826093Z \
    results/eos_source_20261108
```

Choose a fresh destination if it already exists. Download `eos_source_20261108.zip`
and retain a local backup. It excludes tensor weights and optimizer states, which
stay in the original Amarel run folder. Local files downloaded from the older
November 4 run must not be substituted for these records.

## 2. Allocate GPUs and reconnect Jupyter

Request one node with two Ampere GPUs and a six-hour limit. The `gpu` partition
keeps this experiment on the Piscataway side, where its scratch cache was created.
The preflight checks the actual allocated memory before loading weights.

```bash
srun --account=general --partition=gpu --constraint=ampere \
    --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G \
    --gres=gpu:2 --time=06:00:00 --job-name=opd-eos --pty bash -l
```

Inside the allocation, activate the existing user environment and inspect hardware.
Keep SLURM's `CUDA_VISIBLE_DEVICES` intact.

```bash
source "$HOME/venvs/on-policy/bin/activate"
cd "$HOME/on-policy-distillation"
hostname -s
python -c 'import torch; print(torch.cuda.device_count())'
nvidia-smi
```

The notebook requirements preserve CUDA-enabled PyTorch. Install them in this
user environment if needed. The complete CPU QA lockfile targets a different
Python version and is separate from the Python 3.11 GPU setup.

```bash
"$HOME/.local/bin/uv" pip install --python "$VIRTUAL_ENV/bin/python" \
    -r requirements-opd.txt pandas matplotlib bottleneck
python -m jupyterlab --no-browser --ip=127.0.0.1 --port=8888 \
    --ServerApp.port_retries=0
```

Connect your laptop's SSH tunnel to the printed compute hostname through Amarel,
then select the existing Jupyter server in VS Code. Keep the token private.
Run inside the SLURM allocation throughout model execution.

## 3. Configure and run the notebook in order

Restart the kernel before the new code. Section 2 sets the Hugging Face cache to
`/scratch/<username>/huggingface` before library imports when it detects a SLURM
allocation and personal scratch directory. Existing explicit cache overrides are
respected. No separate scratch-cache cell is required. Downloaded teacher files
can be reused. Scratch is temporary, unbacked-up storage.

Use these settings for a new comparison.

```python
MODE = "train"
RUN_STAGE = "eos_comparison"
EOS_LOSS_MODE = "raw_token"  # The comparison constructs both interpretations.
RESUME_RUN_DIRECTORY = None
HISTORICAL_RUN_DIRECTORY = PROJECT_ROOT / "results/on_policy_distillation_20261108T184539826093Z"
```

`HISTORICAL_RUN_DIRECTORY` must point to the original November 8 source run.
Code, 03 inputs, new results, and recovery snapshots stay in the home checkout.
Check your personal storage quota before starting. The filesystem-space check
cannot determine that quota. Two retained snapshots per new arm require roughly
30 GiB in addition to existing runs, dependencies, and source checkpoints.

Run every section in order after restart.

1. Sections 2–4 prepare paths, verify inputs/resources, rebuild data, and explain losses.
2. Section 5 verifies tokenizer identity and measures the shared starting student.
3. Section 6 verifies and displays the historical SFT control, including its recipe.
4. Section 7 loads the teacher once, qualifies alignment, probes 03/16/22 snapshots,
   and finishes the interrupted update-22 validation into the new diagnosis folder.
5. Section 8 trains two independent student policies with matched settings. Only
   EOS interpretation differs. Each ends at 32 updates or a health/walltime pause.
6. Section 9 final-test cells stay gated during this comparison.
7. Section 10 exports the small comparison report, including both arm histories.

Validation occurs at updates 8, 16, 24, and 32 and stopping boundaries. Both KL
definitions are recorded on identical fixed prefixes. Lower grouped KL alone
cannot establish an improvement because grouping changes the objective.

## 4. Recover interrupted work and review

The SLURM deadline guard reserves ten minutes for checkpoint publication.
Pause reason, update count, and snapshot path are written before follow-up
validation. Each completed validation response is saved atomically. The same
checkpoint, questions, tokenizer, and decoding policy are required for cache reuse.

Each pilot writes `progress.jsonl` under its `eos_comparison/raw_token` or
`eos_comparison/semantic_stop` folder. From that folder, run `tail -n 10 progress.jsonl`
to see the latest update and stage, or `tail -f progress.jsonl` to follow the log live.
Stage entries are written before generation, teacher scoring and backpropagation,
optimizer steps, checkpointing, and validation begin. After an interruption, the
last `started` entry shows which operation the notebook reached. Use `manifest.json`
and the saved snapshots to determine how far training can resume.

After an allocation ends, set `RESUME_RUN_DIRECTORY` to the comparison's parent
run folder and keep `RUN_STAGE = "eos_comparison"`. Preserve the same code and
recipe. A stopped raw arm can finish pending validation without new updates.
An unfinished healthy arm resumes its own optimizer/RNG state. Each arm retains
its own two most recent recovery snapshots. Keep the entire run folder available.

Review `eos_comparison_gates.json`, all sampled text, and validation results.
The grouped pilot must finish 32 updates, keep its final cap rate at or below 25%,
keep reference NLL within 10% of the baseline, and reach at least 32/64 strict
validation matches. The matched cap-rate reduction must be at least 25 percentage
points. These are engineering review gates on one training seed.

If the result is incomplete, degraded, or inconclusive, preserve it and review the
hypothesis. A later learning-rate or temperature experiment is a separate change.
The raw health-paused policy must not continue optimizer updates.

After a passing reviewed comparison, restart the kernel and select the semantic
arm for continuation.

```python
RUN_STAGE = "full"
EOS_LOSS_MODE = "semantic_stop"
RESUME_RUN_DIRECTORY = Path("<comparison run>/eos_comparison/semantic_stop")
```

The original 128-update scheduler continues from the saved 32-update boundary.
Final held-out generation occurs after full completion. Numeric accuracy, format,
EOS/cap behavior, and changed answers are reviewed separately. A passing stopping
comparison does not establish that distillation beats the recorded SFT control.

## Optional hardware requalification

If the device layout changes, `scripts/smoke_opd_gpu.py` exercises one disposable
raw-token update with peak memory reporting. Its constant learning rate and single
response differ from the eight-response pilot schedule. Run with no other model
copies loaded and retain the output as hardware evidence. A short smoke response
cannot establish the memory requirement of every maximum-length sequence.


## Guarded reproduction policy

A fresh 04 run signs its safety policy into experiment and checkpoint metadata.
It pauses above 8/32 sampled caps, above 110% of starting NLL, or after a strict
accuracy drop exceeding five percentage points. It preserves a last acceptable
checkpoint, publishes pause metadata before validation, and packages a small
report automatically. A quality pause permits evaluation recovery. Review the
recipe before starting any new optimization from the original 03 checkpoint.

To collect a completed comparison, use the same collector with that run directory.
It discovers both arms and their actual retained snapshots dynamically.

```bash
python scripts/collect_opd_evidence.py \
  results/on_policy_distillation_20261116T022121085005Z \
  results/eos_evidence_fresh_copy
```
