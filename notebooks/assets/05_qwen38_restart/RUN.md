# Run the new model checks

The active model profile is `configs/small_experiment.yaml`:

- Student: `Qwen/Qwen3.5-0.8B-Base`, revision `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`.
- Teacher: `Qwen/Qwen3.8-27B`, revision `37561816e66b5b9153b1b64c10968c18c1e6f4a0`.

The existing 01-04 notebooks, original 03 archives, and EOS pilot snapshots belong
to the historical Qwen3 experiment. Notebook 04 reads `configs/qwen3_legacy.yaml`
so changing the active profile cannot silently resume an old run with new models.
Its teachers and checkpoint signatures remain unchanged.

## 1. Sync the complete project

Run from the laptop terminal after saving notebook outputs:

```bash
rsync -avP \
  --exclude='.git/' --exclude='.cache/' --exclude='.ruff_cache/' \
  --exclude='results/' --exclude='artifacts/' --exclude='.venv/' \
  --exclude='__pycache__/' \
  /home/isaac/Code/on-policy/on-policy-distillation/ \
  igg11@amarel.rutgers.edu:~/on-policy-distillation/
```

No `--delete` is used. Existing remote results and checkpoint archives remain in place.

## 2. Create a separate user environment

On Amarel, these commands create only a new user environment. They do not change
the old `~/venvs/on-policy` environment or install system packages. The explicit
CUDA wheel matches the previously working driver/runtime combination.

```bash
cd "$HOME/on-policy-distillation"
"$HOME/.local/bin/uv" venv --python "$HOME/venvs/on-policy/bin/python" \
  "$HOME/venvs/on-policy-qwen38"
"$HOME/.local/bin/uv" pip install \
  --python "$HOME/venvs/on-policy-qwen38/bin/python" \
  torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
"$HOME/.local/bin/uv" pip install \
  --python "$HOME/venvs/on-policy-qwen38/bin/python" \
  -r requirements-qwen38.txt
```

The constraints retain PyTorch 2.10.0, including the compatible `+cu128` build.
The CPU inspection below downloads tokenizer/configuration files but no model weights:

```bash
export HF_HOME="/scratch/$USER/huggingface"
"$HOME/venvs/on-policy-qwen38/bin/python" scripts/inspect_qwen38.py
```

Expected status: `identical_tokenizers_verified`. A fresh report is written under
`results/qwen38_readiness_*`. The script does not require a GPU unless `--gpu` is supplied.

## 3. Request GPUs through SLURM

Use the previously accepted account, partitions, and feature constraints. Run the
request inside tmux on a known login node so an SSH disconnect does not terminate it:

```bash
tmux new-session -s qwen38
srun --account=general --partition=gpu,cgpu \
  --constraint='ampere|adalovelace' \
  --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G \
  --gres=gpu:2 --time=06:00:00 --job-name=qwen38-check --pty bash -l
```

The six-hour request covers setup, downloads, and qualification. Actual runtime
depends on the allocated GPUs and the attention implementation.
Partition availability, account limits, hardware, and queue times must be rechecked.
After allocation, inspect `nvidia-smi` and PyTorch's visible GPUs. Preserve SLURM's
`CUDA_VISIBLE_DEVICES`. An Ampere feature can identify different GPU models; the
memory/BF16 guard still needs to pass.

## 4. Start Jupyter or run the script

Inside the allocated shell:

```bash
source "$HOME/venvs/on-policy-qwen38/bin/activate"
cd "$HOME/on-policy-distillation"
export HF_HOME="/scratch/$USER/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
python -m jupyterlab --no-browser --ip=0.0.0.0 --port=8888 \
  --ServerApp.port_retries=0
```

On the laptop, replace `ALLOCATED_NODE` with the new hostname. The former
`gpuk011` allocation has ended; do not assume that node is allocated again.

```bash
ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
  -L 8890:ALLOCATED_NODE:8888 igg11@amarel.rutgers.edu
```

In VS Code select the existing server at
`http://127.0.0.1:8890/?token=YOUR_PRIVATE_TOKEN`. Open `05_qwen38_restart.ipynb`,
first use `MODE = "inspect"`, then use `MODE = "gpu_check"` and run from the top.
Alternatively, from the allocated shell run:

```bash
python -u scripts/inspect_qwen38.py --gpu --qualify
```

The first new teacher download is about 55.6 GB. The old Qwen3-32B cache cannot
substitute for it. Check personal disk quota, cache capacity, and allocation time.
Optional fast DeltaNet kernels are not installed by this setup. The reference
PyTorch path is suitable for checking correctness but may be substantially slower.

## 5. Interpret and preserve the result

The scoring phase should report `new_model_scoring_passed`, finite aligned scores,
exact response IDs, and per-GPU memory. The additional qualification reports
`disposable_gpu_qualification_passed` after synthetic long-shape probes, one disposable
update, cached generation, and reload verification. Temporary updated weights are
discarded. These probes establish execution and memory for the tested shapes.

Continue in [03](../../03_sft_cold_start.ipynb) with `MODEL_PROFILE = "qwen38"` and
`MODE = "train"`. Its checkpoint and report stay in the persistent run folder.
Then use [04](../../04_on_policy_distillation.ipynb) for the two fixed learning-rate
pilots and matched reference-prefix control. Follow the [final checklist](../../../FINAL_RUN.md).
Old checkpoints remain historical evidence and have a different model architecture.

## Source references

- [Teacher card](https://huggingface.co/Qwen/Qwen3.8-27B)
- [Student card](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base)
- [Text-model implementation at the tested Transformers release](https://github.com/huggingface/transformers/blob/v5.3.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py)
