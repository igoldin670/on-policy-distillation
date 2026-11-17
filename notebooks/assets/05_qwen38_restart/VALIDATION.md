> Earlier revision record. The current profile-aware workflow and release checks
> are documented in the [project validation record](../../../VALIDATION.md).
> Original GPU executions remain preserved in the historical evidence.

# Model-upgrade validation

Reviewed November 16–17, 2026. Actual pinned-file inspection, tiny CPU tests, and
user-run GPU experiments have distinct scopes.

## Pinned model and tokenizer checks

`tokenizer_verification.json` records revisions and downloaded metadata hashes.

- Student `Qwen/Qwen3.5-0.8B-Base` at `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`.
- Teacher `Qwen/Qwen3.8-27B` at `37561816e66b5b9153b1b64c10968c18c1e6f4a0`.
- All 248,077 resolved token mappings and BPE processing match.
- Both output heads contain 248,320 rows, including 243 padded rows.
- Encoding/decoding matches on 1,008 probes covering numbers, whitespace,
  punctuation, Unicode, thinking delimiters, and tool/audio tokens.
- Ending IDs are 248044 and 248046.
- The non-thinking teacher template returns a flat 20-token prefix for the
  inspection question `What is 2 + 2?`.

The student's official tokenizer configuration supplies eleven definitions beyond
its raw tokenizer JSON. Standard loading resolves the apparent size difference.
No weights were downloaded for this inspection. The author independently ran
`inspect_qwen38.py` on Amarel and reported `identical_tokenizers_verified`, with
`real_model_scoring` still null.

## Executed local checks

Local QA uses Python 3.14.7 and PyTorch 2.10.0+cpu. New architecture tests run
with Transformers 5.3.0, Hub 1.5.0, and Tokenizers 0.22.2. Historical QA retains
Transformers 4.57.6. The intended Amarel environment uses Python 3.11 and CUDA.

- The historical regression suite ran 157 tests, with 156 passing and one
  architecture class skipped in the older Transformers environment.
- All nine new-model tests passed under Transformers 5.3.0. Tiny random models
  cover tied/untied text-weight loading, hybrid cached generation, causal alignment,
  response-only backpropagation, finite gradients, and checkpoint round trips.
- The complete one-response scoring helper ran with tiny random CPU models and
  a CPU precision context. This verifies its code path using test fixtures.
- A cache test verifies that unrelated cached revisions cannot reduce the required
  space estimate. Personal filesystem quota still needs separate HPC inspection.
- Notebook 05 executed all ten code cells in a clean CPU kernel, with zero errors
  and no pretrained model weights. Source outputs remain empty. Separate notebook,
  HTML, and execution report files are under `results/qwen38_upgrade_qa`.
- All five notebooks passed schema, syntax, comment, relative-link, equation, and
  HTML-export checks. All 50 PNG/SVG assets passed integrity or XML checks. The new
  diagrams and measured pilot-history figure received direct visual inspection.
- Ruff lint/format checks and `git diff --check` passed for the intended changes.
  Notebook 03 has pre-existing local edits and remains outside this publication.

The tests caught two Transformers 5 API differences. New generation omits the
removed `use_model_defaults` argument. Teacher chat tokenization explicitly sets
`return_dict=False`. The CPU reference implementation supports hybrid attention
without optional compiled acceleration. Real runtime must be measured on the GPUs.

Private clean-kernel QA used working local Jupyter IPC after a sandboxed attempt
stalled. No saved GPU output was replaced by CPU inspection.

## Historical failure and revised safeguards

The downloaded old-model EOS comparison now passed local file-inventory, trajectory,
and checker auditing. All 400 trajectories are consistent. Strict accuracy first
declines at update 8 in both arms. The new baseline-relative guard requests a pause
at that boundary when replayed on those records. Raw-token training ended at 18
updates. Grouped-stop training reached 32, with better stopping and lower answer
accuracy than its starting checkpoint. Both failed their original engineering gates.
See [the evidence record](../04_on_policy_distillation/recorded_eos_comparison/README.md).

Current safeguards preserve the last acceptable checkpoint, verify recovered
measurements before further updates, check finite parameters, log failure stages,
reserve SLURM checkpoint time, and package pause reports. Old 04 sources are archived
and match the actual GPU run's signature. New-model helpers live under `src/qwen38/`.
The source notebook shows the guard on a labeled synthetic example.

## Pending GPU checks

- Real pretrained student generation and frozen teacher scoring.
- Peak GPU memory, hybrid attention runtime, and longer response behavior.
- A fresh new-model baseline and supervised checkpoint.
- Revised safety behavior during actual optimization.
- Any improvement in stopping, arithmetic, formatting, or held-out performance.

The original 03 checkpoint belongs to the older Qwen3 architecture. Historical
archives remain preserved. This work creates no new pretrained or trained weights.
