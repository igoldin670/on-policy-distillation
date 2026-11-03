# Notebook 04 validation

**Status: implemented and locally validated. Real Qwen distillation is pending.**

Validation date: 2026-11-02. The source notebook contains actual **CPU inspection**
outputs, explicitly labelled as such. They are not Qwen model-generation or
training results. No local test substitutes for the A100 experiment.

## Passed checks

| Check | What actually ran |
| :--- | :--- |
| Unit and integration regressions | **72 tests passed**, no failures, errors, or skips; includes the existing 41 tests |
| Clean-kernel 04 inspection | All **15 code cells** executed sequentially without errors or stderr; real pinned tokenizer/data/configuration, no model weights |
| Real source isolation | 1,024 new training questions, 64 prior validation questions, 256 new primary test questions, 32 separately reported historical test questions |
| Tokenizer compatibility | Full token-to-ID mappings match; **2,180 text probes** encode/decode consistently; output heads configured for 151,936 columns |
| Tensor boundaries | Prompt masking, response-only labels, padding, genuine EOS targets, and the `P - 1` causal prediction boundary |
| Exact KL implementation | Chunked full-vocabulary forward KL and its student gradient match a dense reference; teacher gradients are absent |
| Actual notebook training code | Both arms ran with tiny **randomly initialized** 32-token Qwen3 models; includes a partial accumulation group, checkpoint/tokenizer reload, and frozen-teacher checks |
| Actual notebook final cells | Paired evaluation, changed-answer analysis, both plot exports, and checksummed report export ran with those tiny test models |
| Restart behavior | Optimizer/scheduler/RNG reload reproduces the next CPU optimizer update; stale/missing/reordered rollout logs are rejected; interrupted finalization can be recovered |
| Checkpoint/export behavior | Saved inference defaults retain both stop IDs; two owned restart snapshots are retained; unrelated directories survive; repeated report export preserves completed evidence |
| Earlier notebook regressions | 02: clean 15-cell CPU replay; 03: clean 11-cell CPU inspection; saved notebooks 01, 02, and existing local 03 remain byte-identical |
| Presentation | Notebook JSON, Python syntax, guidance comments, anchors, relative links, three PNG/SVG pairs, embedded-image HTML export, and **12 KaTeX-rendered equations** |
| Code review | Ruff lint/format checks and `git diff --check`; no credential-token patterns found in intended text changes |

Tiny random models test the implementation. Their generated text, losses, and
zero-shot scores are not reported as pretrained-Qwen performance. Test artifacts
are temporary; actual Qwen performance figures are created only in CUDA mode.

## What changed for reproducibility

- Student and dataset revisions remain pinned to 02/03. The teacher is pinned to
  `9216db5781bf21249d130ec9da846c4624c16137`.
- Student sampling uses temperature 1, `top_k=0`, and `top_p=1`. 04 explicitly sets
  `use_model_defaults=False`, preventing Transformers from replacing these values
  with a model's stored sampling defaults. Paired evaluation does the same.
- 04 recognizes both `151643` and `151645` as generation stops for all three arms.
  The token meanings match; their default EOS roles differ. Historical 02/03
  scores used a different stopping policy and are contextual comparisons only.
- Both 04 training losses average within each response, then across responses.
  Validation reference NLL retains the pooled-token definition from 03. These
  quantities have different weighting and should not be conflated.
- All eight trajectories for an update come from the unchanged current student.
  They are discarded after that update. Incorrect/capped responses are retained;
  capped responses receive no invented EOS.
- Teacher parameters are frozen and excluded from the optimizer. Response-position
  chunking retains the entire vocabulary; it is not a top-k approximation.
- Reports record upstream SFT split evidence, model/checkpoint fingerprints,
  code/protocol identity, raw responses, trajectories, environment, and observed
  resource measurements. Weights and optimizer state are separate.
- Shared checksum operations now stream large files rather than reading multi-GB
  checkpoints into RAM. Existing evaluator semantics are unchanged by formatting.

## Upstream evidence and preservation

The supplied Notebook 03 `report.zip` has **14 valid inventory hashes**. Its
before-validation, after-validation, and final-test scores reproduce with the
unchanged answer-checker logic. `cold_start_lineage.json` records its metadata,
split indices, score aggregates, and expected checkpoint fingerprints.

That report contains **no model weights**. The actual user-run 03 checkpoint was
not available locally during this review. 04 will verify the real checkpoint
against the report on Colab before loading it. No claim is made that its weights
were locally backed up or that its CUDA forward pass was rerun here.

An ignored backup of the pre-edit notebooks, README, and SFT helper is under
`.cache/backups/pre_notebook04/`. Existing local 03 download-cell edits were
preserved, along with its saved A100 outputs. No original 01/02 evidence changed.

## Still needs the Colab run

The actual 32B teacher load, A100 memory headroom, mixed-precision training,
real sampled trajectories, wall time, final checkpoint reload on CUDA, and
generated-answer performance remain unverified. The preflight guard estimates
resource needs; it is not an execution guarantee.

Run 04 from the top in the VS Code Colab kernel with an almost-empty A100 80GB,
`MODE = "train"`, and 03's matching report/checkpoint folders or ZIPs. Keep the
fixed experiment unchanged when using the primary held-out test. Download both
the small report and final weights, and save the executed notebook.

This is one training seed and a custom subset. The bootstrap interval measures
question-sampling uncertainty, not training-seed variability. The SFT comparison
also changes the learning signal and compute cost, so it is not an isolated
on-policy-versus-off-policy KD ablation. Teacher accuracy under the shared plain
prompt has not been measured separately.

## Reproduce local checks

After following the CPU environment setup in the repository README:

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/validate_notebook02.py
.venv/bin/python scripts/validate_notebook03.py
.venv/bin/python scripts/validate_notebook04.py
.venv/bin/python scripts/render_notebook04_assets.py
.venv/bin/python -m ruff check src scripts tests notebooks/04_on_policy_distillation.ipynb
.venv/bin/python -m ruff format --check src scripts tests notebooks/04_on_policy_distillation.ipynb
git diff --check
```

Validators write separate executed notebook copies and HTML previews under ignored
`results/notebook02_qa/`, `results/notebook03_qa/`, and `results/notebook04_qa/`.
`--save-inspection` explicitly saves genuine inspection outputs to 04. The earlier
notebooks' saved GPU outputs are preserved by the default validation commands.
Exact local facts and verification scope are in `validation_summary.json`.
