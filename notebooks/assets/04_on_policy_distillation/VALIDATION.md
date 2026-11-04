# Notebook 04 validation record

Reviewed on 2026-11-04. **The corrected experiment is locally validated and awaits
a staged Qwen run on Colab.** The source notebook contains genuine CPU inspection
outputs. The archived legacy experiment and teacher probe are actual user-run
A100 evidence with separate provenance.

## Evidence boundaries

| Evidence | Scope |
| :--- | :--- |
| Interrupted raw-context run | 52 downloaded updates and 416 trajectories; later console output through update 61 |
| User-run frozen-teacher probe | One response under two teacher contexts; strong stopping-mass contrast |
| Corrected teacher scoring | Pinned non-thinking chat prefix; identical sampled response IDs; separate causal offsets |
| Local regression suite | Exact KL/gradients, frozen teacher, both stop IDs, training, staging, recovery, and archives |
| Clean 04 CPU inspection | Real pinned tokenizer/data/configuration; no teacher or student weights loaded |
| Recovered real 03 student | CPU load, 676 tokenizer probes, finite `[1,16,151936]` logits; no training |
| Corrected Qwen training | Pending Colab pilot and full-run qualification |

The investigation and source records are in [INVESTIGATION.md](INVESTIGATION.md)
and [recorded_failure](recorded_failure/README.md). The template change supports a
new experimental hypothesis. One stopping probe does not establish improved accuracy.

## Reproducibility and regression checks

The experiment signature includes pinned model identities, actual checkpoint
fingerprints, split identities, student evaluation policy, teacher-context version,
chat-template fingerprint, and implementation code. Old raw-context snapshots are
incompatible with the corrected experiment.

Student sampling uses temperature 1, `top_k=0`, `top_p=1`, both stop IDs, and
`use_model_defaults=False`. Response IDs are appended directly to the teacher's
chat prefix. Each model's own `P - 1` shift yields `[1,R,V]` response logits.
Full-vocabulary KL uses FP32 softmax and detached teacher targets at temperature 0.1.
Both training objectives use a mean within each response, then across responses.

The actual notebook loop is exercised with tiny random CPU Qwen3 models. Tests
cover the default 32-of-128 pilot boundary, partial groups, exact equality between
paused/resumed and uninterrupted weights, trajectories, scheduler and RNG state,
health-pause snapshots outside periodic save points, and blocked unsafe continuation.
The teacher's end ID is tested through the actual Transformers generation entry point.

Pilot and full reports have separate immutable archive names. Weights stay outside
reports. Export checks compare entry hashes with the source inventory and expose a
final archive only after packaging completes. Corrupt existing ZIPs produce a clear
error while preserving source files.

## Presentation and operational checks

All notebooks have schema/syntax, guidance-comment, navigation, relative-asset,
HTML-export, image, and equation checks. Updated diagrams explicitly distinguish
question contexts from response prefixes. Historical model outputs are retained in
01–03; 04's old errors and mixed-mode outputs are archived separately.

The browser preview permission was declined. Offline HTML exports and direct PNG
inspection were completed; live desktop/mobile GitHub layout remains unverified.
CPU checks use PyTorch 2.10.0+cpu; the user-run GPU records report 2.11.0+cu130.
Cross-hardware or cross-version bitwise identity is not claimed.

## Next GPU acceptance checks

Follow [NEXT_RUN.md](NEXT_RUN.md). Qualify teacher validation performance and response
alignment, inspect the 32-update pilot and stop behavior, then explicitly continue
its saved optimizer boundary. The primary exam stays gated until full completion.
A health-paused recipe requires diagnosis and a fresh revised experiment.

Observed accuracy improvement, mixed-precision memory headroom, timings for the
corrected context, full CUDA checkpoint reload, and final held-out results remain
unverified. Higher teacher stopping mass and lower KL cannot establish these outcomes.

## Reproduce local checks

```bash
python -m unittest discover -s tests -v
python scripts/validate_notebook02.py
python scripts/validate_notebook03.py
python scripts/validate_notebook04.py
python scripts/validate_notebook_series.py
python scripts/render_notebook04_assets.py
python -m ruff check src scripts tests notebooks/*.ipynb
python -m ruff format --check src scripts tests notebooks/*.ipynb
git diff --check
```

For independent equation rendering, use `quickjs==1.19.4` in the QA environment and
pass the official KaTeX bundle to `validate_notebook_series.py --katex-js PATH`.
The recorded review used KaTeX 0.16.22. `validation_summary.json` records the exact
checks and their limits. Local validators preserve source evidence by default.
