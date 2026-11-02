# Notebook 03 final validation record

Notebook 03 has a completed user-run A100 supervised experiment. The notebook's
saved model outputs are preserved exactly; local CPU QA writes separate artifacts.
No teacher or on-policy loss was used in this supervised stage.

## Observed A100 run

The saved execution reports an A100-SXM4-80GB, PyTorch 2.11.0+cu130, FP32 parameter
storage with BF16 autocast, 256 practice examples, 64 separate validation examples,
and 32 optimizer updates in 82.2 seconds (excluding generation and export).
All 11 code cells completed without a saved exception.

| Validation measurement | Before | After |
| :--- | ---: | ---: |
| Required format | 0/64 | 59/64 |
| Strict numeric matches | 0/64 | 35/64 |
| Exploratory prose numeric matches | 31/64 | 35/64 |
| Response-target NLL | 0.776109 | 0.667247 |
| Token-limit stops | 0/64 | 5/64 |

Final reserved-test results were 14/32 strict matches, 32/32 format compliance,
and 13/32 exploratory prose matches. Historical 02 prose matching was 17/32 under
different execution provenance. This small experiment demonstrates strong format
learning and mixed reasoning results, not broad reasoning improvement. The saved
watermelon example regressed from a correct 10 to an incorrect formatted 26.

## Preservation and limits

An ignored local backup of the pre-finalization notebook (including A100 outputs),
training helper, and README is under `.cache/finalize_notebook03/`, with hashes.
The code and saved outputs are also versioned. The actual Colab report ZIP and
checkpoint weights are **not present locally** and have not been independently
audited here. A notebook is not a weight backup; preserve both Colab downloads
before the runtime expires. Notebook 04 remains a scaffold.

The saved checkpoint message came from the original vocabulary/logit check.
That check passed but also emitted a tokenizer regex warning. The final helper
adds text-to-token and token-to-text checks; this enhanced check has not been
rerun against the actual Colab checkpoint by local QA.

## Tokenizer warning investigation

Transformers 4.57.6 can classify a locally saved Qwen3 config in its Mistral regex
migration logic. We reproduced the warning with the pinned Qwen tokenizer saved
beside minimal Qwen3 identifying config, without student weights. For confirmed
`model_type = "qwen3"`, the loader explicitly sets `fix_mistral_regex=False`.
It never applies Mistral's regex to Qwen3 and leaves other model families alone.

The tokenizer-only check compares all selected training and validation prompts
and reference answers, the 32 reserved prompts, and whitespace, number,
punctuation, Unicode, and final-marker examples. It checks both special-token
encoding modes, decoding, vocabulary, special-token meanings, and saved IDs.
See `tokenizer_verification.json` for the actual probe count and result.

## Local checks

- 41 tests pass, including the 25 earlier baseline tests and 16 SFT tests.
- The actual notebook training loop is exercised with a tiny random CPU Qwen3
  model; only CUDA housekeeping is mocked. This is not pretrained-student training.
- Unequal-length gradient accumulation matches a combined batch; padding and
  end-token handling, causal shift, source isolation, generation input, optimizer
  changes, checkpoint reload, changed tokenizer encodings, and report integrity
  have dedicated checks.
- All 11 cells run in a separate clean CPU inspection kernel without errors or
  stderr. No `--save-inspection` flag is used during finalization.
- Schema, syntax, Markdown/navigation links, HTML export, math rendering, artwork,
  Ruff, and Git diff checks are performed. No live GitHub renderer screenshot is claimed.
- Notebooks 01 and 02, their evidence, and Notebook 04 are unchanged.

## Reproduce

Use an isolated environment with PyTorch, `requirements-notebook.txt`, and
`requirements-sft.txt`. Preserve Colab's existing PyTorch/CUDA build.

```bash
python -m unittest discover -s tests -v
python scripts/validate_notebook03.py
python scripts/verify_sft_tokenizer.py
```

The notebook validator writes a separate executed inspection notebook and HTML
under ignored `results/notebook03_qa/`. Its optional `--save-inspection` flag would
replace source outputs and must not be used to preserve the saved training run.
The tokenizer verifier needs the pinned public tokenizer/data or their cache;
it does not load student weights. See `validation_summary.json` for final QA.
