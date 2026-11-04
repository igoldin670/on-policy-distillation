# Notebook 03 validation record

Reviewed on 2026-11-04. The notebook preserves the user's latest completed A100
supervised execution. Local CPU inspection writes separate QA artifacts.

## Latest observed A100 run

The supplied report records an A100-SXM4-80GB, PyTorch 2.11.0+cu130, FP32 parameter
storage with BF16 autocast, 256 practice examples, 64 separate validation examples,
and 32 optimizer updates in 80.54 seconds, excluding generation and export.

| Validation measurement | Before | After |
| :--- | ---: | ---: |
| Required format | 0/64 | 60/64 |
| Strict numeric matches | 0/64 | 32/64 |
| Exploratory prose numeric matches | 31/64 | 30/64 |
| Response-target NLL | 0.776109 | 0.666930 |
| Token-limit stops | 0/64 | 4/64 |

Final historical-test results were 18/32 strict matches, 32/32 format compliance,
and 17/32 exploratory prose matches. The displayed watermelon response regressed
from a correct 10 to a formatted, incorrect 26. These small results support strong
format learning and mixed reasoning outcomes.

This is the latest supplied run. The earlier 82.2-second run had different
observed scores and remains preserved in Git history.

## Artifact verification

The [recorded report](recorded_run/manifest.json) passed its CRC and all 15
inventory hashes. The reconstructed checkpoint archive passed CRC and all 10
checkpoint-file hashes, including model weights. Both archives are present locally.
Weights are excluded from Git.

A separate local CPU check loaded the recovered real student, verified tokenizer
round trips on 676 representative texts, and produced finite logits shaped
`[1, 16, 151936]`. It performed no training or task-accuracy evaluation. CUDA reload
of the recovered copy remains part of the next Colab session.

## Local checks

- The full regression suite covers baseline extraction, SFT masks, padding and
  genuine EOS labels, causal shifts, partial accumulation groups, optimizer
  changes, tokenizer round trips, checkpoints, and export integrity.
- The actual 03 training loop is tested using a tiny randomly initialized CPU
  model. Its outputs are test machinery, not pretrained-Qwen task results.
- All 11 notebook code cells completed in a clean CPU inspection kernel with
  zero errors or stderr. Saved A100 outputs were preserved.
- The all-series QA checks notebook schemas, syntax, comments, anchors, image paths,
  HTML exports, PNG/SVG validity, and KaTeX equation rendering.
- Archive packaging now verifies existing ZIPs and atomically publishes new ZIPs.
  Interrupted packaging and valid-CRC/bad-hash cases have dedicated tests.
- Live browser preview permission was declined. Desktop/mobile browser rendering
  remains unverified; offline HTML and figure checks were completed.

## Tokenizer compatibility

For confirmed `model_type = "qwen3"`, checkpoint tokenizers explicitly set
`fix_mistral_regex=False`. This preserves Qwen's regex in Transformers 4.57.6.
The recovered-checkpoint test checks text encoding/decoding, special tokens,
vocabulary identity, whitespace, numbers, punctuation, and Unicode.

## Reproduce

```bash
python -m unittest discover -s tests -v
python scripts/validate_notebook03.py
python scripts/validate_notebook_series.py
```

Use the pinned analysis/model dependencies in an isolated environment. Validators
write ignored artifacts under `results/`. Preserve the saved GPU outputs by
leaving the optional `--save-inspection` switch unused for 03.
