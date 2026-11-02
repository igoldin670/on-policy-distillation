# Notebook 03 validation boundaries

This notebook implements reference-solution supervised fine-tuning of the original
Qwen3-0.6B-Base student. Inspection checks the real pinned tokenizer and GSM8K
training data; it does not load the student weights. Real CUDA training and any
resulting accuracy improvement remain unverified until the notebook is run on Colab.

CPU unit tests use a tiny **randomly initialized** Qwen3 causal model. They are
checks of token alignment, response-only loss, token-weighted gradient accumulation,
optimizer updates, checkpoint reload, and archive integrity. They are not model
performance measurements. Training and validation questions are disjoint and
normalized overlaps with the reserved Notebook 02 questions are excluded.

The small fixed recipe is not a reproduction of the paper's hyperparameters,
models, generated training solutions, or reported results. No teacher model,
LoRA adapter, or on-policy loss is used in Notebook 03.

The completed local checks are:

- All 11 notebook code cells ran sequentially in a clean CPU inspection kernel,
  with zero exceptions and zero stderr outputs.
- All 39 tests passed, including the existing baseline regressions and 14 new SFT
  checks. The actual notebook training loop ran with a tiny random CPU model;
  only CUDA synchronization/cache housekeeping was mocked in that test.
- The real pinned tokenizer and training split produced 256 practice and 64
  validation examples. Joint tokenization preserved every selected prompt boundary.
- Equation rendering with KaTeX, notebook/README HTML export, relative links,
  diagram SVG parsing, Ruff lint/formatting, and Git whitespace checks passed.
- The banner and both diagrams were visually inspected. This is not a live
  GitHub browser-renderer screenshot review.
- Notebooks 01 and 02 and the original baseline evidence remain unchanged.

See `validation_summary.json` for machine-readable checks and execution status.
The validators never turn CPU inspection into a claim that the pretrained student
was trained. No placeholder accuracy or learning-curve output is provided.

## Reproduce

Use an isolated environment with PyTorch, `requirements-notebook.txt`, and
`requirements-sft.txt`. Preserve Colab's own PyTorch/CUDA stack.

```bash
python -m unittest discover -s tests -v
python scripts/validate_notebook03.py
```

The second command uses a clean CPU inspection kernel and downloads the public
pinned tokenizer/data if needed. It writes a separate HTML preview and executed
inspection copy under ignored `results/notebook03_qa/`. Use `--save-inspection`
only when intentionally updating the source notebook's inspection outputs.

For actual training, sync the complete project to the Colab runtime, open 03,
choose `MODE = "train"`, restart when dependency installation requests it, then
run all cells from the top. Save the separate report and large checkpoint before
the runtime disconnects. Notebook 04 will load that checkpoint from local files.
