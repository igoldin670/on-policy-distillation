# Notebook 02 assets

`recorded_run/` contains the four original files downloaded from the author's
Colab A100 run `baseline_20261027T170926655320Z`, plus `SHA256SUMS.json`.
The original files were copied byte-for-byte. They include every selected
question, reference number, response, token ID list, and recorded environment.

The model and dataset revisions, selected test indices, prompt, and greedy
generation settings are preserved in `recorded_run/manifest.json`. Dataset
questions and reference answers originate from
[OpenAI GSM8K](https://github.com/openai/grade-school-math), distributed under
the MIT license; see the accompanying `GSM8K_LICENSE.txt`.

`figures/` contains PNG images for notebook/GitHub display and SVG versions
for export. `src/baseline_plots.py` regenerates every figure:

- `distillation_context`: conceptual teacher/student training architecture;
  it is not an observed training trace.
- `evaluation_workflow`: replay versus fresh inference and the shared evaluation path.
- `metric_comparison`: counts from the same 32 recorded responses, under two checkers.
- `response_lengths`: recorded response-token counts and stopping behavior.

The notebook's checked-in execution is a clean-kernel CPU replay. It validates
and analyzes recorded inference; it does not rerun the A100 model. Derived
reports and HTML validation previews live under ignored `results/` folders.
No model weights, credentials, caches, or temporary environments are versioned here.
