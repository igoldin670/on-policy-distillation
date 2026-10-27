# Validation record

The portfolio notebook is executed in **replay mode**. All visible analysis
outputs are computed in a clean kernel from the versioned recorded run.
The recorded A100 inference and the CPU analysis environment are labeled separately.

The validation performed for this revision includes:

- 20 unit tests covering strict/prose extraction, contradictory answers,
  reference blindness, denominator preservation, KL values and support boundaries,
  prompt/response boundaries, corruption detection, and report round trips.
- Sequential execution of all 15 code cells in a new kernel, with zero cell
  errors and zero stderr outputs.
- Notebook schema, syntax, cell-ID, navigation, and relative-file checks.
- Four PNG/SVG figures generated from the source code. PNGs were visually
  inspected for readable labels, layout, and data consistency.
- All four display equations successfully rendered with KaTeX 0.16.22.
- Standalone HTML export with embedded figure outputs. The browser surfaces
  available to this session did not support a live GitHub-renderer screenshot;
  the HTML and asset checks are not represented as such a screenshot review.
- Independent verification against the pinned public Hugging Face sources:
  32 prompt tokenizations, 32 response-ID decodes, and 32 reference numbers,
  plus the seeded question selection. See `source_verification.json`.
- Confirmation that the four original downloaded files are unchanged and that
  the original strict score remains 0/32. Prose v2 gives 17/32, with 28 parsed
  responses; its post-hoc status remains explicit.

Fresh CUDA model inference was **not rerun** by this local validation session.
The A100 connection selected in VS Code/Colab was not exposed as an executable
kernel endpoint to the available tools. The optional `generate` path is retained,
its pure boundary logic and CPU fail-fast guard are tested, and its full GPU
execution remains a separate validation step. No model output was fabricated.

## Reproduce the checks

From the project root, in the environment installed from `requirements-notebook.txt`:

```bash
python -m unittest discover -s tests -v
ruff check --select E4,E7,E9,F,I src/evaluation.py src/baseline.py src/baseline_plots.py tests scripts
ruff format --check src/evaluation.py src/baseline.py src/baseline_plots.py tests scripts
python scripts/validate_notebook02.py
```

The final command writes an HTML preview and execution report to ignored
`results/notebook02_qa/`. It starts an isolated kernel using that command's Python
interpreter and does not change the user's VS Code kernel selection.

The optional pinned-source check requires the Hugging Face dependencies in
`requirements-generation.txt` and network access, but no model weights or GPU:

```bash
python scripts/verify_baseline_sources.py
```
