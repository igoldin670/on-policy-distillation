# Notebook 02 validation record

Notebook 02 is complete for the baseline-evaluation milestone. It evaluates an
unchanged student and preserves the original strict-format metric alongside a
separate exploratory prose metric. No teacher is loaded and no weights are updated.

## Evidence and execution provenance

Two kinds of evidence are deliberately kept separate:

- **Saved Colab execution:** the author ran `generate` mode with Python 3.13.16
  and pandas 3.1.0. Every code cell has an execution count, all four figures are
  present, no cell contains an exception, and the output reports fresh inference
  over 32 questions and an analysis export. The last download cell was rerun, so
  its execution count is 16. Model and metric outputs are preserved, not replaced by CPU QA. The conceptual
  workflow figure was regenerated to remove outdated default-mode labels.
- **Local final QA:** all 15 code cells ran sequentially in a new CPU kernel in
  `replay` mode, using the immutable recorded A100 run. This does not perform new
  inference. The separate executed replay notebook, HTML preview, and execution
  report are written under ignored `results/notebook02_qa/`.

Both displayed evaluations have 0/32 strict matches, 17/32 exploratory numeric
matches, 28 prose-parsed responses, two token-limit stops, and a mean response
length of 214.25 tokens. Matching aggregate metrics do not establish identical
responses or token IDs across runs. The immutable source files provide full raw
evidence for the original run. The newer Colab raw run and ZIP are not present in
this local checkout and have not been independently audited here.

The saved GPU run preceded the final dependency/export improvements. Its earlier
ZIP contains the analysis; new exports also include a checksummed `raw_run/`
with the exact manifest, responses and token IDs, results, and summary. After
updating the runtime's project checkout and loading the new helper, export the
existing raw run again to obtain the expanded bundle. Editing the local VS Code
notebook alone does not update an existing `/content/` checkout or loaded Python
module. The old ZIP is not retroactively changed.

## Checks performed

- 25 unit tests: strict/prose extraction, contradictory answers, reference
  blindness, denominator preservation, KL values/support boundaries, token
  alignment, corrupted evidence, and ZIP round trips. Colab setup tests cover
  the outdated bottleneck upgrade, mixed loaded/installed library versions,
  restart instructions, and generation dependencies installed before imports.
- All 15 code cells executed sequentially in an isolated CPU replay kernel,
  with zero exceptions and zero stderr outputs.
- Notebook schema, Python syntax, unique cell IDs, Markdown fences, navigation,
  relative links, and HTML export checks.
- Four PNG/SVG figures generated and checked for readable labels and consistency.
  The generated PNGs match the versioned figures byte for byte in the pinned
  CPU environment; SVGs parse as XML and reproduce deterministically.
- Display and inline TeX checked with KaTeX 0.16.22. The HTML export embeds the
  four figure outputs. This is not a live GitHub-renderer screenshot review.
- Ruff lint and formatting checks on the baseline helpers, tests, and scripts.
- All four original evidence files retain their recorded SHA-256 fingerprints.
  The original strict score remains unchanged.
- The independent pinned-source report reproduces all 32 original prompt
  tokenizations, response-ID decodes, and reference numbers, plus seeded question
  selection. This is a source check, not a new model forward pass.

## What the strict score means

Every original prompt requested a final `#### <number>` line. None of its 32
responses contains `####`; 30 ended naturally. The strict extractor accepts a
valid synthetic marker, so zero parsing is an observed format-compliance failure,
not a broken parser. The prose checker is post-hoc and measures numeric agreement;
it cannot verify reasoning or unit scope.

## Remaining limits

The final public-download authentication arguments and progress messages have not
been rerun on a GPU by this local QA session. The saved user-run outputs establish
that the earlier generation path completed in Colab. The expanded raw-run export
was verified locally by unzipping it and revalidating all source artifacts and
scores. No fresh model results were invented. Colab package installation and
restarting still need a live runtime; mock tests do not replace that check.

The sample is small and the secondary checker was developed after inspecting it.
Keep test examples out of training and use untouched examples with frozen rules
for stronger future comparisons. No post-training quality improvement is claimed.

## Reproduce local QA

From an environment installed from `requirements-notebook.txt`:

```bash
python -m unittest discover -s tests -v
ruff check --select E4,E7,E9,F,I src/evaluation.py src/baseline.py src/baseline_plots.py tests scripts
ruff format --check src/evaluation.py src/baseline.py src/baseline_plots.py tests scripts
python scripts/validate_notebook02.py
```

The validator preserves the source notebook by default. Its optional
`--update-notebook` flag explicitly replaces the saved outputs with CPU replay.
The default command changes neither the VS Code kernel selection nor CUDA state.

The optional pinned-source check requires the Hugging Face dependencies in
`requirements-generation.txt` and network access, but no weights or GPU:

```bash
python scripts/verify_baseline_sources.py
```
