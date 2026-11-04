# Notebook 04 assets and evidence

Explanatory diagrams use repository-native Matplotlib drawings with PNG and SVG
exports. Regenerate them with `python scripts/render_notebook04_assets.py`.

- `on_policy_workflow` shows fresh sampling, model-specific question contexts,
  identical response prefixes, and student-only learning.
- `token_alignment` shows separate student and teacher prompt offsets that yield
  equal response-logit shapes.
- `controlled_comparison` shows the supervised checkpoint and the three student arms.
- `observed_failure` plots actual interrupted-run records and the one-response
  teacher context probe. It contains no simulated performance measurements.

`recorded_failure/` preserves the interrupted experiment, successful user-run A100
probe, and their checksums. `INVESTIGATION.md` explains the diagnosis and limits.
`NEXT_RUN.md` gives the Colab qualification and continuation procedure.

`cold_start_lineage.json` contains metadata, split identities, evaluation aggregates,
and checkpoint fingerprints from the latest verified 03 report. Inspection reads
this metadata without loading weights. Training verifies the actual matching report
and checkpoint. The full latest report is preserved in the 03 recorded-run assets.

`VALIDATION.md` and `validation_summary.json` distinguish offline checks, the
observed legacy GPU experiment, and corrected training that still needs Colab.
