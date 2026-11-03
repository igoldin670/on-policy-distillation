# Notebook 04 assets

The three explanatory figures are repository-native Matplotlib drawings, exported
as PNG for GitHub notebooks with SVG copies. Regenerate them with:

```bash
.venv/bin/python scripts/render_notebook04_assets.py
```

- `on_policy_workflow`: fresh sampling, identical-prefix teacher scoring, student update.
- `token_alignment`: the response starts at input position P and logits position P - 1.
- `controlled_comparison`: one saved SFT checkpoint, three comparison arms.

`cold_start_lineage.json` records metadata, split indices, evaluation aggregates,
and checkpoint fingerprints from the user's completed Notebook 03 A100 report.
The report ZIP's SHA-256 is included. It contains no weights and is not a 04 result.
In training mode, 04 verifies the actual report and checkpoint, not this metadata
as a substitute for weights. Archived source files are never executed.

Actual training curves and performance figures are generated only from completed
runtime records under ignored `results/`. See [VALIDATION.md](VALIDATION.md) for
the exact verification scope.
