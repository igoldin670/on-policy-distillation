# Latest recorded supervised run

`recorded_run/` preserves the user-supplied report that matches the recovered
Notebook 03 checkpoint. Its original `SHA256SUMS.json` covers 15 files. Model
weights are stored separately in ignored local storage.

The run used an A100-SXM4-80GB, 256 practice questions, 64 validation questions,
and 32 optimizer updates. Validation format compliance reached 60/64, strict
numeric matching 32/64, and exploratory prose matching 30/64. The final historical
test had 18/32 strict matches and 17/32 prose matches.

The report's code snapshots describe that actual execution and are evidence only.
The public notebook has subsequent presentation and archive-transfer improvements.
The earlier supervised run's differing scores remain available in Git history.
GSM8K questions use the [existing MIT license](../02_baseline_evaluation/GSM8K_LICENSE.txt).
