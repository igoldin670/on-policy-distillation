# Interrupted raw-prompt experiment

These are historical user-run A100 records. The corrected notebook uses different
teacher conditioning and starts a new experiment from the supervised checkpoint.

- `training_trajectories.jsonl` and `on_policy_history.json` cover 52 updates and 416 responses.
- `notebook04_snapshot.json` preserves the entire notebook as supplied, including
  later console output through update 61 and the successful teacher probe.
- `teacher_probe.txt` preserves the original printed table. `teacher_probe.json`
  parses its rounded numeric values for plotting; it contains one boundary probe.
- `manifest.json` distinguishes the partial run from a completed experiment.
- `SHA256SUMS.json` fingerprints all evidence files.

The first generated response ends with `#### 360` on a question whose correct
answer is 30. A teacher's willingness to end that response is not a correctness
judgment. The probe isolates how teacher context changes next-token preferences.

Archived code is evidence and is never imported or executed by the notebooks.
GSM8K source questions are covered by the [existing MIT license](../../02_baseline_evaluation/GSM8K_LICENSE.txt).
