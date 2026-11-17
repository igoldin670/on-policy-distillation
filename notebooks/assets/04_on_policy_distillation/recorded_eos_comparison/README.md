# Recorded EOS comparison

The author ran both pilots on Amarel. This folder preserves the source and a
locally audited summary of run `on_policy_distillation_20261116T022121085005Z`.

## Provenance

The downloaded `eos_evidence_20261116_v2.zip` is 3,484,347 bytes. Its SHA-256 is
`b1dbdd2b6ec9588cf244aa6301b3139b4c25e2d48144dae6fcf55d181d7d2b6d`.
The local audit verified all 69 collected file checksums. The archive includes
full training token arrays and evaluation records. Compact histories and decoded
validation responses are retained under [audited](audited/).

`source/` preserves the exact historical root-level helper files, dependencies,
and notebook cell source. [The source match record](audited/historical_source_match.json)
checks those helpers against the GPU run's experiment signature. These archived
files describe the past experiment. Current notebooks import the live `src/`.

The source-side collector verified inventories for raw-token updates 16 and 18
and grouped-stop updates 24 and 32. Tensor weights and optimizer states remain on
Amarel. This checkout contains their recorded inventories. Local QA checked the
collected metadata and has no copy of these model tensors to rehash or reload.

## Results

Starting strict validation accuracy was **33/64**. Starting reference NLL was
**0.667651**, taken from the saved starting-validation notebook output.

| Arm | Updates | Strict matches | Greedy validation caps | Final sampled caps | Reference NLL |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Raw-token | 18 | 1/64 | 53/64 | 24/32 | 0.863818 |
| Grouped stop | 32 | 23/64 | 4/64 | 19/32 | 0.963980 |

Training used sampled responses with a 256-token limit. Validation used greedy
responses with a 512-token limit. These cap counts measure different situations.
Both validations completed before the later SLURM timeout. Both pilots failed
the original acceptance gates.

## What local replay verified

- All 400 trajectories have consistent prompt/response boundaries, question
  order, current-policy steps, stopping flags, and recorded update token counts.
- The frozen strict/prose checkers reproduce every saved validation boundary.
- The first rollout groups match between arms.
- The first measured accuracy decline occurs at update 8 in both arms.
- Replaying the new baseline-relative guard requests a pause at update 8 in both.
- All recorded optimizer updates have finite gradient norms above the clipping
  threshold. The local audit reads these recorded norms.

Saved NLL, KL, and frozen-model stopping probabilities are GPU measurements from
the archive. Local checks reproduce metrics from text and metadata. Recomputing
model forwards requires the corresponding weights and GPU execution.

At update 8, grouped-stop validation loses nine previously correct questions and
gains five. One answer invents an extra flight and changes 24 to 25. Another
changes the correct sum 20 + 20 to the incorrect 5 + 15. Format alone therefore
cannot explain the decline. The exact contribution of temperature, learning rate,
and student capacity needs a controlled follow-up.

## Reproduce the audit

From the repository root, choose a fresh destination directory.

```bash
python scripts/audit_eos_comparison.py /path/to/eos_evidence_20261116_v2.zip \
  results/eos_audit_replay --baseline-nll 0.6676512348477289
```

`SHA256SUMS.json` verifies this folder. The inner source checksum file refers to
files in the downloaded archive. Keep that ZIP for a full token-level replay.
