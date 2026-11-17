# Method-to-code review

## The paper's central method

GKD trains on next-token distributions at response prefixes. Its on-policy setting
samples responses from the current student and treats those samples as fixed when
differentiating the loss. Token divergences are averaged within each response and
then across the batch. The paper also studies reference-prefix distillation,
alternative divergences, and mixed data sources. Its GSM8K experiment uses teacher
temperature 0.1 and greedy evaluation. [GKD Algorithm 1, Equations 2–4, Appendix A.4](https://arxiv.org/html/2306.13649v3)

## Correspondence in this project

| Method property | Actual implementation |
| :--- | :--- |
| Supervised starting point | 03 uses worked GSM8K responses and response-only cross-entropy |
| Current-policy responses | `sample_trajectory` runs under no-grad before the group's optimizer update |
| Same preceding response IDs | `trajectory_tensors` appends original response IDs to model-specific question prefixes |
| One causal shift | `response_prediction_logits` starts at each model's prompt length minus one |
| Full next-token feedback | `forward_kl_loss` compares all output-head columns, with FP32 probability arithmetic |
| Frozen teacher | Loading disables gradients, and each update checks that teacher gradients remain absent |
| Response-wise averaging | `backward_opd_group` averages response means |
| Reference-prefix comparison | The `supervised_kd` arm uses the same loss and recipe on reference responses |
| Fair final comparison | Both KD arms start at the same SFT checkpoint and use the selected fixed recipe |

These functions are shared by the notebooks and CPU integration tests. The
`qwen38` profile pins Qwen3.5-0.8B-Base and Qwen3.8-27B. The `legacy` profile keeps
the original model identities. New-profile checkpoints require a fresh SFT run.

## Explicit adaptations

- The experiment uses decoder-only Qwen text models, GSM8K reference solutions,
  AdamW, smaller data selections, and one training seed. These choices differ
  from the published models, rationale data, optimizer, and scale.
- Student and teacher use their own question formatting. The teacher receives its
  pinned non-thinking chat prefix. Response IDs and preceding response prefixes
  are shared exactly.
- Both KD arms group the two ending spellings into a single stopping action after
  temperature scaling and normalization. Ordinary vocabulary contributions stay
  intact. This semantic-STOP objective is an added adaptation.
- Engineering safeguards can pause deteriorating training. Validation chooses
  between two predeclared learning rates before held-out evaluation.
- Notebook 01 illustrates reverse KL on the historical model pair. The final
  controlled study uses forward KL.

The project recreates the student-generated-feedback mechanism and tests a small
controlled instance. It does not claim the paper's benchmark scores.
