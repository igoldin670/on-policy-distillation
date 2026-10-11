# On-Policy Distillation

A step-by-step reproduction of [Generalized Knowledge Distillation (GKD)](https://arxiv.org/abs/2306.13649), starting with one student-generated response.

The student is `Qwen/Qwen3-0.6B-Base`; the teacher is `Qwen/Qwen3-32B`.
The notebook generates a student trajectory, checks token compatibility,
scores the same response with a frozen teacher, and measures reverse KL at
the student-generated response positions. Saved outputs show Parts 1–6
passing on an A100 80GB: 36 prompt tokens, 128 response tokens, matching
student and teacher scoring shapes, and a mean reverse KL of about 0.1592
nats for the sampled response. All 310 student parameter tensors received
finite, nonzero gradients while the teacher received none. Part 7 adds an
unrun, guarded one-step student update.

## Structure

- `notebooks/01_reverse_kl_demo.ipynb`: generate a student response, score it with the student, check teacher token compatibility, score it with a frozen teacher, measure reverse KL, verify student-only gradient flow, and test one guarded optimizer step.
- `notebooks/02_baseline_evaluation.ipynb`: evaluate the baseline model.
- `notebooks/03_sft_cold_start.ipynb`: prepare a supervised fine-tuning cold start.
- `notebooks/04_on_policy_distillation.ipynb`: run on-policy distillation.
- `src/`: shared data, generation, log-probability, loss, evaluation, and utility code.
- `configs/small_experiment.yaml`: student and teacher model choices for a small experiment.
- `results/`: experiment outputs.

The first notebook covers student trajectory generation, student token log
probabilities, teacher compatibility, frozen teacher scoring, a read-only
reverse-KL calculation, a gradient-flow checkpoint, and a single-update
checkpoint. The other notebooks and source modules are scaffolds for later
steps.
