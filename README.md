# On-Policy Distillation

A step-by-step reproduction of [Generalized Knowledge Distillation (GKD)](https://arxiv.org/abs/2306.13649), starting with one student-generated response.

The student is `Qwen/Qwen3-0.6B-Base`; the teacher is `Qwen/Qwen3-32B`.
The notebook generates a student trajectory, checks token compatibility, and
scores the same response with a frozen teacher. Teacher scoring requires an
A100 80GB runtime and has not yet been run in this repository's saved outputs.

## Structure

- `notebooks/01_reverse_kl_demo.ipynb`: generate a student response, score it with the student, check teacher token compatibility, and score it with a frozen teacher.
- `notebooks/02_baseline_evaluation.ipynb`: evaluate the baseline model.
- `notebooks/03_sft_cold_start.ipynb`: prepare a supervised fine-tuning cold start.
- `notebooks/04_on_policy_distillation.ipynb`: run on-policy distillation.
- `src/`: shared data, generation, log-probability, loss, evaluation, and utility code.
- `configs/small_experiment.yaml`: student and teacher model choices for a small experiment.
- `results/`: experiment outputs.

The first notebook covers student trajectory generation, student token log
probabilities, teacher compatibility, and frozen teacher scoring. The other notebooks and
source modules are scaffolds for later steps.
