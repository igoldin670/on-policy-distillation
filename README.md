# On-Policy Distillation

A project for recreating on-policy distillation.

## Structure

- `notebooks/01_reverse_kl_demo.ipynb`: explore reverse KL divergence.
- `notebooks/02_baseline_evaluation.ipynb`: evaluate the baseline model.
- `notebooks/03_sft_cold_start.ipynb`: prepare a supervised fine-tuning cold start.
- `notebooks/04_on_policy_distillation.ipynb`: run on-policy distillation.
- `src/`: shared data, generation, log-probability, loss, evaluation, and utility code.
- `configs/small_experiment.yaml`: configuration placeholder for a small experiment.
- `results/`: experiment outputs.

This initial scaffold contains empty notebooks and source modules. Implementation and dependencies will be added as the project develops.
