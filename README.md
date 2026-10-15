# On-Policy Distillation

A step-by-step reproduction of [Generalized Knowledge Distillation (GKD)](https://arxiv.org/abs/2306.13649), starting with one student-generated response.

The student is `Qwen/Qwen3-0.6B-Base`; the teacher is `Qwen/Qwen3-32B`.
The notebook generates a student trajectory, checks token compatibility,
scores the same response with a frozen teacher, and measures reverse KL at
the student-generated response positions. Saved A100 80GB outputs show all
seven mechanics checkpoints passing: token alignment, matching tokenizer
meanings, teacher scoring, student-only gradients, and one student update.
The update increased mean reverse KL from about 0.1592 to 3.9137 nats on
the saved response. This verifies the mechanics, not a quality improvement.

## Structure

- `notebooks/01_reverse_kl_demo.ipynb`: generate a student response, score it with the student, check teacher token compatibility, score it with a frozen teacher, measure reverse KL, verify student-only gradient flow, and test one guarded optimizer step.
- `notebooks/02_baseline_evaluation.ipynb`: evaluate a fresh student on 32 fixed GSM8K test questions, inspect accuracy and answer-format failures, and download reproducible results.
- `notebooks/03_sft_cold_start.ipynb`: prepare a supervised fine-tuning cold start.
- `notebooks/04_on_policy_distillation.ipynb`: run on-policy distillation.
- `src/`: shared data, generation, log-probability, loss, evaluation, and utility code.
- `configs/small_experiment.yaml`: student and teacher model choices for a small experiment.
- `results/`: experiment outputs.

The first notebook covers student trajectory generation, student token log
probabilities, teacher compatibility, frozen teacher scoring, a read-only
reverse-KL calculation, a gradient-flow checkpoint, and a single-update
checkpoint. Notebook 02 now contains the baseline evaluation protocol and
local scoring checks; its model evaluation still needs to run in Colab.
Notebooks 03 and 04 and the source modules remain scaffolds for later steps.

## Run the baseline

Open `notebooks/02_baseline_evaluation.ipynb` in a fresh Colab GPU runtime
and run from top to bottom. It loads only the original 0.6B student and
uses greedy decoding with a 512-token response limit. The teacher and the
student update from notebook 01 are not reused.

The evaluation uses a reproducible 32-question sample from the
[GSM8K test split](https://huggingface.co/datasets/openai/gsm8k). The student
must end each response with `#### <number>`. Accuracy includes every
question; missing or malformed final answers count as incorrect. Format
success and token-limit stops are reported separately. This is a small,
custom baseline, not an official benchmark score.

The notebook saves source revisions, question IDs, prompts, responses,
token IDs, metrics, and environment information under `results/`, then
offers a ZIP download. Download the ZIP and save the executed notebook
before disconnecting from Colab. Keep the selected test questions out of
later training and tuning, and reuse the same evaluation rules for the
eventual before/after comparison.
