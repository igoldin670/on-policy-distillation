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
probabilities, teacher compatibility, frozen teacher scoring, reverse KL,
student-only gradients, and a single guarded update. Notebook 02 turns the
recorded A100 baseline into a reproducible experiment with versioned evidence,
two explicit scoring contracts, mathematical context, and data-backed figures.
Notebooks 03 and 04 remain scaffolds for later steps.

## Run the baseline

Open [Notebook 02](notebooks/02_baseline_evaluation.ipynb) and run from top to
bottom. Its default **replay** mode uses the checksummed recorded run and
needs no GPU, model weights, or live dataset download. The notebook's saved
outputs are explicitly labeled as replay, not new inference.

For an isolated local environment (validated on Linux with Python 3.14;
the pinned analysis packages require Python 3.12+):

```bash
uv venv
uv pip install -r requirements-notebook.txt
.venv/bin/python scripts/validate_notebook02.py
```

The validator starts a clean kernel, executes all cells sequentially,
rejects errors or unexpected stderr, and writes an HTML preview under
`results/notebook02_qa/`. Select the installed environment in Jupyter or
VS Code for interactive use. In Colab, the first cell finds or clones the
public project without Google Drive and installs the analysis libraries.

For fresh inference, use an A100 or L4 Colab runtime and set `MODE` to
`"generate"`. The optional path loads only the original 0.6B student,
reuses the pinned questions and decoding settings, and writes a new run.
`requirements-generation.txt` preserves the runtime's working PyTorch/CUDA
stack. The teacher and notebook 01's updated student are never reused.

The evaluation uses a reproducible 32-question sample from the
[GSM8K test split](https://huggingface.co/datasets/openai/gsm8k). The student
must end each response with `#### <number>`. Accuracy includes every
question; missing or malformed final answers count as incorrect. Format
success and token-limit stops are reported separately. This is a small,
custom baseline, not an official benchmark score.

The immutable run is stored under
`notebooks/assets/02_baseline_evaluation/recorded_run/`; derived reports
are written under ignored `results/` folders. The archive records source
fingerprints, both scores, exact checker source, code hashes, and separate
recorded-inference and current-analysis environments. Download it and save
the executed notebook before disconnecting from Colab. Keep the selected
test questions out of later training and tuning.

## Rescore saved answers without a GPU

Notebook 02 adds a conservative checker for explicit answers
such as `Final answer: 21` and clear concluding sentences. It reads the
response without seeing the reference answer, records its evidence, and
rejects unclear answers and token-limit stops. Numeric agreement does not
certify the reasoning or units. The original strict score stays unchanged.

The original strict score is **0/32**. The exploratory prose score is
**17/32 (53.125%)**, with 28 extracted answers, 11 parsed mismatches, and
4 abstentions. The same recorded responses underlie both scores; this is
not a training improvement or verified reasoning accuracy. The notebook
includes a concrete example where a matching number has the wrong unit scope.

The equivalent local command uses only Python's standard library:

```bash
python3 -m src.evaluation /path/to/downloaded/baseline --output-directory results/prose_review
```

The output directory must be new. It contains a separate JSON/CSV report
and a sibling ZIP, with source fingerprints and checker version
`clear_final_numeric_answer_v2`. Version 2 rejects contradictions between
prose conclusions and a final strict marker; the recorded run's counts are
unchanged from v1. The original baseline files are never
overwritten. This metric is **exploratory** because the checker was designed
after inspecting the first test run. Freeze its rules and use untouched
examples for a stronger future comparison; do not tune the checker against
the current test labels.

Run the extraction and report-integrity tests with:

```bash
python3 -m unittest discover -s tests -v
```

The mathematical reference code, token-boundary guards, metric partitions,
source-integrity checks, and report round trips have dedicated tests.
Fresh CUDA generation must be validated separately on the connected GPU;
a successful CPU replay is not evidence that a new model forward pass ran.
