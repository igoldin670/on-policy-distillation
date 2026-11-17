# Fresh Colab EOS comparison

Use the updated project bundle, the verified 03 checkpoint ZIP, and its matching
report ZIP. The default fresh comparison starts both pilots from 03. The old
Amarel checkpoints can be investigated later using the optional historical path.

## Upload and extract

Upload `opd_colab_project.zip`, `checkpoint.zip`, and `report.zip` to `/content`
using your connected runtime's file upload facility. Run this in the Colab kernel.

```python
from pathlib import Path
import zipfile

# This bundle contains the project folder and the current experiment helpers.
with zipfile.ZipFile('/content/opd_colab_project.zip') as project_archive:
    project_archive.extractall('/content')

project_root = Path('/content/on-policy-distillation')
assert (project_root / 'src/opd_experiment.py').is_file()
assert (project_root / 'src/opd_diagnostics.py').is_file()
assert Path('/content/checkpoint.zip').is_file()
assert Path('/content/report.zip').is_file()
print('Project and 03 archives are available.')
```

Restart the kernel after replacing project files, then open the updated 04 and
select the A100 80GB runtime. Section 2 installs dependencies and requests a
further kernel restart if packages changed. Keep the default settings below.

```python
RUN_STAGE = "eos_comparison"
EOS_LOSS_MODE = "raw_token"
RESUME_RUN_DIRECTORY = None
HISTORICAL_RUN_DIRECTORY = None
```

The setup defaults to training mode on CUDA. Run every cell in order. The notebook
verifies the 03 archives, rebuilds the fixed data splits, evaluates the starting
student, qualifies the frozen teacher, and probes starting-student EOS behavior.
Section 8 trains independent raw-token and semantic-STOP pilots for at most 32
updates each. Both use the same question order, initial seed, and 128-update
scheduler. Checkpoints, incremental validation, and progress logs are saved under
the printed `results/` run folder. Final test evaluation occurs after a later fixed
full run. Section 10 packages the comparison report.

## Preserve and continue

Download the comparison report and preserve the full run folder, including
`eos_comparison/*/resume`, before ending the Colab session. The small report carries
the text and metrics. Resuming training also requires the model and optimizer files.
For an interrupted comparison, set `RESUME_RUN_DIRECTORY` to its parent run folder.

After reviewing a passing comparison, choose `RUN_STAGE = "full"`,
`EOS_LOSS_MODE = "semantic_stop"`, and set the resume path to the comparison's
`eos_comparison/semantic_stop` folder. The fresh workflow trains a new SFT control
before loading the teacher, then continues the semantic pilot's saved optimizer.
It evaluates the new control and completed KD student on the reserved test set.

If the original Amarel run is available, set `HISTORICAL_RUN_DIRECTORY` to its
complete server-side folder to enable the historical control and 16/22 probes.
Those measurements retain their original provenance.
