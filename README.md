# Evaluating Click Models for Logging-Policy Bias

## Setup

### 1. Virtual environment with Conda

Dependency management

1. Setup [conda](https://www.anaconda.com/)
   / [miniconda](https://docs.conda.io/en/latest/miniconda.html) on your device.
2. Create environment and install dependencies: `conda env create -f environment.yaml`
3. Activating environment: `conda activate cm-bias-evaluation`

### 2. Pre-commit

Automatically format and lint modified files in commit.

1. Make sure you activate your environment
2. Initialize pre-commit: `pre-commit install`
3. (Optional) Run on checks against all files (not just
   changed): `pre-commit run --all-files`

### 3. Datasets

The project automatically downloads public datasets used in this work.

We need to specify the location for storing datasets using the environment:

1. Open the file `config/env.yaml` to edit configurations for your local machine (i.e.,
   edits are not tracked on git)
2. The specified the location for storing datasets, e.g.: `base_dir: "~/.ltr_datasets/"`
3. Ask git to ignore all changes made to the file to avoid committing your local
   configs: `git update-index --skip-worktree config/env.yaml`
4. To avoid downloading datasets, you can directly place the original .zip file into
   the `download` subdirectory, e.g.:
   `~/.ltr_datasets/download/MSLR-WEB30K.zip`

### 4. Logging

Log loss and metrics with [Weights & Biases](https://github.com/wandb/wandb).

1. Make sure you activate your environment
2. Log into Weights & Biases before your first run: `wandb login`
