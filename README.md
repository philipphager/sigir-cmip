# Evaluating Click Models for Logging-Policy Bias
## Setup
### 1. Virtual environment with Conda
Dependency management

1. Setup [conda](https://www.anaconda.com/) / [miniconda](https://docs.conda.io/en/latest/miniconda.html) on your device.
2. Create environment and install dependencies: `conda env create -f environment.yaml`
3. Activating environment: `conda activate cm-bias-evaluation`

### 2. Pre-commit
Automatically formats, lints changed files in commit.

1. Make sure you activate your environment
2. Initialize pre-commit: `pre-commit install`
3. (Optional) Run on checks against all files (not just changed): `pre-commit run --all-files`

### 2. Datasets
