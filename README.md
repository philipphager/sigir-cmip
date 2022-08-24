# Evaluating Click Models for Logging-Policy Bias
## Setup
### 1. Virtual environment with Conda
Dependency management

1. Setup [conda](https://www.anaconda.com/) / [miniconda](https://docs.conda.io/en/latest/miniconda.html) on your device.
2. Create environment and install dependencies: `conda env create -f environment.yaml`
3. Activating environment: `conda activate cm-bias-evaluation`

### 2. Pre-commit
Automatically format and lint modified files in commit.

1. Make sure you activate your environment
2. Initialize pre-commit: `pre-commit install`
3. (Optional) Run on checks against all files (not just changed): `pre-commit run --all-files`

### 3. Datasets
The project automatically downloads public datasets used in this work.

1. Specify the location for storing datasets using the environment, e.g.: `export LTR_DATASETS="~/.ltr_datasets/"`
2. The specified directory will be automatically created, if not existing.
3. To avoid downloading datasets, you can directly place their original .zip file into the `download` subdirectory, e.g.:
   `$LTR_DATASETS/download/MSLR-WEB30K.zip`
