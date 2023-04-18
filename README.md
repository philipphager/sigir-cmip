# An Offline Metric for the Debiasedness of Click Models
Source code for the SIGIR 2023 paper `An Offline Metric for the Debiasedness of Click Models`. For a standalone implementation of the proposed CMIP metric, [see this repository](https://github.com/philipphager/CMIP).

## Hyperparameters and configuration
You can find a list of model parameters and training configurations under `config/`.

## Setup
### 1. Virtual environment with Conda

Dependency management

1. Setup [conda](https://www.anaconda.com/)
   / [miniconda](https://docs.conda.io/en/latest/miniconda.html) on your device.
2. Create environment and install dependencies: `conda env create -f environment.yaml`
3. Activating environment: `conda activate sigir-cmip`

### 2. Experiments

All experimental runs are documented inside the `scripts/` directory. To execute an experiment: 

1. Make the scripts executable: `chmod +x ./scripts/*`
2. Run a script locally use, e.g.: `./scripts/graded-pbm.sh`
3. To execute a script on a [SLURM cluster](https://slurm.schedmd.com/documentation.html) add: `./scripts/graded-pbm.sh +launcher=slurm`
4. You can configure the SLURM resources in: `config/launcher/slurm.yaml`

Documentation of each experiment can be found inside the scripts.

### 3. Pre-commit

Automatically format and lint modified files in commit.

1. Make sure you activate your environment
2. Initialize pre-commit: `pre-commit install`
3. (Optional) Run on checks against all files (not just
   changed): `pre-commit run --all-files`

### 4. Datasets

The project automatically downloads the dataset used in this work to: `~/.ltr_datasets`.

1. You can change the directory by modifying the `base_dir` variable in: `config/env.yaml`
2. To avoid downloading datasets, you can directly place the original .zip file into
   the `download` subdirectory, e.g.:
   `~/.ltr_datasets/download/MSLR-WEB30K.zip`

### 5. Logging

Log metrics with [Weights & Biases](https://github.com/wandb/wandb).

1. Make sure you activate your environment
2. Log into Weights & Biases before your first run: `wandb login`

## Reference
```
@inproceedings{Deffayet2023Debiasedness,
  author = {Romain Deffayet and Philipp Hager and Jean-Michel Renders and Maarten de Rijke},
  title = {An Offline Metric for the Debiasedness of Click Models},
  booktitle = {Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR`23)},
  organization = {ACM},
  year = {2023},
}
```

## License
This project uses the [MIT license](https://github.com/philipphager/sigir-cmip/blob/main/LICENSE).
