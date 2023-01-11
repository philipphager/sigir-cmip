#!/bin/bash

python main.py -m \
  hydra/launcher=submitit_slurm \
  hydra.launcher.mem_gb=32 \
  hydra.launcher.cpus_per_task=4 \
  hydra.launcher.array_parallelism=4 \
  hydra.launcher.partition=gpu \
  hydra.launcher.gres=gpu:1 \
  $@
