#!/bin/bash

datetime=`date +'%s'`
experiment_name="cmip-sigir"
run_name="carousel"
user_model="graded-carousel"
query_dist="uniform"

# Train all seven click models on user behavior following a carousel layout of multiple
# rows of items.
# Each model is trained on three logging policies:
# Uniform shuffling, a feature-based lambda-mart ranker, or a near-optimal oracle policy
# And evaluated on a different policy: uniform shuffling or noisy-oracle to simulate
# a covariate shift in the ranking distribution.
# All simulations are repeated over 10 random seeds.

echo "datetime:" "$datetime"
echo "experiment_name: $experiment_name"
echo "run_name: $run_name"
echo "user_model: $user_model"
echo "query_dist: $query_dist"

python main.py -m \
  experiment_name="$experiment_name" \
  run_name="$run_name" \
  +datetime="$datetime" \
  data/user_model@data.train_simulator.user_model="$user_model" \
  data/user_model@data.val_simulator.user_model="$user_model" \
  data/user_model@data.test_simulator.user_model="$user_model" \
  data/query_dist@data.train_simulator.query_dist="$query_dist" \
  data/query_dist@data.val_simulator.query_dist="$query_dist" \
  data/query_dist@data.test_simulator.query_dist="$query_dist" \
  data/logging_policy@data.train_policy=lambda-mart,noisy-oracle \
  data/logging_policy@data.test_policy=uniform,lambda-mart,noisy-oracle \
  data.val_simulator.user_model.random_state_increment=1 \
  data.test_simulator.user_model.random_state_increment=2 \
  data.val_simulator.query_dist.random_state_increment=1 \
  data.test_simulator.query_dist.random_state_increment=2 \
  model=dctr,ranked-dctr,pbm,ubm,dbn,cacm-minus,ncm \
  random_state=2023,3901,2837,47969,3791,3807,8963,11289,75656,31277 \
  $@
