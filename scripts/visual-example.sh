#!/bin/bash

datetime=`date +'%s'`
experiment_name="cm-bias"
run_name="visual-example"
user_model="graded-pbm"
query_dist="uniform"

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
  data/logging_policy@data.train_policy=noisy-oracle \
  data/logging_policy@data.test_policy=uniform \
  data.train_simulator.temperature=1.0 \
  data.val_simulator.temperature=1.0 \
  model=dctr,pbm \
  random_state=30219,76665,88914,11656,43670 \
  $@
