#!/bin/bash

# Bash script for evaluating models for experiments 1 and 2

echo "Evaluating experiment 1"

# Experiment 1 parameters
results_folder="./results_ex1"
steps_list=("5000" "10000" "100000")
models=("pretrained" "random")

# Loop through configurations for Experiment 1
for model_type in "${models[@]}"; do
  for num_steps in "${steps_list[@]}"; do
    python3 experiments/evaluate_me.py $results_folder 1 $model_type $num_steps
  done
done

echo "Evaluating experiment 2"

# Experiment 2 parameters
results_folder="./results_ex2"
num_steps=10000

# Loop through configurations for Experiment 2
for model_type in "${models[@]}"; do
  python3 experiments/evaluate_me.py $results_folder 2 $model_type $num_steps
done

echo "Finished evaluation."
