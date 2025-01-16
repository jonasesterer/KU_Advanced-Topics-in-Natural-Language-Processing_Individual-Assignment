#!/bin/bash

# Bash script for training models for experiments 1 and 2

echo "Training models for experiment 1"

# Experiment 1 parameters
results_folder="./results_ex1"
steps_list=("5000" "10000" "100000")
models=("pretrained" "random")

# Loop through configurations for Experiment 1
for model_type in "${models[@]}"; do
  for num_steps in "${steps_list[@]}"; do
    python3 experiments/train_me.py 1 ./data/datafiles/simple_split/size_variations/tasks_train_simple_p1.txt ./data/datafiles/simple_split/size_variations/tasks_test_simple_p1.txt $results_folder $model_type $num_steps
    python3 experiments/train_me.py 1 ./data/datafiles/simple_split/size_variations/tasks_train_simple_p2.txt ./data/datafiles/simple_split/size_variations/tasks_test_simple_p2.txt $results_folder $model_type $num_steps
    python3 experiments/train_me.py 1 ./data/datafiles/simple_split/size_variations/tasks_train_simple_p4.txt ./data/datafiles/simple_split/size_variations/tasks_test_simple_p4.txt $results_folder $model_type $num_steps
    python3 experiments/train_me.py 1 ./data/datafiles/simple_split/size_variations/tasks_train_simple_p8.txt ./data/datafiles/simple_split/size_variations/tasks_test_simple_p8.txt $results_folder $model_type $num_steps
    python3 experiments/train_me.py 1 ./data/datafiles/simple_split/size_variations/tasks_train_simple_p16.txt ./data/datafiles/simple_split/size_variations/tasks_test_simple_p16.txt $results_folder $model_type $num_steps
    python3 experiments/train_me.py 1 ./data/datafiles/simple_split/size_variations/tasks_train_simple_p32.txt ./data/datafiles/simple_split/size_variations/tasks_test_simple_p32.txt $results_folder $model_type $num_steps
    python3 experiments/train_me.py 1 ./data/datafiles/simple_split/size_variations/tasks_train_simple_p64.txt ./data/datafiles/simple_split/size_variations/tasks_test_simple_p64.txt $results_folder $model_type $num_steps
    python3 experiments/train_me.py 1 ./data/datafiles/simple_split/tasks_train_simple.txt ./data/datafiles/simple_split/tasks_test_simple.txt $results_folder $model_type $num_steps
  done
done

echo "Training models for experiment 2"

# Experiment 2 parameters
results_folder="./results_ex2"
num_steps=10000

# Loop through configurations for Experiment 2
for model_type in "${models[@]}"; do
  python3 experiments/train_me.py 2 ./data/datafiles/length_split/tasks_train_length.txt ./data/datafiles/length_split/tasks_test_length.txt $results_folder $model_type $num_steps
done

echo "Finished training."
echo "Run the eval_me.sh file to get evaluation results."
