echo "Training models for experiment 1"
python3 experiments/train.py 1 ./data/datafiles/simple_split/size_variations/tasks_train_simple_p1.txt ./data/datafiles/simple_split/size_variations/tasks_test_simple_p1.txt ./results_ex1
python3 experiments/train.py 1 ./data/datafiles/simple_split/size_variations/tasks_train_simple_p2.txt ./data/datafiles/simple_split/size_variations/tasks_test_simple_p2.txt ./results_ex1
python3 experiments/train.py 1 ./data/datafiles/simple_split/size_variations/tasks_train_simple_p4.txt ./data/datafiles/simple_split/size_variations/tasks_test_simple_p4.txt ./results_ex1
python3 experiments/train.py 1 ./data/datafiles/simple_split/size_variations/tasks_train_simple_p8.txt ./data/datafiles/simple_split/size_variations/tasks_test_simple_p8.txt ./results_ex1
python3 experiments/train.py 1 ./data/datafiles/simple_split/size_variations/tasks_train_simple_p16.txt ./data/datafiles/simple_split/size_variations/tasks_test_simple_p16.txt ./results_ex1
python3 experiments/train.py 1 ./data/datafiles/simple_split/size_variations/tasks_train_simple_p32.txt ./data/datafiles/simple_split/size_variations/tasks_test_simple_p32.txt ./results_ex1
python3 experiments/train.py 1 ./data/datafiles/simple_split/size_variations/tasks_train_simple_p64.txt ./data/datafiles/simple_split/size_variations/tasks_test_simple_p64.txt ./results_ex1
python3 experiments/train.py 1 ./data/datafiles/simple_split/tasks_train_simple.txt ./data/datafiles/simple_split/tasks_test_simple.txt ./results_ex1

echo "Training models for experiment 2"
python3 experiments/train.py 2 ./data/datafiles/length_split/tasks_train_length.txt ./data/datafiles/length_split/tasks_test_length.txt ./results_ex2

echo "Training models from experiment 3"
python3 experiments/train.py 3 ./data/datafiles/add_prim_split/with_additional_examples/tasks_train_addprim_complex_jump_num1_rep1.txt ./data/datafiles/add_prim_split/with_additional_examples/tasks_test_addprim_complex_jump_num1_rep1.txt ./results_ex3
python3 experiments/train.py 3 ./data/datafiles/add_prim_split/with_additional_examples/tasks_train_addprim_complex_jump_num2_rep1.txt ./data/datafiles/add_prim_split/with_additional_examples/tasks_test_addprim_complex_jump_num2_rep1.txt ./results_ex3
python3 experiments/train.py 3 ./data/datafiles/add_prim_split/with_additional_examples/tasks_train_addprim_complex_jump_num4_rep1.txt ./data/datafiles/add_prim_split/with_additional_examples/tasks_test_addprim_complex_jump_num4_rep1.txt ./results_ex3
python3 experiments/train.py 3 ./data/datafiles/add_prim_split/with_additional_examples/tasks_train_addprim_complex_jump_num8_rep1.txt ./data/datafiles/add_prim_split/with_additional_examples/tasks_test_addprim_complex_jump_num8_rep1.txt ./results_ex3
python3 experiments/train.py 3 ./data/datafiles/add_prim_split/with_additional_examples/tasks_train_addprim_complex_jump_num16_rep1.txt ./data/datafiles/add_prim_split/with_additional_examples/tasks_test_addprim_complex_jump_num16_rep1.txt ./results_ex3
python3 experiments/train.py 3 ./data/datafiles/add_prim_split/with_additional_examples/tasks_train_addprim_complex_jump_num32_rep1.txt ./data/datafiles/add_prim_split/with_additional_examples/tasks_test_addprim_complex_jump_num32_rep1.txt ./results_ex3

echo "Finished training."
echo "Run the eval.sh file to get evaluation results"
