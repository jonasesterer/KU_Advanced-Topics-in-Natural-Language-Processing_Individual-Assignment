from experiments.evaluate import main_eval

num_experiment = 3
train_file = "./data/datafiles/simple_split/tasks_train_simple.txt"
test_file = "./data/datafiles/simple_split/tasks_test_simple.txt "
save_path = "./results_ex3"

# main(num_experiment, train_file, test_file, save_path)

main_eval(save_path, num_experiment)
