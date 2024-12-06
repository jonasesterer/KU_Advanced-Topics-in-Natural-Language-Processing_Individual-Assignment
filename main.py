from experiments.train import main
from experiments.ex3 import main_eval

num_experiment = 1
train_file = "./data/datafiles/simple_split/tasks_train_simple.txt"
test_file = "./data/datafiles/simple_split/tasks_test_simple.txt "
save_path = "./results"

# main(num_experiment, train_file, test_file, save_path)

main_eval(save_path, test_file)