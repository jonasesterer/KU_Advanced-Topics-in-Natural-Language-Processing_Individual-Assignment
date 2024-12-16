echo "Evaluating experiment 1"
python3 experiments/evaluate.py ./results_ex1/ 1

echo "Evaluating experiment 2"

python3 experiments/evaluate.py ./results_ex2/ 2

echo "Evaluating experiment 3"
python3 experiments/evaluate.py ./results_ex3_p1/ 3
python3 experiments/evaluate.py ./results_ex3_p2/ 3
