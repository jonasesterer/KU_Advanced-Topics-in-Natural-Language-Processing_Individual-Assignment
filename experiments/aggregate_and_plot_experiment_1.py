import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from glob import glob

plt.rcParams.update({
    'font.size': 14,       # Default font size
    'axes.titlesize': 18,  # Title font size
    'axes.labelsize': 18,  # X and Y label font size
    'xtick.labelsize': 16, # X-tick label font size
    'ytick.labelsize': 16, # Y-tick label font size
    'legend.fontsize': 14  # Legend font size
})

def load_experiment_1_results() -> dict:
    """
    Load all results files for Experiment 1.

    Returns:
        dict: Nested dictionary with model type as keys,
              step counts as second keys, and results data.
    """
    results = {}
    file_pattern = "Results_Individual_1_*_*.pkl"  # Only load files for Experiment 1
    result_files = glob(file_pattern)

    for file_path in result_files:
        # Extract model type and num_steps from the filename
        parts = os.path.basename(file_path).split("_")
        model_type = parts[3]
        num_steps = int(parts[4].split(".")[0])

        # Load the results
        with open(file_path, "rb") as f:
            data = pickle.load(f)

        if model_type not in results:
            results[model_type] = {}
        results[model_type][num_steps] = data

    return results

def plot_experiment_1_results(results: dict):
    """
    Generate and save a line plot for Experiment 1 aggregated results.

    Args:
        results (dict): Nested dictionary with model type as keys,
                        step counts as second keys, and results data.
    """
    plt.figure(figsize=(12, 6))
    color_schemes = {
        "pretrained": plt.cm.Blues,
        "random": plt.cm.Oranges,
    }
    commands = sorted(next(iter(next(iter(results.values())).values())).keys())  # Sorted command labels
    x_positions = np.arange(len(commands))  # Ensure bar-like spacing for x-axis

    for model_type, steps_results in results.items():
        steps = sorted(steps_results.keys())

        # Plot each step configuration
        for i, step in enumerate(steps):
            accuracies = [steps_results[step].get(cmd, (0,))[0] for cmd in commands]
            color_map = color_schemes[model_type]
            line_color = color_map(0.4 + 0.5 * (i / (len(steps) - 1)))  # Restrict range for stronger colors
            plt.plot(
                x_positions,
                accuracies,
                label=f"{model_type} (Steps {step})",
                color=line_color,
                linewidth=2,
            )

    # X-axis with commands
    plt.xticks(x_positions, [f"{cmd}%" for cmd in commands])
    plt.xlabel("Commands Used")
    plt.ylabel("Token-Level Accuracy (%)")
    plt.yticks(range(0, 101, 20))
    plt.grid(axis="y", linestyle="-", linewidth=1, alpha=0.7)
    plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1))

    # Save plot
    plot_path = "Plot_Aggregated_Experiment_1.png"
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"Plot saved as {plot_path}")

    # Save agrgegated results
    aggregated_path = "Results_Aggregated_Experiment_1.pkl"
    with open(aggregated_path, "wb") as f:
        pickle.dump(results, f)
    print(f"Aggregated results saved as {aggregated_path}")

if __name__ == "__main__":
    results = load_experiment_1_results()
    plot_experiment_1_results(results)
