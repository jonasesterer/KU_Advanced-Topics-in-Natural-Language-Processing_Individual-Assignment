import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Tuple, List

import os
import pickle
from glob import glob

plt.rcParams.update({
    'font.size': 14,       # Default font size
    'axes.titlesize': 18,  # Title font size
    'axes.labelsize': 18,  # X and Y label font size
    'xtick.labelsize': 12, # X-tick label font size
    'ytick.labelsize': 12, # Y-tick label font size
    'legend.fontsize': 14  # Legend font size
})

def load_experiment_2_results() -> dict:
    """
    Load all results files for Experiment 2.

    Returns:
        dict: Nested dictionary with model type as keys,
              step counts as second keys, and results data.
    """
    results = {}
    file_pattern = "Results_Individual_2_*_*.pkl"  # Only load files for Experiment 2
    result_files = glob(file_pattern)

    for file_path in result_files:
        # Extract model type and num_steps from the filename
        parts = os.path.basename(file_path).split("_")
        # Example filename: "Results_Individual_2_pretrained_10000.pkl"
        # parts -> ["Results", "Individual", "2", "pretrained", "10000.pkl"]
        model_type = parts[3]
        num_steps = int(parts[4].split(".")[0])

        # Load the results
        with open(file_path, "rb") as f:
            data = pickle.load(f)

        if model_type not in results:
            results[model_type] = {}
        results[model_type][num_steps] = data

    return results

def plot_experiment_2_results(all_results: dict):
    """
    Plots aggregated Experiment 2 results, splitting into:
      - Standard  (Figure 1: 1×2 subplots)
      - Oracle    (Figure 2: 2×2 subplots)
    Each model/step combination is overlaid as a line.
    
    Requested tweaks:
      - No key suffix in legend labels
      - Legend only in the Oracle figure's 4th subplot (sequence-level vs input length)
      - Specific color schemes for each model type (Blues for pretrained, Oranges for random)
      - Exact x-ticks from the data, y-ticks matching the original style
    """

    # ---------------------------------------------------------
    # 1) Setup Figures
    #    Figure A (Standard): 1 row × 2 columns
    fig_std, axs_std = plt.subplots(1, 2, figsize=(12, 6))
    axs_std = axs_std.ravel()

    #    Figure B (Oracle): 2 rows × 2 columns
    fig_oracle, axs_oracle = plt.subplots(2, 2, figsize=(12, 10))
    axs_oracle = axs_oracle.ravel()

    # ---------------------------------------------------------
    # 2) Prepare color schemes
    color_schemes = {
        "pretrained": plt.cm.Blues,
        "random": plt.cm.Oranges,
    }

    # ---------------------------------------------------------
    # 3) We'll gather x-values in sets so we can set ticks at the end
    #    Standard figure
    std_tgt_xvals = set()  # token-level vs. target length
    std_inp_xvals = set()  # token-level vs. input length

    #    Oracle figure
    oracle_tgt_token_xvals = set()  # token-level vs. target length
    oracle_inp_token_xvals = set()  # token-level vs. input length
    oracle_tgt_seq_xvals   = set()  # seq-level vs. target length
    oracle_inp_seq_xvals   = set()  # seq-level vs. input length

    # ---------------------------------------------------------
    # 4) Helper to compute accuracies
    def compute_accuracy(stats_dict: dict) -> (np.ndarray, np.ndarray):
        """
        Given stats_dict[length] = {'correct': int, 'total': int},
        return (sorted_lengths, list_of_accuracies).
        """
        if not stats_dict:
            return np.array([]), np.array([])

        lengths = sorted(stats_dict.keys())
        accuracies = [
            100.0 * (stats_dict[l]["correct"] / stats_dict[l]["total"]) 
            for l in lengths
        ]
        return np.array(lengths), np.array(accuracies)

    # ---------------------------------------------------------
    # 5) Iterate over the entire results structure
    #    We'll track step_counts so we can do color mapping
    #    for each model type. We want to plot different steps
    #    with different shades from the color map.
    for model_type, steps_dict in all_results.items():
        # Gather all step_counts in a sorted list so we have a stable color order
        step_counts_sorted = sorted(steps_dict.keys())

        for i, step_count in enumerate(step_counts_sorted):
            # Decide on a color from the color map
            color_map = color_schemes[model_type]
            # If there's only one step, we just use some middle color
            if len(step_counts_sorted) == 1:
                line_color = color_map(0.6)  # single color
            else:
                # Vary from 0.4 to 0.9 in the color scale
                fraction = i / (len(step_counts_sorted) - 1)
                line_color = color_map(0.4 + 0.5 * fraction)

            subresults_dict = steps_dict[step_count]
            
            # Typically subresults_dict = {0: ( (dict,dict,dict,dict), (dict,dict,dict,dict) )}
            # We loop over each key in subresults_dict (usually just 0).
            for key_id, items_tuple in subresults_dict.items():
                # items_tuple might have [0] for Standard, [1] for Oracle, etc.
                # We'll label lines just as "model_type step_count"
                label_str = f"{model_type} {step_count}"

                # --- 5A) Plot "Standard" (items_tuple[0]) ---
                if len(items_tuple) > 0:
                    std_inp_stats,  std_tgt_stats,  std_inp_seq_stats, std_tgt_seq_stats = items_tuple[0]
                    
                    # Token-level vs target length
                    tgt_lengths_std, tgt_token_acc_std = compute_accuracy(std_tgt_stats)
                    # Token-level vs input length
                    inp_lengths_std, inp_token_acc_std = compute_accuracy(std_inp_stats)

                    # Add to sets for tick usage
                    std_tgt_xvals.update(tgt_lengths_std.tolist())
                    std_inp_xvals.update(inp_lengths_std.tolist())

                    # Plot lines
                    if len(tgt_lengths_std) > 0:
                        axs_std[0].plot(
                            tgt_lengths_std, tgt_token_acc_std, 
                            marker='o', color=line_color, label=label_str
                        )
                    if len(inp_lengths_std) > 0:
                        axs_std[1].plot(
                            inp_lengths_std, inp_token_acc_std, 
                            marker='o', color=line_color, label=label_str
                        )

                # --- 5B) Plot "Oracle" (items_tuple[1]) ---
                if len(items_tuple) > 1:
                    oracle_inp_stats, oracle_tgt_stats, oracle_inp_seq_stats, oracle_tgt_seq_stats = items_tuple[1]

                    # Token-level
                    tgt_lengths_oracle, tgt_token_acc_oracle = compute_accuracy(oracle_tgt_stats)
                    inp_lengths_oracle, inp_token_acc_oracle = compute_accuracy(oracle_inp_stats)
                    # Sequence-level
                    tgt_lengths_seq_oracle, tgt_seq_acc_oracle = compute_accuracy(oracle_tgt_seq_stats)
                    inp_lengths_seq_oracle, inp_seq_acc_oracle = compute_accuracy(oracle_inp_seq_stats)

                    # Add to sets for tick usage
                    oracle_tgt_token_xvals.update(tgt_lengths_oracle.tolist())
                    oracle_inp_token_xvals.update(inp_lengths_oracle.tolist())
                    oracle_tgt_seq_xvals.update(tgt_lengths_seq_oracle.tolist())
                    oracle_inp_seq_xvals.update(inp_lengths_seq_oracle.tolist())

                    # Subplot 0 (Token-Level vs Target Length)
                    if len(tgt_lengths_oracle) > 0:
                        axs_oracle[0].plot(
                            tgt_lengths_oracle, tgt_token_acc_oracle,
                            marker='o', color=line_color, label=label_str
                        )
                    # Subplot 1 (Token-Level vs Input Length)
                    if len(inp_lengths_oracle) > 0:
                        axs_oracle[1].plot(
                            inp_lengths_oracle, inp_token_acc_oracle,
                            marker='o', color=line_color, label=label_str
                        )
                    # Subplot 2 (Sequence-Level vs Target Length)
                    if len(tgt_lengths_seq_oracle) > 0:
                        axs_oracle[2].plot(
                            tgt_lengths_seq_oracle, tgt_seq_acc_oracle,
                            marker='o', color=line_color, label=label_str
                        )
                    # Subplot 3 (Sequence-Level vs Input Length)
                    if len(inp_lengths_seq_oracle) > 0:
                        axs_oracle[3].plot(
                            inp_lengths_seq_oracle, inp_seq_acc_oracle,
                            marker='o', color=line_color, label=label_str
                        )

    # ---------------------------------------------------------
    # 6) Final adjustments: ticks, titles, labels, grids, legends, etc.

    # --- Standard Figure (1x2) ---
    # Subplot 0: Token-Level vs. Target Length
    axs_std[0].set_title("Standard")
    axs_std[0].set_xlabel("Ground-Truth Action Sequence Length (words)")
    axs_std[0].set_ylabel("Token-Level Accuracy (%)")
    axs_std[0].grid(axis="y", linestyle="--", alpha=0.7)
    axs_std[0].set_xticks(sorted(std_tgt_xvals))
    axs_std[0].set_yticks(range(0, 101, 20))

    # Subplot 1: Token-Level vs. Input Length
    axs_std[1].set_title("Standard")
    axs_std[1].set_xlabel("Command Length (words)")
    axs_std[1].set_ylabel("Token-Level Accuracy (%)")
    axs_std[1].grid(axis="y", linestyle="--", alpha=0.7)
    axs_std[1].set_xticks(sorted(std_inp_xvals))
    axs_std[1].set_yticks(range(0, 101, 20))

    axs_std[0].legend()

    # We do NOT show a legend in the Standard figure
    fig_std.tight_layout()
    fig_std.savefig("Plot_Overlayed_Standard_Experiment2.png")
    print("Saved Figure (Standard): 'Plot_Overlayed_Standard_Experiment2.png'")

    # --- Oracle Figure (2x2) ---
    # Subplot 0: Token-Level vs. Target Length
    axs_oracle[0].set_title("Oracle")
    axs_oracle[0].set_xlabel("Ground-Truth Action Sequence Length (words)")
    axs_oracle[0].set_ylabel("Token-Level Accuracy (%)")
    axs_oracle[0].grid(axis="y", linestyle="--", alpha=0.7)
    axs_oracle[0].set_xticks(sorted(oracle_tgt_token_xvals))
    axs_oracle[0].set_yticks(range(0, 101, 20))

    # Subplot 1: Token-Level vs. Input Length
    axs_oracle[1].set_title("Oracle")
    axs_oracle[1].set_xlabel("Command Length (words)")
    axs_oracle[1].set_ylabel("Token-Level Accuracy (%)")
    axs_oracle[1].grid(axis="y", linestyle="--", alpha=0.7)
    axs_oracle[1].set_xticks(sorted(oracle_inp_token_xvals))
    axs_oracle[1].set_yticks(range(0, 101, 20))

    # Subplot 2: Sequence-Level vs. Target Length
    axs_oracle[2].set_title("Oracle")
    axs_oracle[2].set_xlabel("Ground-Truth Action Sequence Length (words)")
    axs_oracle[2].set_ylabel("Sequence-Level Accuracy (%)")
    axs_oracle[2].grid(axis="y", linestyle="--", alpha=0.7)
    axs_oracle[2].set_xticks(sorted(oracle_tgt_seq_xvals))
    axs_oracle[2].set_yticks(range(0, 71, 10))

    # Subplot 3: Sequence-Level vs. Input Length
    axs_oracle[3].set_title("Oracle")
    axs_oracle[3].set_xlabel("Command Length (words)")
    axs_oracle[3].set_ylabel("Sequence-Level Accuracy (%)")
    axs_oracle[3].grid(axis="y", linestyle="--", alpha=0.7)
    axs_oracle[3].set_xticks(sorted(oracle_inp_seq_xvals))
    axs_oracle[3].set_yticks(range(0, 71, 10))

    # Place the legend ONLY in subplot 3
    axs_oracle[2].legend()

    fig_oracle.tight_layout()
    fig_oracle.savefig("Plot_Overlayed_Oracle_Experiment2.png")
    print("Saved Figure (Oracle): 'Plot_Overlayed_Oracle_Experiment2.png'")

    # Save agrgegated results
    aggregated_path = "Results_Aggregated_Experiment_2.pkl"
    with open(aggregated_path, "wb") as f:
        pickle.dump(all_results, f)
    print(f"Aggregated results saved as {aggregated_path}")

if __name__ == "__main__":
    results = load_experiment_2_results()
    plot_experiment_2_results(results)
