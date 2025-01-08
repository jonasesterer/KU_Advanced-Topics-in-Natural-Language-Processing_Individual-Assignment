import torch
from torch.utils.data import DataLoader
from tokenizers import Tokenizer
from models.transformer import Transformer
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple

plt.rcParams.update({
    'font.size': 14,       # Default font size
    'axes.titlesize': 18,  # Title font size
    'axes.labelsize': 18,  # X and Y label font size
    'xtick.labelsize': 12, # X-tick label font size
    'ytick.labelsize': 12, # Y-tick label font size
    'legend.fontsize': 14  # Legend font size
})

def argsort(seq):
    return sorted(range(len(seq)), key=seq.__getitem__)


def apply_sort(list, seq):
    return [list[i] for i in seq]


def extract_label_from_path(path: Path | str) -> int | float | str:
    return 0


def evaluate(
    model: Transformer,
    dataloader: DataLoader,
    tokenizer: Tokenizer,
    device: torch.device,
) -> Tuple[
    Tuple[
        Dict[str, Dict[str, int]],
        Dict[str, Dict[str, int]],
        Dict[str, Dict[str, int]],
        Dict[str, Dict[str, int]],
    ],
    Tuple[
        Dict[str, Dict[str, int]],
        Dict[str, Dict[str, int]],
        Dict[str, Dict[str, int]],
        Dict[str, Dict[str, int]],
    ],
]:
    normal = inner_evaluate(model, dataloader, tokenizer, device)
    set_length = inner_evaluate(model, dataloader, tokenizer, device, oracle=True)
    return (normal, set_length)


def inner_evaluate(
    model: Transformer,
    dataloader: DataLoader,
    tokenizer: Tokenizer,
    device: torch.device,
    oracle: bool = False,
) -> Tuple[
    Dict[str, Dict[str, int]],
    Dict[str, Dict[str, int]],
    Dict[str, Dict[str, int]],
    Dict[str, Dict[str, int]],
]:
    input_length_stats = {}  # {input_length: {"correct": int (tokens), "total": int (tokens)}}
    target_length_stats = {}  # {target_length: {"correct": int (tokens), "total": int (tokens)}}
    input_length_seq_stats = {}  # {input_length: {"correct": int (sequences), "total": int (sequences)}}
    target_length_seq_stats = {}  # {target_length: {"correct": int (sequences), "total": int (sequences)}}

    eos_id: int = tokenizer.token_to_id("[EOS]")
    sos_id: int = tokenizer.token_to_id("[SOS]")

    with torch.inference_mode():
        for batch in dataloader:
            src = batch["src"]
            tgt = batch["tgt"]

            src_encoded = torch.tensor(
                list(map(lambda x: x.ids, tokenizer.encode_batch(src)))
            ).to(device)
            tgt_encoded = torch.tensor(
                list(map(lambda x: x.ids, tokenizer.encode_batch(tgt)))
            ).to(device)

            # Compute the fixed lengths based on the position of the [EOS] token
            # We assume each target sequence has an [EOS] token. The length includes the EOS.
            eos_positions = (tgt_encoded == eos_id).float().argmax(dim=1)
            fixed_lengths = eos_positions + 1

            if oracle:
                prediction = model.fixed_length_greedy(
                    src_encoded,
                    sos_id,
                    eos_id,
                    device,
                    fixed_lengths=fixed_lengths,
                )
            else:
                prediction = model.inference_forward_greedy(
                    src_encoded,
                    sos_id,
                    eos_id,
                    device,
                    max_len=tgt_encoded.size(-1),
                )

            tgt_decoded = tokenizer.decode_batch(tgt_encoded.cpu().tolist())
            prediction_decoded = tokenizer.decode_batch(prediction)
            for t_src, t_tgt, p in zip(src, tgt_decoded, prediction_decoded):
                seq_correct = int(t_tgt == p)
                t_tokens = t_tgt.split()
                p_tokens = p.split()

                length_correct = sum(
                    1 for t_tok, p_tok in zip(t_tokens, p_tokens) if t_tok == p_tok
                )

                input_len = len(t_src.split())
                if input_len not in input_length_stats:
                    input_length_stats[input_len] = {"correct": 0, "total": 0}

                # Track input length stats (sequence-level)
                if input_len not in input_length_seq_stats:
                    input_length_seq_stats[input_len] = {"correct": 0, "total": 0}

                # Track target length stats (token-level)
                target_len = len(t_tokens)
                if target_len not in target_length_stats:
                    target_length_stats[target_len] = {"correct": 0, "total": 0}
                # Track target length stats (sequence-level)
                if target_len not in target_length_seq_stats:
                    target_length_seq_stats[target_len] = {"correct": 0, "total": 0}

                # Update token-level stats by length
                length_total = len(t_tokens)
                input_length_stats[input_len]["correct"] += length_correct
                input_length_stats[input_len]["total"] += length_total

                target_length_stats[target_len]["correct"] += length_correct
                target_length_stats[target_len]["total"] += length_total

                # Update sequence-level stats by length
                input_length_seq_stats[input_len]["correct"] += seq_correct
                input_length_seq_stats[input_len]["total"] += 1

                target_length_seq_stats[target_len]["correct"] += seq_correct
                target_length_seq_stats[target_len]["total"] += 1

    return (
        input_length_stats,
        target_length_stats,
        input_length_seq_stats,
        target_length_seq_stats,
    )


def plot(
    results: Dict[
        str,
        Tuple[
            Tuple[
                Dict[str, Dict[str, int]],
                Dict[str, Dict[str, int]],
                Dict[str, Dict[str, int]],
                Dict[str, Dict[str, int]],
            ],
            Tuple[
                Dict[str, Dict[str, int]],
                Dict[str, Dict[str, int]],
                Dict[str, Dict[str, int]],
                Dict[str, Dict[str, int]],
            ],
        ],
    ],
):
    assert (
        len(results.keys()) == 1
    ), "For experiment 2, we expect to only evaluate one model"

    values = list(results.values())[0]
    # Unpack results for each model
    model_1_results = values[0]
    model_2_results = values[1]

    print(f"model Standard results: {model_1_results}")
    print(f"model Oracle results: {model_2_results}")

    # Define a function to process results and generate plots
    def prepare_and_plot(unpacked_results, title_suffix):
        input_length_stats = unpacked_results[0]
        target_length_stats = unpacked_results[1]
        input_length_seq_stats = unpacked_results[2]
        target_length_seq_stats = unpacked_results[3]

        # Prepare data for token-level accuracy by target length
        target_lengths = sorted(target_length_stats.keys())
        target_accuracies_token = [
            100.0
            * (
                target_length_stats[length]["correct"]
                / target_length_stats[length]["total"]
            )
            for length in target_lengths
        ]

        # Prepare data for token-level accuracy by input length
        input_lengths = sorted(input_length_stats.keys())
        input_accuracies_token = [
            100.0
            * (
                input_length_stats[length]["correct"]
                / input_length_stats[length]["total"]
            )
            for length in input_lengths
        ]

        # Prepare data for sequence-level accuracy by target length
        target_accuracies_seq = [
            100.0
            * (
                target_length_seq_stats[length]["correct"]
                / target_length_seq_stats[length]["total"]
            )
            for length in target_lengths
        ]

        # Prepare data for sequence-level accuracy by input length
        input_accuracies_seq = [
            100.0
            * (
                input_length_seq_stats[length]["correct"]
                / input_length_seq_stats[length]["total"]
            )
            for length in input_lengths
        ]

        if title_suffix == "Standard":
            fig, axs = plt.subplots(1, 2, figsize=(12, 10))
        else:
            fig, axs = plt.subplots(2, 2, figsize=(12, 10))

        # Top-left: Token-Level Accuracy by Target Length
        axs[0, 0].bar(
            target_lengths, target_accuracies_token, color="skyblue", edgecolor="black"
        )
        axs[0, 0].set_xticks(target_lengths)
        axs[0, 0].set_xlabel("Ground-Truth Action Sequence Length (words)")
        axs[0, 0].set_ylabel("Token-Level Accuracy (%)")
        #axs[0, 0].set_title(
        #    f"Token-Level Accuracy by Target Length ({title_suffix})",
        #    fontsize=14,
        #    fontweight="bold",
        #)
        axs[0, 0].grid(axis="y", linestyle="--", alpha=0.7)

        # Top-right: Token-Level Accuracy by Input Length
        axs[0, 1].bar(
            input_lengths, input_accuracies_token, color="skyblue", edgecolor="black"
        )
        axs[0, 1].set_xticks(input_lengths)
        axs[0, 1].set_xlabel("Command Length (words)")
        axs[0, 1].set_ylabel("Token-Level Accuracy (%)")
        #axs[0, 1].set_title(
        #    f"Token-Level Accuracy by Input Length ({title_suffix})",
        #    fontsize=14,
        #    fontweight="bold",
        #)
        axs[0, 1].grid(axis="y", linestyle="--", alpha=0.7)

        if title_suffix =! "Standard":
            # Bottom-left: Sequence-Level Accuracy by Target Length
            axs[1, 0].bar(
                target_lengths, target_accuracies_seq, color="lightcoral", edgecolor="black"
            )
            axs[1, 0].set_xticks(target_lengths)
            axs[1, 0].set_xlabel("Ground-Truth Action Sequence Length (words)")
            axs[1, 0].set_ylabel("Sequence-Level Accuracy (%)")
            #axs[1, 0].set_title(
            #    f"Sequence-Level Accuracy by Target Length ({title_suffix})",
            #    fontsize=14,
            #    fontweight="bold",
            #)
            axs[1, 0].grid(axis="y", linestyle="--", alpha=0.7)
    
            # Bottom-right: Sequence-Level Accuracy by Input Length
            axs[1, 1].bar(
                input_lengths, input_accuracies_seq, color="lightcoral", edgecolor="black"
            )
            axs[1, 1].set_xticks(input_lengths)
            axs[1, 1].set_xlabel("Command Length (words)")
            axs[1, 1].set_ylabel("Sequence-Level Accuracy (%)")
            #axs[1, 1].set_title(
            #    f"Sequence-Level Accuracy by Input Length ({title_suffix})",
            #    fontsize=14,
            #    fontweight="bold",
            #)
            axs[1, 1].grid(axis="y", linestyle="--", alpha=0.7)

        plt.tight_layout()
        #plt.show()

        plot_path = f"Plot_Group_2_{title_suffix}.png"
        plt.savefig(plot_path)
        print(f"Plot saved as {plot_path}")

    # Plot results for the first model
    prepare_and_plot(model_1_results, title_suffix="Standard")

    # Plot results for the second model with "oracle lengths"
    prepare_and_plot(model_2_results, title_suffix="Oracle Lengths")
