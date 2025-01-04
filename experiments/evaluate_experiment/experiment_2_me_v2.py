import torch
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration

import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple

matplotlib.use("Agg")

def argsort(seq):
    return sorted(range(len(seq)), key=seq.__getitem__)

def apply_sort(list, seq):
    return [list[i] for i in seq]

def extract_label_from_path(path: Path | str) -> int | float | str:
    return 0

def evaluate(
    model: T5ForConditionalGeneration,
    dataloader: DataLoader,
    tokenizer: T5Tokenizer,
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
    model: T5ForConditionalGeneration,
    dataloader: DataLoader,
    tokenizer: T5Tokenizer,
    device: torch.device,
    oracle: bool = False,
) -> Tuple[
    Dict[str, Dict[str, int]],
    Dict[str, Dict[str, int]],
    Dict[str, Dict[str, int]],
    Dict[str, Dict[str, int]],
]:
    input_length_stats = {}
    target_length_stats = {}
    input_length_seq_stats = {}
    target_length_seq_stats = {}

    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id

    with torch.inference_mode():
        for batch in dataloader:
            src = batch["src"]
            tgt = batch["tgt"]

            encoded_src = tokenizer(
                src, return_tensors="pt", padding=True, truncation=True
            ).to(device)
            encoded_tgt = tokenizer(
                tgt, return_tensors="pt", padding=True, truncation=True
            ).to(device)

            tgt_ids = encoded_tgt["input_ids"]

            eos_positions = (tgt_ids == eos_token_id).float().argmax(dim=1)
            fixed_lengths = eos_positions + 1

            if oracle:
                outputs = model.generate(
                    input_ids=encoded_src["input_ids"],
                    max_length=fixed_lengths.max().item(),
                )
            else:
                outputs = model.generate(
                    input_ids=encoded_src["input_ids"],
                    max_length=tgt_ids.size(-1),
                )

            decoded_tgt = tokenizer.batch_decode(tgt_ids, skip_special_tokens=True)
            decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            for t_src, t_tgt, t_out in zip(src, decoded_tgt, decoded_outputs):
                seq_correct = int(t_tgt == t_out)
                t_tokens = t_tgt.split()
                o_tokens = t_out.split()

                length_correct = sum(
                    1 for t_tok, o_tok in zip(t_tokens, o_tokens) if t_tok == o_tok
                )

                input_len = len(t_src.split())
                if input_len not in input_length_stats:
                    input_length_stats[input_len] = {"correct": 0, "total": 0}

                if input_len not in input_length_seq_stats:
                    input_length_seq_stats[input_len] = {"correct": 0, "total": 0}

                target_len = len(t_tokens)
                if target_len not in target_length_stats:
                    target_length_stats[target_len] = {"correct": 0, "total": 0}
                if target_len not in target_length_seq_stats:
                    target_length_seq_stats[target_len] = {"correct": 0, "total": 0}

                input_length_stats[input_len]["correct"] += length_correct
                input_length_stats[input_len]["total"] += len(t_tokens)

                target_length_stats[target_len]["correct"] += length_correct
                target_length_stats[target_len]["total"] += len(t_tokens)

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

        fig, axs = plt.subplots(2, 2, figsize=(12, 10))

        # Top-left: Token-Level Accuracy by Target Length
        axs[0, 0].bar(
            target_lengths, target_accuracies_token, color="skyblue", edgecolor="black"
        )
        axs[0, 0].set_xticks(target_lengths)
        axs[0, 0].set_xlabel("Ground-Truth Action Sequence Length (words)")
        axs[0, 0].set_ylabel("Token-Level Accuracy (%)")
        axs[0, 0].set_title(
            f"Token-Level Accuracy by Target Length {title_suffix}",
            fontsize=14,
            fontweight="bold",
        )
        axs[0, 0].grid(axis="y", linestyle="--", alpha=0.7)

        # Top-right: Token-Level Accuracy by Input Length
        axs[0, 1].bar(
            input_lengths, input_accuracies_token, color="skyblue", edgecolor="black"
        )
        axs[0, 1].set_xticks(input_lengths)
        axs[0, 1].set_xlabel("Command Length (words)")
        axs[0, 1].set_ylabel("Token-Level Accuracy (%)")
        axs[0, 1].set_title(
            f"Token-Level Accuracy by Input Length {title_suffix}",
            fontsize=14,
            fontweight="bold",
        )
        axs[0, 1].grid(axis="y", linestyle="--", alpha=0.7)

        # Bottom-left: Sequence-Level Accuracy by Target Length
        axs[1, 0].bar(
            target_lengths, target_accuracies_seq, color="lightcoral", edgecolor="black"
        )
        axs[1, 0].set_xticks(target_lengths)
        axs[1, 0].set_xlabel("Ground-Truth Action Sequence Length (words)")
        axs[1, 0].set_ylabel("Sequence-Level Accuracy (%)")
        axs[1, 0].set_title(
            f"Sequence-Level Accuracy by Target Length {title_suffix}",
            fontsize=14,
            fontweight="bold",
        )
        axs[1, 0].grid(axis="y", linestyle="--", alpha=0.7)

        # Bottom-right: Sequence-Level Accuracy by Input Length
        axs[1, 1].bar(
            input_lengths, input_accuracies_seq, color="lightcoral", edgecolor="black"
        )
        axs[1, 1].set_xticks(input_lengths)
        axs[1, 1].set_xlabel("Command Length (words)")
        axs[1, 1].set_ylabel("Sequence-Level Accuracy (%)")
        axs[1, 1].set_title(
            f"Sequence-Level Accuracy by Input Length {title_suffix}",
            fontsize=14,
            fontweight="bold",
        )
        axs[1, 1].grid(axis="y", linestyle="--", alpha=0.7)

        plt.tight_layout()
        plt.savefig("2_evaluation_plot_t5_200000.png")  # Save instead of showing
        #plt.show()

    # Plot results for the first model
    prepare_and_plot(model_1_results, title_suffix="(Standard)")

    # Plot results for the second model with "oracle lengths"
    prepare_and_plot(model_2_results, title_suffix="(Oracle Lengths)")
