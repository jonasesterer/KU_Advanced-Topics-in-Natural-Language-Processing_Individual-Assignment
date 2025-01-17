import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import T5Tokenizer, T5ForConditionalGeneration
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple
import numpy as np

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
        for batch_idx, batch in enumerate(dataloader):
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
            print(f"Batch {batch_idx} - Fixed lengths: {fixed_lengths}")
    
            if oracle:
                outputs = []
                max_length_in_batch = int(fixed_lengths.max().item())  # Find the maximum oracle length in the batch
            
                for i, (input_ids, length) in enumerate(zip(encoded_src["input_ids"], fixed_lengths)):
                    # Generate sequence for each input with a length based on oracle
                    generated = model.generate(
                        input_ids=input_ids.unsqueeze(0),  # Add batch dimension
                        max_length=max_length_in_batch,  # Set maximum length to the longest in the batch
                        min_length=length.item(),  # Ensure a minimum length specific to this sample
                        early_stopping=False,  # Disable early stopping
                    )
            
                    # Debugging: Log the generated shape before adjustment
                    #print(f"Index {i} - Generated shape before adjustment: {generated.shape}")
            
                    # Pad or truncate to the batch's maximum length
                    current_length = generated.size(1)
                    if current_length < max_length_in_batch:
                        pad_size = max_length_in_batch - current_length
                        pad_tensor = torch.full(
                            (generated.size(0), pad_size),
                            pad_token_id,
                            device=generated.device,
                            dtype=generated.dtype,
                        )
                        generated = torch.cat((generated, pad_tensor), dim=1)
                    elif current_length > max_length_in_batch:
                        generated = generated[:, :max_length_in_batch]
            
                    # Debugging: Log the final shape after adjustment
                    #print(f"Index {i} - Final generated shape: {generated.shape}, Max length in batch: {max_length_in_batch}")
            
                    outputs.append(generated)
                    
                # Debugging: Log the first 10 output shapes
                print(f"Number of elements in outputs: {len(outputs)}")
                print(f"Shapes of the first 10 outputs before concatenation:")
                for j, output in enumerate(outputs[:10]):  # Print only the first 10 elements
                    print(f"  Output {j}: {output.shape}")
            
                # Concatenate outputs after ensuring consistency in lengths
                outputs = torch.cat(outputs, dim=0)
                print(f"Batch {batch_idx} - Final concatenated output shape: {outputs.shape}")
   
            else:
                outputs = model.generate(
                    input_ids=encoded_src["input_ids"],
                    max_length=tgt_ids.size(-1),
                )
                print(f"Batch {batch_idx} - Non-oracle generated shape: {outputs.shape}")

            #decoded_tgt = tokenizer.batch_decode(tgt_ids, skip_special_tokens=True)
            decoded_tgt = [
                tokenizer.decode(ids, skip_special_tokens=True)
                for ids in tgt_ids
            ]
            #decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            decoded_outputs = [
                tokenizer.decode(ids, skip_special_tokens=True)
                for ids in outputs
            ]

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
model_type: str, num_steps: int):
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

        # Choose the subplot layout
        if title_suffix == "Standard":
            fig, axs = plt.subplots(1, 2, figsize=(12, 6))  # 1x2 layout
        else:
            fig, axs = plt.subplots(2, 2, figsize=(12, 10))  # 2x2 layout

        axs = np.ravel(axs)  # Flatten axs for consistent 1D indexing

        # Plot 1: Token-Level Accuracy by Target Length
        axs[0].bar(
            target_lengths, target_accuracies_token, color="skyblue", edgecolor="black"
        )
        axs[0].set_xticks(target_lengths)
        axs[0].set_yticks(range(0, 101, 20))
        axs[0].set_xlabel("Ground-Truth Action Sequence Length (words)")
        axs[0].set_ylabel("Token-Level Accuracy (%)")
        axs[0].grid(axis="y", linestyle="--", alpha=0.7)
    
        # Plot 2: Token-Level Accuracy by Input Length
        axs[1].bar(
            input_lengths, input_accuracies_token, color="skyblue", edgecolor="black"
        )
        axs[1].set_xticks(input_lengths)
        axs[1].set_yticks(range(0, 101, 20))
        axs[1].set_xlabel("Command Length (words)")
        axs[1].set_ylabel("Token-Level Accuracy (%)")
        axs[1].grid(axis="y", linestyle="--", alpha=0.7)
    
        if title_suffix == "Oracle Lengths":
            # Plot 3: Sequence-Level Accuracy by Target Length
            axs[2].bar(
                target_lengths, target_accuracies_seq, color="lightcoral", edgecolor="black"
            )
            axs[2].set_xticks(target_lengths)
            axs[2].set_yticks(range(0, 71, 10))
            axs[2].set_xlabel("Ground-Truth Action Sequence Length (words)")
            axs[2].set_ylabel("Sequence-Level Accuracy (%)")
            axs[2].grid(axis="y", linestyle="--", alpha=0.7)
    
            # Plot 4: Sequence-Level Accuracy by Input Length
            axs[3].bar(
                input_lengths, input_accuracies_seq, color="lightcoral", edgecolor="black"
            )
            axs[3].set_xticks(input_lengths)
            axs[2].set_yticks(range(0, 71, 10))
            axs[3].set_xlabel("Command Length (words)")
            axs[3].set_ylabel("Sequence-Level Accuracy (%)")
            axs[3].grid(axis="y", linestyle="--", alpha=0.7)

        plt.tight_layout()
        plot_path = f"Plot_Individual_2_{model_type}_{num_steps}_{title_suffix}.png"
        plt.savefig(plot_path)
        print(f"Plot saved as {plot_path}")

    # Plot results for the first model
    prepare_and_plot(model_1_results, title_suffix="Standard")

    # Plot results for the second model with "oracle lengths"
    prepare_and_plot(model_2_results, title_suffix="Oracle Lengths")
