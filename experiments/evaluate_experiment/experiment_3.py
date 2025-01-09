import torch
from torch.utils.data import DataLoader
from tokenizers import Tokenizer
from models.transformer import Transformer
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, Tuple

plt.rcParams.update({
    'font.size': 14,       # Default font size
    'axes.titlesize': 18,  # Title font size
    'axes.labelsize': 18,  # X and Y label font size
    'xtick.labelsize': 16, # X-tick label font size
    'ytick.labelsize': 16, # Y-tick label font size
    'legend.fontsize': 14  # Legend font size
})


def argsort(seq):
    return sorted(range(len(seq)), key=seq.__getitem__)


def apply_sort(list, seq):
    return [list[i] for i in seq]


def extract_label_from_path(path: Path | str) -> int | float | str:
    search_start = "num"
    search_end = "_rep"
    start_idx = str(path).find(search_start)
    end_idx = str(path).find(search_end)

    if (start_idx == -1) or (end_idx == -1):
        if "jump" in str(path):
            return "jump"
        elif "turn_left" in str(path):
            return "turn_left"
        else:
            raise NotImplementedError(
                "Cannot evaluate the model for experiment 3, check which experiment the model trained to do."
            )

    number = int(str(path)[start_idx + len(search_start) : end_idx])
    return number


def evaluate(
    model: Transformer,
    dataloader: DataLoader,
    tokenizer: Tokenizer,
    device: torch.device,
) -> Tuple[float, float]:
    total_tokens = 0
    correct_tokens = 0
    total_sequences = 0
    correct_sequences = 0

    with torch.inference_mode():
        for batch in dataloader:
            src = batch["src"]
            tgt = batch["tgt"]

            src = torch.tensor(
                list(map(lambda x: x.ids, tokenizer.encode_batch(src)))
            ).to(device)
            tgt = torch.tensor(
                list(map(lambda x: x.ids, tokenizer.encode_batch(tgt)))
            ).to(device)

            prediction = model.inference_forward_greedy(  # .greedy_decode
                src,
                tokenizer.token_to_id("[SOS]"),
                tokenizer.token_to_id("[EOS]"),
                device,
                max_len=tgt.size(-1),
            )

            tgt_decoded = tokenizer.decode_batch(tgt.cpu().tolist())
            prediction_decoded = tokenizer.decode_batch(prediction)

            for t, p in zip(tgt_decoded, prediction_decoded):
                total_sequences += 1
                correct_sequences += int(t == p)

                t_tokens = t.split()
                p_tokens = p.split()

                total_tokens += len(t_tokens)
                correct_tokens += sum(
                    1 for t_tok, p_tok in zip(t_tokens, p_tokens) if t_tok == p_tok
                )

    token_accuracy = (correct_tokens / total_tokens) * 100
    sequence_accuracy = (correct_sequences / total_sequences) * 100

    print(token_accuracy)
    print(sequence_accuracy)
    return token_accuracy, sequence_accuracy


def plot(results: Dict[str, Tuple[float, float]]):
    # Extract statistics
    model_labels = list(results.keys())
    token_accuracies = list(map(lambda x: x[0], results.values()))
    sequence_accuracies = list(map(lambda x: x[1], results.values()))

    # Sort
    idxs = argsort(model_labels)
    model_labels = apply_sort(model_labels, idxs)
    token_accuracies = apply_sort(token_accuracies, idxs)
    sequence_accuracies = apply_sort(sequence_accuracies, idxs)

    x = np.arange(len(model_labels))

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Token-level accuracy
    ax1.bar(x, token_accuracies, color="skyblue", edgecolor="black")
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_labels, ha="right", fontsize=10)
    ax1.set_xlabel("Number of Composed Commands Used For Training", fontsize=12)
    ax1.set_ylabel("Token-Level Accuracy on New Commands (%)", fontsize=12)
    #ax1.set_title("Token-Level Accuracy", fontsize=14, fontweight="bold")
    ax1.grid(axis="y", linestyle="--", alpha=0.7)

    # Sequence-level accuracy
    ax2.bar(x, sequence_accuracies, color="lightcoral", edgecolor="black")
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_labels, ha="right", fontsize=10)
    ax2.set_xlabel("Number of Composed Commands Used For Training", fontsize=12)
    ax2.set_ylabel("Sequence-Level Accuracy on New Commands (%)", fontsize=12)
    #ax2.set_title("Sequence-Level Accuracy", fontsize=14, fontweight="bold")
    ax2.grid(axis="y", linestyle="--", alpha=0.7)

    # Adjust layout
    plt.tight_layout()
    plot_path = "Plot_Group_3"
    plt.savefig(plot_path)
    print(f"Plot saved as {plot_path}")
