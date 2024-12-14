import torch
from torch.utils.data import DataLoader

from tokenizers import Tokenizer

from models.transformer import Transformer

import matplotlib.pyplot as plt

import numpy as np

from pathlib import Path
from typing import Dict, Tuple


def argsort(seq):
    return sorted(range(len(seq)), key=seq.__getitem__)


def apply_sort(list, seq):
    return [list[i] for i in seq]


def extract_label_from_path(path: Path | str) -> int | float | str:
    path = str(path)
    if "_p" in path:
        start_idx = path.find("_p") + len("_p")
        end_idx = path.find(".txt")
        return int(path[start_idx:end_idx])
    else:
        return 100


def evaluate(
    model: Transformer,
    dataloader: DataLoader,
    tokenizer: Tokenizer,
    device: torch.device,
) -> Tuple[float]:
    total_tokens = 0
    correct_tokens = 0

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
                t_tokens = t.split()
                p_tokens = p.split()

                total_tokens += len(t_tokens)
                correct_tokens += sum(
                    1 for t_tok, p_tok in zip(t_tokens, p_tokens) if t_tok == p_tok
                )

    token_accuracy = (correct_tokens / total_tokens) * 100

    return (token_accuracy,)


def plot(results: Dict[str, Tuple[float]]):
    # Extract statistics
    model_labels = list(results.keys())
    token_accuracies = list(map(lambda x: x[0], results.values()))

    # Sort
    idxs = argsort(model_labels)
    sorted_model_labels = apply_sort(model_labels, idxs)
    sorted_token_accuracies = apply_sort(token_accuracies, idxs)

    x = np.arange(len(sorted_model_labels))

    # Add " (%)" to model labels
    model_labels_with_percent = [str(int(label)) + "%" for label in sorted_model_labels]

    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.bar(x, sorted_token_accuracies)
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_labels_with_percent)
    ax1.set_xlabel("Commands Used")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title("TExperiment 1")

    # Add vertical gridlines at intervals of 20
    ax1.set_yticks(range(0, 101, 20))
    ax1.yaxis.grid(True, linestyle="-", linewidth=1)
    ax1.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig("experiments/results/results_1.png")
    plt.show()
