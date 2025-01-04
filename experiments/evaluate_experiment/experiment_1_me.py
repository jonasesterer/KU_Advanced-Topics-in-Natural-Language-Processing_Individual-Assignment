import torch
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration

#from tokenizers import Tokenizer
#from models.transformer import Transformer

import matplotlib.pyplot as plt
import numpy as np
import time

from pathlib import Path
from typing import Dict, Tuple
from tqdm import tqdm


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
        return 80 #100


def evaluate(
    model: T5ForConditionalGeneration,
    dataloader: DataLoader,
    #tokenizer: Tokenizer,
    device: torch.device,
) -> Tuple[float]:
    tokenizer = T5Tokenizer.from_pretrained("t5-small")

    total_tokens = 0
    correct_tokens = 0

    with torch.inference_mode():
        #for batch in dataloader:
        for batch in tqdm(dataloader, desc="Evaluating Batches", leave=False):
            src = batch["src"]
            tgt = batch["tgt"]

            #src = torch.tensor(
            #    list(map(lambda x: x.ids, tokenizer.encode_batch(src)))
            #).to(device)
            #tgt = torch.tensor(
            #    list(map(lambda x: x.ids, tokenizer.encode_batch(tgt)))
            #).to(device)

            # Tokenize source and target
            start_time = time.time()
            input_ids = tokenizer(
                src, padding=True, truncation=True, return_tensors="pt"
            ).input_ids.to(device)
            target_ids = tokenizer(
                tgt, padding=True, truncation=True, return_tensors="pt"
            ).input_ids.to(device)
            print(f"Time tokenizing: {time.time() - start_time}")
            
            #prediction = model.inference_forward_greedy(  # .greedy_decode
            #    src,
            #    tokenizer.token_to_id("[SOS]"),
            #    tokenizer.token_to_id("[EOS]"),
            #    device,
            #    max_len=tgt.size(-1),
            #)

            # Generate predictions ? use length of test set ? 
            #print(f"target size 1: {target_ids.size(1)}")
            #print(f"target size -1: {target_ids.size(-1)}")
            start_time = time.time()
            outputs = model.generate(input_ids, max_length=target_ids.size(1))
            #print(outputs)
            print(f"Time generating: {time.time() - start_time}")

            # Decode target and prediction
            start_time = time.time()
            tgt_decoded = [
                tokenizer.decode(ids, skip_special_tokens=True) for ids in target_ids
            ]
            prediction_decoded = [
                tokenizer.decode(ids, skip_special_tokens=True) for ids in outputs
            ]
            print(f"Time decoding: {time.time() - start_time}")

            start_time = time.time()
            for t, p in zip(tgt_decoded, prediction_decoded):
                t_tokens = t.split()
                p_tokens = p.split()

                total_tokens += len(t_tokens)
                correct_tokens += sum(
                    1 for t_tok, p_tok in zip(t_tokens, p_tokens) if t_tok == p_tok
                )
            print(f"Time evaluating: {time.time() - start_time}")
            
    token_accuracy = (correct_tokens / total_tokens) * 100

    return (token_accuracy,)

def plot(results: Dict[int, Tuple[float]]):
    labels = sorted(results.keys())
    accuracies = [results[label][0] for label in labels]

    x = np.arange(len(labels))

    plt.figure(figsize=(12, 6))
    plt.bar(x, accuracies, color="skyblue", edgecolor="black")
    plt.xticks(x, [f"{label}%" for label in labels])
    plt.xlabel("Commands Used")
    plt.ylabel("Accuracy (%)")
    plt.title("Experiment 1")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig("1_evaluation_plot.png")  # Save instead of showing

def plot_old(results: Dict[str, Tuple[float]]):
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

    ax1.bar(x, sorted_token_accuracies, color="skyblue", edgecolor="black")
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_labels_with_percent)
    ax1.set_xlabel("Commands Used")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title("Experiment 1")

    # Add vertical gridlines at intervals of 20
    ax1.set_yticks(range(0, 101, 20))
    ax1.yaxis.grid(True, linestyle="-", linewidth=1, alpha=0.7)
    ax1.set_axisbelow(True)

    plt.tight_layout()
    plt.show()
