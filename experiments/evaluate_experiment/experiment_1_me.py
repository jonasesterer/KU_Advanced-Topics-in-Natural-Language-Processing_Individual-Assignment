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
    tokenizer: T5Tokenizer,
    device: torch.device,
) -> Tuple[float]:

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
            #start_time = time.time()
            input_ids = tokenizer(
                src, padding=True, truncation=True, return_tensors="pt"
            ).input_ids.to(device)
            target_ids = tokenizer(
                tgt, padding=True, truncation=True, return_tensors="pt"
            ).input_ids.to(device)
            #print(f"Time tokenizing: {time.time() - start_time}")
            
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
            #start_time = time.time()
            outputs = model.generate(input_ids, max_length=target_ids.size(1))
            #print(outputs)
            #print(f"Time generating: {time.time() - start_time}")

            # Decode target and prediction
            #start_time = time.time()
            tgt_decoded = [
                tokenizer.decode(ids, skip_special_tokens=True) for ids in target_ids
            ]
            prediction_decoded = [
                tokenizer.decode(ids, skip_special_tokens=True) for ids in outputs
            ]
            #print(f"Time decoding: {time.time() - start_time}")

            #start_time = time.time()
            for t, p in zip(tgt_decoded, prediction_decoded):
                t_tokens = t.split()
                p_tokens = p.split()

                total_tokens += len(t_tokens)
                correct_tokens += sum(
                    1 for t_tok, p_tok in zip(t_tokens, p_tokens) if t_tok == p_tok
                )
            #print(f"Time evaluating: {time.time() - start_time}")
            
    token_accuracy = (correct_tokens / total_tokens) * 100

    return (token_accuracy,)

def plot(results: Dict[int, Tuple[float]], model_type: str, num_steps: int):
    labels = sorted(results.keys())
    accuracies = [results[label][0] for label in labels]

    x = np.arange(len(labels))

    plt.figure(figsize=(12, 6))
    ax = plt.gca()  # Get the current Axes object
    ax.bar(x, accuracies, color="skyblue", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{label}%" for label in labels])
    ax.set_xlabel("Commands Used")
    ax.set_ylabel("Token-Level Accuracy (%)")
    ax.set_yticks(range(0, 101, 20))  # Corrected line
    #ax.set_title("Experiment 1")
    ax.grid(axis="y", linestyle="-", linewidth=1, alpha=0.7)
                   
    plt.tight_layout()
    plot_path = "Plot_Individual_1_{model_type}_{num_steps}.png"
    plt.savefig(plot_path)  # Save instead of showing
    print(f"Plot saved as {plot_path}")
