import torch
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration

import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple
import time
from tqdm import tqdm


def evaluate(
    model: T5ForConditionalGeneration,
    dataloader: DataLoader,
    tokenizer: T5Tokenizer,
    device: torch.device,
    oracle: bool = False,
) -> Tuple[Dict[int, float], Dict[int, float]]:
    input_length_stats = {}
    target_length_stats = {}

    model.eval()
    eos_id = tokenizer.eos_token_id

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating Batches"):
            src = batch["src"]
            tgt = batch["tgt"]

            start_time = time.time()
            # Tokenize inputs
            encoded_input = tokenizer(
                src,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(device)
            encoded_target = tokenizer(
                tgt,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(device)
            tokenization_time = time.time() - start_time

            tgt_ids = encoded_target["input_ids"]

            # Calculate oracle lengths if applicable
            max_lengths = (
                (tgt_ids == eos_id).nonzero(as_tuple=True)[1] + 1
                if oracle
                else None
            )

            # Generate predictions
            start_time = time.time()
            predictions = model.generate(
                input_ids=encoded_input["input_ids"],
                attention_mask=encoded_input["attention_mask"],
                max_length=max_lengths.max().item() if oracle else 512,
            )
            generation_time = time.time() - start_time

            # Decode predictions
            start_time = time.time()
            tgt_decoded = [
                tokenizer.decode(ids, skip_special_tokens=True)
                for ids in tgt_ids
            ]
            pred_decoded = [
                tokenizer.decode(ids, skip_special_tokens=True)
                for ids in predictions
            ]
            decoding_time = time.time() - start_time

            # Evaluate accuracy
            for src_text, tgt_text, pred_text in zip(src, tgt_decoded, pred_decoded):
                t_tokens = tgt_text.split()
                p_tokens = pred_text.split()

                length_correct = sum(
                    1 for t_tok, p_tok in zip(t_tokens, p_tokens) if t_tok == p_tok
                )
                input_len = len(src_text.split())
                target_len = len(t_tokens)

                if input_len not in input_length_stats:
                    input_length_stats[input_len] = {"correct": 0, "total": 0}
                if target_len not in target_length_stats:
                    target_length_stats[target_len] = {"correct": 0, "total": 0}

                input_length_stats[input_len]["correct"] += length_correct
                input_length_stats[input_len]["total"] += len(t_tokens)

                target_length_stats[target_len]["correct"] += length_correct
                target_length_stats[target_len]["total"] += len(t_tokens)

            print(
                f"Batch timings - Tokenization: {tokenization_time:.2f}s, "
                f"Generation: {generation_time:.2f}s, Decoding: {decoding_time:.2f}s"
            )

    return input_length_stats, target_length_stats


def plot(
    input_length_stats: Dict[int, float],
    target_length_stats: Dict[int, float],
    oracle: bool = False,
):
    input_lengths = sorted(input_length_stats.keys())
    target_lengths = sorted(target_length_stats.keys())

    input_accuracies = [
        100.0
        * (
            input_length_stats[length]["correct"]
            / input_length_stats[length]["total"]
        )
        for length in input_lengths
    ]
    target_accuracies = [
        100.0
        * (
            target_length_stats[length]["correct"]
            / target_length_stats[length]["total"]
        )
        for length in target_lengths
    ]

    plt.figure(figsize=(12, 6))
    plt.plot(input_lengths, input_accuracies, label="Input Length Accuracy")
    plt.plot(target_lengths, target_accuracies, label="Target Length Accuracy")
    plt.xlabel("Sequence Length")
    plt.ylabel("Accuracy (%)")
    plt.title(f"Experiment 2 Accuracy {'(Oracle)' if oracle else ''}")
    plt.legend()
    plt.grid()
    plt.show()


def main():
    # Paths
    model_path = Path("path_to_trained_model.pt")
    data_path = Path("path_to_test_data.txt")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and tokenizer
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    # Load test data
    dataset_test = SCANDataset(data_path)
    dataloader_test = DataLoader(dataset_test, batch_size=32)

    # Evaluate
    input_stats, target_stats = evaluate(
        model, dataloader_test, tokenizer, device, oracle=False
    )

    # Plot
    plot(input_stats, target_stats, oracle=False)


if __name__ == "__main__":
    main()
