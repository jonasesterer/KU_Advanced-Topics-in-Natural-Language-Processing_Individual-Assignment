import torch
from experiments.tokenizer_dataloader import SCANDataset
from tokenizers import Tokenizer
from torch.utils.data import DataLoader
from models.transformer import Transformer
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from experiments.experiment import ConfigExperiment
from typing import Dict, Tuple


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
            # Has to be rewritten to account for strings
            for t, p in zip(tgt_decoded, prediction_decoded):
                # TODO inspect
                total_sequences += 1
                correct_sequences += int(t == p)  # only true if sequence is 100p match

                t_tokens = t.split()
                p_tokens = p.split()

                total_tokens += len(t_tokens)
                correct_tokens += sum(
                    1 for t_tok, p_tok in zip(t_tokens, p_tokens) if t_tok == p_tok
                )

    token_accuracy = (correct_tokens / total_tokens) * 100
    sequence_accuracy = (correct_sequences / total_sequences) * 100

    return token_accuracy, sequence_accuracy


def main_eval(models_folder: str, experiment_number: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    path_tokenizer = str(Path(__file__).parent.parent / "custom_tokenizer.json")
    tokenizer: Tokenizer = Tokenizer.from_file(path_tokenizer)

    models_folder_path = Path(models_folder)

    results = {}

    for model_file in models_folder_path.glob("*-model-*.pt"):
        config_file = Path(
            str(model_file).replace("model", "config").replace(".pt", ".json")
        )
        config = ConfigExperiment.from_pretrained(config_file)

        model = load_model(model_file, config, device)

        test_file = config.training.file_path_test

        dataset_test = SCANDataset(test_file)
        dataloader_test = DataLoader(dataset_test, batch_size=256, shuffle=False)

        label = extract_label_from_path(test_file)
        results[label] = evaluate(model, dataloader_test, tokenizer, device)

    plot(results)


def argsort(seq):
    # Returns the indices that would sort the list
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


def load_model(
    model_path: Path, config: ConfigExperiment, device: torch.device
) -> Transformer:
    model = Transformer(**config.model.model_dump())

    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


def plot(results: Dict[str, Tuple[float, float]]):
    # model_labels, token_accuracies, sequence_accuracies

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
    ax1.set_ylabel("Accuracy on New Commands (%)", fontsize=12)
    ax1.set_title("Token-Level Accuracy", fontsize=14, fontweight="bold")
    ax1.grid(axis="y", linestyle="--", alpha=0.7)

    # Sequence-level accuracy
    ax2.bar(x, sequence_accuracies, color="lightcoral", edgecolor="black")
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_labels, ha="right", fontsize=10)
    ax2.set_xlabel("Number of Composed Commands Used For Training", fontsize=12)
    ax2.set_ylabel("Accuracy on New Commands (%)", fontsize=12)
    ax2.set_title("Sequence-Level Accuracy", fontsize=14, fontweight="bold")
    ax2.grid(axis="y", linestyle="--", alpha=0.7)

    # Adjust layout
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import sys

    args = sys.argv[1:]

    assert len(args) == 2, "Provide: <PATH-MODELS+CONFIGS> <EXPERIMENT-NUMBER>"

    models_folder = sys.argv[1]
    experiment_number = int(sys.argv[2])

    main_eval(models_folder, experiment_number)
