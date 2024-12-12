import torch
from experiments.tokenizer_dataloader import SCANDataset
from tokenizers import Tokenizer
from torch.utils.data import DataLoader
from models.transformer import Transformer
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from experiments.experiment import ConfigExperiment


def evaluate_model(
    model: Transformer,
    dataloader: DataLoader,
    tokenizer: Tokenizer,
    device: torch.device,
) -> tuple:
    total_tokens = 0
    correct_tokens = 0
    total_sequences = 0
    correct_sequences = 0

    # loss_fn = torch.nn.CrossEntropyLoss(ignore_index=model.tgt_pad_idx)

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

    token_accuracies = []
    sequence_accuracies = []
    model_labels = []

    for model_file in models_folder_path.glob("*-model-*.pt"):
        config_file = Path(
            str(model_file).replace("model", "config").replace(".pt", ".json")
        )
        config = ConfigExperiment.from_pretrained(config_file)

        model = load_model(model_file, config, device)

        test_file = config.training.file_path_test

        dataset_test = SCANDataset(test_file)
        dataloader_test = DataLoader(dataset_test, batch_size=256, shuffle=False)

        token_acc, sequence_acc = evaluate_model(
            model, dataloader_test, tokenizer, device
        )

        token_accuracies.append(token_acc)
        sequence_accuracies.append(sequence_acc)
        # model_labels.append(model_file.stem.split('-')[1]) # UUID
        model_labels.append(extract_num_from_path(test_file, experiment_number))  # UUID

    idxs = np.argsort(model_labels)
    token_accuracies = np.array(token_accuracies)[idxs].tolist()
    model_labels = np.array(model_labels)[idxs].tolist()
    sequence_accuracies = np.array(sequence_accuracies)[idxs].tolist()
    plot_results(model_labels, token_accuracies, sequence_accuracies)


def extract_num_from_path(
    path: Path | str, experiment_number: int
) -> int | float | str:
    if experiment_number == 1:
        return 0

    elif experiment_number == 2:
        search_start = "num"
        search_end = "_rep"
        start_idx = str(path).find(search_start)
        end_idx = str(path).find(search_end)

        number = int(str(path)[start_idx + len(search_start) : end_idx])
        return number

    elif experiment_number == 3:
        search_start = "num"
        search_end = "_rep"
        start_idx = str(path).find(search_start)
        end_idx = str(path).find(search_end)

        number = int(str(path)[start_idx + len(search_start) : end_idx])
        return number

    elif experiment_number == 4:
        if "jump" in str(path):
            return "jump"
        elif "turn_left" in str(path):
            return "turn_left"
        else:
            raise NotImplementedError(
                "Only works for jump or turn_left, TODO refactor this evaluation code to be for evaluating a specific thing rather than for an experiment"
            )

    else:
        raise NotImplementedError(
            f"No extracting function is implemented for experiment {experiment_number}"
        )


def load_model(
    model_path: Path, config: ConfigExperiment, device: torch.device
) -> Transformer:
    model = Transformer(**config.model.model_dump())
    # model = Transformer(**config_experiment["model"])

    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


def plot_results2(model_labels, token_accuracies, sequence_accuracies):
    x = np.arange(len(model_labels))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.bar(x, token_accuracies)
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_labels)
    ax1.set_xlabel("Number of Composed Commands Used For Training")
    ax1.set_ylabel("Accuracy on new commands (%)")
    ax1.set_title("Token-Level Accuracy")

    ax2.bar(x, sequence_accuracies)
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_labels)
    ax2.set_xlabel("Number of Composed Commands Used For Training")
    ax2.set_ylabel("Accuracy on new commands (%)")
    ax2.set_title("Sequence-Level Accuracy")

    plt.tight_layout()
    plt.show()


def plot_results(model_labels, token_accuracies, sequence_accuracies):
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
