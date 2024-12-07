import torch
from experiments.tokenizer_dataloader import SCANDataset
from tokenizers import Tokenizer
from torch.utils.data import DataLoader
from models.transformer import Transformer
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import os
import json





def evaluate_model(
    model: Transformer, dataloader: DataLoader, tokenizer: Tokenizer, device: torch.device
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

            prediction = model.inference_forward_greedy( # .greedy_decode
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
                correct_sequences += int(t == p) # only true if sequence is 100p match

                t_tokens = t.split()
                p_tokens = p.split()

                total_tokens += len(t_tokens)
                correct_tokens += sum(
                    1 for t_tok, p_tok in zip(t_tokens, p_tokens) if t_tok == p_tok 
                )

    token_accuracy = (correct_tokens / total_tokens) * 100
    sequence_accuracy = (correct_sequences / total_sequences) * 100

    return token_accuracy, sequence_accuracy


def main_eval(models_folder: str, test_file: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_test = SCANDataset(test_file)
    dataloader_test = DataLoader(dataset_test, batch_size=32, shuffle=False)

    path = str(Path(__file__).parent.parent / "custom_tokenizer.json")
    tokenizer: Tokenizer = Tokenizer.from_file(path)

    models_folder_path = Path(models_folder)

    token_accuracies = []
    sequence_accuracies = []
    model_labels = []

    for model_file in models_folder_path.glob("*-model-*.pt"):
        config_file = Path(str(model_file).replace('model', 'config').replace(".pt", ".json"))
        model = load_model(model_file, config_file, device)
        token_acc, sequence_acc = evaluate_model(model, dataloader_test, tokenizer, device)

        token_accuracies.append(token_acc)
        sequence_accuracies.append(sequence_acc)
        model_labels.append(model_file.stem.split('-')[0])

    plot_results(model_labels, token_accuracies, sequence_accuracies)

def load_model(model_path: Path, config_path: Path, device: torch.device) -> Transformer:
    with open(config_path, 'r') as f:
        config_experiment = json.load(f)

    model = Transformer(**config_experiment["model"])
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    # model.load_state_dict(torch.load(model_path, weights_only=True))
    model = model.to(device)
    model.eval()
    return model

def plot_results(model_labels, token_accuracies, sequence_accuracies):
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


if __name__ == "__main__":
    import sys

    models_folder = sys.argv[1]
    test_file = sys.argv[2]
    main_eval(models_folder, test_file)
