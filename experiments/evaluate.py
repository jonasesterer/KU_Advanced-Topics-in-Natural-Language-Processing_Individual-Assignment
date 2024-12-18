import torch
from experiments.tokenizer_dataloader import SCANDataset
from tokenizers import Tokenizer
from torch.utils.data import DataLoader
from models.transformer import Transformer
from pathlib import Path
from experiments.experiment import ConfigExperiment


def main(models_folder: str, experiment_number: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    path_tokenizer = str(Path(__file__).parent.parent / "custom_tokenizer.json")
    tokenizer: Tokenizer = Tokenizer.from_file(path_tokenizer)

    models_folder_path = Path(models_folder)

    match experiment_number:
        case 1:
            from experiments.evaluate_experiment.experiment_1 import (
                extract_label_from_path,
                evaluate,
                plot,
            )
        case 2:
            from experiments.evaluate_experiment.experiment_2 import (
                extract_label_from_path,
                evaluate,
                plot,
            )
        case 3:
            from experiments.evaluate_experiment.experiment_3 import (
                extract_label_from_path,
                evaluate,
                plot,
            )
        case _:
            raise NotImplementedError(
                f"Evaluation is not implemented for experiment {experiment_number}"
            )

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


def load_model(
    model_path: Path, config: ConfigExperiment, device: torch.device
) -> Transformer:
    model = Transformer(**config.model.model_dump())

    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


if __name__ == "__main__":
    import sys

    args = sys.argv[1:]

    assert len(args) == 2, "Provide: <PATH-MODELS+CONFIGS> <EXPERIMENT-NUMBER>"

    models_folder = sys.argv[1]
    experiment_number = int(sys.argv[2])

    main(models_folder, experiment_number)
