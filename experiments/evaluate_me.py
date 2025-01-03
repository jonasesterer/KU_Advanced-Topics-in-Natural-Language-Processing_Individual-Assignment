import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, set_seed
from experiments.tokenizer_dataloader import SCANDataset
#from tokenizers import Tokenizer
from torch.utils.data import DataLoader
#from models.transformer import Transformer
from pathlib import Path
from experiments.experiment import ConfigExperiment
#from transformers import set_seed
from typing import Union, Dict


def main(models_folder: str, experiment_number: int):
    set_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #path_tokenizer = str(Path(__file__).parent.parent / "custom_tokenizer.json")
    #tokenizer: Tokenizer = Tokenizer.from_file(path_tokenizer)

    models_folder_path = Path(models_folder)

    # TODO in these we can have an expected results that is contingent on the
    # label, si
    # plot should take an optional argument, which is expected results
    match experiment_number:
        case 1:
            from experiments.evaluate_experiment.experiment_1_me import (
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
        print(f"Evaluating model from file: {model_file}")  # Added print statement
        config_file = Path(
            str(model_file).replace("model", "config").replace(".pt", ".json")
        )
        config = ConfigExperiment.from_pretrained(config_file)
        # ? with or without config ? 
        model = load_model(model_file, device)

        test_file = config.training.file_path_test

        dataset_test = SCANDataset(test_file)
        dataset_length = len(dataset_test)
        print(f"Dataset length: {dataset_length}")

        # Calculate expected number of batches
        batch_size = 1024  # Ensure this matches the batch size used
        expected_batches = (dataset_length + batch_size - 1) // batch_size  # Ceiling division
        print(f"Expected number of batches (batch size {batch_size}): {expected_batches}")

        dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
        # ? Dataset or DataLoader ?
        label = extract_label_from_path(test_file)
        results[label] = evaluate(model, dataloader_test, device)

    plot(results)


def load_model(
    model_path: Path, device: torch.device
) -> T5ForConditionalGeneration: #Transformer:
    #model = Transformer(**config.model.model_dump())
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
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
