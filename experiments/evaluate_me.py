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
import time
import pickle


def main(models_folder: str, experiment_number: int, model_type: str, num_steps: int):
    start_time = time.time()
    set_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #path_tokenizer = str(Path(__file__).parent.parent / "custom_tokenizer.json")
    #tokenizer: Tokenizer = Tokenizer.from_file(path_tokenizer)

    models_folder_path = Path(models_folder) / f"{model_type}_{num_steps}"
    assert models_folder_path.exists(), f"Folder {models_folder_path} does not exist."

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
            from experiments.evaluate_experiment.experiment_2_me_v2 import (
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
        # Print dataset paths from config
        print(f"Using dataset for evaluation: {config.training.file_path_test}")

        model = load_model(model_file, device)
        tokenizer = T5Tokenizer.from_pretrained("t5-small")

        test_file = config.training.file_path_test

        dataset_test = SCANDataset(test_file)
        dataset_length = len(dataset_test)
        #print(f"Dataset length: {dataset_length}")

        # Calculate expected number of batches
        batch_size = 1024  # Ensure this matches the batch size used
        expected_batches = (dataset_length + batch_size - 1) // batch_size  # Ceiling division
        #print(f"Expected number of batches (batch size {batch_size}): {expected_batches}")

        dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

        label = extract_label_from_path(test_file)
        results[label] = evaluate(model, dataloader_test, tokenizer, device)
        print(results)
        
    # Save results (Added) 
    pickle_file_path = f"Results_Individual_{experiment_number}_{model_type}_{num_steps}.pkl"
    with open(pickle_file_path, "wb") as f:
        pickle.dump(results, f)
    print(f"Results saved as pickle to {pickle_file_path}")

    plot(results, model_type, num_steps)

    print(f"Evaluation time: {time.time() - start_time}")


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

    #args = sys.argv[1:]

    #assert len(args) == 2, "Provide: <PATH-MODELS+CONFIGS> <EXPERIMENT-NUMBER>"

    models_folder = sys.argv[1]
    experiment_number = int(sys.argv[2])
    model_type = sys.argv[3]
    num_steps = int(sys.argv[4])

    main(models_folder, experiment_number, model_type, num_steps)
