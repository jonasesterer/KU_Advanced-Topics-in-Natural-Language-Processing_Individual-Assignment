import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

import sys
import os
from pathlib import Path
from uuid import uuid4
from typing import Union

# ↓↓↓ NEW: import T5
from transformers import set_seed, T5Tokenizer, T5ForConditionalGeneration

from experiments.experiment import BuilderConfigExperiment
from experiments.tokenizer_dataloader import SCANDataset

from tqdm import tqdm


def main():
    set_seed(0)

    # CLI: e.g. python train.py 1 train.txt test.txt out_dir
    num_experiment, train_file, test_file, save_path = sys.argv[1:]
    save_path = Path(save_path)
    os.makedirs(save_path, exist_ok=True)

    # Prepare dataset
    dataset_train = SCANDataset(train_file)
    dataset_test = SCANDataset(test_file)

    # Build config (from your experiment.py)
    config_experiment = (
        BuilderConfigExperiment()
        .select_experiment(int(num_experiment))
        # we pass in dummy vocab sizes—these won't really matter for T5
        .set_vocab_sizes(32128, 32128)  
        # set the [PAD] index to 0 for logging, again not strictly used by T5
        .set_padding_indices(0, 0)
        .set_max_len(max(dataset_train.max_len, dataset_test.max_len))
        .set_path_train(os.path.abspath(train_file))
        .set_path_test(os.path.abspath(test_file))
        .build()
    )

    # ↓↓↓ NEW: Instead of your custom tokenizer.json, use T5’s tokenizer
    tokenizer = T5Tokenizer.from_pretrained("t5-small")

    # ↓↓↓ NEW: Initialize T5 model
    model = T5ForConditionalGeneration.from_pretrained("t5-small")

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Use AdamW with the learning rate from your config
    optimizer = AdamW(model.parameters(), lr=config_experiment.training.lr)

    # DataLoaders
    dataloader_train = DataLoader(
        dataset_train, 
        batch_size=config_experiment.training.batch_size, 
        shuffle=True
    )
    dataloader_test = DataLoader(
        dataset_test, 
        batch_size=config_experiment.training.batch_size, 
        shuffle=True
    )

    # Number of epochs (example: total steps / dataset size)
    steps_per_epoch = len(dataloader_train)
    epochs = int(config_experiment.training.num_steps // len(dataset_train))
    print(f"Training for {epochs} epochs...")

    # Save the config
    uuid_ = uuid4()
    save_path_config = save_path / f"{num_experiment}-config-{uuid_}.json"
    config_experiment.save(save_path_config)
    print(f"Saved Experiment config at: {save_path_config}")

    # Train
    trained_model = train(
        model,
        tokenizer,
        device,
        dataloader_train,
        dataloader_test,
        optimizer,
        grad_clip=config_experiment.training.grad_clip,
        epochs=epochs,
        max_length=config_experiment.model.max_len,
    )

    # Save
    # Option 1: Standard PyTorch state_dict:
    torch.save(trained_model.state_dict(), save_path / f"{num_experiment}-model-{uuid_}.pt")
    # Option 2: Hugging Face style:
    # trained_model.save_pretrained(save_path / f"{num_experiment}-model-{uuid_}")

    print("Done!")


def train(
    model: T5ForConditionalGeneration,
    tokenizer: T5Tokenizer,
    device: Union[torch.device, str],
    dataloader_train: DataLoader,
    dataloader_test: DataLoader,
    optimizer: AdamW,
    grad_clip: float,
    epochs: int,
    max_length: int,
):
    """
    Train loop for T5 in a seq2seq manner. T5 internally handles
    the causal mask for the decoder if you pass `labels=...`.
    """
    model.train()

    for epoch in range(epochs):
    #for epoch in tqdm(range(epochs), desc="Training", unit="epoch"):

        epoch_loss = 0.0
        #for batch in dataloader_train:
        for batch in tqdm(dataloader_train, desc=f"Epoch {epoch+1}", leave=False):

            src_texts = batch["src"]  # list of strings
            tgt_texts = batch["tgt"]  # list of strings

            # 1. Encode the inputs and targets
            # 'padding=True' will pad to the longest example in the batch
            # set 'max_length' to your config_experiment.model.max_len or so
            encoded_input = tokenizer(
                src_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded_target = tokenizer(
                tgt_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )

            # 2. Move to GPU (if available)
            input_ids = encoded_input["input_ids"].to(device)
            attention_mask = encoded_input["attention_mask"].to(device)
            labels = encoded_target["input_ids"].to(device)

            # 3. Forward pass (T5 does the masked self-attention internally)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss  # CrossEntropy inside T5

            # 4. Backprop
            optimizer.zero_grad()
            loss.backward()

            # optional: gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

            optimizer.step()

            epoch_loss += loss.item()

        # End of epoch
        avg_loss = epoch_loss / len(dataloader_train)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

        # Optional: Evaluate or sample from the model
        if epoch == (epochs - 1):
            model.eval()
            with torch.inference_mode():
                sample_batch = next(iter(dataloader_test))
                src_texts = sample_batch["src"]
                tgt_texts = sample_batch["tgt"]

                # Encode the source
                encoded_input = tokenizer(
                    src_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                ).to(device)

                # Generate predictions
                pred_ids = model.generate(
                    input_ids=encoded_input["input_ids"],
                    attention_mask=encoded_input["attention_mask"],
                    max_length=max_length,
                )
                # Decode predictions & ground truth
                pred_str = tokenizer.decode(pred_ids[0], skip_special_tokens=True)
                gt_str = tgt_texts[0]

                print(f"\nSample Source: {src_texts[0]}")
                print(f"Model Prediction: {pred_str}")
                print(f"Ground Truth: {gt_str}\n")

            model.train()

    return model


if __name__ == "__main__":
    main()
