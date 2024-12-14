import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

from tokenizers import Tokenizer

import sys
import os
from pathlib import Path
from uuid import uuid4
from typing import Union

from models.transformer import Transformer

from experiments.experiment import BuilderConfigExperiment
from experiments.tokenizer_dataloader import SCANDataset

from tqdm import tqdm


def main():
    # Setup
    num_experiment, train_file, test_file, save_path = sys.argv[1:]

    save_path = Path(save_path)

    dataset_train = SCANDataset(train_file)
    dataset_test = SCANDataset(test_file)

    path = str(Path(__file__).parent.parent / "custom_tokenizer.json")
    tokenizer: Tokenizer = Tokenizer.from_file(path)

    config_experiment = (
        BuilderConfigExperiment()
        .select_experiment(int(num_experiment))
        .set_vocab_sizes(tokenizer.get_vocab_size(), tokenizer.get_vocab_size())
        .set_padding_indices(
            src_pad_idx=tokenizer.token_to_id("[PAD]"),
            tgt_pad_idx=tokenizer.token_to_id("[PAD]"),
        )
        .set_max_len(max(dataset_train.max_len, dataset_test.max_len))
        .set_path_train(
            os.path.abspath(train_file)
        )  # TODO if this has to be more dynamic we need to rewrite this.
        .set_path_test(
            os.path.abspath(test_file)
        )  # TODO if this has to be more dynamic we need to rewrite this.
        .build()
    )

    os.makedirs(save_path, exist_ok=True)

    uuid = uuid4()
    save_path_config = save_path / f"{num_experiment}-config-{uuid}.json"

    config_experiment.save(save_path_config)

    print(f"Saved Experiment config at: {save_path_config}")

    dataloader_train = DataLoader(
        dataset_train, batch_size=config_experiment.training.batch_size, shuffle=True
    )

    dataloader_test = DataLoader(
        dataset_test, batch_size=config_experiment.training.batch_size, shuffle=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Transformer(**config_experiment.model.model_dump())
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=config_experiment.training.lr)

    # Train
    trained_model: torch.nn.Module = train(
        model,
        optimizer,
        tokenizer,
        device,
        dataloader_train,
        dataloader_test,
        grad_clip=config_experiment.training.grad_clip,
        epochs=int(config_experiment.training.num_steps // len(dataset_train)),
    )

    # Save
    torch.save(
        trained_model.state_dict(), save_path / f"{num_experiment}-model-{uuid}.pt"
    )


def train(
    model: Transformer,
    optimizer: AdamW,
    tokenizer: Tokenizer,
    device: Union[torch.device, str],
    dataloader_train: DataLoader,
    dataloader_test: DataLoader,
    grad_clip: float,
    epochs: int = 10,
):
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=model.tgt_pad_idx)

    for epoch in tqdm(range(epochs), desc="Training", unit="epoch"):
        model.train()
        for batch in dataloader_train:
            optimizer.zero_grad()

            src = batch["src"]
            tgt = batch["tgt"]

            src = torch.tensor(
                list(map(lambda x: x.ids, tokenizer.encode_batch(src)))
            ).to(device)

            tgt = torch.tensor(
                list(map(lambda x: x.ids, tokenizer.encode_batch(tgt)))
            ).to(device)

            outputs = model.train_forward(
                src=src, tgt=tgt, sos_token=tokenizer.token_to_id("[SOS]")
            )

            loss = loss_fn(
                outputs.view(-1, outputs.size(-1)), tgt[:, 1:].contiguous().view(-1)
            )
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

            optimizer.step()

        if (epoch % 5 == 0) or (epoch == epochs):
            with torch.inference_mode():
                item = next(iter(dataloader_test))
                src = item["src"]
                tgt = item["tgt"]

                src = torch.tensor(
                    list(map(lambda x: x.ids, tokenizer.encode_batch(src)))
                )

                tgt = torch.tensor(
                    list(map(lambda x: x.ids, tokenizer.encode_batch(tgt)))
                )

                prediction = model.inference_forward_greedy(
                    src,
                    tokenizer.token_to_id("[SOS]"),
                    tokenizer.token_to_id("[EOS]"),
                    device,
                    max_len=tgt.size(-1),
                )

                print(
                    f"Model prediction: {tokenizer.decode_batch(prediction, skip_special_tokens=True)[0]}\n"
                )
                print(f"GT: {tokenizer.decode_batch(tgt.cpu().tolist())[0]}\n")

    return model


if __name__ == "__main__":
    main()
