from experiments.experiment import BuilderConfigExperiment
from experiments.tokenizer_dataloader import SCANDataset
from torch.utils.data import DataLoader
from tokenizers import Tokenizer
import sys
import torch
from pathlib import Path
import os
from uuid import uuid4
from models.transformer import Transformer
from torch.optim import AdamW
from typing import Union


def main():
    # Setup
    num_experiment, train_file, test_file, save_path = sys.argv[1:]
    # num_experiment, train_file, test_file, save_path = sys.argv[-1].split(" ")

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
        num_steps=config_experiment.training.num_steps,
        grad_clip=config_experiment.training.grad_clip,
    )

    # Save
    torch.save(trained_model, save_path / f"{num_experiment}-model-{uuid}.pt")


def train(
    model: Transformer,
    optimizer: AdamW,
    tokenizer: Tokenizer,
    device: Union[torch.device, str],
    dataloader_train: DataLoader,
    dataloader_test: DataLoader,
    num_steps: int,
    grad_clip: float,
):
    # TODO add logging
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=model.tgt_pad_idx)

    # _epochs = 0
    # while num_steps:
    for _epochs in range(10):
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
            # outputs = model(src=src, tgt=tgt)

            loss = loss_fn(
                outputs.view(-1, outputs.size(-1)), tgt[:, 1:].contiguous().view(-1)
            )
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

            optimizer.step()

            num_steps -= len(batch)

            if num_steps <= 0:
                num_steps = False
                break

        if _epochs % 5 == 0 or not num_steps:
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
                #                prediction = greedy_decode(
                #                    model,
                #                    src,
                #                    tgt.size(-1),
                #                    tokenizer.token_to_id("[SOS]"),
                #                    tokenizer.token_to_id("[EOS]"),
                #                    device,
                #                )
                print(
                    # f"Model prediction: {tokenizer.decode_batch(prediction.cpu().tolist(), skip_special_tokens=False)[0]}\n"
                    f"Model prediction: {tokenizer.decode_batch(prediction, skip_special_tokens=False)[0]}\n"
                )
                print(f"GT: {tokenizer.decode_batch(tgt.cpu().tolist())[0]}\n")
        _epochs += 1
    return model


if __name__ == "__main__":
    main()
