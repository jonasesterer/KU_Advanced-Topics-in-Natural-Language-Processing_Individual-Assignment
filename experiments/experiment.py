from pathlib import Path
from pydantic import BaseModel
from typing import Self


class ModelConfig(BaseModel):
    src_vocab_size: int
    tgt_vocab_size: int
    src_pad_idx: int
    tgt_pad_idx: int
    emb_dim: int
    num_layers: int
    num_heads: int
    forward_dim: int
    dropout: float
    max_len: int


class TrainingConfig(BaseModel):
    batch_size: int
    lr: float
    grad_clip: float
    num_steps: int = 10000 #10**5
    file_path_train: str
    file_path_test: str


class ConfigExperiment(BaseModel):
    model: ModelConfig
    training: TrainingConfig

    @classmethod
    def from_pretrained(cls, path: str | Path) -> Self:
        json_config = None
        with open(path, "r") as f:
            json_config = f.read()

        return cls.model_validate_json(json_config)

    def save(self, path: str | Path):
        with open(path, "w") as f:
            f.write(self.model_dump_json(indent=4))


class BuilderConfigExperiment:
    def __init__(self):
        self.config_model = {}
        self.config_training = {}

    def select_experiment(self, num: int):
        if num == 1:
            self.config_model["emb_dim"] = 128
            self.config_model["num_layers"] = 1
            self.config_model["num_heads"] = 8
            self.config_model["forward_dim"] = 512
            self.config_model["dropout"] = 0.05

            self.config_training["lr"] = 7e-4
            self.config_training["batch_size"] = 64  # 64
            self.config_training["grad_clip"] = 1

        elif (num == 2) or (num == 3):
            self.config_model["emb_dim"] = 128
            self.config_model["num_layers"] = 2
            self.config_model["num_heads"] = 8
            self.config_model["forward_dim"] = 256
            self.config_model["dropout"] = 0.15

            self.config_training["lr"] = 2e-4
            self.config_training["batch_size"] = 16  # 16
            self.config_training["grad_clip"] = 1
        else:
            raise NotImplementedError(f"Experiment number: {num} does not exist")

        if num == 2:
            self.config_training["num_steps"] = 10**5 * 2

        return self

    def set_vocab_sizes(self, src_vocab_size, tgt_vocab_size):
        self.config_model["src_vocab_size"] = src_vocab_size
        self.config_model["tgt_vocab_size"] = tgt_vocab_size
        return self

    def set_padding_indices(self, src_pad_idx, tgt_pad_idx):
        self.config_model["src_pad_idx"] = src_pad_idx
        self.config_model["tgt_pad_idx"] = tgt_pad_idx
        return self

    def set_max_len(self, max_len):
        self.config_model["max_len"] = max_len
        return self

    def set_path_train(self, path: Path | str):
        self.config_training["file_path_train"] = path
        return self

    def set_path_test(self, path: Path | str):
        self.config_training["file_path_test"] = path
        return self

    def build(self) -> ConfigExperiment:
        model = ModelConfig(**self.config_model)
        training = TrainingConfig(**self.config_training)
        return ConfigExperiment(model=model, training=training)


if __name__ == "__main__":
    experiment_config = (
        BuilderConfigExperiment()
        .select_experiment(1)
        .set_vocab_sizes(2, 2)
        .set_padding_indices(0, 0)
        .set_max_len(100)
        .set_path_train("train.txt")
        .set_path_test("test.txt")
        .build()
    )

    from torch.optim import AdamW
    from models.transformer import Transformer

    model = Transformer(**experiment_config.model.model_dump())
    optimizer = AdamW(model.parameters(), lr=experiment_config.training.lr)

    from uuid import uuid4

    file = Path(__file__).parent / f"{uuid4()}-test_experiment.json"
    experiment_config.save(file)
    ConfigExperiment.from_pretrained(file)

    import os

    os.remove(file)
