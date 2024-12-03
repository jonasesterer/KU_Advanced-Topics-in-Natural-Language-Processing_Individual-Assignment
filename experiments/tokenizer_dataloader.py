from tokenizers import Tokenizer
from pathlib import Path
from torch.utils.data import Dataset, DataLoader


class SCANDataset(Dataset):
    def __init__(self, datafile: Path | str):
        self.data: list[dict[str, str]] = self.extract_samples(datafile)
        self.max_len: int = max(
            map(lambda x: max(len(x["src"]), len(x["tgt"])), self.data)
        )

    def extract_samples(self, file: Path | str) -> list[dict[str, str]]:
        all_lines = []
        with open(file, "r") as tf:
            for line in tf.readlines():
                lines = line.split("IN:")[-1].split("OUT:")
                lines = list(map(lambda x: x.strip(), lines))
                all_lines.append({"src": lines[0], "tgt": lines[1]})

        return all_lines

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


if __name__ == "__main__":
    path = str(Path(__file__).parent.parent / "custom_tokenizer.json")

    tokenizer: Tokenizer = Tokenizer.from_file(path)

    dataset = SCANDataset(
        Path(__file__).parent.parent
        / "data/datafiles/add_prim_split"
        / "tasks_test_addprim_jump.txt"
    )
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    item = next(iter(dataloader))

    src = item["src"]
    tgt = item["tgt"]

    # it will auto pad, because I trained it to lol
    tgt_tokens_all = list(map(lambda x: x.ids, tokenizer.encode_batch(src)))

    # it will auto pad, because I trained it to lol
    src_tokens_all = list(map(lambda x: x.ids, tokenizer.encode_batch(tgt)))
    print(src_tokens_all)
