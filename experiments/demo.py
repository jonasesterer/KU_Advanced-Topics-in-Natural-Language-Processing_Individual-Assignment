# THIS FILE DOES A FULL MODEL PASS FROM TEXT TO TRANSFORMER


from tokenizers import Tokenizer
from pathlib import Path
import torch
from models.transformer import Transformer

path = str(Path(__file__).parent.parent / "custom_tokenizer.json")

tokenizer: Tokenizer = Tokenizer.from_file(path)

tgt_sentences = [
    "I_TURN_RIGHT I_TURN_RIGHT I_RUN I_TURN_RIGHT I_TURN_RIGHT I_WALK I_TURN_RIGHT I_TURN_RIGHT I_WALK I_TURN_RIGHT I_TURN_RIGHT I_WALK",
    "I_TURN_RIGHT I_TURN_RIGHT I_RUN I_TURN_RIGHT I_TURN_RIGHT I_RUN I_TURN_RIGHT I_TURN_RIGHT I_TURN_RIGHT I_TURN_RIGHT I_TURN_RIGHT I_TURN_RIGHT I_TURN_RIGHT I_TURN_RIGHT I_TURN_RIGHT I_TURN_RIGHT I_TURN_RIGHT I_TURN_RIGHT",
    "I_TURN_RIGHT I_WALK I_TURN_RIGHT I_WALK I_TURN_RIGHT I_WALK I_TURN_RIGHT I_WALK I_TURN_RIGHT I_WALK I_TURN_RIGHT I_WALK I_TURN_RIGHT I_WALK I_TURN_RIGHT I_WALK I_TURN_LEFT I_TURN_LEFT I_LOOK I_TURN_LEFT I_TURN_LEFT I_LOOK",
]

src_sentences = [
    "walk opposite right thrice after run opposite right",
    "turn around right thrice after run opposite right twice",
    "look opposite left twice after walk around right twice",
]


# GET INDICIES IN TWO DIFFERENT PARTS, SO WE AVOID DOING UNECCESARY PADDING
# I.E A SMALL OPTIMIZATION.

# it will auto pad, because I trained it to lol
tgt_tokens_all = list(map(lambda x: x.ids, tokenizer.encode_batch(tgt_sentences)))


# it will auto pad, because I trained it to lol
src_tokens_all = list(map(lambda x: x.ids, tokenizer.encode_batch(src_sentences)))


print(tokenizer.decode_batch(tgt_tokens_all))


tgt_tokens_all = torch.tensor(tgt_tokens_all)
src_tokens_all = torch.tensor(src_tokens_all)


print(tokenizer.get_vocab_size())
model = Transformer(
    src_vocab_size=tokenizer.get_vocab_size(),
    tgt_vocab_size=tokenizer.get_vocab_size(),
    src_pad_idx=tokenizer.token_to_id("[PAD]"),
    tgt_pad_idx=tokenizer.token_to_id("[PAD]"),
    emb_dim=512,
    num_layers=6,
    num_heads=8,
    forward_dim=2048,
    dropout=0,
    max_len=128,  # We need to find this for the dataset, maybe they set this as a hyperparmeter in the assignment text
)

out = model.forward(
    src=src_tokens_all, tgt=tgt_tokens_all
)  # outputs in log space, so to get probabilities, we need to call softmax
# however this is desirable as we want to use the logit loss as it is more numerically stable.
assert out.isnan().sum() == 0
