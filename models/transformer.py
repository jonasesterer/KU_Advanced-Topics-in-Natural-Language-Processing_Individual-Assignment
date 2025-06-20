import math

import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim: int, num_heads: int):
        super().__init__()
        assert (
            emb_dim >= num_heads
        ), "The embedding dimension cannot be smaller than the number of heads"
        head_dim = emb_dim // num_heads

        #        B           S      num_heads  head_dim -> # B x num_heads x S x head_dim
        def split(x):
            return x.view(x.size(0), x.size(1), num_heads, head_dim).transpose(1, 2)

        self.split = split
        # self.split = lambda x: x.view(
        # x.size(0), x.size(1), num_heads, head_dim
        # ).transpose(1, 2)
        # -> B x S x num_heads x head_dim # -> B x S x num_heads * head_dim
        # avoid copy
        self.collect = (
            lambda x: x.transpose(1, 2)
            .contiguous()
            .view(x.size(0), x.size(2), num_heads * head_dim)
        )

        self.key_linear_in = nn.Linear(emb_dim, head_dim * num_heads)
        self.query_linear_in = nn.Linear(emb_dim, head_dim * num_heads)
        self.value_linear_in = nn.Linear(emb_dim, head_dim * num_heads)

        self.linear_out = nn.Linear(head_dim * num_heads, emb_dim)

        self.factor = head_dim**0.5

    def forward(self, query, key, value, mask=None):
        """
        Forward pass for multi-head attention.

        Args:
            query: Tensor of shape (batch_size, seq_len, emb_dim)
            key: Tensor of shape (batch_size, seq_len, emb_dim)
            value: Tensor of shape (batch_size, seq_len, emb_dim)
            mask: Optional mask tensor of shape (batch_size, 1, 1, seq_len)

        Returns:
            Tensor of shape (batch_size, seq_len, emb_dim)
        """
        # query, key, value: B x S x E
        key = self.key_linear_in(key)  # B x S x head_dim * num_heads
        query = self.query_linear_in(query)  # B x S x head_dim * num_heads
        value = self.value_linear_in(value)  # B x S x head_dim * num_heads

        # split into num_heads -> B x num_heads x S x head_dim
        key = self.split(key)
        query = self.split(query)
        value = self.split(value)

        # row wise softmax because we want to weight each value embedding -> dim = -1
        #                               transpose -> B x num_heads x head_dim x S
        energy = torch.softmax(
            self.apply_energy_mask(
                query @ key.transpose(-2, -1) / self.factor, mask=mask
            ),
            dim=-1,
        )  # B x num_heads x S x S

        attention = energy @ value  # B x num_heads x S x head_dim
        attention = self.collect(attention)

        query = self.linear_out(attention)

        return query

    def apply_energy_mask(self, tensor: torch.Tensor, mask=None):
        """
        Mask must be shape B x S, B is batch , S is the sequence length
        We fill the values with -infinity, because exp(-infinity) = 0,
        thus the energy at the masked points will be zero.
        """
        if mask is not None:
            # mask = mask.view(mask.size(0), 1, 1, mask.size(1)) we just receive it in this size
            tensor = tensor.masked_fill(~mask, -torch.inf)

        return tensor


class TransformerBlock(nn.Module):
    def __init__(self, emb_dim, num_heads, dropout, forward_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, forward_dim),
            nn.ReLU(),
            nn.Linear(forward_dim, emb_dim),
        )
        self.attention_multihead = MultiHeadAttention(emb_dim, num_heads)
        self.dropout = nn.Dropout(p=dropout)

        self.layer_norm_1 = nn.LayerNorm(emb_dim, eps=1e-6)
        self.layer_norm_2 = nn.LayerNorm(emb_dim, eps=1e-6)

    def forward(self, query, key, value, mask=None):
        """
        Forward pass through the Transformer block.

        Args:
            x: Tensor of shape (batch_size, seq_len, emb_dim)
            mask: Optional mask tensor

        Returns:
            Tensor of shape (batch_size, seq_len, emb_dim)
        """

        # B x S x E
        query = query + self.attention_multihead(query, key, value, mask)
        query = self.dropout(query)
        query = self.layer_norm_1(query)

        # B x S x E
        query = query + self.mlp(query)
        query = self.dropout(query)
        query = self.layer_norm_2(query)

        return query


def get_sinusoid_table(max_len, emb_dim):
    def get_angle(pos, i, emb_dim):
        return pos / 10000 ** ((2 * (i // 2)) / emb_dim)

    sinusoid_table = torch.zeros(max_len, emb_dim)
    for pos in range(max_len):
        for i in range(emb_dim):
            if i % 2 == 0:
                sinusoid_table[pos, i] = math.sin(get_angle(pos, i, emb_dim))
            else:
                sinusoid_table[pos, i] = math.cos(get_angle(pos, i, emb_dim))
    return sinusoid_table


class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        emb_dim,
        num_layers,
        num_heads,
        forward_dim,
        dropout,
        max_len,
        padding_idx=0,
    ):
        super().__init__()
        # Get embeddings

        self.token_embeddings = nn.Embedding(
            vocab_size, emb_dim, padding_idx=padding_idx
        )

        # max_len + 3, because we have padding tokens, sos, and eos tokens.
        self.sinusoid_table = nn.Embedding.from_pretrained(
            get_sinusoid_table(max_len + 3, emb_dim), freeze=True
        )

        self.dropout = nn.Dropout(p=dropout)

        # num_layers transformer blocks
        self.blocks_transformer = nn.ModuleList(
            [
                TransformerBlock(
                    emb_dim=emb_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    forward_dim=forward_dim,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, mask=None):
        """
        Forward pass for the encoder.

        Args:
            x: Tensor of shape (batch_size, seq_len)
            mask: Optional mask tensor

        Returns:
            Tensor of shape (batch_size, seq_len, emb_dim)
        """

        device = x.device

        token_emb = self.token_embeddings(x)

        # current largest sequence length
        position_indices = (
            torch.arange(1, x.size(1) + 1, device=device)
            .unsqueeze(0)
            .expand(x.size(0), -1)
        )

        # Mask position indices where padding tokens are present
        padding_mask = (x != 0).long()
        position_indices = position_indices * padding_mask
        sinus_emb = self.sinusoid_table(position_indices)

        """
        I don't see why this is neccessary, so I don't do it. (like there is no difference between indexing somewhere else right?)
        (I understand that you want the model to learn some temporal information, maybe I'll do this later)

        Make sure to shift each index by +1 (and the max_len in the creation of the sinusoidal table, too)
        This is done because index 0 is usually reserved for special tokens like [PAD], which don't need a positional encoding.
        """

        embedding = token_emb + sinus_emb
        embedding = self.dropout(embedding)

        x = embedding
        # previously: query = key = value = embedding

        for block in self.blocks_transformer:
            x = block(x, x, x, mask)

        return x


class DecoderBlock(nn.Module):
    def __init__(self, emb_dim, num_heads, forward_dim, dropout):
        super().__init__()

        self.layer_norm_1 = nn.LayerNorm(emb_dim, eps=1e-6)
        self.attention = MultiHeadAttention(emb_dim, num_heads)

        self.transformer_block = TransformerBlock(
            emb_dim, num_heads=num_heads, dropout=dropout, forward_dim=forward_dim
        )
        self.dropout = nn.Dropout(p=dropout)

    # x is the encoder output
    def forward(self, x, value, key, src_mask, tgt_mask):
        """
        Forward pass through the decoder block.

        Args:
            x: Tensor of shape (batch_size, tgt_seq_len, emb_dim)
            encoder_out: Encoder output tensor
            src_mask: Source mask tensor
            tgt_mask: Target mask tensor

        Returns:
            Tensor of shape (batch_size, tgt_seq_len, emb_dim)
        """
        # x: is from the decoder
        # value, key: B x S x E is from the encoder

        x = x + self.attention(x, x, x, tgt_mask)
        x = self.dropout(x)
        x = self.layer_norm_1(x)

        x = self.transformer_block(x, key, value, src_mask)

        return x


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        emb_dim,
        num_layers,
        num_heads,
        forward_dim,
        dropout,
        max_len,
        padding_idx=0,
    ):
        super().__init__()

        self.token_embeddings = nn.Embedding(
            vocab_size, emb_dim, padding_idx=padding_idx
        )

        # TODO: Remove sos?
        # max_len + 3, because we have sos, and eos tokens.
        self.position_embeddings = nn.Embedding(max_len + 2, emb_dim)

        self.dropout = nn.Dropout(p=dropout)

        # num_layers transformer blocks
        self.blocks_decoders = nn.ModuleList(
            [
                DecoderBlock(
                    emb_dim=emb_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    forward_dim=forward_dim,
                )
                for _ in range(num_layers)
            ]
        )

        self.linear_final = nn.Linear(emb_dim, vocab_size)

    def forward(self, x, encoder_out, src_mask, tgt_mask):
        """
        Forward pass for the decoder.

        Args:
            x: Tensor of shape (batch_size, tgt_seq_len)
            encoder_out: Encoder output tensor
            src_mask: Source mask tensor
            tgt_mask: Target mask tensor

        Returns:
            Tensor of shape (batch_size, tgt_seq_len, vocab_size)
        """
        device = x.device

        token_emb = self.token_embeddings(x)

        # current largest sequence length
        position_indices = (
            torch.arange(1, x.size(1) + 1, device=device)
            .unsqueeze(0)
            .expand(x.size(0), -1)
        )

        pos_emb = self.position_embeddings(position_indices)

        embeddings = token_emb + pos_emb

        embeddings = self.dropout(embeddings)

        x = embeddings

        for block in self.blocks_decoders:
            x = block(x, encoder_out, encoder_out, src_mask, tgt_mask)

        x = self.linear_final(x)
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        src_pad_idx,
        tgt_pad_idx,
        emb_dim=512,
        num_layers=6,
        num_heads=8,
        forward_dim=2048,
        dropout=0.0,
        max_len=128,
    ):
        super().__init__()

        self.encoder = Encoder(
            vocab_size=src_vocab_size,
            emb_dim=emb_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            forward_dim=forward_dim,
            dropout=dropout,
            max_len=max_len,
            padding_idx=src_pad_idx,
        )

        self.decoder = Decoder(
            vocab_size=tgt_vocab_size,
            emb_dim=emb_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            forward_dim=forward_dim,
            dropout=dropout,
            max_len=max_len,
            padding_idx=tgt_pad_idx,
        )

        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx

        # + 2 for sos and eos tokens
        self.max_len = max_len + 2

    def create_src_mask(self, src):
        device = src.device
        # (batch_size, 1, 1, src_seq_len)
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask.to(device)

    def create_tgt_mask(self, tgt):
        device = tgt.device
        batch_size, tgt_len = tgt.shape
        tgt_mask = (tgt != self.tgt_pad_idx).unsqueeze(1).unsqueeze(2)
        tgt_mask = tgt_mask * torch.tril(torch.ones((tgt_len, tgt_len))).expand(
            batch_size, 1, tgt_len, tgt_len
        ).to(device)
        return tgt_mask

    def forward(self, src, tgt):
        """
        Forward pass through the Transformer model.

        Args:
            src: Tensor of shape (batch_size, src_seq_len)
            tgt: Tensor of shape (batch_size, tgt_seq_len)

        Returns:
            Tensor of shape (batch_size, tgt_seq_len, tgt_vocab_size)
        """
        src_mask = self.create_src_mask(src)
        tgt_mask = self.create_src_mask(tgt)
        encoder_out = self.encoder.forward(src, src_mask)
        decoder_out = self.decoder.forward(tgt, encoder_out, src_mask, tgt_mask)

        return decoder_out

    def train_forward(self, src, tgt):
        src_mask = self.create_src_mask(src)
        encoder_out = self.encoder.forward(src, src_mask)

        predicted_logits = []

        for i in range(1, tgt.size(1)):
            tgt_mask = self.create_src_mask(tgt[:, :i])
            decoder_out = self.decoder.forward(
                tgt[:, :i], encoder_out, src_mask, tgt_mask
            )
            predicted_logits.append(decoder_out[:, -1, :])

        return torch.stack(predicted_logits, dim=1)

    @torch.inference_mode()
    def inference_forward_greedy(
        self, src, sos_token, eos_token, device, max_len=None, return_up_to_eos=True
    ):
        self.eval()

        src = src.to(device)

        max_len = self.max_len if max_len is None else max_len

        # Create the source mask
        src_mask = self.create_src_mask(src)

        # Pass the source through the encoder
        encoder_out = self.encoder.forward(src, src_mask)

        # Prepare target sequence starting with the start symbol
        batch_size = src.size(0)
        tgt = torch.full((batch_size, 1), sos_token, dtype=torch.long, device=device)

        eos_tracker = torch.zeros((batch_size,), device=device)

        for _ in range(max_len - 1):
            # Create the target mask
            tgt_mask = self.create_src_mask(tgt)

            # Pass through the decoder
            logits = self.decoder.forward(tgt, encoder_out, src_mask, tgt_mask)

            # Take the last token's logits
            next_token_logits = logits[:, -1, :]  # Shape: (batch_size, tgt_vocab_size)
            next_token_logits = torch.nn.functional.softmax(next_token_logits, dim=-1)
            # Get the most likely next token
            next_token = torch.argmax(next_token_logits, dim=-1)  # Shape: (batch_size,)

            eos_tracker += next_token == eos_token

            # Append the next token to the target sequence
            tgt = torch.cat([tgt, next_token.unsqueeze(1)], dim=1)

            # Check for EOS token in all sequences and stop decoding for finished sequences
            if eos_tracker.all():
                break
        if return_up_to_eos:
            list_target = tgt.cpu().tolist()
            indicies = list(map(lambda x: try_find(x, eos_token), list_target))
            return list(map(lambda x: x[0][: x[1]], zip(list_target, indicies)))

        return tgt

    @torch.inference_mode()
    def fixed_length_greedy(
        self,
        src: torch.Tensor,
        sos_token: int,
        eos_token: int,
        device: torch.device,
        fixed_lengths: torch.Tensor,
        return_up_to_eos=True,
    ):
        self.eval()

        src = src.to(device)
        fixed_lengths = fixed_lengths.to(device)
        max_len = int(fixed_lengths.max().item())

        # Encode
        src_mask = self.create_src_mask(src)
        encoder_out = self.encoder.forward(src, src_mask)

        batch_size = src.size(0)
        tgt = torch.full((batch_size, 1), sos_token, dtype=torch.long, device=device)

        for i in range(max_len - 1):
            tgt_mask = self.create_src_mask(tgt)
            logits = self.decoder.forward(tgt, encoder_out, src_mask, tgt_mask)

            # Take the last token's logits
            next_token_logits = logits[:, -1, :]
            next_token_probs = torch.nn.functional.softmax(next_token_logits, dim=-1)

            top_values, top_indices = torch.topk(next_token_probs, k=2, dim=-1)
            chosen_tokens = torch.empty(batch_size, dtype=torch.long, device=device)

            for k in range(batch_size):
                desired_length = fixed_lengths[k].item()

                if i < desired_length - 2:
                    # Not the final token yet; must not choose EOS if top.
                    chosen = top_indices[k, 0]
                    if chosen == eos_token:
                        chosen = top_indices[k, 1]
                    chosen_tokens[k] = chosen

                elif i == desired_length - 2:
                    # This is the exact final token required (the EOS position).
                    chosen_tokens[k] = eos_token

                else:
                    # Beyond the desired length, produce EOS as padding.
                    chosen_tokens[k] = eos_token

            tgt = torch.cat([tgt, chosen_tokens.unsqueeze(1)], dim=1)

        if return_up_to_eos:
            list_target = tgt.cpu().tolist()
            indicies = list(map(lambda x: try_find(x, eos_token), list_target))
            return list(map(lambda x: x[0][: x[1]], zip(list_target, indicies)))

        return tgt


def try_find(array: list, item):
    try:
        res = array.index(item) + 1
    except Exception:
        res = None

    return res


if __name__ == "__main__":
    # general test case
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Transformer(
        src_vocab_size=200,
        tgt_vocab_size=220,
        src_pad_idx=0,
        tgt_pad_idx=0,
    ).to(device)

    # source input: batch size 4, sequence length of 75
    src_in = torch.randint(0, 200, (4, 75)).to(device)

    # target input: batch size 4, sequence length of 80
    tgt_in = torch.randint(0, 220, (4, 80)).to(device)

    # expected output shape of the model
    expected_out_shape = torch.Size([4, 80, 220])

    with torch.no_grad():
        out = model(src_in, tgt_in)
    assert (
        out.shape == expected_out_shape
    ), f"wrong output shape, expected: {expected_out_shape}"
    print(model.train_forward(src_in, tgt_in, 1).shape)
    print(tgt_in.shape)
# %%
