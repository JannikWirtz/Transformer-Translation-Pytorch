import torch
import torch.nn as nn
from torch.utils.data import Dataset


class BilingualDataset(Dataset):
    def __init__(self, dataset, tok_src, tok_tgt, src_lang, tgt_lang, seq_len) -> None:
        super().__init__()

        self.dataset = dataset
        self.tok_src = tok_src
        self.tok_tgt = tok_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len

        self.sos_token = torch.tensor([tok_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tok_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tok_tgt.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        src_tgt_pair = self.dataset[idx]
        src_text = src_tgt_pair["translation"][self.src_lang]
        tgt_text = src_tgt_pair["translation"][self.tgt_lang]

        enc_input_tokens = self.tok_src.encode(src_text).ids
        dec_input_tokens = self.tok_tgt.encode(tgt_text).ids

        # padding to fill input to seq_len
        enc_pad_size = self.seq_len - len(enc_input_tokens) - 2  # SOS + EOS
        dec_pad_size = self.seq_len - len(dec_input_tokens) - 1  # only SOS

        assert enc_pad_size > 0 and dec_pad_size > 0, "Input sequence is too long."

        # Add tokens to inputs
        enc_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_pad_size, dtype=torch.int64),
            ]
        )

        dec_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_pad_size, dtype=torch.int64),
            ]
        )

        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_pad_size, dtype=torch.int64),
            ]
        )

        assert enc_input.shape[0] == self.seq_len
        assert dec_input.shape[0] == self.seq_len
        assert label.shape[0] == self.seq_len

        return {
            "enc_input": enc_input,
            "enc_mask": (enc_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),  # (1, 1, seq_len)
            "dec_input": dec_input,
            "dec_mask": (dec_input != self.pad_token).unsqueeze(0).unsqueeze(0).int()
            & causal_mask(dec_input.shape[0]),  # (1, seq_len) & (1, seq_len, seq_len)
            "label": label,
            "src_text": src_text,
            "tgt_text": tgt_text,
        }


def causal_mask(seq_len):
    """Make sure that decoder can only see previous tokens"""
    mask = torch.triu(torch.ones(1, seq_len, seq_len), diagonal=1).type(torch.int)
    return mask == 0  # negate mask
