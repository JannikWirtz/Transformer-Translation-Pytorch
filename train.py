from pathlib import Path
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, random_split

from config import getConfig, get_weights_file_path
from dataset import BilingualDataset, causal_mask
from datasets import load_dataset
from Transformer import Transformer, build_transformer
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from torch.utils.tensorboard import SummaryWriter


def get_all_sentences(dataset, lang="en"):
    """Generator to get sentences from a dataset"""
    for item in dataset:
        yield item["translation"][lang]


def get_tokenizer(config, dataset, lang="en"):
    tokenizer_path = Path(config["tokenizer_file"].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(
            WordLevel(unk_token="[UNK]")
        )  # token to replace words not in vocab
        tokenizer.pre_tokenizer = Whitespace()  # split by whitespace
        trainer = WordLevelTrainer(
            special_tokens=["[UNK]", "[SOS]", "[EOS]", "[PAD]"], min_frequency=2
        )
        tokenizer.train_from_iterator(get_all_sentences(dataset, lang), trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


def get_dataset(config):
    dataset = load_dataset(
        "opus_books", f'{config["lang_src"]}-{config["lang_tgt"]}', split="train"
    )
    tokenizer_src = get_tokenizer(config, dataset, config["lang_src"])
    tokenizer_tgt = get_tokenizer(config, dataset, config["lang_tgt"])

    # train/val split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = random_split(dataset, [train_size, val_size])

    train_set = BilingualDataset(
        train_data,
        tokenizer_src,
        tokenizer_tgt,
        config["lang_src"],
        config["lang_tgt"],
        config["seq_len"],
    )
    val_set = BilingualDataset(
        val_data,
        tokenizer_src,
        tokenizer_tgt,
        config["lang_src"],
        config["lang_tgt"],
        config["seq_len"],
    )

    max_len_src = 0
    max_len_tgt = 0

    for item in dataset:
        src_ids = tokenizer_src.encode(item["translation"][config["lang_src"]]).ids
        tgt_ids = tokenizer_src.encode(item["translation"][config["lang_tgt"]]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f"Max len of source sentence: {max_len_src}")
    print(f"Max len of target sentence: {max_len_tgt}")

    train_dataloader = DataLoader(
        train_set, batch_size=config["batch_size"], shuffle=True
    )
    val_dataloader = DataLoader(val_set, batch_size=config["batch_size"])

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


def get_model(config, vocab_src_len, vocab_tgt_len):
    return build_transformer(
        vocab_src_len,
        vocab_tgt_len,
        config["seq_len"],
        config["seq_len"],
        config["d_model"],
    )


def train_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_dataset(config)
    model = get_model(
        config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()
    ).to(device)

    # Tensorboard
    writer = SummaryWriter(config["experiment_name"])

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    init_epoch = 0
    global_step = 0
    if config["preload"] is not None:
        raise NotImplementedError

    # label smoothing regularization: https://arxiv.org/pdf/1512.00567.pdf
    loss_fn = nn.CrossEntropyLoss(
        ignore_index=tokenizer_tgt.token_to_id("[PAD]"), label_smoothing=0.1
    ).to(device)

    for epoch in range(init_epoch, config["num_epochs"]):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch}")
        for batch in batch_iterator:
            encoder_input = batch["enc_input"].to(device)  # (batch_size, seq_len)
            decoder_input = batch["dec_input"].to(device)  # (batch_size, seq_len)
            enc_mask = batch["enc_mask"].to(device)  # (batch_size, 1, 1, seq_len)
            dec_mask = batch["dec_mask"].to(device)  # (batch_size, 1, seq_len, seq_len)
            label = batch["label"].to(device)

            encoder_output = model.encode(
                encoder_input, enc_mask
            )  # (batch_size, seq_len, d_model)
            decoder_output = model.decode(
                decoder_input, encoder_output, enc_mask, dec_mask
            )
            output = model.project(
                decoder_output
            )  # (batch_size, seq_len, d_model) -> (batch_size, seq_len, tgt_vocab_size)

            # (batch_size, seq_len, tgt_vocab_size) -> (batch_size * seq_len, tgt_vocab_size)
            loss = loss_fn(output.view(-1, output.size(-1)), label.view(-1))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1

            # logging
            batch_iterator.set_postfix(loss=loss.item())
            writer.add_scalar("Loss/train", loss.item(), global_step)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optim_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
        },
        get_weights_file_path(config, epoch),
    )


if __name__ == "__main__":
    config = getConfig()
    train_model(config)
