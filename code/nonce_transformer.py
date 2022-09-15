# -*- coding: utf-8 -*-
"""translation_transformer.ipynb
 Nonce generation using transformers
"""

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline

"""Nonce generation  with nn.Transformer and torchtext
======================================================
"""
import os
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import Multi30k
from typing import Iterable, List
import json

config_file = "../config/n6_e512_ff2048_h8_d1.json"
# config_file = "../config/n6_e1024_ff4096_h16_d1.json"
with open(config_file, "r") as f:
    config = json.load(f)


# Define special symbols and indices
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
# Make sure the tokens are in order of their indices to properly insert them in vocab
special_symbols = ["<unk>", "<pad>", "<bos>", "<eos>"]

"""Seq2Seq Network using Transformer
---------------------------------

Transformer is a Seq2Seq model introduced in `“Attention is all you
need” <https://papers.nips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf>`__
paper for solving machine translation tasks. 
Below, we will create a Seq2Seq network that uses Transformer. The network
consists of three parts. First part is the embedding layer. This layer converts tensor of input indices
into corresponding tensor of input embeddings. These embedding are further augmented with positional
encodings to provide position information of input tokens to the model. The second part is the 
actual `Transformer <https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html>`__ model. 
Finally, the output of Transformer model is passed through linear layer
that give un-normalized probabilities for each token in the target language. 
"""

from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import Transformer
import math
from torch.utils.data import Dataset
import csv

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DataGenerator
class NonceDataset(Dataset):
    def __init__(self, split):
        self.inputs, self.targets = self.get_data(split)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input = self.inputs[idx]
        target = self.targets[idx]

        return input, target

    def get_data(self, split):
        tstart = config["TSTART"]
        tb = config["IBYTES"]
        # train start -> count
        if split == "train":
            start = tstart
            end = tstart + config['COUNT']
        else:  # valid train end -> start + 63
            start = tstart + config['COUNT'] + 1
            end = start + 63

        output_csv = "../data/nonce_train.csv"
        infile = open(output_csv, newline="")
        reader = csv.DictReader(infile)

        inputs = []
        targets = []
        print("Loading: ", split)
        print("data Start: {}".format(start))
        print("data End: {}".format(end))
        for i, row in enumerate(reader):
            if i < start:
                continue
            input = row["input"]
            target = row["target"]
            inlist = []
            outlist = []
            # split hex string into char integers
            # convert strings to tensors
            while input:
                inlist.append(int(input[:4], 16) + 4)  # adding 4 as 4 tokens are taken
                input = input[4:]

            inputs.append(inlist)

            while target:
                outlist.append(
                    int(target[:2], 16) + 4
                )  # adding 4 as 4 tokens are taken
                target = target[2:]

            targets.append(outlist)
            if i == end:
                break

        return inputs, targets


# helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pos_embedding", pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(
            token_embedding + self.pos_embedding[: token_embedding.size(0), :]
        )


# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


# Seq2Seq Network
class Seq2SeqTransformer(nn.Module):
    def __init__(
        self,
        num_encoder_layers: int,
        num_decoder_layers: int,
        emb_size: int,
        nhead: int,
        src_vocab_size: int,
        tgt_vocab_size: int,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = Transformer(
            d_model=emb_size,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)

    def forward(
        self,
        src: Tensor,
        trg: Tensor,
        src_mask: Tensor,
        tgt_mask: Tensor,
        src_padding_mask: Tensor,
        tgt_padding_mask: Tensor,
        memory_key_padding_mask: Tensor,
    ):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outs = self.transformer(
            src_emb,
            tgt_emb,
            src_mask,
            tgt_mask,
            None,
            src_padding_mask,
            tgt_padding_mask,
            memory_key_padding_mask,
        )
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(
            self.positional_encoding(self.src_tok_emb(src)), src_mask
        )

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(
            self.positional_encoding(self.tgt_tok_emb(tgt)), memory, tgt_mask
        )


"""During training, we need a subsequent word mask that will prevent model to look into
the future words when making predictions. We will also need masks to hide
source and target padding tokens. Below, let's define a function that will take care of both. 



"""


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask


def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=DEVICE).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


"""Let's now define the parameters of our model and instantiate the same. Below, we also 
define our loss function which is the cross-entropy loss and the optmizer used for training.
"""

torch.manual_seed(0)

SRC_VOCAB_SIZE = 65540  # byte 255 + 4 tokens
TGT_VOCAB_SIZE = 260
EMB_SIZE = config["EMBDINGS"]
NHEAD = config["HEADS"]
FFN_HID_DIM = config["FF"]
BATCH_SIZE = 128
EVAL_BATCH_SIZE = 64
NUM_ENCODER_LAYERS = config["ELAYERS"]
NUM_DECODER_LAYERS = config["DLAYERS"]
DROPOUT = config["DROP"]

transformer = Seq2SeqTransformer(
    NUM_ENCODER_LAYERS,
    NUM_DECODER_LAYERS,
    EMB_SIZE,
    NHEAD,
    SRC_VOCAB_SIZE,
    TGT_VOCAB_SIZE,
    FFN_HID_DIM,
    DROPOUT,
)

for p in transformer.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

transformer = transformer.to(DEVICE)

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

LR = config["LR"]
optimizer = torch.optim.Adam(
    transformer.parameters(), lr=LR, betas=(0.9, 0.98), eps=1e-9
)

"""Collation
---------

As seen in the ``Data Sourcing and Processing`` section, our data iterator yields a pair of raw strings. 
We need to convert these string pairs into the batched tensors that can be processed by our ``Seq2Seq`` network 
defined previously. Below we define our collate function that convert batch of raw strings into batch tensors that
can be fed directly into our model.   



"""

from torch.nn.utils.rnn import pad_sequence

# function to add BOS/EOS and create tensor for input sequence indices
def tensor_transform(token_ids: List[int]):
    return torch.cat(
        (torch.tensor([BOS_IDX]), torch.tensor(token_ids), torch.tensor([EOS_IDX]))
    )


# function to collate data samples into batch tesors
# This funtion is needed
def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(tensor_transform(src_sample))
        tgt_batch.append(tensor_transform(tgt_sample))

    # data is of same size. No need for padding
    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch, tgt_batch


"""Let's define training and evaluation loop that will be called for each 
epoch.



"""

from torch.utils.data import DataLoader

# Load ddata
train_iter = NonceDataset("train")
train_dataloader = DataLoader(
    train_iter,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=8,
    collate_fn=collate_fn,
)

val_iter = NonceDataset("valid")
val_dataloader = DataLoader(val_iter, batch_size=EVAL_BATCH_SIZE, collate_fn=collate_fn)


def train_epoch(model, optimizer):
    model.train()
    losses = 0

    for src, tgt in train_dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
            src, tgt_input
        )

        logits = model(
            src,
            tgt_input,
            src_mask,
            tgt_mask,
            src_padding_mask,
            tgt_padding_mask,
            src_padding_mask,
        )

        optimizer.zero_grad()

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()

    return losses / len(train_dataloader)


def evaluate(model):
    model.eval()
    losses = 0
    char = 0
    word = 0

    for src, tgt in val_dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
            src, tgt_input
        )

        logits = model(
            src,
            tgt_input,
            src_mask,
            tgt_mask,
            src_padding_mask,
            tgt_padding_mask,
            src_padding_mask,
        )

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()
        # accuracy: number of values that are same
        vals = torch.argmax(logits, dim=2)
        # cehck number of digits that are correct
        char += torch.numel(tgt_out) - torch.count_nonzero(vals - tgt_out)
        # check if any whole word is correct
        word += EVAL_BATCH_SIZE - torch.count_nonzero(
            torch.sum(torch.abs(vals - tgt_out), dim=0)
        )

    return losses / len(val_dataloader), word, char


"""Now we have all the ingredients to train our model. Let's do it!



"""

from timeit import default_timer as timer

last_epoch = 1
chkpoint_file = os.path.join(
    "../output/{}.chk".format(os.path.basename(config_file).split(".")[0])
)
if os.path.exists(chkpoint_file):
    print("Loading checkpoint : {}".format(chkpoint_file))
    checkpoint = torch.load(chkpoint_file)
    transformer.load_state_dict(checkpoint["transformer"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    last_epoch = checkpoint["epoch"]

NUM_EPOCHS = 1000
prev_loss = 100
prev_acc = 0
for epoch in range(last_epoch, NUM_EPOCHS + 1):
    start_time = timer()
    train_loss = train_epoch(transformer, optimizer)
    val_loss, wacc, cacc = evaluate(transformer)
    end_time = timer()
    print(
        (
            f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, wAcc: {wacc}, cAcc: {cacc}, "
            f"Epoch time = {(end_time - start_time):.3f}s"
        )
    )
    if train_loss < prev_loss or wacc > prev_acc:
        # save checkpoint
        if wacc > prev_acc:
            prev_acc = wacc
            chkpoint_file = os.path.join(
                "../output/{}.chk".format(
                    os.path.basename(config_file).split(".")[0] + "_{}".format(wacc)
                )
            )
        else:
            prev_loss = train_loss
            chkpoint_file = os.path.join(
                "../output/{}.chk".format(os.path.basename(config_file).split(".")[0])
            )

        torch.save(
            {
                "transformer": transformer.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            },
            chkpoint_file,
        )
