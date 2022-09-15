# -*- coding: utf-8 -*-
"""translation_transformer.ipynb
 Nonce generation using transformers
"""

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline

"""Nonce generation with Bert model from Hugginface
https://huggingface.co/transformers/model_doc/bert.html 
======================================================
"""
import os
import numpy as np
from typing import Iterable, List
import json
from transformers import BertModel, BertConfig

config_file = "../config/bert_default.json"
with open(config_file, "r") as f:
    config = json.load(f)


# Define special symbols and indices
PAD_IDX, CLS, SEP = 0, 101, 102
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
TOKEN_SIZE = 105
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
            end = tstart + config["COUNT"]
        else:  # valid train end -> start + 63
            start = tstart + config["COUNT"] + 1
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
                inlist.append(
                    # int(input[:4], 16) + TOKEN_SIZE
                    0
                )  # adding 4 as 4 tokens are taken
                input = input[4:]

            inputs.append(inlist)

            while target:
                outlist.append(
                    int(target[:2], 16) + TOKEN_SIZE
                )  # adding 4 as 4 tokens are taken
                target = target[2:]

            targets.append(outlist)
            if i == end:
                break

        return inputs, targets


"""
# tokenizer structure
{'input_ids': tensor([[  101,  1046,  2232, 10023, 10023,  2243,   102,     0,     0,     0],
        [  101,  1046,  2243,  3501,  2232,  2243,  3501,  2232,  2243,   102]]), 'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])}
[CLS] 101, [SEP] 102
"""
# function to add CLS/SEP and create tensor for input sequence indices
def tensor_transform(token_ids: List[int]):
    return np.concatenate((np.array([CLS]), np.array(token_ids), np.array([SEP])))


def pad_tgt(tgt, length):
    pad_len = length - len(tgt)
    return np.concatenate((np.array(tgt), np.zeros(pad_len, dtype=np.int)))


# function to collate batch in to tokenizer struct
def collate_fn(batch):
    src_batch, tgt_batch, mask, type = [], [], [], []
    token = {}

    for src_sample, tgt_sample in batch:
        src_batch.append(tensor_transform(src_sample))
        tgt = tensor_transform(tgt_sample)
        tgt = pad_tgt(tgt, len(src_batch[0]))
        tgt_batch.append(tgt)
        mask.append(np.ones((len(src_batch[0])), dtype=np.int))
        type.append(np.zeros((len(src_batch[0])), dtype=np.int))

    token["input_ids"] = torch.tensor(src_batch)
    token["token_type_ids"] = torch.tensor(type)
    token["attention_mask"] = torch.tensor(mask)
    tgt_batch = torch.tensor(tgt_batch)

    return token, tgt_batch


# Seq2Seq Network
class Seq2SeqTransformer(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
    ):
        super(Seq2SeqTransformer, self).__init__()
        # conf = BertConfig(vocab_size=src_vocab_size)
        conf = BertConfig()
        emb_size = conf.hidden_size
        self.transformer = BertModel(conf)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)

    def forward(
        self,
        src: Tensor,
    ):
        outs = self.transformer(**src)
        return self.generator(outs.last_hidden_state)


"""Let's now define the parameters of our model and instantiate the same. Below, we also 
define our loss function which is the cross-entropy loss and the optmizer used for training.
"""

torch.manual_seed(0)

SRC_VOCAB_SIZE = 65535 + 105  # byte 255 + 4 tokens
TGT_VOCAB_SIZE = 255 + 105
BATCH_SIZE = 1
EVAL_BATCH_SIZE = 64

transformer = Seq2SeqTransformer(
    SRC_VOCAB_SIZE,
    TGT_VOCAB_SIZE,
)
"""
for p in transformer.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)
"""
transformer = transformer.to(DEVICE)

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
loss_fn = loss_fn.to(DEVICE)

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
        src["input_ids"] = src["input_ids"].to(DEVICE)
        src["token_type_ids"] = src["token_type_ids"].to(DEVICE)
        src["attention_mask"] = src["attention_mask"].to(DEVICE)
        tgt = tgt.to(DEVICE)

        logits = model(src)

        optimizer.zero_grad()

        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1))
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
