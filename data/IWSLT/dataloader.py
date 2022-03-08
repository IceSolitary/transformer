# For data loading.
import os

import torch
import numpy as np
from torch.utils.data import Dataset
from torchtext.legacy import data, datasets
from torchtext.legacy.data import BucketIterator


def getData(max_len, path):
    path = os.getcwd() + path + "\.data"
    if True:
        import spacy
        spacy_de = spacy.load('de_core_news_sm')
        spacy_en = spacy.load('en_core_web_sm')

        def tokenize_de(text):
            return [tok.text for tok in spacy_de.tokenizer(text)]

        def tokenize_en(text):
            return [tok.text for tok in spacy_en.tokenizer(text)]

        BOS_WORD = '<bos>'
        EOS_WORD = '<eos>'
        BLANK_WORD = "<pad>"
        SRC = data.Field(tokenize=tokenize_de, pad_token=BLANK_WORD, batch_first=True)
        TGT = data.Field(tokenize=tokenize_en, init_token=BOS_WORD,
                         eos_token=EOS_WORD, pad_token=BLANK_WORD, batch_first=True)

        MAX_LEN = max_len
        train, val, test = datasets.IWSLT.splits(
            exts=('.de', '.en'), fields=(SRC, TGT), root=path,
            filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and
                                  len(vars(x)['trg']) <= MAX_LEN)
        MIN_FREQ = 2
        SRC.build_vocab(train.src, min_freq=MIN_FREQ)
        TGT.build_vocab(train.trg, min_freq=MIN_FREQ)

    return train, val, test, SRC, TGT


if __name__ == '__main__':
    train, val, test, SRC, TGT = getData(50, "")

    train_iterator = BucketIterator(train, batch_size=32, shuffle=True,
                                    device=torch.device("cuda"))




