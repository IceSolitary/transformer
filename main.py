import time
import os

import torch
import torch.nn as nn
import torch.optim as optim
from data.IWSLT.dataloader import getData
from torchtext.legacy.data import BucketIterator
from model import transformer
import torch.nn.functional as F
from tqdm import tqdm
from utils import greedy_decoder
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def main(data_path=r"/data/IWSLT",
         batch_size=32,
         max_len=30,
         epochs=20,
         learning_rate=0.001,
         d_model=32,
         n_head=8,
         d_fnn=32,
         n_enc_layers=2,
         n_dec_layers=2,
         max_gradient_norm=5.0
         ):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(10*"-", "preparing data", "-"*10)
    train, val, test, SRC, TGT = getData(max_len, path=data_path)
    src_vocab = SRC.vocab
    tgt_vocab = TGT.vocab

    train_iterator = BucketIterator(train, batch_size=batch_size, shuffle=True,
                                    device=device)
    val_iterator = BucketIterator(val, batch_size=1, shuffle=False,
                                  device=device)

    test_iterator = BucketIterator(test, batch_size=1, shuffle=False,
                                   device=device)

    pad_idx = SRC.vocab.stoi["<pad>"]

    model = transformer(src_vocab_size=len(src_vocab),
                        tgt_vocab_size=len(tgt_vocab),
                        d_model=d_model,
                        n_head=n_head,
                        d_fnn=d_fnn,
                        n_enc_layers=n_enc_layers,
                        n_dec_layers=n_dec_layers,
                        max_len=max_len + 2,
                        pad_idx=pad_idx,
                        device=device).to(device)

    loss_function = nn.CrossEntropyLoss(ignore_index=1)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)

    # warmup_steps = 4000
    # lr_lambda = lambda epoch: min(1 / math.sqrt(epoch * len(train_iterator)),
    #                               epoch * len(train_iterator) * (warmup_steps ** -1.5))
    # scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    epoch_train_losses = []
    epoch_val_losses = []
    epoch_val_bleu_scores = []
    for epoch in range(epochs):

        start_time = time.time()
        running_train_loss = 0.0
        running_val_loss = 0.0
        running_bleu = 0.0

        # train
        # model.train()
        # train_batch_iterator = tqdm(train_iterator)
        # for batch_idx, batch in enumerate(train_batch_iterator):
        #     optimizer.zero_grad()
        #     src = batch.src.to(device)
        #     tgt = batch.trg.to(device)
        #
        #     tgt_input = tgt[:, :-1].to(device)
        #     tgt_output = tgt[:, 1:].to(device)
        #
        #     logits = model(src, tgt_input)
        #
        #     loss = loss_function(logits, tgt_output.view(-1))
        #     running_train_loss += loss.item()
        #
        #     loss.backward()
        #
        #     nn.utils.clip_grad_norm_(model.parameters(), max_gradient_norm)
        #     optimizer.step()
        #     # scheduler.step()
        #
        train_time = time.time()
        #
        # epoch_train_loss = running_train_loss / len(train_iterator)
        # epoch_train_losses.append(epoch_train_loss)
        # print("-> Training time: {:.4f}s, loss = {:.4f}"
        #       .format(train_time-start_time, epoch_train_loss))

        # valid
        model.eval()
        val_batch_iterator = tqdm(val_iterator)
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_batch_iterator):
                optimizer.zero_grad()
                src = batch.src.to(device)
                tgt = batch.trg.to(device)

                tgt_input = tgt[:, :-1].to(device)
                tgt_output = tgt[:, 1:].to(device)

                logits = model(src, tgt_input)
                loss = loss_function(logits, tgt_output.view(-1))
                running_val_loss += loss.item()

                pre = greedy_decoder(model, src,
                                     start_symbol=TGT.vocab.stoi["<bos>"], end_symbol=TGT.vocab.stoi["<eos>"],
                                     max_len=max_len, device=device)
                pre = pre.squeeze(0)
                tgt = tgt.cpu().numpy()
                pre = pre.cpu().numpy()

                bleu_score = sentence_bleu(tgt, pre,
                                           weights=(0.25, 0.25, 0.25, 0.25))
                running_bleu += bleu_score

            valid_time = time.time()

            epoch_val_loss = running_val_loss / len(val_iterator)
            epoch_val_bleu = running_bleu / len(val_iterator)
            epoch_val_losses.append(epoch_val_loss)
            epoch_val_bleu_scores.append(epoch_val_bleu)
            print("-> Validating time: {:.4f}s, loss = {:.4f}, bleu-score: {:.4f}"
                  .format(valid_time - train_time, epoch_val_loss, epoch_val_bleu))


if __name__ == '__main__':
    main()




