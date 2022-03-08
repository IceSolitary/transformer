import math
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from utils import greedy_decoder
from nltk.translate.bleu_score import sentence_bleu

# S: Symbol that shows starting of decoding input
# E: Symbol that shows starting of decoding output
# P: Symbol that will fill in blank sequence if current batch data size is short than time steps
from model import transformer

sentences = [
    # enc_input           dec_input         dec_output
    ['ich mochte ein bier P', 'S i want a beer .', 'i want a beer . E'],
    ['ich mochte ein cola P', 'S i want a coke .', 'i want a coke . E']
]

# Padding Should be Zero
src_vocab = {'P': 0, 'ich': 1, 'mochte': 2, 'ein': 3, 'bier': 4, 'cola': 5}
src_vocab_size = len(src_vocab)

tgt_vocab = {'P': 0, 'i': 1, 'want': 2, 'a': 3, 'beer': 4, 'coke': 5, 'S': 6, 'E': 7, '.': 8}
idx2word = {i: w for i, w in enumerate(tgt_vocab)}
tgt_vocab_size = len(tgt_vocab)

src_len = 5  # enc_input max sequence length
tgt_len = 6  # dec_input(=dec_output) max sequence length


def make_data(sentences):
    enc_inputs, dec_inputs, dec_outputs = [], [], []
    for i in range(len(sentences)):
        enc_input = [[src_vocab[n] for n in sentences[i][0].split()]]  # [[1, 2, 3, 4, 0], [1, 2, 3, 5, 0]]
        dec_input = [[tgt_vocab[n] for n in sentences[i][1].split()]]  # [[6, 1, 2, 3, 4, 8], [6, 1, 2, 3, 5, 8]]
        dec_output = [[tgt_vocab[n] for n in sentences[i][2].split()]]  # [[1, 2, 3, 4, 8, 7], [1, 2, 3, 5, 8, 7]]

        enc_inputs.extend(enc_input)
        dec_inputs.extend(dec_input)
        dec_outputs.extend(dec_output)

    return torch.LongTensor(enc_inputs), torch.LongTensor(dec_inputs), torch.LongTensor(dec_outputs)


enc_inputs, dec_inputs, dec_outputs = make_data(sentences)


class MyDataSet(Data.Dataset):
    def __init__(self, enc_inputs, dec_inputs, dec_outputs):
        super(MyDataSet, self).__init__()
        self.enc_inputs = enc_inputs
        self.dec_inputs = dec_inputs
        self.dec_outputs = dec_outputs

    def __len__(self):
        return self.enc_inputs.shape[0]

    def __getitem__(self, idx):
        return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx]


loader = Data.DataLoader(MyDataSet(enc_inputs, dec_inputs, dec_outputs), 2, True)

device = torch.device("cpu")

pad_idx = 0

model = transformer(src_vocab_size=len(src_vocab),
                    tgt_vocab_size=len(tgt_vocab),
                    d_model=32,
                    n_head=8,
                    d_fnn=32,
                    n_enc_layers=6,
                    n_dec_layers=6,
                    max_len=10,
                    pad_idx=pad_idx,
                    device=device).to(device)

loss_function = nn.CrossEntropyLoss(ignore_index=0)

optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9)

epochs = 30
model.train()
for epoch in range(epochs):
    run_loss = 0.0
    for batch in loader:
        enc_inputs, dec_inputs, dec_outputs = batch
        optimizer.zero_grad()
        logits = model(enc_inputs, dec_inputs)
        loss = loss_function(logits, dec_outputs.view(-1))
        loss.backward()
        optimizer.step()
        run_loss += loss.item()

    print(f"epoch{epoch}: loss: {run_loss/len(loader)}")

loader2 = Data.DataLoader(MyDataSet(enc_inputs, dec_inputs, dec_outputs), 1, True)
model.eval()
with torch.no_grad():
    for batch in loader2:
        enc_inputs, dec_inputs, dec_outputs = batch
        print(enc_inputs)
        pre = greedy_decoder(model, enc_inputs,6,7,10,device).squeeze(0)
        dec_outputs_ = torch.cat([torch.tensor([6]), dec_outputs.squeeze(0)])
        dec_outputs_ = dec_outputs_.cpu().numpy()
        pre = pre.cpu().numpy()
        bleu = sentence_bleu([dec_outputs_], pre)
        print(bleu)


