import torch
import torch.nn as nn
from layers import Encoder, Decoder


class transformer(nn.Module):

    def __init__(self,
                 src_vocab_size,
                 tgt_vocab_size,
                 d_model,
                 n_head,
                 d_fnn,
                 n_enc_layers,
                 n_dec_layers,
                 pad_idx,
                 device,
                 max_len):
        super(transformer, self).__init__()

        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.d_model = d_model
        self.d_fnn = d_fnn
        self.device = device
        self.encoder = Encoder(src_vocab_size,
                               d_model,
                               n_head,
                               d_fnn,
                               n_enc_layers,
                               pad_idx,
                               device,
                               max_len)
        self.decoder = Decoder(tgt_vocab_size,
                               d_model,
                               n_head,
                               d_fnn,
                               n_dec_layers,
                               pad_idx,
                               device,
                               max_len)

        self.fnn = nn.Linear(d_model, tgt_vocab_size, bias=False)

    def forward(self, src_input, tgt_input):
        """
        :param src_input: 编码器输入，shape:[batch_size, src_length]
        :param tgt_input: 解码器输入，shape:[batch_size, tgt_length]
        :return:
        """

        enc_outputs = self.encoder(src_input)
        dec_outputs = self.decoder(tgt_input, enc_outputs, src_input)
        logits = self.fnn(dec_outputs)

        return logits.view(-1, logits.size(-1))



