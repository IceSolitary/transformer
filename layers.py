import torch.nn as nn
import torch
import numpy as np
import math
from utils import get_pad_mask, get_subsequence_mask


class EmbeddingLayer(nn.Module):

    def __init__(self, vocab_size, d_model):

        super(EmbeddingLayer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len, device, dropout=0.1):
        """
        :param d_model: 编码的维度
        :param dropout: 辍学率
        :param max_len: 最大句长
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        self.pe = torch.zeros(max_len, d_model).to(device)
        position = torch.arange(0, max_len, step = 1, dtype=torch.float).unsqueeze(1)
        div_sin = torch.exp((torch.arange(0., d_model, 2).float()/d_model)*
                        - math.log(10000.0))
        div_cos = torch.exp((torch.arange(1., d_model, 2).float()/d_model)*
                        - math.log(10000.0))
        self.pe[:, 0::2] = torch.sin(position * div_sin)
        self.pe[:, 1::2] = torch.cos(position * div_cos)

        self.pe = self.pe .unsqueeze(0)
        self.register_buffer('my_pe', self.pe)

    def forward(self, input):

        """
        :param input: 输入序列 shape:[batch_size, seq_length, d_model]
        :return: output: 位置编码与词向量编码的和 shape:[batch_size, seq_length, d_model]
        """


        batch_size, seq_length, d_model = input.size()

        input = input + self.pe[:, :input.size(1), :].expand(batch_size, seq_length, d_model)
        return self.dropout(input)


class ScaledDotProductAttention(nn.Module):

    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, q, k, v, mask):
        """
        :param q: query vector, shape:[batch_size, n_head, seq_length_q, d_k]
        :param k: key vector， shape:[batch_size, n_head, seq_length_k, d_k]
        :param v: value vector, shape:[batch_size, n_head, seq_length_k, d_v]
        :param mask: attention mask, shape:[batch_size, n_head, seq_length, seq_length]
        :return: attention scores
        """
        d_k = q.size(-1)

        scores = torch.matmul(q, k.transpose(-1, -2)) / np.sqrt(d_k)

        if mask is not None:
            mask = mask.bool()
            scores.masked_fill_(mask, -1e9)

        attention = nn.functional.softmax(scores, dim=-1)

        attention_scores = attention.matmul(v)

        return attention_scores


class MultiHeadAttentionLayer(nn.Module):

    def __init__(self, n_head, d_model, dropout=0.1):
        super(MultiHeadAttentionLayer, self).__init__()
        self.n_head = n_head
        self.d_k = self.d_v = d_model//n_head
        self.W_Q = nn.Linear(d_model, n_head*self.d_k, bias=False)
        self.W_K = nn.Linear(d_model, n_head*self.d_k, bias=False)
        self.W_V = nn.Linear(d_model, n_head*self.d_v, bias=False)
        self.W_O = nn.Linear(n_head*self.d_v, d_model, bias=False)
        self.dropout = nn.Dropout(p=dropout)
        self.layerNorm = nn.LayerNorm(d_model)

    def forward(self, input_Q, input_K, input_V, mask):
        '''
        :param input_Q: 输入的query 向量， shape:[batch_size, seq_length_q, d_model]
        :param input_K: 输入的key 向量， shape:[batch_size, seq_length_k, d_model]
        :param input_V: 输入的value向量， shape:[batch_size, seq_length, d_model]
        :param mask: 注意力掩码， shape:[batch_size, seq_length_q, seq_length_v]
        :return: output_normed 单个多头注意力模块的输出
        '''

        batch_size = input_Q.size(0)
        res = input_Q

        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        K = self.W_K(input_K).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        V = self.W_V(input_V).view(batch_size, -1, self.n_head, self.d_v).transpose(1, 2)

        multi_head_mask = mask.unsqueeze(1).repeat(1, self.n_head, 1, 1)

        attention_scores = ScaledDotProductAttention()(Q, K, V, multi_head_mask)

        attention_scores = attention_scores.transpose(1, 2).reshape(batch_size, -1, self.n_head*self.d_v)

        output = self.W_O(attention_scores)

        output_normed = self.layerNorm(output + res)

        return output_normed


class PositionWiseFFN(nn.Module):

    def __init__(self, d_model, d_ffn):
        super(PositionWiseFFN, self).__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn, bias=False),
            nn.ReLU(),
            nn.Linear(d_ffn, d_model, bias=False)
        )
        self.layerNorm = nn.LayerNorm(d_model)

    def forward(self, inputs):
        """
        :param inputs: 多头注意力的输出， shape:[batch_size, seq_length, d_model]
        :return: output_normed 层归一化后的ffn输出， shape:[batch_size, seq_length, d_model]
        """
        res = inputs
        output = self.ffn(inputs)
        output_normed = self.layerNorm(output + res)
        return output_normed


class EncoderLayer(nn.Module):

    def __init__(self, d_model, n_head, d_ffn):
        super(EncoderLayer, self).__init__()
        self.multiHeadAttention = MultiHeadAttentionLayer(n_head=n_head, d_model=d_model)
        self.ffn = PositionWiseFFN(d_model=d_model, d_ffn=d_ffn)

    def forward(self, enc_inputs, enc_mask):
        """
        :param enc_inputs: 编码器的输入， shape:[batch_size, src_length]
        :param enc_mask: 输入的注意力的掩码，shape:[batch_size, src_length, src_length]
        :return: enc_outputs:单层编码器的输出, shape:[batch_size, src_length, d_model]
        """

        enc_outputs = self.multiHeadAttention(enc_inputs,enc_inputs,enc_inputs,enc_mask)

        enc_outputs = self.ffn(enc_outputs)

        return enc_outputs


class DecoderLayer(nn.Module):

    def __init__(self, d_model, n_head, d_ffn):
        super(DecoderLayer, self).__init__()
        self.maskedMultiHeadAttention = MultiHeadAttentionLayer(n_head=n_head, d_model=d_model)
        self.multiHeadAttention = MultiHeadAttentionLayer(n_head=n_head, d_model=d_model)
        self.ffn = PositionWiseFFN(d_model=d_model, d_ffn=d_ffn)

    def forward(self, enc_outputs, dec_inputs, dec_mask, dec_enc_mask):
        """
        :param enc_outputs: 编码器的输出， shape:[batch_size, src_length, d_model]
        :param dec_inputs: 解码器的输入， shape:[batch_size, tgt_length, d_model]
        :param dec_mask: 解码器的注意力掩码， shape:[batch_size, tgt_length, tgt_length]
        :param dec_enc_mask: 交叉注意掩码， shape:[batch_size, tgt_length, src_length]
        :return: dec_outputs: 解码器的输出， shape:[batch_size, tgt_length, d_model]
        """

        dec_outputs = self.maskedMultiHeadAttention(dec_inputs, dec_inputs, dec_inputs, dec_mask)
        dec_outputs = self.multiHeadAttention(dec_outputs, enc_outputs, enc_outputs, dec_enc_mask)
        dec_outputs = self.ffn(dec_outputs)

        return  dec_outputs


class Encoder(nn.Module):

    def __init__(self,
                 vocab_size,
                 d_model,
                 n_head,
                 d_fnn,
                 n_enc_layers,
                 pad_idx,
                 device,
                 max_len):
        super(Encoder, self).__init__()
        self.pad_idx = pad_idx
        self.pos_embedding = PositionalEncoding(d_model=d_model, max_len=max_len, device=device)
        self.seq_embedding = EmbeddingLayer(vocab_size=vocab_size, d_model=d_model)
        self.encoderLayers = nn.ModuleList(
            [EncoderLayer(d_model=d_model, n_head=n_head, d_ffn=d_fnn) for i in range(n_enc_layers)]
        )

    def forward(self, enc_inputs):
        """
        :param enc_inputs: 编码器的输入，shape[batch_size, src_length]
        :return:
        """
        enc_embedded = self.seq_embedding(enc_inputs)
        enc_pos_embedded = self.pos_embedding(enc_embedded)

        enc_outputs = enc_pos_embedded

        enc_mask = get_pad_mask(enc_inputs, enc_inputs, pad_idx=self.pad_idx)

        for layer in self.encoderLayers:
            enc_outputs = layer(enc_outputs, enc_mask)

        return enc_outputs


class Decoder(nn.Module):

    def __init__(self,
                 vocab_size,
                 d_model,
                 n_head,
                 d_fnn,
                 n_dec_layers,
                 pad_idx,
                 device,
                 max_len):
        super(Decoder, self).__init__()
        self.pad_idx = pad_idx
        self.device = device
        self.pos_embedding = PositionalEncoding(d_model=d_model, max_len=max_len, device=device)
        self.seq_embedding = EmbeddingLayer(vocab_size=vocab_size, d_model=d_model)
        self.decoderLayers = nn.ModuleList(
            [DecoderLayer(d_model=d_model, n_head=n_head, d_ffn=d_fnn) for i in range(n_dec_layers)]
        )

    def forward(self, dec_inputs, enc_outputs, enc_inputs):
        """
        :param enc_inputs: 编码器输入， shape[batch_size, src_length]
        :param dec_inputs: 解码器的输入，shape[batch_size, tgt_length]
        :param enc_outputs: 编码器的输出，shape[batch_size, src_length, d_model]
        :return:
        """
        dec_embedded = self.seq_embedding(dec_inputs)
        dec_pos_embedded = self.pos_embedding(dec_embedded)

        dec_outputs = dec_pos_embedded

        dec_mask = get_pad_mask(dec_inputs, dec_inputs, pad_idx=self.pad_idx)
        dec_enc_mask = get_pad_mask(enc_inputs, dec_inputs, pad_idx=self.pad_idx)
        dec_subsequence_mask = get_subsequence_mask(dec_inputs).to(self.device)
        dec_mask_ = dec_subsequence_mask & dec_mask

        for layer in self.decoderLayers:
            dec_outputs = layer(enc_outputs, dec_outputs, dec_mask_, dec_enc_mask)

        return dec_outputs











