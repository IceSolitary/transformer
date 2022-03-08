import torch
import numpy as np


def get_pad_mask(input_k, input_q, pad_idx):
    """
    :param input_k: 源向量， shape:[batch_size, seq_length_k]
    :param input_q: 目标向量，shape:[batch_size, seq_length_q]
    :return: 掩码， shape:[batch_size, seq_length_q, seq_length_k]

    注意力掩码表示, 因为在qk相乘后 shape变为[batch_size,seq_length_q,seq_length_k],然后送入softmax按行计算，每行表示
    query中的词对key中的每个词的关注程度，本身query中的pad也应该被mask的，此处直接扩展而没有mask对最终结果影响不大，query中
    每个表示pad的行是稀疏向量，表示pad对其他词的关注程度，本身较大无意义。
    """
    batch_size, length_k = input_k.size()
    batch_size, length_q = input_q.size()

    mask = input_k.data.eq(pad_idx).unsqueeze(1).expand(batch_size, length_q, length_k)

    return mask


def get_subsequence_mask(inputs):
    """
    :param inputs: 输入向量， shape[batch_size, seq_length]
    :return: seq_mask: 序列掩码，用于避免解码时模型得到预测词之后的信息， shape[batch_size, seq_length, seq_length]
    """

    batch_size, seq_length = inputs.size()
    seq_mask = np.triu(np.ones([batch_size, seq_length, seq_length]), k=1)
    seq_mask = torch.from_numpy(seq_mask).byte()

    return seq_mask


def greedy_decoder(model, enc_input, start_symbol, end_symbol, max_len, device):
    """

    :param model: 模型
    :param enc_input: 源语言序列，shape:[1,src_length]
    :param start_symbol: 开始标记，int
    :param end_symbol: 结束标记，int
    :param max_len: 最大长度，int
    :return: dec_input: 预测的句子，shape:[1, tgt_length]
    """
    dec_input = torch.tensor(start_symbol).to(device)
    dec_input = dec_input.unsqueeze(0).unsqueeze(0)
    is_end = True
    while is_end:
        logits = model(enc_input, dec_input)
        pre = torch.max(logits[-1, :], dim=-1)[1].unsqueeze(0).unsqueeze(0)
        dec_input = torch.cat([dec_input, pre], dim=-1).to(device)
        pre_token = pre[0]

        if pre_token == end_symbol or len(dec_input[0]) > max_len:
            is_end = False

    return dec_input


if __name__ == '__main__':
    input_q = torch.tensor([[4,5,6,7,1,1,1,1]])
    input_k = torch.tensor([[5,6,5,1,1]])

    mask = get_pad_mask(input_k, input_q, 1)
    seq_mask = get_subsequence_mask(input_k)
    mask2 = get_pad_mask(input_k,input_k, 1)
    print(mask)
    print(mask2)
    print(seq_mask&mask2)
    print(seq_mask)



