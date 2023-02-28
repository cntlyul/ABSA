# -*- coding: utf-8 -*-
# file: attention.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, embed_dim, hidden_dim=None, out_dim=None, n_head=1, score_function='dot_product', dropout=0):
        ''' Attention Mechanism
        :param embed_dim:
        :param hidden_dim:
        :param out_dim:
        :param n_head: num of head (Multi-Head Attention)
        :param score_function: scaled_dot_product / mlp (concat) / bi_linear (general dot)
        :return (?, q_len, out_dim,)
        '''
        super(Attention, self).__init__()
        if hidden_dim is None:
            hidden_dim = embed_dim // n_head # 隐藏层维度，默认为词嵌入维度除以注意力头数
        if out_dim is None:
            out_dim = embed_dim  # 输出维度
        self.embed_dim = embed_dim # 词嵌入维度
        self.hidden_dim = hidden_dim #隐藏层维度
        self.n_head = n_head # 多头注意力机制头数
        self.score_function = score_function # 得分函数，没太懂
        self.w_k = nn.Linear(embed_dim, n_head * hidden_dim) # 注意力公式中的K [e_dim,n_head * hidden_dim]
        self.w_q = nn.Linear(embed_dim, n_head * hidden_dim) # 注意力公式中的Q [e_dim,n_head * hidden_dim]
        self.proj = nn.Linear(n_head * hidden_dim, out_dim) # 最后输出的概率用一个全连接层 [n_head * hidden_dim,e_dim]
        self.dropout = nn.Dropout(dropout)
        if score_function == 'mlp':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim*2))
        elif self.score_function == 'bi_linear':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        else:  # dot_product / scaled_dot_product
            self.register_parameter('weight', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.hidden_dim)
        if self.weight is not None:
            self.weight.data.uniform_(-stdv, stdv)

    def forward(self, k, q):
        if len(q.shape) == 2:  # q_len missing
            q = torch.unsqueeze(q, dim=1) # [src_len,1,batch_size]
        if len(k.shape) == 2:  # k_len missing
            k = torch.unsqueeze(k, dim=1) # [src_len,1,batch_size]
        mb_size = k.shape[0]  # src_len
        k_len = k.shape[1] # batch_size
        q_len = q.shape[1] # batch_size
        # k: (?, k_len, embed_dim,)
        # q: (?, q_len, embed_dim,)
        # kx: (n_head*?, k_len, hidden_dim)
        # qx: (n_head*?, q_len, hidden_dim)
        # score: (n_head*?, q_len, k_len,)
        # output: (?, q_len, out_dim,)
        kx = self.w_k(k).view(mb_size, k_len, self.n_head, self.hidden_dim) # [src_len,batch_size,num_head,hidden_dim]
        kx = kx.permute(2, 0, 1, 3).contiguous().view(-1, k_len, self.hidden_dim) # kw [num_head,src_len,batch_size,hidden_dim] -> [num_head*src_len,batch_size,hidden_size]
        qx = self.w_q(q).view(mb_size, q_len, self.n_head, self.hidden_dim) # [src_len,batch_size,num_head,hidden_dim]
        qx = qx.permute(2, 0, 1, 3).contiguous().view(-1, q_len, self.hidden_dim) # kw [num_head,src_len,batch_size,hidden_dim] -> [num_head*src_len,batch_size,hidden_size]
        if self.score_function == 'dot_product':
            kt = kx.permute(0, 2, 1) # [num_head*batch_size,hidden_size,src_len]
            score = torch.bmm(qx, kt) # 高维度矩阵乘法
        elif self.score_function == 'scaled_dot_product':
            kt = kx.permute(0, 2, 1)
            qkt = torch.bmm(qx, kt)
            score = torch.div(qkt, math.sqrt(self.hidden_dim))
        elif self.score_function == 'mlp':
            kxx = torch.unsqueeze(kx, dim=1).expand(-1, q_len, -1, -1) # [num_head*batch_size(e_dim),src_len,hidden_size] -> [num_head*batch_size(e_dim),1,src_len,hidden_size]
            # [num_head*batch_size(e_dim),src_len,hidden_size]
            qxx = torch.unsqueeze(qx, dim=2).expand(-1, -1, k_len, -1)
            kq = torch.cat((kxx, qxx), dim=-1)  # (n_head*?, q_len, k_len, hidden_dim*2)
            # kq = torch.unsqueeze(kx, dim=1) + torch.unsqueeze(qx, dim=2)
            score = F.tanh(torch.matmul(kq, self.weight))
        elif self.score_function == 'bi_linear':
            qw = torch.matmul(qx, self.weight)
            kt = kx.permute(0, 2, 1)
            score = torch.bmm(qw, kt)
        else:
            raise RuntimeError('invalid score_function')
        score = F.softmax(score, dim=-1)
        output = torch.bmm(score, kx)  # (n_head*?, q_len, hidden_dim)
        output = torch.cat(torch.split(output, mb_size, dim=0), dim=-1)  # (?, q_len, n_head*hidden_dim)
        output = self.proj(output)  # (?, q_len, out_dim)
        output = self.dropout(output)
        return output, score


class NoQueryAttention(Attention):
    '''q is a parameter'''
    def __init__(self, embed_dim, hidden_dim=None, out_dim=None, n_head=1, score_function='dot_product', q_len=1, dropout=0):
        super(NoQueryAttention, self).__init__(embed_dim, hidden_dim, out_dim, n_head, score_function, dropout)
        self.q_len = q_len
        self.q = nn.Parameter(torch.Tensor(q_len, embed_dim))
        self.reset_q()

    def reset_q(self):
        stdv = 1. / math.sqrt(self.embed_dim)
        self.q.data.uniform_(-stdv, stdv)

    def forward(self, k, **kwargs):
        mb_size = k.shape[0]
        q = self.q.expand(mb_size, -1, -1)
        return super(NoQueryAttention, self).forward(k, q)


if __name__ == '__main__':
    embed_dim = 32
    n_head = 2
    score_function = 'dot_product'
    dropout = 0
    test =  Attention(embed_dim=embed_dim,n_head=n_head,score_function=score_function,dropout=dropout)
    x = torch.arange(64,dtype=torch.float)
    x = x.reshape(2,32)
    z = torch.tensor(x,dtype=torch.float)
    y = test(z,z)
    print(z.shape)