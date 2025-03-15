import math
from idlelib.pyparse import trans

from sympy import transpose
from torch import nn
import torch
import tools

class DotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,Q,K,V,valid_len):
        w = Q.shape[-1]
        S = torch.bmm(Q,K.transpose(1,2))/math.sqrt(w)
        Attention_weights = tools.masked_softmax(S,valid_len)
        return torch.bmm(Attention_weights,V)

class MultiHeadAttention(nn.Module):
    def __init__(self,hidden,heads,bias=False):
        super().__init__()
        self.heads = heads
        self.Wq = nn.LazyLinear(hidden,bias)
        self.Wk = nn.LazyLinear(hidden,bias)
        self.Wv = nn.LazyLinear(hidden,bias)
        self.attention = DotProductAttention()
        self.Wo = nn.LazyLinear(hidden,bias)
    def forward(self,X,valid_lens=None):
        Q = self.transpose_qkv(self.Wq(X))
        K = self.transpose_qkv(self.Wk(X))
        V = self.transpose_qkv(self.Wv(X))

        if valid_lens is not None:
            # On axis 0, copy the first item (scalar or vector) for num_heads
            # times, then copy the next item, and so on
            valid_lens = torch.repeat_interleave(
                valid_lens, repeats=self.num_heads, dim=0)
        O = self.transpose_output(self.attention(Q,K,V,valid_lens))
        Output = self.Wo(O)
        
        return Output

    def transpose_qkv(self, X):
        """Transposition for parallel computation of multiple attention heads.

        Defined in :numref:`sec_multihead-attention`"""
        # Shape of input X: (batch_size, no. of queries or key-value pairs,
        # num_hiddens). Shape of output X: (batch_size, no. of queries or
        # key-value pairs, num_heads, num_hiddens / num_heads)
        X = X.reshape(X.shape[0], X.shape[1], self.num_heads, -1)
        # Shape of output X: (batch_size, num_heads, no. of queries or key-value
        # pairs, num_hiddens / num_heads)
        X = X.permute(0, 2, 1, 3)
        # Shape of output: (batch_size * num_heads, no. of queries or key-value
        # pairs, num_hiddens / num_heads)
        return X.reshape(-1, X.shape[2], X.shape[3])

    def transpose_output(self, X):
        """Reverse the operation of transpose_qkv.

        Defined in :numref:`sec_multihead-attention`"""
        X = X.reshape(-1, self.num_heads, X.shape[1], X.shape[2])
        X = X.permute(0, 2, 1, 3)
        return X.reshape(X.shape[0], X.shape[1], -1)